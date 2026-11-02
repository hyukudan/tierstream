// Bounded startup loader shared by CPU fixtures and the CUDA context.
#pragma once
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include "desc.hpp"
#include "source_file.hpp"
#include "xxh64.hpp"
namespace ts {
// Pageable aligned bounce storage for the covering reads of unaligned parts: never an H2D source. Each reader owns one
// (no reader waits for another's).
struct DgBounce {
  char* p; size_t bytes, alignment;
  DgBounce(size_t n, size_t a)
      : p(static_cast<char*>(::operator new(n, std::align_val_t(a)))), bytes(n), alignment(a) {}
  ~DgBounce() { ::operator delete(p, std::align_val_t(alignment)); }
  DgBounce(const DgBounce&) = delete;
  DgBounce& operator=(const DgBounce&) = delete;
};
struct DgLoadUnit {
  int layer = 0, unit = 0;
  int32_t slot = -1;
  char* host = nullptr;
  // A host unit with host == nullptr: its pinned block's index in the loader's DgBlockAlloc, the pointer resolved when
  // the unit is read (only once the block's chunk is allocated).
  int64_t block = -1;
};
struct DgLoadCopy { int32_t slot; size_t part; const char* data; uint64_t bytes; };
struct DgLoadStats {
  uint64_t payload_bytes = 0, read_bytes = 0, read_calls = 0, bounce_bytes = 0, h2d_bytes = 0;
  uint64_t pieces = 0;  // dg_read_unit sub-reads issued (read_calls counts one per source batch of them)
  // read_seconds is wall time through the last completed batch (includes hashing
  // and buffer waits); hash_seconds sums reader CPU wall times. bounce_bytes
  // counts payload routed through fallback storage, excluding alignment padding.
  double read_seconds = 0, hash_seconds = 0;
  // resident_seconds: through the last completed resident batch. alloc_wait_seconds: wall time some reader waited for
  // a pinned block's chunk with no resident batch left to take (0 without a DgBlockAlloc).
  double resident_seconds = 0, alloc_wait_seconds = 0;
};

// The pinned host blocks allocated chunk by chunk on a helper thread while the startup loader already reads the
// resident units (DG_LOAD_OVERLAP_ALLOC). Chunk c holds chunk_blocks[c] blocks of `stride` bytes, the blocks numbered
// in chunk order. alloc(c, bytes) returns chunk c's memory or throws; release frees one chunk. The helper publishes
// chunk c (ready() > c) only after its memory exists; block() resolves a pointer of a published chunk. An alloc error
// stops the helper: failed() turns true, error() holds it and finish() rethrows it. The destructor stops and joins the
// helper and frees every chunk finish() did not hand over.
class DgBlockAlloc {
 public:
  using Alloc = std::function<void*(size_t chunk, uint64_t bytes)>;
  using Release = std::function<void(void*)>;
  DgBlockAlloc(std::vector<uint64_t> chunk_blocks, uint64_t stride, Alloc alloc, Release release);
  ~DgBlockAlloc();
  DgBlockAlloc(const DgBlockAlloc&) = delete;
  DgBlockAlloc& operator=(const DgBlockAlloc&) = delete;
  void start();  // the helper thread (once)
  void cancel();  // no further chunk is allocated
  size_t chunks() const { return counts_.size(); }
  uint64_t total() const;  // blocks
  size_t ready() const { return ready_.load(std::memory_order_acquire); }
  bool failed() const { return failed_.load(std::memory_order_acquire); }
  std::exception_ptr error() const { return failed() ? error_ : nullptr; }
  size_t chunk_of(uint64_t b) const;
  char* block(uint64_t b) const;  // its chunk is published
  // Called under the allocator's lock after every publish, after a failure and when the helper ends (the loader's
  // wake-up; it must not call back into the allocator). nullptr unsubscribes; once that returns no call is running.
  void subscribe(std::function<void()> notify);
  // Joins the helper, appends every allocated chunk and its blocks (in order) to *chunks and *blocks (the caller owns
  // them from then on), then rethrows the helper's error if any.
  void finish(std::vector<void*>* out_chunks, std::vector<char*>* out_blocks);
  double seconds() const { return seconds_; }  // the helper's allocation time (after finish)
  std::chrono::steady_clock::time_point done_at() const { return done_at_; }  // the helper's end (after finish)

 private:
  void run();
  std::vector<uint64_t> counts_, first_;
  uint64_t stride_;
  Alloc alloc_;
  Release release_;
  std::vector<void*> ptrs_;
  std::atomic<size_t> ready_{0};
  std::atomic<bool> failed_{false}, cancel_{false};
  std::exception_ptr error_;
  std::mutex mu_;
  std::function<void()> notify_;
  std::thread thread_;
  bool handed_ = false;
  double seconds_ = 0;
  std::chrono::steady_clock::time_point done_at_{};
};
// Buffers are caller-owned. recycle waits for the previous transfer from this buffer;
// submit must enqueue all copies before returning. Workers never publish placement.
// Optional context sources keep one shared MFile session alive across all worker adapters.
// blocks (the overlap): host units with host == nullptr land in their block, read only once its chunk is published;
// resident batches are taken first, a reader waits for a chunk only when no resident batch is left to take. A helper
// error stops the readers and is rethrown.
DgLoadStats dg_load_pipeline(const Desc& d, const std::vector<DgLoadUnit>& units,
    const std::vector<char*>& buffers, uint64_t buffer_bytes, uint32_t readers,
    bool direct, bool verify, const std::function<void(size_t)>& recycle,
    const std::function<void(size_t, const std::vector<DgLoadCopy>&)>& submit,
    const std::vector<std::unique_ptr<SourceFile>>* sources = nullptr, DgBlockAlloc* blocks = nullptr);

// One unit's bytes (its parts back to back) into dst, from the open sources of d. A part read in place (no direct I/O,
// or aligned offset, bytes and destination) lands directly; a part whose destination has its file offset's
// misalignment reads its aligned middle in place and only its head and tail blocks through `bounce`; any other goes
// through `bounce` (bounce_bytes, aligned) whole. `bounce` is the caller's own: concurrent readers pass their own
// buffers (nothing here is shared between calls but the open sources). parallel: the in-place parts of one
// multi-location session (MFileSource) are issued as one MFile::preadv, so they run on the drives' pools together;
// the bytes are the same. subread > 0 (DG_UNIT_SUBREAD_KIB): every in-place part, of any source, is cut into pieces
// of subread bytes (never across a session's chunk), and all pieces of the unit on one source are in flight together
// (SourceFile::read_pieces: overlapped reads on a file, one preadv on a session); 0 = one request per part. hash: the
// xxh64 of the unit's bytes into digest (computed once every part landed, in part order).
void dg_read_unit(const Desc& d, const std::vector<std::unique_ptr<SourceFile>>& files, int layer, int unit,
                  void* dst, void* bounce, uint64_t bounce_bytes, bool hash, uint64_t* digest,
                  DgLoadStats* stats = nullptr, bool parallel = true, uint64_t subread = 0);

// One unit read as a sequence of chunks (a landing read that yields its reader between them, dg_landing_chunk): the
// same bytes into the same destination as dg_read_unit, part by part in order, each part cut at file offsets that are
// multiples of the step (chunk rounded down to the source's alignment, at least one alignment unit; on a session that
// re-checks whole stripe chunks as they are read (SourceFile::verify_grid) a multiple of that chunk, at least one, so
// every stripe chunk a part covers whole is still read whole and checked). A chunk never spans two parts. Each chunk
// reads like a part of dg_read_unit: in place, its aligned middle in place and its head / tail sectors through the
// caller's bounce buffer, or bounced whole; subread > 0 cuts its in-place bytes into pieces in flight together. Cut at
// aligned file offsets, the head and tail sectors of an unaligned part are bounced once each, as in one read. hash:
// the xxh64 of the unit's bytes, fed chunk by chunk as each landed (the parts in order: the digest of dg_read_unit).
// parallel batching of several parts on one session (dg_read_unit's preadv) is not used: a chunk is one part's.
struct DgUnitChunks {
  int layer = 0, unit = 0;
  char* dst = nullptr;
  uint64_t chunk = 0;  // bytes, a multiple of 4096
  bool hash = false;
  // progress: the part being read, the bytes of it read, the bytes of the unit read; the digest so far
  size_t part = 0;
  uint64_t at = 0, done = 0;
  Xxh64 digest;
};
// Reads c's next chunk (with `bounce`, the caller's own); returns the unit's bytes still to read (0: complete, its
// digest in *digest when c.hash, else the empty digest). Throws as dg_read_unit (the progress is then undefined).
uint64_t dg_read_unit_chunk(const Desc& d, const std::vector<std::unique_ptr<SourceFile>>& files, DgUnitChunks& c,
                            void* bounce, uint64_t bounce_bytes, uint64_t* digest, DgLoadStats* stats = nullptr,
                            uint64_t subread = 0);

// The three-tier file reads in flight (dg_disk_queue): `threads` workers run the submitted reads, oldest first. wait()
// blocks until that read finished, rethrows its error and returns its own time in ns; every handle is waited for
// exactly once. Destruction drops the reads not started, stops a chunked one at its next chunk and waits for the
// running ones.
// Landing reads (dg_landing_blocks) wait in a second queue: a worker takes one only while no demand read is queued and
// (reserve) fewer than landing_cap landing reads run and one reader stays free of them (at most threads - 1 landing
// reads at once; one thread: a landing read only while nothing else runs); without reserve (dg_landing_chunk
// 0xFFFFFFFF: the scheduler before) fewer than landing_cap reads run, demand and landing together. The pool does not
// know the drives: landing_cap (dg_landing_max_queued) bounds the reads of all of them together. A chunked landing
// read (submit_steps) runs one chunk per step: between two chunks, with a demand read queued that no idle worker takes
// (more queued than idle workers), it yields its worker (back at the FRONT of the landing queue, its progress kept:
// landing_yields) and the worker takes the demand read; a whole landing read (submit_indexed) is never preempted. A wait() for a landing read
// makes it a demand: still queued (never started or part-way) it moves to the back of the demand queue, running it no
// longer yields, and it runs to its end. cancel() drops one in the landing queue (never started, or yielded part-way);
// abort() stops one at its next chunk.
class DgReadPool {
 public:
  explicit DgReadPool(uint32_t threads, uint32_t landing_cap = 1, bool reserve = true);
  ~DgReadPool();
  DgReadPool(const DgReadPool&) = delete;
  DgReadPool& operator=(const DgReadPool&) = delete;
  void* submit(std::function<void()> read);
  // the same, the read given the index of the worker that runs it (0 .. threads - 1; e.g. its own bounce buffer);
  // landing: into the landing queue
  void* submit_indexed(std::function<void(uint32_t)> read, bool landing = false);
  // A read in steps (a chunked landing read): step(worker) reads the next chunk and returns the bytes still to read
  // (0: complete); bytes = the read's total. The steps of one read never run concurrently; each may run on another
  // worker (a worker's own buffers serve one step only).
  void* submit_steps(std::function<uint64_t(uint32_t)> step, uint64_t bytes, bool landing = true);
  // How a landing read ended: aborted (abort() stopped it before its end: its bytes are partial, aborted_bytes not
  // read); the times it yielded its worker to a demand read.
  struct End {
    uint64_t aborted_bytes = 0;
    uint32_t yields = 0;
    bool aborted = false;
  };
  // (`end` is filled before an error is rethrown; an aborted read waited for without `end` throws SS_E_IO: its bytes
  // are partial)
  uint64_t wait(void* handle, End* end = nullptr);
  bool done(void* handle);  // finished (wait would not block)
  // A landing read in the landing queue (never started, or yielded part-way) is removed: true, and the handle is gone
  // (never waited for; nothing writes into its destination any more; `end`: a part-way one's bytes not read and its
  // yields); false: it runs (or finished, or is a demand read): wait for it as usual.
  bool cancel(void* handle, End* end = nullptr) noexcept;
  // A landing read that is not a demand (never waited for) stops: still queued, at once; a chunked one running, at its
  // next chunk boundary (a whole read runs to its end). Then done; wait for it as usual (End::aborted).
  void abort(void* handle) noexcept;
  // Test observation: true while the read is a landing read (a wait has not made it a demand), queued or running.
  bool is_landing(void* handle);
  // Test observation: the reads running, the landing reads among them, the demand and landing reads queued.
  struct Counts {
    uint32_t running = 0, landing_running = 0;
    size_t queued = 0, landing_queued = 0;
  };
  Counts counts();

 private:
  struct Job {
    std::function<void(uint32_t)> read;
    std::function<uint64_t(uint32_t)> step;  // chunked (read empty)
    std::exception_ptr error;
    uint64_t ns = 0, left = 0;  // left: a chunked read's bytes still to read
    uint32_t yields = 0;
    // landing: in the landing queue or running as a landing read (a wait made it a demand: false)
    bool done = false, landing = false, started = false, stop = false, aborted = false;
  };
  void work(uint32_t worker);
  void* push(std::unique_ptr<Job> j);
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  bool landing_ready() const {
    if (!queue_.empty() || landing_.empty()) return false;
    if (!reserve_) return running_ < landing_cap_;
    if (threads_n_ <= 1) return running_ == 0;
    return landing_running_ < landing_cap_ && landing_running_ + 1 < threads_n_;
  }
  std::deque<Job*> queue_, landing_;  // demand reads, landing reads
  std::vector<std::unique_ptr<Job>> jobs_;  // submitted, not waited for yet
  uint32_t running_ = 0, landing_running_ = 0, landing_cap_ = 1, threads_n_ = 1;
  bool stop_ = false, reserve_ = true;
  std::vector<std::thread> threads_;
};
}  // namespace ts
