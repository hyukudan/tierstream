// The runtime: one engine's streaming state on one device (docs/design.md §4).
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "backend.hpp"
#include "desc.hpp"
#include "file_reader.hpp"
#include "plan.hpp"
#include "resident_set.hpp"
#include "strix_stream/ss.h"

namespace ss {

struct RaEntry;

struct Job {
  enum State { Queued = 0, Running = 1, Done = 2, Cancelled = 3, Failed = 4 };
  enum RanAs : int8_t { AsDemand = 0, AsSpec = 1, AsRa = 2 };
  int layer = 0;
  int unit = 0;
  bool spec = false;
  bool from_pinned = false;
  bool host_dst = false;             // dst is host memory (pinned tier fill)
  void* dst = nullptr;              // device address
  const void* host_src = nullptr;   // pinned tier address (from_pinned)
  void* dst_release_ev = nullptr;   // writes to dst wait for it
  void* dst_wb_ev = nullptr;        // exclusive: writes to dst also wait for the write-back of its previous unit
  void* src_wait_ev = nullptr;      // exclusive: host_src is a block whose write-back of this forward lands first
  void* done_ev = nullptr;          // non-unified: recorded on the copy stream after the last piece
  std::atomic<int> state{Queued};
  ss_status err_status = SS_OK;
  std::string err;
  // Host read-ahead: a producer reads the unit's file into an RA buffer; a child copies that buffer into a ring slot.
  bool ra_producer = false;
  bool ra_child = false;
  bool issued_piece = false;          // child: at least one copy out of the buffer was issued
  int8_t ran_as = AsDemand;           // the queue a worker took it from (its running counter)
  std::shared_ptr<RaEntry> ra;        // keeps the entry alive through the worker's last access
};
using JobPtr = std::shared_ptr<Job>;

// One host read-ahead request of a forward (design §8-11). Every field is read and written under Ctx::q_mu_ once the
// entry is published, except buf, which only the engine thread sets before the producer is queued.
struct RaEntry {
  enum St { Waiting, Reading, Ready, Landing, Consumed, Failed, Abandoned };
  St st = Waiting;
  int layer = 0;
  int unit = 0;
  int32_t buf = -1;              // RA buffer index, -1 = none (never held / taken back)
  bool quarantined = false;      // a copy out of buf failed after issuing pieces: buf is never reused
  bool child_queued = false;     // the child was pushed to the demand queue
  JobPtr producer, child;
};

class Ctx {
 public:
  Ctx(const Desc& d, const ss_budget& b, const Plan& p, const ss_options& o);
  ~Ctx();

  void seed(int layer, const int32_t* units, int n);
  void begin(const uint32_t* rows, int n_layers);
  void routes(int layer, const int32_t* ids, int n);
  void routes_dev(int layer, const int32_t* ids_dev, int n, void* stream);
  void routes_all(int layer);
  void prefetch(int layer, const int32_t* ids, int n, bool speculative);
  void prefetch_layer(int layer);
  void acquire(int layer, void* stream, bool wait_all, ss_layer_view* out);
  void group_wait(int layer, int group, void* stream);
  void release(int layer, void* stream);
  void end();
  void check_unit(int layer, int unit);

  const ss_layer_stats& stats(int layer) const { return stats_[size_t(layer)]; }
  ss_layer_stats total() const;
  void stats_reset();
  void set_hook(ss_stats_hook h, void* user) { hook_ = h; hook_user_ = user; }
  int n_layers() const { return int(d_.layers.size()); }
  int layer_units(int layer) const { return d_.layers[size_t(layer)].n_units; }
  // After an allocation failure or an unexpected exception inside a call, the context's bookkeeping may be
  // half-updated (a unit admitted without its read, a job nobody owns): it refuses every further call but close.
  void poison() noexcept { poisoned_ = true; }
  bool poisoned() const { return poisoned_; }
  bool source_direct(int i) const { return files_[size_t(i)]->direct(); }
  int n_sources() const { return int(files_.size()); }
  Json stats_json() const;
  ss_exclusive_stats exclusive_stats() const;
  // Exclusive tier mode, between forwards (tests): every unit in a slot XOR a block XOR only its file, every block
  // owned exactly once (a unit or a free list). Throws SS_E_INVALID naming the first broken rule.
  void check_exclusive() const;
  void host_prefetch(int layer, const int32_t* ids, int n);
  // Tests: the write-back stream waits for this event before its next copies (makes a pending write-back certain).
  void test_gate_writeback(void* ev);
  // Tests: reader threads take no job while held (queued file reads wait; pinned copies are issued at submission).
  void test_hold_readers(bool hold);
  // Tests: one-shot read-ahead fault injection (1 = queueing a read throws, 2 = queueing a landing throws,
  // 3 = the next landing copy fails before any copy, 4 = it fails after its first piece, 5 = the next read fails).
  void test_inject(int what) { test_inject_.store(what); }
  // Tests: fill every free pinned block with `byte` (known stale bytes); host-wait the copy stream.
  void test_fill_free_blocks(int byte);
  void test_sync_copy();

 private:
  enum Loc : int8_t { None = 0, Persistent = 1, Ring = 2 };
  struct RingSlot {
    void* dev = nullptr;
    uint64_t stride = 0;
    int owner = -1;  // layer holding it this forward, -1 = free
    JobPtr job;
  };
  struct Staging {
    void* host = nullptr;
    void* ev = nullptr;  // non-unified: last copy out of this buffer
  };
  struct LayerState {
    std::unique_ptr<ResidentSet> rs;
    void* pool = nullptr;                // persistent slots (capacity x stride)
    void* release_ev = nullptr;          // recorded at ss_release
    void* ready_ev = nullptr;            // copy stream, at ss_end: last persistent copy issued for this layer
    bool ready_wait = false;             // the next acquire waits for ready_ev
    void* table_dev[2] = {nullptr, nullptr};
    uint64_t* table_host[2] = {nullptr, nullptr};
    void* table_ev[2] = {nullptr, nullptr};
    // Per forward.
    std::vector<int8_t> loc;
    std::vector<int32_t> slot;
    std::vector<JobPtr> job;
    std::vector<char> spec_unclaimed;
    std::vector<int32_t> routed;         // distinct routed units this forward
    std::vector<uint32_t> count;         // occurrences per unit in the routes (group order)
    std::vector<int32_t> pending;        // units waiting for a ring slot
    std::vector<int32_t> hits;           // landed before this forward
    std::vector<std::vector<int32_t>> groups;
    std::vector<ss_group> group_view;
    bool routes_given = false, acquired = false, released = false, skip = false;
    int32_t* ids_host = nullptr;         // pinned buffer for device routes
    size_t ids_cap = 0;
    int ids_n = -1;                      // pending device routes (>= 0)
    std::vector<std::shared_ptr<RaEntry>> ra;  // host read-ahead entry per unit this forward (tombstones until end)
    void* ids_ev = nullptr;
    int version = 0;
  };

  void init();
  void cleanup();
  void start_workers();
  void stop_workers();
  void worker();
  void run_job(Job& j);
  void read_unit_host(int layer, int unit, void* dst, bool hash, uint64_t* digest);
  void read_unit_device(Job& j);
  void submit(const JobPtr& j);
  void promote(const JobPtr& j);
  void fail_job(Job& j, ss_status st, const char* msg) noexcept;
  class StagingLease;
  void wait_job(Job& j);
  JobPtr make_job(int layer, int unit, void* dst, bool spec);
  void fetch_into_persistent(int layer, int unit, int slot, bool spec, void* dst_wb_ev = nullptr);
  bool fetch_into_ring(int layer, int unit, bool spec);
  void dispatch_pending();
  void ingest_routes(int layer, const int32_t* ids, int n);
  void ingest_dev_routes(int layer);
  void check_layer(int layer, const char* what) const;
  void check_stream(void* stream, const char* what);
  void* unit_addr(int layer, int unit) const;
  void open_forward();
  void retire_ring(int layer);
  void wait_job_quiet(Job& j);
  void* write_back(int layer, const Admission& a);
  void fetch_admitted(int layer, const Admission& a, bool spec);
  int eviction_cap(int layer, bool speculative) const;
  void free_block(int layer, int unit);
  int block_class(int layer) const;
  // Host read-ahead (all engine-thread unless *_locked: then q_mu_ is held).
  bool ra_on() const { return !ra_host_.empty(); }
  bool ra_eligible(int layer, int unit, bool pending_ok) const;
  void ra_request(int layer, int unit, bool pending_ok);
  void ra_pump();
  bool ra_land(int layer, int unit, const JobPtr& child);
  void ra_take_buf_locked(RaEntry& e);
  void ra_abandon_locked(RaEntry& e);
  void ra_producer_epilogue_locked(Job& j);
  void ra_child_epilogue_locked(Job& j);
  void ra_sweep_locked();
  void ra_drain_and_reconcile();
  void fail_job_locked(Job& j, ss_status st, const char* msg) noexcept;
  void run_ra_child(Job& j);
  int64_t gidx(int layer, int unit) const { return d_.unit_base[size_t(layer)] + unit; }

  Desc d_;
  Plan plan_;
  ss_options o_;
  std::unique_ptr<Backend> be_;
  std::vector<std::unique_ptr<File>> files_;
  std::vector<LayerState> L_;
  std::vector<RingSlot> ring_;
  std::vector<void*> ring_owner_ev_;  // release event of the layer that last read each ring slot
  std::vector<void*> ev_pool_;        // recycled job events
  struct RingRange {
    uint64_t stride;
    size_t first, count, cursor;
    void* base;
  };
  std::vector<RingRange> ring_classes_;
  std::vector<void*> pinned_chunks_;
  std::vector<const void*> pinned_of_;  // per global unit (nullptr when not in the pinned tier); exclusive: its block

  // Exclusive tier mode (design.md §3.4).
  bool excl_ = false;
  void* wb_stream_ = nullptr;               // backend write-back stream (D2H)
  std::vector<uint64_t> block_bytes_;       // per block class: host stride
  std::vector<int> layer_class_;            // per layer: block class, -1 = none
  std::vector<std::vector<void*>> free_blocks_;  // per class, LIFO
  std::vector<void*> spare_chunks_;
  std::vector<void*> wb_pending_;           // per global unit: write-back event of this forward into its block
  std::vector<int64_t> wb_units_;           // units with wb_pending_ set this forward
  std::vector<void*> wb_evs_;               // write-back events of this forward
  void* blocks_ev_ = nullptr;               // copy stream, at ss_end: after every H2D that read a freed block
  void* wb_mark_ev_ = nullptr;              // write-back stream, at ss_end: after the forward's write-backs
  bool wb_gate_ = false;                    // the next write-back waits blocks_ev_ first
  uint64_t n_blocks_ = 0;                   // every block (pinned tier + spares)
  std::vector<uint64_t> spares_of_;         // per class: spare blocks the plan placed (the speculation reserve)
  ss_exclusive_stats xst_{};
  std::vector<Staging> staging_;
  std::vector<size_t> staging_free_;
  std::mutex staging_mu_;
  std::condition_variable staging_cv_;
  std::mutex copy_mu_;  // groups wait + pieces + record on the copy stream

  std::mutex q_mu_;
  std::condition_variable q_cv_;
  std::condition_variable done_cv_;
  std::deque<JobPtr> demand_q_, spec_q_, ra_q_;
  int demand_running_ = 0;
  int ra_running_ = 0;               // unattached read-ahead reads running (capped at max(1, readers - 1))
  bool stop_ = false;
  bool hold_ = false;  // test_hold_readers
  std::atomic<int> test_inject_{0};

  // Host read-ahead buffers (readahead_units; empty = off).
  std::vector<void*> ra_host_;
  std::vector<void*> ra_ev_;          // per buffer: last H2D out of it (dedicated, never pooled)
  std::vector<uint32_t> ra_free_;     // capacity reserved at open: a push never allocates
  uint64_t ra_bytes_ = 0;
  std::deque<std::shared_ptr<RaEntry>> ra_wait_;  // engine thread: requests waiting for a buffer, in call order
  std::vector<JobPtr> ra_jobs_;       // producers of this forward (auxiliary: never reported by ss_end)
  bool ra_frozen_ = false;
  struct RaStats {
    uint64_t requested = 0, read = 0, landed = 0, unused = 0, failed = 0, fallback = 0, bytes = 0;
  } ra_st_;
  std::vector<std::thread> workers_;

  void* fwd_stream_ = nullptr;
  bool fwd_stream_set_ = false;
  bool poisoned_ = false;
  bool in_forward_ = false;
  bool fwd_open_ = false;  // per-forward state reset (prefetch may open the next forward before ss_begin)
  uint64_t forward_ = 0;
  std::vector<JobPtr> fwd_jobs_;
  std::vector<ss_layer_stats> stats_, fwd_stats_;
  ss_stats_hook hook_ = nullptr;
  void* hook_user_ = nullptr;
};

}  // namespace ss
