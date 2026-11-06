#include "mfile.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <map>
#include <thread>

#include "errors.hpp"
#include "json.hpp"
#include "sha256.hpp"
#include "xxh64.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace ts {
namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
std::atomic<uint64_t> sessions_created{0};
uint64_t elapsed(Clock::time_point start) {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
}

// Links resolved (each file is then opened through a BoundDir: resolved once, then bound).
std::string canonical(const std::string& path) {
  std::error_code ec;
  fs::path p = fs::weakly_canonical(fs::absolute(fs::u8path(path)), ec);
  if (ec) p = fs::absolute(fs::u8path(path)).lexically_normal();
#if defined(_WIN32)
  std::wstring w = p.wstring();
  if (!w.empty()) CharLowerBuffW(&w[0], DWORD(w.size()));
  p = fs::path(w);
#endif
  return p.generic_u8string();
}

std::string joined(const std::string& dir, const std::string& name) {
  return (fs::u8path(dir) / fs::u8path(name)).generic_u8string();
}

bool digest_valid(const std::string& s) {
  return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}

std::string read_all(const RawFile& f, uint64_t limit) {
  const std::string& path = f.path();
  auto n = f.size();
  require(n <= limit, SS_E_DESC, "file exceeds size limit: " + path);
  std::string s(size_t(n), '\0');
  require(!n || f.read(&s[0], n, 0) == n, SS_E_IO, "cannot read " + path);
  return s;
}

// A cache file of this user's (not a model file: opened by path).
std::string read_text(const std::string& path, uint64_t limit) { return read_all(RawFile(path, false, false), limit); }

struct Buffer {
  char* p;
  explicit Buffer(size_t n) {
#if defined(_WIN32)
    p = static_cast<char*>(_aligned_malloc(n, MFile::kAlign));
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, MFile::kAlign, n)) raw = nullptr;
    p = static_cast<char*>(raw);
#endif
    require(p != nullptr, SS_E_NOMEM, "stripe verification buffer");
  }
  ~Buffer() {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

class Pool {
 public:
  explicit Pool(uint32_t n) {
    try {
      for (uint32_t i = 0; i < n; ++i) threads_.emplace_back([this] { worker(); });
    } catch (...) { stop(); throw; }
  }
  ~Pool() { stop(); }
  std::future<void> submit(std::function<void()> fn) {
    auto task = std::make_shared<std::packaged_task<void()>>(std::move(fn));
    auto future = task->get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      queue_.push_back([task] { (*task)(); });
    }
    cv_.notify_one();
    return future;
  }
 private:
  void stop() {
    { std::lock_guard<std::mutex> lock(mu_); stopping_ = true; }
    cv_.notify_all();
    for (auto& t : threads_) if (t.joinable()) t.join();
  }
  void worker() {
    for (;;) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (queue_.empty()) return;
        fn = std::move(queue_.front());
        queue_.pop_front();
      }
      fn();
    }
  }
  std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::function<void()>> queue_;
  std::vector<std::thread> threads_;
  bool stopping_ = false;
};

std::atomic<uint64_t> pools_created{0};
std::shared_ptr<Pool> drive_pool(uint64_t volume, uint32_t readers) {
  static std::mutex mu;
  static std::map<std::pair<uint64_t, uint32_t>, std::weak_ptr<Pool>> pools;
  std::lock_guard<std::mutex> lock(mu);
  auto& slot = pools[{volume, readers}];
  auto pool = slot.lock();
  if (!pool) {
    pool = std::make_shared<Pool>(readers);
    slot = pool;
    pools_created.fetch_add(1);
  }
  return pool;
}

void finish(std::vector<std::future<void>>& jobs, std::exception_ptr error = {}) {
  for (auto& j : jobs) {
    try { j.get(); } catch (...) { if (!error) error = std::current_exception(); }
  }
  if (error) std::rethrow_exception(error);
}

// STRIX_MFILE_ROUTE_LEGACY, read at each open (any case): 1 / true / on restores the routing before balanced routing;
// 0 / false / off, unset or empty keeps balanced routing; anything else refuses the open (SS_E_INVALID).
bool legacy_routing_env() {
  const char* v = std::getenv("STRIX_MFILE_ROUTE_LEGACY");
  if (!v || !*v) return false;
  std::string s = v;
  for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "1" || s == "true" || s == "on") return true;
  require(s == "0" || s == "false" || s == "off", SS_E_INVALID,
          "mfile: invalid STRIX_MFILE_ROUTE_LEGACY (use 0, 1, false, true, off or on, in any case)");
  return false;
}

int strictness(int mode) { return mode == SS_MFILE_VERIFY_ALWAYS ? 2 : mode == SS_MFILE_VERIFY_CACHED ? 1 : 0; }

std::string id_text(const FileIdentity& id) {
  return std::to_string(id.volume) + ":" + std::to_string(id.index) + ":" + std::to_string(id.size) + ":" +
         std::to_string(id.mtime) + ":" + std::to_string(id.ctime);
}

void cache_write(const std::string& path, const std::string& text) {
  // RawFile's atomic writer uses a fixed .tmp name. Give concurrent processes different staging names, then publish.
  static std::atomic<uint64_t> serial{0};
#if defined(_WIN32)
  uint64_t pid = GetCurrentProcessId();
#else
  uint64_t pid = uint64_t(getpid());
#endif
  std::string staging = path + "." + std::to_string(pid) + "." + std::to_string(serial.fetch_add(1));
  try {
    write_file_atomic(staging, text);
#if defined(_WIN32)
    require(MoveFileExW(fs::u8path(staging).c_str(), fs::u8path(path).c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0, SS_E_IO, "publish verification cache");
#else
    require(std::rename(staging.c_str(), path.c_str()) == 0, SS_E_IO, "publish verification cache");
#endif
  } catch (...) {
    std::error_code ec;
    fs::remove(fs::u8path(staging), ec);
    fs::remove(fs::u8path(staging + ".tmp"), ec);
    throw;
  }
}

}  // namespace

// Concurrent parts of one chunk-sized piece: queue depth on a drive even when a call holds only a couple of chunks.
constexpr uint64_t kSubRead = uint64_t(4) << 20;

// Which requests use the stripe. The primary holds every byte, so a stripe only pays when both drives work on the
// same request at once: a request of at least kSplitBytes (its stripe chunks read in parallel with the primary's), or a
// batch of at least kSplitBatch scattered reads (every other one from the stripe). Smaller requests and short batches
// (a decode step's 24-48 Engram pages) are latency bound: a split batch waits for the slower drive's page, so they stay
// on the primary, which serves them as single-drive reading always did.
constexpr uint64_t kSplitBytes = uint64_t(8) << 20;
// Requests of kBalanceBytes up to kSplitBytes are never split, but a request whose chunks all have a stripe copy on one
// location goes whole to whichever of the primary and that location has fewer bytes in flight (balanced routing; the
// legacy routing: only while other requests are in flight, to the less loaded by the large-request shares).
constexpr uint64_t kBalanceBytes = uint64_t(1) << 20;
constexpr uint32_t kSplitBatch = 64;

struct MFile::Impl {
  struct Loc {
    std::string path, manifest_path, manifest_text, manifest_digest, open_reason;
    FileIdentity id, manifest_id;
    std::unique_ptr<RawFile> file;
    std::unique_ptr<RawFile> manifest_file;  // held for the session's lifetime (Windows: no writer, no rename)
    StripeManifest manifest;
    std::atomic<bool> enabled{false};
    std::atomic<uint32_t> errors{0};
    std::atomic<uint64_t> bytes{0}, reads{0}, busy_ns{0}, serving{0};
    // Bytes planned onto this location and not finished (under plan_mu): every request in balanced routing, medium and
    // large requests in the legacy routing.
    uint64_t inflight = 0;
    uint64_t turn = 0;  // balanced routing, under plan_mu: equal bytes in flight take turns
    std::mutex injection_mu;
    int injection = 0;
    uint64_t after = 0;
    // test hook (a mock drive): every piece of at least 1 MiB run() serves from here takes delay_us per MiB longer, one
    // piece at a time
    std::atomic<uint64_t> delay_us{0};
    std::mutex delay_mu;
    bool manifest_absent = false;
    std::shared_ptr<Pool> pool;
    int fault() {
      std::lock_guard<std::mutex> lock(injection_mu);
      if (after) { --after; return 0; }
      return injection;
    }
  };
  struct Piece {
    size_t loc;
    uint64_t logical, phys, bytes;
    char* dst;
    size_t flight = SIZE_MAX;  // its entry in the call's Flight (SIZE_MAX: not reserved)
  };
  Options options;
  uint64_t size = 0, chunk = uint64_t(16) << 20;
  std::string primary, bound_digest, report;
  std::vector<std::unique_ptr<Loc>> locs;
  std::vector<uint8_t> chunk_loc;
  std::vector<uint32_t> chunk_slot;
  // 0 = primary (terminal), 1 = verified secondary, 2 = unverified secondary (tests).
  std::unique_ptr<std::atomic<uint8_t>[]> states;
  std::atomic<uint64_t> fallback{0}, demoted{0}, verify_fail{0}, read_errors{0}, verified{0}, verify_ns{0};
  std::atomic<bool> prefer_stripe{false};  // test hook

  // Adaptive shares of large requests (kSplitBytes and up). Each location's share follows its measured throughput
  // (its bytes of a request over the time until its last piece of that request finished); every kSingleEvery-th large
  // request reads the primary alone, so a fresh single-drive baseline is always known, and while split requests measure
  // slower than that baseline, splitting stays off for kSplitOff large requests (then it is tried again).
  // Balanced routing compares split and primary-only requests only within one load class (the bytes other reads had in
  // flight when the request was planned: under a quarter of its size, so record pages count as none, under its size,
  // under four times its size, more), so a primary-only sample taken alone is never held against split requests that
  // queued behind other reads.
  static constexpr uint64_t kSingleEvery = 16, kSplitOff = 64;
  // Balanced routing: a stripe location whose rate was not sampled for kProbeAfter large requests or kProbeNs is offered
  // at the best rate of the others again, so a drive measured slow once (busy for a moment, or the OS drive) is measured
  // again instead of keeping a share too small to be given a piece; really slow, it measures slow and its share falls
  // back. The time limit matters when no large request comes at all (decode reads one expert's parts, medium requests):
  // the medium rule's slow-stripe guard reads these shares and would otherwise keep a rate learned at load for good.
  static constexpr uint64_t kProbeAfter = 32;
  static constexpr int64_t kProbeNs = int64_t(10) * 1000 * 1000 * 1000;
  static constexpr size_t kLoadClasses = 4;
  std::mutex share_mu, plan_mu;
  std::vector<double> rate;           // bytes per ns per location (EWMA)
  std::vector<uint64_t> sampled;      // balanced routing: `large` at each location's last rate sample
  std::vector<int64_t> sampled_ns;    // and its time (now_ns)
  std::atomic<int64_t> probe_ns{kProbeNs};  // kProbeNs (test hook: shorter)
  double split_rate = 0, single_rate = 0;  // whole large requests, split / primary only (EWMA, bytes per ns)
  uint64_t large = 0, split_samples = 0, split_off_until = 0;
  struct LoadClass {
    double split_rate = 0, single_rate = 0;
    uint64_t split_samples = 0;
  } classes[kLoadClasses];            // balanced routing (under share_mu)
  bool legacy = false;                // options.legacy_routing
  struct Route {
    bool split = false;
    std::vector<double> share;  // per location, sums to 1 over the locations with a measured rate
  };

  ~Impl() {
    // Every call has waited for its own jobs before releasing the session; pools may serve other sessions too.
    for (auto& loc : locs) loc->pool.reset();
  }

  bool reusable(const Options& o) const {
    if (options.drive_readers != o.drive_readers || options.legacy_routing != o.legacy_routing ||
        strictness(options.verify_mode) < strictness(o.verify_mode) ||
        (!o.primary_sha256.empty() && o.primary_sha256 != bound_digest)) return false;
    try {
      for (const auto& loc : locs) {
        if (loc->manifest_absent) {
          std::error_code ec;
          if (fs::exists(fs::u8path(loc->manifest_path), ec) || ec) return false;
          continue;
        }
        // Other rejection reasons must be retried on another open.
        if (!loc->file || identity_of(loc->path) != loc->id) return false;
        // the manifest: the same file (volume, index, size; its bytes are compared next, so a timestamp-only change
        // such as a metadata event moving the Linux ctime does not keep the stripe from serving again)
        if (!loc->manifest_path.empty()) {
          if (!loc->manifest_file) return false;
          const FileIdentity now = identity_of(loc->manifest_path), held = loc->manifest_file->identity();
          if (now.volume != held.volume || now.index != held.index || now.size != held.size ||
              read_all(*loc->manifest_file, uint64_t(256) << 20) != loc->manifest_text)
            return false;
        }
      }
    } catch (const Error&) { return false; }
    return true;
  }

  void demote_chunk(uint64_t c) {
    if (states[c].exchange(0) != 0) {
      demoted.fetch_add(1);
      locs[chunk_loc[size_t(c)]]->serving.fetch_sub(1);
    }
  }

  void failure(const Piece& p) {
    for (uint64_t c = p.logical / chunk; c <= (p.logical + p.bytes - 1) / chunk; ++c) demote_chunk(c);
    if (locs[p.loc]->errors.fetch_add(1) + 1 >= 3) locs[p.loc]->enabled.store(false);
  }

  void account(size_t loc, uint64_t got, Clock::time_point start) {
    locs[loc]->bytes.fetch_add(got);
    locs[loc]->reads.fetch_add(1);
    locs[loc]->busy_ns.fetch_add(elapsed(start));
  }

  uint64_t logical_bytes(const Piece& p) const { return std::min(p.bytes, size - p.logical); }

  void zero_tail(void* dst, uint64_t n, uint64_t off) const {
    uint64_t logical = off < size ? std::min(n, size - off) : 0;
    if (n > logical) std::memset(static_cast<char*>(dst) + logical, 0, size_t(n - logical));
  }

  void primary_read(const Piece& p) {
    auto start = Clock::now();
    uint64_t got = locs[0]->file->read_split(p.dst, p.bytes, p.logical, kSubRead, [](const char*, uint64_t) {});
    account(0, got, start);
    require(got >= logical_bytes(p), SS_E_IO, "short primary read");
  }

  bool use_secondary(const Piece& p) const {
    if (!locs[p.loc]->enabled.load()) return false;
    for (uint64_t c = p.logical / chunk; c <= (p.logical + p.bytes - 1) / chunk; ++c)
      if (!states[c].load()) return false;
    return true;
  }

  // A piece never crosses a chunk boundary (plan); it covers its whole chunk when it starts at the chunk and reaches
  // the chunk's logical end.
  bool whole_chunk(const Piece& p) const { return p.logical % chunk == 0 && p.bytes >= std::min(chunk, size - p.logical); }
  // The inline re-check of whole-chunk reads runs with VERIFY_ALWAYS (and in tests, OFF): CACHED serves chunks the
  // open-time verification of this data-file version passed, under the platform writer contract, without read-path hashing.
  bool recheck(const Piece& p) const { return options.verify_mode != SS_MFILE_VERIFY_CACHED && whole_chunk(p); }

  // digest: xxh64 of the chunk's logical bytes as read (whole-chunk pieces only)
  bool complete_secondary(const Piece& p, uint64_t got, bool ok, int fault, uint64_t digest) {
    if (!ok || got != p.bytes || fault == 1 || fault == 2) {
      read_errors.fetch_add(1);
      failure(p);
      return false;
    }
    uint64_t c = p.logical / chunk;
    if (recheck(p) && digest != locs[p.loc]->manifest.hashes[chunk_slot[size_t(c)]]) {
      verify_fail.fetch_add(1);
      failure(p);
      return false;
    }
    return true;
  }

  void fallback_read(const Piece& p) {
    fallback.fetch_add(logical_bytes(p));
    primary_read(p);
  }

  size_t run(const Piece& p) {
    // (pieces under 1 MiB are not delayed: a sleep of a few microseconds takes a whole timer tick on Windows)
    if (uint64_t d = p.bytes >= kBalanceBytes ? locs[p.loc]->delay_us.load() : 0) {
      std::lock_guard<std::mutex> lock(locs[p.loc]->delay_mu);
      std::this_thread::sleep_for(std::chrono::microseconds(d * p.bytes >> 20));
    }
    if (!p.loc) { primary_read(p); return 0; }
    if (!use_secondary(p)) { fallback_read(p); return 0; }
    Loc& loc = *locs[p.loc];
    int fault = loc.fault();
    uint64_t got = 0;
    bool ok = true, whole = recheck(p);
    // the whole-chunk check hashes each part as it lands, while the later parts are still being read
    Xxh64 h;
    uint64_t left = whole ? logical_bytes(p) : 0;
    bool first = true;
    auto start = Clock::now();
    try {
      if (fault != 1)
        got = loc.file->read_split(p.dst, p.bytes, p.phys, kSubRead, [&](const char* part, uint64_t n) {
          // the test hook corrupts a whole chunk only: partial reads deliberately rely on open-time verification
          if (first && fault == 3 && whole && n) p.dst[0] ^= 1;
          first = false;
          uint64_t take = std::min(n, left);
          if (take) h.update(part, size_t(take));
          left -= take;
        });
      else ok = false;
    } catch (const Error&) { ok = false; }
    account(p.loc, got, start);
    if (!complete_secondary(p, got, ok, fault, whole ? h.digest() : 0)) {
      fallback_read(p);
      return 0;
    }
    return p.loc;
  }

  // allow_stripe: the request may use the stripe copies of its chunks (the caller's size rule, kSplitBytes /
  // kSplitBatch); otherwise every piece is read from the primary, which holds every byte.
  // share (large requests): a stripe chunk goes to whichever of its two homes has the least of (bytes already
  // planned onto it and not finished, by this and concurrent large requests, + this chunk) / its share; a location
  // with share 0 gets nothing. Concurrent one-chunk requests therefore spread over the drives by their shares.
  uint64_t plan(void* dst, uint64_t n, uint64_t off, std::vector<Piece>& out, bool allow_stripe,
                const std::vector<double>* share = nullptr) const {
    std::vector<uint64_t> given(locs.size(), 0);
    if (share)
      for (size_t l = 0; l < locs.size(); ++l) given[l] = locs[l]->inflight;
    require(off % kAlign == 0 && n % kAlign == 0 && reinterpret_cast<uintptr_t>(dst) % kAlign == 0 &&
            (dst || !n), SS_E_INVALID, "mfile read requires aligned offset, bytes and destination");
    uint64_t end = (size + kAlign - 1) / kAlign * kAlign;
    require(off <= end && n <= end - off, SS_E_INVALID, "mfile read exceeds aligned EOF");
    uint64_t existing = off < size ? std::min(n, size - off) : 0;
    if (!existing) return 0;
    uint64_t stop = off + (existing + kAlign - 1) / kAlign * kAlign;
    char* p = static_cast<char*>(dst);
    for (uint64_t at = off; at < stop;) {
      uint64_t c = at / chunk, bytes = std::min(stop - at, chunk - at % chunk);
      size_t loc = (allow_stripe || prefer_stripe.load()) && !chunk_loc.empty() && states[c].load() &&
                           locs[chunk_loc[size_t(c)]]->enabled.load()
                       ? chunk_loc[size_t(c)] : 0;
      if (loc && share && !prefer_stripe.load()) {
        double s0 = (*share)[0], s1 = (*share)[loc];
        if (s1 <= 0 || (s0 > 0 && double(given[0] + bytes) / s0 <= double(given[loc] + bytes) / s1)) loc = 0;
      }
      given[loc] += bytes;
      uint64_t phys = loc ? uint64_t(chunk_slot[size_t(c)]) * chunk + at % chunk : at;
      // one piece per chunk: a whole-chunk read stays recognisable for the re-check, and demotion stays per chunk
      out.push_back(Piece{loc, at, phys, bytes, p + (at - off)});
      at += bytes;
    }
    return existing;
  }

  static int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
  }

  // plan_mu held: p's bytes are in flight on its location.
  void reserve_locked(const Piece& p) {
    Loc& l = *locs[p.loc];
    l.inflight += p.bytes;
  }

  // plan_mu held: p (reserved) finished.
  void release_locked(const Piece& p) { locs[p.loc]->inflight -= p.bytes; }

  // The in-flight bytes of one call's pieces: reserved as they are planned (plan_mu held), each released when it
  // finished (balanced routing; the legacy routing releases them all when the call ends, as before), and whatever is
  // still reserved when the call ends, on every path (errors, exceptions, fallback). Every job of the call has been
  // waited for before the destructor runs.
  struct Flight {
    Impl& f;
    std::vector<Piece> pieces;  // as reserved: the location planned, even when a fallback read the primary
    std::vector<uint8_t> done;  // 1: released (or not yet reserved)
    explicit Flight(Impl& impl) : f(impl) {}
    Flight(const Flight&) = delete;
    Flight& operator=(const Flight&) = delete;
    // plan_mu held: reserves p and records its entry in p.flight
    void add_locked(Piece& p) {
      p.flight = pieces.size();
      pieces.push_back(p);
      done.push_back(1);
      f.reserve_locked(p);
      done.back() = 0;
    }
    // A piece's job finished (the second argument, the location that read it, is not needed for the release).
    void finished(const Piece& p, size_t) {
      if (f.legacy || p.flight == SIZE_MAX) return;
      std::lock_guard<std::mutex> lock(f.plan_mu);
      f.release_locked(pieces[p.flight]);
      done[p.flight] = 1;
    }
    // Several pieces finished together, released under one lock: (p.flight, the location that read it).
    void finished(const std::vector<std::pair<size_t, size_t>>& batch) {
      if (f.legacy || batch.empty()) return;
      std::lock_guard<std::mutex> lock(f.plan_mu);
      for (const auto& b : batch) {
        f.release_locked(pieces[b.first]);
        done[b.first] = 1;
      }
    }
    ~Flight() {
      if (pieces.empty()) return;
      std::lock_guard<std::mutex> lock(f.plan_mu);
      for (size_t i = 0; i < done.size(); ++i)
        if (!done[i]) f.release_locked(pieces[i]);
    }
  };

  // The current shares without counting a large request.
  Route route_large_shares() {
    std::lock_guard<std::mutex> lock(share_mu);
    return shares_locked();
  }

  Route route_large() {
    std::lock_guard<std::mutex> lock(share_mu);
    ++large;
    Route r = shares_locked();
    r.split = large % kSingleEvery != 0 && large >= split_off_until;
    return r;
  }

  Route shares_locked() {
    Route r;
    if (rate.empty()) rate.assign(locs.size(), 6.0);  // 6 GB/s until measured
    if (sampled.empty()) sampled.assign(locs.size(), 0);
    if (sampled_ns.empty()) sampled_ns.assign(locs.size(), 0);
    std::vector<double> eff = rate;
    if (!legacy) {
      double best = 0;
      for (size_t l = 0; l < locs.size(); ++l)
        if (locs[l]->enabled.load() || !l) best = std::max(best, rate[l]);
      const int64_t now = now_ns();
      for (size_t l = 1; l < locs.size(); ++l)
        if (large - sampled[l] >= kProbeAfter || now - sampled_ns[l] >= probe_ns.load())
          eff[l] = std::max(rate[l], best);  // re-probe a stale location
    }
    double sum = 0;
    for (size_t l = 0; l < locs.size(); ++l) sum += locs[l]->enabled.load() || !l ? eff[l] : 0.0;
    r.share.assign(locs.size(), 0.0);
    for (size_t l = 0; l < locs.size(); ++l)
      if ((locs[l]->enabled.load() || !l) && sum > 0) r.share[l] = eff[l] / sum;
    return r;
  }

  // load: the request's load class (balanced routing)
  void learn_large(const Route& r, uint64_t bytes, int64_t t0, const std::vector<Piece>& pieces,
                   const std::vector<int64_t>& done_at, const std::vector<size_t>& served_by, size_t load) {
    int64_t end = t0;
    std::vector<uint64_t> loc_bytes(locs.size(), 0);
    std::vector<int64_t> loc_end(locs.size(), t0);
    for (size_t i = 0; i < pieces.size(); ++i) {
      end = std::max(end, done_at[i]);
      loc_bytes[served_by[i]] += pieces[i].bytes;
      loc_end[served_by[i]] = std::max(loc_end[served_by[i]], done_at[i]);
    }
    if (end <= t0) return;
    std::lock_guard<std::mutex> lock(share_mu);
    auto ewma = [](double& v, double sample) { v = v > 0 ? v * 0.75 + sample * 0.25 : sample; };
    for (size_t l = 0; l < locs.size(); ++l)
      if (loc_bytes[l] && loc_end[l] > t0) {
        ewma(rate[l], double(loc_bytes[l]) / double(loc_end[l] - t0));
        if (l < sampled.size()) sampled[l] = large;
        if (l < sampled_ns.size()) sampled_ns[l] = now_ns();
      }
    double whole = double(bytes) / double(end - t0);
    if (!legacy) {
      LoadClass& c = classes[load];
      if (!r.split) {
        ewma(c.single_rate, whole);
        return;
      }
      ewma(c.split_rate, whole);
      if (++c.split_samples >= 4 && c.single_rate > 0 && c.split_rate < 0.98 * c.single_rate) {
        split_off_until = large + kSplitOff;
        c.split_rate = 0;
        c.split_samples = 0;
      }
      return;
    }
    if (!r.split) {
      ewma(single_rate, whole);
      return;
    }
    ewma(split_rate, whole);
    if (++split_samples >= 4 && single_rate > 0 && split_rate < 0.98 * single_rate) {
      split_off_until = large + kSplitOff;  // splitting measured slower than the primary alone: off for a while
      split_rate = 0;
      split_samples = 0;
    }
  }

  // Balanced routing counts small reads in flight only while a stripe location serves chunks (as MFile::multi()).
  bool track_small() const {
    if (legacy) return false;
    for (size_t l = 1; l < locs.size(); ++l)
      if (locs[l]->enabled.load() && locs[l]->serving.load()) return true;
    return false;
  }

  // Balanced routing, plan_mu held: a medium request, every chunk of which loc (a stripe location) serves, goes to
  // whichever of the primary and loc has fewer bytes in flight (every read counted); with those equal (two idle
  // locations) the two take turns. Across concurrent calls a faster drive empties its queue sooner and so takes more
  // requests by itself (weighting by a measured drive rate fed back on itself: the location that got less traffic
  // measured slower and then got still less). Within one call (a batch's medium ios, or one caller's reads one at a
  // time) the turns split evenly whatever the speeds, so a location whose large-request share (learned from split
  // requests, not from this rule) is under half the primary's gets nothing here.
  bool stripe_first_locked(size_t loc, const Route& r) {
    if (r.share.size() > loc && r.share[loc] < 0.5 * r.share[0]) return false;
    const uint64_t q0 = locs[0]->inflight, ql = locs[loc]->inflight;
    if (q0 != ql) return ql < q0;
    return (locs[loc]->turn++ & 1) != 0;
  }

  // A medium request (kBalanceBytes..kSplitBytes): one location serves all of it, the primary or, when every chunk it
  // touches is served from one stripe location, possibly that location. Balanced routing: stripe_first_locked. The
  // legacy routing: the stripe location only when other reads are in flight on the primary or that location and that
  // location's in-flight bytes per share are lower than the primary's; with nothing in flight the primary. Appends
  // the pieces (the caller reserves them); plan_mu must be held.
  uint64_t plan_balanced(void* dst, uint64_t n, uint64_t off, std::vector<Piece>& out, const Route& r) {
    std::vector<Piece> pieces;
    uint64_t got = plan(dst, n, off, pieces, true);
    size_t loc = pieces.empty() ? 0 : pieces.front().loc;
    for (const Piece& p : pieces) if (p.loc != loc) loc = 0;
    bool stripe = loc && (legacy ? locs[0]->inflight + locs[loc]->inflight > 0 && r.share[loc] > 0 &&
                                       (r.share[0] <= 0 || double(locs[loc]->inflight + n) / r.share[loc] <
                                                               double(locs[0]->inflight + n) / r.share[0])
                                 : stripe_first_locked(loc, r));
    if (!stripe) {
      pieces.clear();
      got = plan(dst, n, off, pieces, false);
    }
    out.insert(out.end(), pieces.begin(), pieces.end());
    return got;
  }

  uint64_t read_balanced(void* dst, uint64_t n, uint64_t off) {
    Route r = route_large_shares();
    Flight flight(*this);
    std::vector<Piece> pieces;
    uint64_t got;
    {
      std::lock_guard<std::mutex> lock(plan_mu);
      got = plan_balanced(dst, n, off, pieces, r);
      flight.pieces.reserve(pieces.size());
      flight.done.reserve(pieces.size());
      for (Piece& p : pieces) flight.add_locked(p);
    }
    if (pieces.size() != 1) {
      parallel(pieces, &flight);
      return got;
    }
    int64_t t0 = now_ns();
    size_t served = run(pieces.front());
    int64_t t1 = now_ns();
    flight.finished(pieces.front(), served);
    // the legacy routing: a single-location read also measures its location; its share follows it like a large
    // request's (balanced routing: the drive rates measure it, and the shares follow large requests only)
    if (legacy && t1 > t0) {
      std::lock_guard<std::mutex> lock(share_mu);
      double sample = double(pieces.front().bytes) / double(t1 - t0);
      rate[served] = rate[served] > 0 ? rate[served] * 0.75 + sample * 0.25 : sample;
    }
    return got;
  }

  // A small request (under kBalanceBytes): read from the primary (or as the test hook routes it); balanced routing
  // counts its bytes in flight.
  uint64_t read_small(void* dst, uint64_t n, uint64_t off) {
    Flight flight(*this);
    std::vector<Piece> pieces;
    uint64_t got = plan(dst, n, off, pieces, false);
    if (track_small()) {
      flight.pieces.reserve(pieces.size());
      flight.done.reserve(pieces.size());
      std::lock_guard<std::mutex> lock(plan_mu);
      for (Piece& p : pieces) flight.add_locked(p);
    }
    if (pieces.size() == 1) flight.finished(pieces.front(), run(pieces.front()));
    else parallel(pieces, &flight);
    return got;
  }

  // A large request: planned with the current shares, timed per piece, and fed back into the shares.
  uint64_t read_large(void* dst, uint64_t n, uint64_t off) {
    Route r = route_large();
    Flight flight(*this);
    std::vector<Piece> pieces;
    uint64_t got;
    size_t load = 0;
    {
      std::lock_guard<std::mutex> lock(plan_mu);
      uint64_t others = 0;
      for (const auto& l : locs) others += l->inflight;
      load = others < n / 4 ? 0 : others < n ? 1 : others / 4 < n ? 2 : 3;
      got = plan(dst, n, off, pieces, r.split, &r.share);
      flight.pieces.reserve(pieces.size());
      flight.done.reserve(pieces.size());
      for (Piece& p : pieces) flight.add_locked(p);
    }
    std::vector<int64_t> done_at(pieces.size(), 0);
    std::vector<size_t> served_by(pieces.size(), 0);
    int64_t t0 = now_ns();
    std::vector<std::future<void>> jobs;
    jobs.reserve(pieces.size());
    try {
      for (size_t i = 0; i < pieces.size(); ++i) {
        Piece p = pieces[i];
        int64_t* at = &done_at[i];
        size_t* served = &served_by[i];
        Flight* fl = &flight;
        jobs.push_back(locs[p.loc]->pool->submit([this, p, at, served, fl] {
          *served = run(p);
          *at = now_ns();
          fl->finished(p, *served);
        }));
      }
    } catch (...) { finish(jobs, std::current_exception()); }
    finish(jobs);
    learn_large(r, n, t0, pieces, done_at, served_by, load);
    return got;
  }

  void parallel(const std::vector<Piece>& pieces, Flight* flight) {
    std::vector<std::future<void>> jobs;
    jobs.reserve(pieces.size());
    try {
      for (const Piece& p : pieces)
        jobs.push_back(locs[p.loc]->pool->submit([this, p, flight] { flight->finished(p, run(p)); }));
    } catch (...) { finish(jobs, std::current_exception()); }
    finish(jobs);
  }

  void verify_open(size_t index) {
    Loc& loc = *locs[index];
    const auto before = loc.file->identity();
    std::string identity = id_text(before);
    std::string key_text = loc.manifest_digest + "\n" + identity;
    std::string key = sha256_hex(key_text.data(), key_text.size());
    std::string dir = cache_dir();
    std::string path = dir.empty() ? std::string() : joined(dir, "stripe-verify-" + key);
    std::string bitmap;
    if (options.verify_mode == SS_MFILE_VERIFY_CACHED && !path.empty()) {
      try {
        std::string cached = read_text(path, key.size() + 1 + loc.manifest.slots());
        if (cached.size() == key.size() + 1 + loc.manifest.slots() && cached.compare(0, key.size() + 1, key + "\n") == 0) {
          bitmap = cached.substr(key.size() + 1);
          if (bitmap.find_first_not_of("01") != std::string::npos) bitmap.clear();
        }
      } catch (const Error&) {}
    }
    auto start = Clock::now();
    if (bitmap.size() != loc.manifest.slots()) {
      bitmap.assign(size_t(loc.manifest.slots()), '0');
      // Bounded jobs and scratch: at most drive_readers chunks are being verified on each location.
      std::atomic<uint64_t> next{0};
      std::vector<uint64_t> slot_chunk(size_t(loc.manifest.slots()));
      for (const auto& r : loc.manifest.runs)
        for (uint64_t k = 0; k < r.count; ++k) slot_chunk[size_t(r.first_slot + k)] = r.first_chunk + k;
      std::vector<std::future<void>> jobs;
      uint32_t workers = uint32_t(std::min<uint64_t>(options.drive_readers, loc.manifest.slots()));
      jobs.reserve(workers);
      try {
        for (uint32_t w = 0; w < workers; ++w) jobs.push_back(loc.pool->submit([&] {
          Buffer buffer{size_t(chunk)};
          for (;;) {
            uint64_t slot = next.fetch_add(1);
            if (slot >= loc.manifest.slots()) break;
            uint64_t bytes = std::min(chunk, size - slot_chunk[size_t(slot)] * chunk);
            try {
              uint64_t got = loc.file->read(buffer.p, chunk, slot * chunk);
              if (got != chunk) { read_errors.fetch_add(1); continue; }
              verified.fetch_add(bytes);
              if (xxh64(buffer.p, size_t(bytes)) == loc.manifest.hashes[size_t(slot)]) bitmap[size_t(slot)] = '1';
            } catch (const Error&) { read_errors.fetch_add(1); }
          }
        }));
      } catch (...) { finish(jobs, std::current_exception()); }
      finish(jobs);
      bool stable = loc.file->identity() == before && identity_of(loc.path) == before;
      if (!stable) { loc.enabled.store(false); loc.open_reason = "data identity changed during verification"; }
      if (stable && !path.empty()) {
        try { cache_write(path, key + "\n" + bitmap); } catch (const Error&) {}
      }
    }
    // Linux readers must re-check identity even on a cache hit.
    if (loc.file->identity() != before || identity_of(loc.path) != before) {
      loc.enabled.store(false);
      loc.open_reason = "data identity changed during verification";
    }
    for (const auto& r : loc.manifest.runs) {
      for (uint64_t k = 0; k < r.count; ++k) {
        if (bitmap[size_t(r.first_slot + k)] != '1') {
          verify_fail.fetch_add(1);
          demote_chunk(r.first_chunk + k);
          if (loc.errors.fetch_add(1) + 1 >= 3) loc.enabled.store(false);
          loc.open_reason = "chunk verification failed";
        }
      }
    }
    verify_ns.fetch_add(elapsed(start));
  }
};

MFile::MFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) { sessions_created.fetch_add(1); }
MFile::~MFile() = default;
uint64_t MFile::test_sessions_created() { return sessions_created.load(); }
uint64_t MFile::test_pools_created() { return pools_created.load(); }
double MFile::test_rate(int loc) const {
  std::lock_guard<std::mutex> lock(impl_->share_mu);
  return impl_->rate.at(size_t(loc));
}
uint64_t MFile::test_inflight(int loc) const {
  std::lock_guard<std::mutex> lock(impl_->plan_mu);
  return impl_->locs.at(size_t(loc))->inflight;
}
void MFile::test_hold(int loc, int64_t bytes) {
  std::lock_guard<std::mutex> lock(impl_->plan_mu);
  impl_->locs.at(size_t(loc))->inflight += uint64_t(bytes);
}
bool MFile::test_split_off() const {
  std::lock_guard<std::mutex> lock(impl_->share_mu);
  return impl_->large + 1 < impl_->split_off_until;  // the next large request reads the primary alone
}

std::vector<std::string> MFile::secondary_dirs(const std::string& primary, const std::string& weight_dirs) {
  std::string parent = canonical(fs::u8path(primary).parent_path().empty() ? "." : fs::u8path(primary).parent_path().u8string());
  std::vector<std::string> dirs;
  for (size_t start = 0; start <= weight_dirs.size();) {
    size_t end = weight_dirs.find(';', start);
    if (end == std::string::npos) end = weight_dirs.size();
    if (end != start) {
      std::string dir = canonical(weight_dirs.substr(start, end - start));
      if (dir != parent && std::find(dirs.begin(), dirs.end(), dir) == dirs.end()) dirs.push_back(std::move(dir));
      require(dirs.size() < SS_MFILE_MAX_LOCATIONS, SS_E_INVALID, "at most seven secondary directories are supported");
    }
    if (end == weight_dirs.size()) break;
    start = end + 1;
  }
  return dirs;
}

std::shared_ptr<MFile> MFile::open(const std::string& primary, const std::string& weight_dirs, const Options& input) {
  require(!primary.empty(), SS_E_INVALID, "mfile primary path is empty");
  Options o = input;
  require(o.drive_readers >= 1 && o.drive_readers <= 32, SS_E_INVALID, "drive_readers must be in 1..32");
  require(o.verify_mode >= SS_MFILE_VERIFY_CACHED && o.verify_mode <= SS_MFILE_VERIFY_OFF, SS_E_INVALID, "invalid verify mode");
  require(o.primary_sha256.empty() || digest_valid(o.primary_sha256), SS_E_INVALID, "invalid primary SHA-256");
  o.legacy_routing = o.legacy_routing || legacy_routing_env();
  std::string path = canonical(primary);
  std::error_code ec;
  require(fs::is_regular_file(fs::u8path(path), ec) && !ec, SS_E_IO, "primary is not a regular file");
  auto dirs = secondary_dirs(path, weight_dirs);
  std::string key = std::to_string(path.size()) + ":" + path + (o.direct ? "1" : "0");
  for (const auto& d : dirs) key += std::to_string(d.size()) + ":" + d;
  struct RegistryEntry {
    std::mutex mu;
    std::vector<std::weak_ptr<MFile>> sessions;
  };
  static std::mutex registry_mu;
  static std::map<std::string, std::shared_ptr<RegistryEntry>> registry;
  std::shared_ptr<RegistryEntry> entry;
  {
    std::lock_guard<std::mutex> lock(registry_mu);
    auto& slot = registry[key];
    if (!slot) slot = std::make_shared<RegistryEntry>();
    entry = slot;
  }
  std::lock_guard<std::mutex> lock(entry->mu);
  auto& entries = entry->sessions;
  for (auto it = entries.begin(); it != entries.end();) {
    if (auto existing = it->lock()) {
      if (existing->impl_->reusable(o)) return existing;
      ++it;
    } else it = entries.erase(it);
  }
  auto impl = std::make_unique<Impl>();
  Impl& f = *impl;
  f.options = o;
  f.legacy = o.legacy_routing;
  f.primary = path;
  auto first = std::make_unique<Impl::Loc>();
  first->path = path;
  first->file = std::make_unique<RawFile>(path, o.direct);
  first->id = first->file->identity();
  first->enabled.store(true);
  first->pool = drive_pool(first->id.volume, o.drive_readers);
  f.size = first->file->size();
  f.locs.push_back(std::move(first));
  // The primary-only session has no verification or chunk-map cost. Existing source readers never create it. The
  // primary is hashed (identity-cached) only once a manifest naming it with its size is found.
  f.bound_digest = o.primary_sha256;
  bool grid_set = false;
  for (const std::string& dir : dirs) {
    auto loc = std::make_unique<Impl::Loc>();
    loc->manifest_path = joined(dir, stripe_manifest_name(fs::u8path(path).filename().u8string()));
    loc->path = dir;
    size_t index = f.locs.size();
    f.locs.push_back(std::move(loc));
    auto& l = *f.locs.back();
    try {
      std::error_code absent_ec;
      l.manifest_absent = !fs::exists(fs::u8path(l.manifest_path), absent_ec) && !absent_ec;
      require(!l.manifest_absent, SS_E_IO, "manifest is absent");
      // read through the held handle, which stays open (bound: no link, no rename while the session lives)
      l.manifest_file = std::make_unique<RawFile>(l.manifest_path, false);
      l.manifest_id = l.manifest_file->identity();
      l.manifest_text = read_all(*l.manifest_file, uint64_t(256) << 20);
      l.manifest = stripe_manifest_parse(l.manifest_text);
      l.manifest_digest = json_str(*json_parse(l.manifest_text).find("digest"), "digest");
      const auto& m = l.manifest;
      require(m.primary_size == f.size && canonical(joined(fs::u8path(path).parent_path().u8string(), m.primary_basename)) == path,
              SS_E_DESC, "primary basename or size mismatch");
      if (f.bound_digest.empty()) f.bound_digest = file_sha256_cached(*f.locs[0]->file);
      require(m.primary_sha256 == f.bound_digest, SS_E_DESC, "primary SHA-256 mismatch");
      require(!grid_set || f.chunk == m.chunk_bytes, SS_E_DESC, "secondary chunk grid mismatch");
      if (grid_set) for (const auto& r : m.runs)
        for (uint64_t k = 0; k < r.count; ++k)
          require(f.chunk_loc[size_t(r.first_chunk + k)] == 0, SS_E_DESC, "chunk already claimed by an earlier directory");
      l.path = joined(dir, m.data_name);
      l.file = std::make_unique<RawFile>(l.path, o.direct);
      l.id = l.file->identity();
      require(l.file->size() == m.data_size, SS_E_DESC, "data file size mismatch");
      for (size_t prev = 0; prev < index; ++prev) {
        const auto& id = f.locs[prev]->id;
        require(!f.locs[prev]->file || id.volume != l.id.volume || id.index != l.id.index,
                SS_E_DESC, "data identity aliases another location");
      }
      l.pool = drive_pool(l.id.volume, o.drive_readers);
      if (!grid_set) {
        f.chunk = m.chunk_bytes;
        size_t count = size_t(f.size / f.chunk + uint64_t(f.size % f.chunk != 0));
        f.chunk_loc.assign(count, 0);
        f.chunk_slot.assign(count, 0);
        f.states = std::make_unique<std::atomic<uint8_t>[]>(count);
        for (size_t c = 0; c < count; ++c) f.states[c].store(0);
        grid_set = true;
      }
      l.enabled.store(true);
      for (const auto& r : m.runs) {
        for (uint64_t k = 0; k < r.count; ++k) {
          size_t c = size_t(r.first_chunk + k);
          f.chunk_loc[c] = uint8_t(index);
          f.chunk_slot[c] = uint32_t(r.first_slot + k);
          l.serving.fetch_add(1);
          f.states[c].store(o.verify_mode == SS_MFILE_VERIFY_OFF ? 2 : 1);
        }
      }
    } catch (const Error& e) {
      l.enabled.store(false);
      l.pool.reset();
      l.file.reset();
      l.manifest_file.reset();
      l.manifest_digest.clear();
      f.report += "location " + std::to_string(index) + ": " + e.what() + "\n";
    }
  }
  require(f.locs[0]->file->identity() == f.locs[0]->id && identity_of(path) == f.locs[0]->id,
          SS_E_IO, "primary changed during open");
  std::vector<std::future<void>> scans;
  // One coordinator per enabled location; each submits bounded scan jobs to its shared volume pool.
  try {
    for (size_t i = 1; i < f.locs.size(); ++i) {
      if (f.locs[i]->enabled.load() && o.verify_mode != SS_MFILE_VERIFY_OFF)
        scans.push_back(std::async(std::launch::async, [&f, i] {
          try { f.verify_open(i); }
          catch (const Error& e) { f.locs[i]->enabled.store(false); f.locs[i]->open_reason = e.what(); }
        }));
    }
  } catch (...) { finish(scans, std::current_exception()); }
  finish(scans);
  for (size_t i = 1; i < f.locs.size(); ++i) {
    const auto& l = *f.locs[i];
    if (!l.open_reason.empty() || (l.file && !l.enabled.load()))
      f.report += "location " + std::to_string(i) + ": " +
                  (l.enabled.load() ? "demoted chunks: " : "disabled: ") + l.open_reason + "\n";
  }
  auto result = std::shared_ptr<MFile>(new MFile(std::move(impl)));
  entries.push_back(result);
  return result;
}

uint64_t MFile::size() const { return impl_->size; }
uint64_t MFile::chunk_bytes() const { return impl_->chunk; }
bool MFile::direct() const { return impl_->locs[0]->file->direct(); }
int MFile::locations() const { return int(impl_->locs.size()); }
bool MFile::multi() const {
  for (size_t i = 1; i < impl_->locs.size(); ++i)
    if (impl_->locs[i]->enabled.load() && impl_->locs[i]->serving.load()) return true;
  return false;
}
std::string MFile::location_path(int loc) const {
  return loc < 0 || size_t(loc) >= impl_->locs.size() ? std::string() : impl_->locs[size_t(loc)]->path;
}
std::string MFile::reasons() const { return impl_->report; }

FileIdentity MFile::primary_identity() const { return impl_->locs[0]->id; }

void MFile::location_info(int loc, ss_mfile_location* out) const {
  require(out && out->struct_size >= 2 * sizeof(uint32_t), SS_E_INVALID, "mfile location info: struct_size not set");
  require(loc >= 0 && size_t(loc) < impl_->locs.size(), SS_E_INVALID, "mfile location info: no location " +
                                                                          std::to_string(loc));
  auto put = [](ss_file_identity& to, const FileIdentity& id) {
    to.volume = id.volume;
    to.index = id.index;
    to.size = id.size;
  };
  const auto& l = *impl_->locs[size_t(loc)];
  ss_mfile_location x{};
  x.struct_size = std::min<uint32_t>(out->struct_size, uint32_t(sizeof x));
  if (l.file) {
    x.flags |= SS_MFILE_LOCATION_OPEN;
    if (l.enabled.load()) x.flags |= SS_MFILE_LOCATION_ENABLED;
    put(x.data, l.id);
  }
  if (l.manifest_file) {
    put(x.manifest, l.manifest_id);
    std::snprintf(x.manifest_digest, sizeof x.manifest_digest, "%s", l.manifest_digest.c_str());
  }
  put(x.primary, impl_->locs[0]->id);
  std::memcpy(out, &x, x.struct_size);
}

uint64_t MFile::pread(void* dst, uint64_t n, uint64_t offset) {
  if (n >= kSplitBytes && impl_->locs.size() > 1) {
    uint64_t got = impl_->read_large(dst, n, offset);
    impl_->zero_tail(dst, n, offset);
    return got;
  }
  if (n >= kBalanceBytes && impl_->locs.size() > 1) {
    uint64_t got = impl_->read_balanced(dst, n, offset);
    impl_->zero_tail(dst, n, offset);
    return got;
  }
  uint64_t got = impl_->read_small(dst, n, offset);
  impl_->zero_tail(dst, n, offset);
  return got;
}

void MFile::preadv(const ss_mfile_io* ios, uint32_t n) {
  require(ios || !n, SS_E_INVALID, "mfile preadv: null ios");
  std::vector<Impl::Piece> pieces;
  // large ios go through the adaptive shares (read_large, after the scattered ios); in a batch of at least
  // kSplitBatch, every other small io comes from the stripe
  // medium ios (kBalanceBytes..kSplitBytes) are routed whole like a medium pread, each reserved in flight before the
  // next is routed; balanced routing also reserves the small ios. Each is released when it finished, at the latest
  // when this call ends.
  bool multi_loc = impl_->locs.size() > 1;
  Impl::Flight flight(*impl_);
  bool any_medium = false;
  for (uint32_t i = 0; i < n; ++i) any_medium |= multi_loc && ios[i].bytes >= kBalanceBytes && ios[i].bytes < kSplitBytes;
  const bool track = impl_->track_small();
  {
    Impl::Route r = any_medium ? impl_->route_large_shares() : Impl::Route{};
    std::unique_lock<std::mutex> lock(impl_->plan_mu, std::defer_lock);
    if (any_medium || track) lock.lock();
    if (any_medium)
      for (uint32_t i = 0; i < n; ++i)
        if (ios[i].bytes >= kBalanceBytes && ios[i].bytes < kSplitBytes) {
          size_t first = pieces.size();
          impl_->plan_balanced(ios[i].dst, ios[i].bytes, ios[i].offset, pieces, r);
          for (size_t k = first; k < pieces.size(); ++k) flight.add_locked(pieces[k]);
        }
    size_t from = pieces.size();
    for (uint32_t i = 0; i < n; ++i)
      if (!(multi_loc && ios[i].bytes >= kSplitBytes) && !(any_medium && ios[i].bytes >= kBalanceBytes))
        impl_->plan(ios[i].dst, ios[i].bytes, ios[i].offset, pieces, n >= kSplitBatch && (i & 1));
    if (track)
      for (size_t k = from; k < pieces.size(); ++k) flight.add_locked(pieces[k]);
  }
  struct Large {
    MFile::Impl& m;
    const ss_mfile_io* ios;
    uint32_t n;
    bool on;
    void run() const {
      for (uint32_t i = 0; i < n; ++i)
        if (on && ios[i].bytes >= kSplitBytes) m.read_large(ios[i].dst, ios[i].bytes, ios[i].offset);
    }
  } large{*impl_, ios, n, multi_loc};
#if defined(_WIN32)
  // Large pieces go to their drives' pools (concurrent sub-reads, the whole-chunk hash on the reader threads); the small
  // ones (record pages) are issued from this thread, all at once. Every pool job is waited for before returning.
  std::vector<std::future<void>> jobs;
  jobs.reserve(pieces.size());  // no allocation between a submit and its future's tracking
  {
    std::vector<Impl::Piece> scattered;
    try {
      for (const auto& p : pieces) {
        if (p.bytes >= kSubRead) {
          Impl* impl = impl_.get();
          Impl::Flight* fl = &flight;
          jobs.push_back(impl->locs[p.loc]->pool->submit([impl, p, fl] { fl->finished(p, impl->run(p)); }));
        } else scattered.push_back(p);
      }
    } catch (...) { finish(jobs, std::current_exception()); }
    pieces.swap(scattered);
  }
  auto issue_small = [&] {
    struct Pending {
      Impl::Piece piece;
      OVERLAPPED ov{};
      uint64_t got = 0;
      bool ok = true, issued = false;
      int fault = 0;
      Clock::time_point start;
      ~Pending() { if (ov.hEvent) CloseHandle(ov.hEvent); }
    };
    std::vector<std::unique_ptr<Pending>> pending;
    pending.reserve(pieces.size());
    // Allocate every event before issuing anything: failures cannot leave an outstanding IO targeting caller memory.
    for (auto p : pieces) {
      if (p.loc && !impl_->use_secondary(p)) { p.loc = 0; p.phys = p.logical; }
      auto q = std::make_unique<Pending>();
      q->piece = p;
      q->ov.Offset = DWORD(p.phys);
      q->ov.OffsetHigh = DWORD(p.phys >> 32);
      q->ov.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
      require(q->ov.hEvent != nullptr, SS_E_IO, "mfile preadv event creation failed");
      pending.push_back(std::move(q));
    }
    for (auto& q : pending) {
      auto& p = q->piece;
      q->fault = p.loc ? impl_->locs[p.loc]->fault() : 0;
      q->start = Clock::now();
      if (q->fault == 1) { q->ok = false; continue; }
      q->issued = true;
      if (!ReadFile(impl_->locs[p.loc]->file->handle(), p.dst, DWORD(p.bytes), nullptr, &q->ov)) {
        DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) { q->ok = error == ERROR_HANDLE_EOF; q->issued = false; }
      }
    }
    // First wait for every secondary completion, then permit primary fallback into any destination. Each piece leaves
    // the bytes in flight once its read completed (a later verification failure still re-reads it from the primary);
    // the completions found together are released under one lock.
    std::vector<std::pair<size_t, size_t>> released;
    released.reserve(pending.size());
    for (size_t i = 0; i < pending.size(); ++i) {
      auto& q = pending[i];
      DWORD got = 0;
      if (q->issued && !GetOverlappedResult(impl_->locs[q->piece.loc]->file->handle(), &q->ov, &got, TRUE))
        q->ok = GetLastError() == ERROR_HANDLE_EOF;
      q->got = got;
      impl_->account(q->piece.loc, q->got, q->start);
      if (q->piece.flight != SIZE_MAX)
        released.emplace_back(q->piece.flight,
                              q->ok && !q->fault && q->got >= impl_->logical_bytes(q->piece) ? q->piece.loc : SIZE_MAX);
      const bool next_done = i + 1 < pending.size() &&
                             (!pending[i + 1]->issued || HasOverlappedIoCompleted(&pending[i + 1]->ov));
      if (!next_done) {
        flight.finished(released);
        released.clear();
      }
    }
    std::exception_ptr error;
    for (auto& q : pending) {
      try {
        const auto& p = q->piece;
        if (p.loc) {
          uint64_t digest = 0;
          if (impl_->recheck(p) && q->ok && q->got == p.bytes) {
            if (q->fault == 3) p.dst[0] ^= 1;  // test hook, as in run()
            digest = xxh64(p.dst, size_t(impl_->logical_bytes(p)));
          }
          if (!impl_->complete_secondary(p, q->got, q->ok, q->fault, digest)) impl_->fallback_read(p);
        } else require(q->ok && q->got >= impl_->logical_bytes(p), SS_E_IO, "mfile preadv primary read failed");
      } catch (...) { if (!error) error = std::current_exception(); }
    }
    if (error) std::rethrow_exception(error);
  };
  try {
    issue_small();
  } catch (...) { finish(jobs, std::current_exception()); }
  finish(jobs);
  large.run();
#else
  impl_->parallel(pieces, &flight);
  large.run();
#endif
  for (uint32_t i = 0; i < n; ++i) impl_->zero_tail(ios[i].dst, ios[i].bytes, ios[i].offset);
}

void MFile::counters(ss_mfile_counters* out) const {
  require(out && out->struct_size >= sizeof(uint32_t), SS_E_INVALID, "mfile counters: struct_size not set");
  ss_mfile_counters c{};
  c.struct_size = std::min<uint32_t>(out->struct_size, uint32_t(sizeof c));
  c.locations = uint32_t(impl_->locs.size());
  c.verify_mode = impl_->options.verify_mode;
  for (size_t i = 0; i < impl_->locs.size(); ++i) {
    const auto& l = *impl_->locs[i];
    c.bytes[i] = l.bytes.load(); c.reads[i] = l.reads.load(); c.busy_ns[i] = l.busy_ns.load();
    if (!l.enabled.load()) c.disabled_mask |= uint32_t(1) << i;
    if (l.file && l.file->direct()) c.direct_mask |= uint32_t(1) << i;
  }
  c.fallback_bytes = impl_->fallback.load(); c.demoted_chunks = impl_->demoted.load();
  c.verify_fail = impl_->verify_fail.load(); c.read_errors = impl_->read_errors.load();
  c.verified_bytes = impl_->verified.load(); c.verify_ns = impl_->verify_ns.load();
  std::memcpy(out, &c, c.struct_size);
}

void MFile::test_prefer_stripe(bool on) { impl_->prefer_stripe.store(on); }
void MFile::test_delay(int loc, uint64_t microseconds) {
  impl_->locs.at(size_t(loc))->delay_us.store(microseconds);
}
void MFile::test_probe_ns(int64_t ns) { impl_->probe_ns.store(ns); }

void MFile::test_inject(int loc, int kind, uint64_t after) {
  require(loc > 0 && size_t(loc) < impl_->locs.size() && kind >= 0 && kind <= 3, SS_E_INVALID, "invalid mfile injection");
  auto& l = *impl_->locs[size_t(loc)];
  std::lock_guard<std::mutex> lock(l.injection_mu);
  l.after = after;
  l.injection = kind;
}

}  // namespace ts
