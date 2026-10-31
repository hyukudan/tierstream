// Device-gated cache core (src/dg_core.cpp) against a scripted device: every copy, table publication, ready write,
// event record / wait and write-back is a pending operation of its stream, and a randomized scheduler completes them
// in any order the stream and event rules allow. A device model plays the engine's graph (export -> wait -> read
// -> next layer, the end edge, the boundary) and checks, at every read, that the slot it resolves holds the unit's
// bytes and that no copy lands in a slot while a layer reads it. check() runs after every service step. Each
// mutation of the core must be caught.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "dg_core.hpp"
#include "dg_score.hpp"
#include "errors.hpp"

using ts::DgConfig;
using ts::DgCore;
using ts::DgTransport;

namespace {

// A violation the model saw (the first one is kept).
struct Violation {
  std::string what;
  void set(const std::string& w) {
    if (what.empty()) what = w;
  }
};

class Scripted final : public DgTransport {
 public:
  struct Ev {
    uint64_t recorded = 0, completed = 0;
  };
  enum Kind { H2D, D2H, Entry, Ready, Record, Wait, Upload, D2HPart, D2D, H2H, Read };
  struct Op {
    Kind kind;
    int32_t a = 0, b = 0;
    uint32_t key = 0;
    Ev* ev = nullptr;
    uint64_t target = 0;
    std::vector<int32_t> table;
  };

  // A block holds one key per part (a write-back may land part by part); a slot is always copied whole.
  Scripted(uint32_t keys, uint32_t slots, uint32_t blocks, uint32_t gated, uint64_t seed, Violation* v,
           uint32_t parts = 3)
      : table_(keys, -1), slot_(slots, -1), block_(size_t(blocks) * parts, -1), ready_(gated, 0), parts_(parts),
        rng_(seed), v_(v) {}

  // ---- DgTransport ----
  void h2d(Stream s, int32_t slot, int32_t block) override {
    if (on_h2d_) on_h2d_();
    kind_fault(kFH2D);
    if (move_faults_) fault("h2d");
    for (const Pend& r : in_flight_)
      if (r.block == block) v_->set("a copy from block " + std::to_string(block) + " before its file read ended");
    push(s, {H2D, slot, block});
  }
  void d2d(int32_t src, int32_t dst) override { fault("d2d"); push(kDem, {D2D, src, dst}); sync(kDem); }
  void h2h(int32_t src, int32_t dst) override { fault("h2h"); push(kDem, {H2H, src, dst}); sync(kDem); }
  void d2h(int32_t slot, int32_t block) override {
    kind_fault(kFD2H);
    if (move_faults_) fault("d2h");
    push(kD2h, {D2H, slot, block});
  }
  void d2h_part(Stream s, int32_t slot, int32_t block, uint32_t part) override {
    if (part >= parts_) v_->set("a write-back part out of range");
    fault("d2h_part");
    Op o{D2HPart, slot, block};
    o.key = part;
    push(s, std::move(o));
  }
  // three-tier: the file read lands at once; no queued copy may still read or write that block
  void read_unit(uint32_t key, int32_t block) override {
    if (fail_read_at_ > 0 && --fail_read_at_ == 0) ts::fail(SS_E_IO, "injected read failure");
    read_target(block);
    issued_.push_back(Read);
    trace_.push_back({-1, int64_t(Read), int64_t(key), block, 0});
    fill_block(block, int32_t(key));
    ++reads_;
  }
  // Asynchronous reads (set_async_reads): a begun read lands at a random later moment (its own random stream: the
  // scheduler's and the routes' draws are unchanged), in any order, at the latest when it is ended. A copy from a block
  // whose read was not ended yet, a second read into a block in flight, or a read ended twice is a violation. A failed
  // read (fail_read) lands garbage and throws at its end.
  void* read_unit_begin(uint32_t key, int32_t block) override {
    if (!async_) return DgTransport::read_unit_begin(key, block);
    read_target(block);
    for (const Pend& r : in_flight_)
      if (r.block == block) v_->set("two file reads in flight into block " + std::to_string(block));
    const bool fails = fail_read_at_ > 0 && --fail_read_at_ == 0;
    issued_.push_back(Read);
    trace_.push_back({-1, int64_t(Read), int64_t(key), block, 0});
    in_flight_.push_back({++read_id_, key, block, false, fails});
    max_in_flight_ = std::max<uint64_t>(max_in_flight_, in_flight_.size());
    land_some();
    return reinterpret_cast<void*>(uintptr_t(read_id_));
  }
  bool read_unit_done(void* h) override {
    if (!async_) return true;
    land_some();
    const uint64_t id = uint64_t(reinterpret_cast<uintptr_t>(h));
    for (const Pend& r : in_flight_)
      if (r.id == id) return r.landed;
    v_->set("a finished-read query of no read in flight");
    return true;
  }
  uint64_t read_unit_end(void* h) override {
    if (!async_) return 0;
    land_some();
    const uint64_t id = uint64_t(reinterpret_cast<uintptr_t>(h));
    auto it = std::find_if(in_flight_.begin(), in_flight_.end(), [&](const Pend& r) { return r.id == id; });
    if (it == in_flight_.end()) {
      v_->set("a file read ended twice, or never begun");
      return 0;
    }
    if (it != in_flight_.begin()) ++out_of_order_;  // (the core ends the oldest first: the reads landed in any order)
    land(*it);
    const bool fails = it->fails;
    in_flight_.erase(it);
    if (fails) ts::fail(SS_E_IO, "injected read failure");
    return 1000;
  }
  void set_async_reads(bool on, uint64_t seed) {
    async_ = on;
    read_rng_.seed(seed);
  }
  void fail_read(int n) { fail_read_at_ = n; }  // the n-th later read fails (0 = off)
  void set_eager(bool on) { eager_ = on; }      // every operation runs as soon as it can (no scheduler draws)
  size_t reads_in_flight() const { return in_flight_.size(); }
  uint64_t max_in_flight() const { return max_in_flight_; }
  uint64_t landed_out_of_order() const { return landed_ooo_; }
  uint64_t reads() const { return reads_; }
  uint32_t parts() const override { return parts_; }
  // unequal parts, as the real gate / up / down (the last one larger)
  uint64_t part_bytes(uint32_t part) const override { return part + 1 == parts_ ? 4096 : 2048; }
  void write_entry(Stream s, uint32_t key, int32_t v) override {
    kind_fault(kFEntry);
    Op o{Entry, v};
    o.key = key;
    push(s, std::move(o));
  }
  void signal_ready(uint32_t g, uint32_t seq) override { push(kDem, {Ready, int32_t(g), int32_t(seq)}); }
  void* event_create() override {
    fault("event_create");
    evs_.push_back(std::make_unique<Ev>());
    return evs_.back().get();
  }
  void event_destroy(void*) override {}
  void record(void* ev, Stream s) override {
    kind_fault(kFRecord);
    fault("record");
    Ev* e = static_cast<Ev*>(ev);
    Op o{Record};
    o.ev = e;
    o.target = ++e->recorded;
    push(s, std::move(o));
  }
  void wait(Stream s, void* ev) override {
    Ev* e = static_cast<Ev*>(ev);
    Op o{Wait};
    o.ev = e;
    o.target = e->recorded;
    push(s, std::move(o));
  }
  bool done(void* ev) override {
    const Ev* e = static_cast<Ev*>(ev);
    return e->completed >= e->recorded;
  }
  void event_sync(void* ev) override {
    kind_fault(kFEventSync);
    while (!done(ev))
      if (!step()) {
        v_->set("event_sync: deadlock");
        return;
      }
  }
  void sync(Stream s) override {
    kind_fault(kFSync);
    while (!q_[s].empty())
      if (!step()) {
        v_->set("sync: deadlock");
        return;
      }
  }
  void upload_table(const int32_t* table, uint32_t n) override {
    kind_fault(kFUpload);
    if (move_faults_) fault("upload_table");
    Op o{Upload};
    o.table.assign(table, table + n);
    push(kDem, std::move(o));
    sync(kDem);
  }

  // ---- scheduler ----
  bool runnable(int s) const {
    if (q_[s].empty() || held_[s]) return false;
    const Op& o = q_[s].front();
    return o.kind != Wait || o.ev->completed >= o.target;
  }
  // Completes one runnable operation of a random stream; false if none can run.
  bool step() {
    land_some();
    int cand[3], n = 0;
    for (int s = 0; s < 3; ++s)
      if (runnable(s)) cand[n++] = s;
    if (!n) return false;
    if (eager_) {
      Op o = std::move(q_[cand[0]].front());
      q_[cand[0]].pop_front();
      run(o);
      return true;
    }
    int s = cand[rng_() % unsigned(n)];
    // slow write-backs: a D2H waits behind the H2D streams most of the time (resurrections, pending at boundaries)
    if (slow_d2h_ && s == kD2h && n > 1 && rng_() % 8 != 0) s = cand[rng_() % unsigned(n - 1)];
    Op o = std::move(q_[s].front());
    q_[s].pop_front();
    run(o);
    return true;
  }
  void steps(unsigned max) {
    const unsigned n = max ? unsigned(rng_() % (max + 1)) : 0u;
    for (unsigned i = 0; i < n && step();) ++i;
  }
  void drain_all() {
    while (step()) {
    }
  }

  // ---- device model ----
  int32_t entry(uint32_t key) const { return table_[key]; }
  uint32_t ready(uint32_t g) const { return ready_[g]; }
  int32_t slot_holds(int32_t s) const { return slot_[size_t(s)]; }
  // the key every part of block b holds, -2 if its parts disagree (a write-back partly landed)
  int32_t block_holds(int32_t b) const {
    if (b == -1) return staging_;
    const int32_t k = block_[size_t(b) * parts_];
    for (uint32_t p = 1; p < parts_; ++p)
      if (block_[size_t(b) * parts_ + p] != k) return -2;
    return k;
  }
  void fill_slot(int32_t s, int32_t key) { slot_[size_t(s)] = key; }
  void fill_block(int32_t b, int32_t key) {
    if (b == -1) { staging_ = key; return; }
    for (uint32_t p = 0; p < parts_; ++p) block_[size_t(b) * parts_ + p] = key;
  }
  void set_table(const std::vector<int32_t>& t) { table_ = t; }
  void set_slow_d2h(bool on) { slow_d2h_ = on; }
  void hold(int s, bool on) { held_[s] = on; }  // a held stream runs nothing (its syncs would deadlock: tests only)
  void begin_read(const std::set<int32_t>& slots) { reading_ = slots; }
  // The kinds of the operations in the order they were issued (not run), since the last clear.
  const std::vector<Kind>& issued() const { return issued_; }
  const std::vector<std::vector<int64_t>>& trace() const { return trace_; }
  void clear_issued() {
    trace_.clear();
    issued_.clear();
    wb_slots_.clear();
  }
  // The source slots of the write-back copies (whole or one part) in issue order, since the last clear.
  const std::vector<int32_t>& wb_slots() const { return wb_slots_; }
  void end_read() { reading_.clear(); }
  std::mt19937_64& rng() { return rng_; }
  // Fault injection: the n-th later call of d2h_part / event_create / record throws (as a failed CUDA call), 0 = off.
  void fail_after(int n) { fail_at_ = n; }
  void fail_reset_after(int n) { fail_at_ = n; move_faults_ = true; }
  // Per-kind fault injection: the n-th later call of that kind throws SS_E_BACKEND (0 = off); fired() says it did.
  enum FaultKind { kFH2D, kFD2H, kFEntry, kFRecord, kFEventSync, kFSync, kFUpload, kFaultKinds };
  void fail_kind(FaultKind k, int n) {
    kind_ = k;
    kind_at_ = n;
    kind_fired_ = false;
  }
  bool kind_fired() const { return kind_fired_; }
  bool read_failure_pending() const { return fail_read_at_ > 0; }  // fail_read's failure has not fired yet
  void set_on_h2d(std::function<void()> f) { on_h2d_ = std::move(f); }  // called at every h2d (before it is queued)

 private:
  struct Pend {
    uint64_t id;
    uint32_t key;
    int32_t block;
    bool landed, fails;
  };
  void fault(const char* what) {
    if (fail_at_ > 0 && --fail_at_ == 0) ts::fail(SS_E_BACKEND, std::string("injected failure: ") + what);
  }
  void kind_fault(FaultKind k) {
    if (kind_at_ > 0 && k == kind_ && --kind_at_ == 0) {
      kind_fired_ = true;
      ts::fail(SS_E_BACKEND, "injected failure of kind " + std::to_string(int(k)));
    }
  }
  std::function<void()> on_h2d_;
  FaultKind kind_ = kFH2D;
  int kind_at_ = 0;
  bool kind_fired_ = false;
  // no queued copy may still read or write a block a file read lands in
  void read_target(int32_t block) {
    for (const auto& q : q_)
      for (const Op& o : q)
        if (((o.kind == H2D || o.kind == D2H || o.kind == D2HPart) && o.b == block) ||
            (o.kind == H2H && (o.a == block || o.b == block)))
          v_->set("a file read overwrote block " + std::to_string(block) + " while a copy still used it");
  }
  void land(Pend& r) {
    if (r.landed) return;
    r.landed = true;
    fill_block(r.block, r.fails ? -3 : int32_t(r.key));
    if (!r.fails) ++reads_;
  }
  void land_some() {  // each read in flight lands now with probability 1/3, any of them first
    for (size_t i = 0; i < in_flight_.size(); ++i)
      if (!in_flight_[i].landed && read_rng_() % 3 == 0) {
        for (size_t j = 0; j < i; ++j) landed_ooo_ += in_flight_[j].landed ? 0 : 1;
        land(in_flight_[i]);
      }
  }
  void push(Stream s, Op o) {
    issued_.push_back(o.kind);
    trace_.push_back({int64_t(s), int64_t(o.kind), o.a, o.b, int64_t(o.key)});
    for (int32_t value : o.table) trace_.back().push_back(value);
    if (o.kind == D2H || o.kind == D2HPart) wb_slots_.push_back(o.a);
    q_[s].push_back(std::move(o));
    if (eager_)
      while (step()) {
      }
  }
  void run(const Op& o) {
    switch (o.kind) {
      case D2D:
        if (reading_.count(o.b)) v_->set("reset D2D overwrote a reader");
        slot_[size_t(o.b)] = slot_[size_t(o.a)];
        break;
      case H2H:
        fill_block(o.b, block_holds(o.a));
        break;
      case H2D:
        if (reading_.count(o.a)) v_->set("a copy landed in slot " + std::to_string(o.a) + " while a layer read it");
        slot_[size_t(o.a)] = block_holds(o.b);
        break;
      case D2H:
        fill_block(o.b, slot_[size_t(o.a)]);
        break;
      case D2HPart:
        block_[size_t(o.b) * parts_ + o.key] = slot_[size_t(o.a)];
        break;
      case Entry:
        table_[o.key] = o.a;
        break;
      case Ready:
        ready_[size_t(o.a)] = uint32_t(o.b);
        break;
      case Record:
        o.ev->completed = std::max(o.ev->completed, o.target);
        break;
      case Wait:
        break;
      case Upload:
        table_ = o.table;
        break;
      case Read:
        break;
    }
  }

  std::deque<Op> q_[3];
  std::vector<std::unique_ptr<Ev>> evs_;
  std::vector<int32_t> table_, slot_, block_;
  std::vector<uint32_t> ready_;
  uint32_t parts_;
  std::set<int32_t> reading_;
  std::vector<Kind> issued_;
  std::vector<std::vector<int64_t>> trace_;
  std::vector<int32_t> wb_slots_;
  bool slow_d2h_ = false;
  bool held_[3] = {false, false, false};
  bool move_faults_ = false;
  int fail_at_ = 0;
  uint64_t reads_ = 0;
  bool async_ = false, eager_ = false;
  int fail_read_at_ = 0;
  std::deque<Pend> in_flight_;
  uint64_t read_id_ = 0, max_in_flight_ = 0, out_of_order_ = 0, landed_ooo_ = 0;
  std::mt19937_64 read_rng_{1};
  int32_t staging_ = -1;
  std::mt19937_64 rng_;
  Violation* v_;
};

struct Shape {
  std::vector<uint32_t> units;
  uint32_t residents, spare, bank, host_spares;
  uint32_t max_ids;
  uint32_t pressure_free = 0, pressure_recent = 0;
  uint32_t predict = 0;  // predicted ids per layer (0 = off)
  uint32_t pf_layer = 4, pf_token = 48;
  bool slow_d2h = false;
  bool defer = false;  // wb_defer, and its options
  bool toggle_defer = false;  // persist: set_wb_defer flipped between random tokens
  bool on_d2h = false;
  uint64_t wb_bytes = 0, wb_out = 0;
  // winpf: windows of 1..max_rows rows of top_k ids (export n = rows x top_k), predictions of rows x predict_k,
  // a random token prior for gated layers 0..1, window hints between tokens, a seed() now and then
  bool winpf = false;
  uint32_t max_rows = 0, top_k = 0, predict_k = 0;
  bool revert = true;
  uint32_t prior_window_us = 1500;
  // scored eviction: every boundary checked against the brute-force oracle below; vary = random keep lists and spare
  // targets between tokens (their own random stream)
  bool scored = false;
  bool vary = false;
  bool boundary_score = false;
  bool scored_auto = false;
  uint32_t auto_rows = 6;  // 0 alternates R1 and R6
  // three-tier: `disk` non-resident keys start file-homed (a random set), staging buffers, prefetch of file keys;
  // inclusive = the clean-copy share (its blocks come out of host_spares)
  uint32_t disk = 0, staging = 3;
  bool disk_tier = false, disk_prefetch = false;
  uint32_t inclusive = 0;
  // overlapped file reads: the core's queue depth (1 = synchronous), the transport's reads asynchronous (landing out
  // of order), an eager device (every operation runs at once: timing-free, for the depth comparisons)
  uint32_t queue = 16;
  bool async_reads = false, eager = false;
  // kept-resident layers (make_resident): episodes between tokens (a layer or part of one made resident and held, or a
  // held one released; their own random stream), a held layer never exported and its kept slots read in every layer;
  // skip = 1 in skip layers of a token are not exported (rows 0)
  bool resident = false;
  uint32_t skip = 0;
};

// What the resident runs exercised (their coverage).
struct ResidentCoverage {
  uint64_t calls = 0, refusals = 0, released = 0, copied = 0, evicted = 0, skipped_layers = 0, held_reads = 0;
};
ResidentCoverage g_res;

// Scored eviction's exhaustive boundary oracle (design scored.md 8.3, 8.5, 10): its own history (transition counts
// saturating at 2^32 - 1, the last 16 unions, demand epochs) from the routes the run serves, row sums and ring counts
// recounted from scratch, every eligible unit scored and fully sorted by (score, demand epoch, recency stamp), then the
// acceptance rule (free + pending + accepted < target, a free block per victim, nothing after the first shortage) and
// the pressure override from the most recent unit. It reads only the cache's homes, write-back flags, recency stamps,
// free slots / blocks, pending write-backs and target at the boundary (the same state), never a score or a list.
struct Oracle {
  std::vector<uint32_t> units, base;
  std::vector<std::vector<uint64_t>> T;
  std::vector<std::deque<std::vector<uint32_t>>> ring;
  std::vector<std::vector<uint32_t>> cur;
  std::vector<uint64_t> epoch_of;
  uint64_t epoch = 0;
  uint64_t wt = 700, wf = 200, wr = 100;
  struct Expect {
    std::vector<uint32_t> victims;
    bool short_blocks = false;
    bool armed = false;
  };
  explicit Oracle(const std::vector<uint32_t>& u) : units(u) {
    base.push_back(0);
    for (uint32_t n : u) base.push_back(base.back() + n);
    for (uint32_t n : u) T.emplace_back(size_t(n) * n, 0);
    ring.resize(u.size());
    cur.resize(u.size());
    epoch_of.assign(base.back(), 0);
  }
  void reset() {
    for (auto& t : T) std::fill(t.begin(), t.end(), 0);
    for (auto& r : ring) r.clear();
    for (auto& c : cur) c.clear();
    std::fill(epoch_of.begin(), epoch_of.end(), 0);
    epoch = 0;
  }
  void begin() { ++epoch; }
  void serve(uint32_t g, const uint32_t* ids, uint32_t n) {
    std::vector<uint32_t> u(ids, ids + n);
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    for (uint32_t r : cur[g])  // cur still holds the previous token's union of g
      for (uint32_t e : u)
        if (T[g][size_t(r) * units[g] + e] < 0xFFFFFFFFull) ++T[g][size_t(r) * units[g] + e];
    ring[g].push_back(u);
    if (ring[g].size() > 16) ring[g].pop_front();
    cur[g] = u;
    for (uint32_t e : u) epoch_of[base[g] + e] = epoch;
  }
  uint64_t score(uint32_t k) const {
    uint32_t g = 0;
    while (base[g + 1] <= k) ++g;
    const uint32_t e = k - base[g], n = units[g];
    uint64_t N = 0, D = 0;
    for (uint32_t r : cur[g]) {
      N += T[g][size_t(r) * n + e];
      for (uint32_t j = 0; j < n; ++j) D += T[g][size_t(r) * n + j];
    }
    uint64_t c = 0;
    for (const auto& un : ring[g]) c += std::count(un.begin(), un.end(), e) ? 1 : 0;
    const uint64_t P = ((uint64_t(cur[g].size()) * N) << 16) / (D ? D : 1);
    return wt * P + wf * ((c << 16) / 16) + wr * ((uint64_t(1) << 16) / (1 + epoch - epoch_of[k]));
  }
  // by_lru: the fallback order (recency tail first)
  bool disk_tier = false;
  Expect boundary(const DgCore& c, const std::vector<uint32_t>& keep, uint32_t pressure_recent, bool by_lru) const {
    Expect x;
    x.armed = true;
    auto elig = [&](uint32_t k) {
      return c.home(k) == DgCore::Home::Device && !c.wb_pending(k) &&
             std::find(keep.begin(), keep.end(), k) == keep.end();
    };
    struct C {
      uint64_t s, ep, st;
      uint32_t k;
    };
    // recency: the scored core's stamps; under LRU (no stamps) the positions of the recency list
    std::vector<uint64_t> pos;
    if (!c.scored()) {
      pos.assign(c.keys(), 0);
      const std::vector<uint32_t> r = c.residency();  // device homes, most recent first
      for (size_t i = 0; i < r.size(); ++i) pos[r[i]] = uint64_t(r.size() - i);
    }
    std::vector<C> all;
    for (uint32_t k = 0; k < c.keys(); ++k)
      if (elig(k))
        all.push_back({by_lru ? 0 : score(k), by_lru ? 0 : epoch_of[k], c.scored() ? c.lru_stamp(k) : pos[k], k});
    std::sort(all.begin(), all.end(), [](const C& a, const C& b) {
      return a.s != b.s ? a.s < b.s : a.ep != b.ep ? a.ep < b.ep : a.st < b.st;
    });
    const size_t base_n = c.free_slots() + c.pending_writebacks();
    size_t blocks = c.free_blocks(), hosts = c.host_units(), got = 0;
    std::vector<uint32_t> taken;
    // a clean copy is dropped (no block); else a free block, else (three-tier) a pinned home dropped to its file, else
    // (three-tier) the victim itself to its file; a two-tier victim without a block stops the acceptance
    auto take = [&](uint32_t k) {
      if (c.backed(k)) {  // its block becomes a pinned home (three-tier: droppable from now on)
        taken.push_back(k);
        ++got;
        if (disk_tier) ++hosts;
        return true;
      }
      if (!blocks && disk_tier && hosts) {
        --hosts;
        ++blocks;
      }
      if (!blocks && disk_tier) {
        taken.push_back(k);
        ++got;
        return true;
      }
      if (!blocks) {
        x.short_blocks = true;
        return false;
      }
      --blocks;
      taken.push_back(k);
      x.victims.push_back(k);
      ++got;
      return true;
    };
    for (const C& a : all) {
      if (base_n + got >= c.spare_target()) break;
      if (!take(a.k)) break;
    }
    if (pressure_recent && !x.short_blocks) {
      std::sort(all.begin(), all.end(), [](const C& a, const C& b) { return a.st > b.st; });
      uint32_t n = 0;
      for (const C& a : all) {
        if (n >= pressure_recent) break;
        if (std::find(taken.begin(), taken.end(), a.k) != taken.end()) continue;
        if (!take(a.k)) break;
        ++n;
      }
    }
    return x;
  }
};

// What the scored runs exercised at their boundaries (the oracle's coverage).
struct OracleCoverage {
  uint64_t boundaries = 0, victims = 0, unsubmitted = 0, partial = 0, recorded = 0, wb_pending_skipped = 0;
  uint64_t kept_skipped = 0, short_boundaries = 0, pressure_boundaries = 0, target_changes = 0, coming_at_score = 0;
  uint64_t lru_boundaries = 0, fallbacks = 0;
};
OracleCoverage g_cov;

struct Outcome {
  std::string violation;
  ts::DgStats st;
  bool threw = false;
  std::vector<std::vector<uint32_t>> victims;
  // the end state: the core's homes (home, slot, block per key) and table, the device's table and slot contents
  std::vector<int64_t> homes;
  std::vector<int32_t> table, device_table, slots;
  uint64_t max_in_flight = 0, landed_out_of_order = 0;
};

std::vector<uint64_t> counters(const ts::DgStats& s) { return {s.tokens, s.lookups, s.misses, s.covered, s.fetched, s.prefetched, s.prefetch_used, s.resurrected, s.bank_serves, s.collected_mid, s.evicted, s.evict_short, s.writebacks_done, s.stalled, s.pred_invalid, s.covered_waits, s.predicted, s.predicted_true, s.miss_layers, s.wb_deferred, s.wb_cancelled, s.wb_forced, s.wb_boundary_flushed, s.wb_issued_bytes, s.wb_completed_bytes, s.wb_cancelled_bytes, s.wb_boundary_flushed_bytes, s.winpf_hints, s.winpf_hint_rows, s.prior_issued, s.prior_used, s.prior_late, s.prior_reverted, s.l1_candidates, s.l1_issued, s.l1_used, s.l1_late, s.l1_reverted, s.l1_priced_out, s.l1_skipped_ahead, s.prior_budget_stops}; }

void verify_bytes(DgCore& core, Scripted& t) {
  core.check();
  for (uint32_t k = 0; k < core.keys(); ++k) {
    CHECK_EQ(t.entry(k), core.table()[k]);
    if (core.home(k) == DgCore::Home::Device) CHECK_EQ(t.slot_holds(core.slot_of(k)), int32_t(k));
    else if (core.home(k) != DgCore::Home::File) {
      CHECK(core.home(k) == DgCore::Home::Host);
      CHECK_EQ(t.block_holds(core.block_of(k)), int32_t(k));
    }
    CHECK(!core.backed(k));  // the open state holds no clean copy
  }
}

uint64_t reset_d2d = 0, reset_h2h = 0, reset_staging = 0;

void reset_compare(DgCore& core, Scripted& t, const DgConfig& c, const std::vector<uint32_t>& order,
                   Violation& v, uint32_t seq, const std::vector<uint8_t>* disk = nullptr) {
  Violation fv;
  Scripted ft(core.keys(), c.persistent_slots + 2 * c.bank_slots, c.blocks + c.disk_staging, core.gated(), 901, &fv);
  DgCore fresh(c, &ft);
  fresh.place(order, disk);
  fresh.set_prior(ts::DgPrior(core.prior()));
  for (uint32_t k = 0; k < fresh.keys(); ++k) {
    if (fresh.slot_of(k) >= 0) ft.fill_slot(fresh.slot_of(k), int32_t(k));
    else if (fresh.block_of(k) >= 0) ft.fill_block(fresh.block_of(k), int32_t(k));
  }
  ft.set_table(fresh.table());
  const auto before = counters(core.stats());
  t.clear_issued();
  core.reset_open();
  for (const auto& op : t.trace()) {
    reset_d2d += op[1] == Scripted::D2D;
    reset_h2h += op[1] == Scripted::H2H;
    reset_staging += (op[1] == Scripted::H2D && op[3] == -1) ||
                     (op[1] == Scripted::D2H && op[3] == -1) ||
                     (op[1] == Scripted::H2H && (op[2] == -1 || op[3] == -1));
  }
  verify_bytes(core, t);
  CHECK(core.same_open_state(fresh));
  const auto after = counters(core.stats());
  for (size_t i = 0; i < before.size(); ++i) CHECK(after[i] >= before[i]);
  t.clear_issued();
  core.reset_open();
  CHECK(core.same_open_state(fresh));
  for (auto op : t.issued())
    CHECK(op != Scripted::H2D && op != Scripted::D2H && op != Scripted::D2HPart &&
          op != Scripted::D2D && op != Scripted::H2H && op != Scripted::Read);

  // Equal scripted scheduler/completion/ahead choices, independent of the randomized prefix. Compare issue
  // arguments (not opaque event handles or the lifetime sequence domain) and every deterministic counter delta.
  auto suffix = [&](DgCore& x, Scripted& tr) {
    tr.rng().seed(901);
    tr.set_slow_d2h(false);
    tr.clear_issued();
    ts::DgWinpf w;
    w.enable = true;
    w.top_k = 1;
    w.l1_pmin = 0.8f;
    x.set_winpf(w);
    for (uint32_t tok = 1; tok <= 6; ++tok) {
      int32_t sym = int32_t(tok % 16);
      x.window_hint(&sym, nullptr, 1);
      x.begin_token(seq + tok);
      for (uint32_t g = 0; g < x.gated(); ++g) {
        uint32_t id = (tok * 7 + g * 3) % x.units(g);
        uint64_t mask = tr.entry(x.key(g, id)) < 0 ? 1 : 0;
        x.serve(g, &id, 1, mask);
        if (g + 1 < x.gated()) {
          uint32_t pred = (tok * 11 + g) % x.units(g + 1);
          x.serve_prediction(g + 1, &pred, 1, 1, tok % 3 == 0);
        }
        x.issue_writebacks(g, [] { return false; });
        x.score_layer(g);
        tr.sync(DgTransport::kDem);
      }
      x.end_token({});
      x.check();
    }
    x.drain();
  };
  auto a0 = counters(core.stats()), b0 = counters(fresh.stats());
  suffix(core, t);
  suffix(fresh, ft);
  CHECK(t.trace() == ft.trace());
  auto a1 = counters(core.stats()), b1 = counters(fresh.stats());
  for (size_t i = 0; i < a0.size(); ++i) CHECK_EQ(a1[i] - a0[i], b1[i] - b0[i]);
  CHECK(v.what.empty());
  CHECK(fv.what.empty());
}

// Runs `tokens` tokens of random routes (a skewed popularity so the cache warms) through the core and the model.
Outcome run(const Shape& sh, int tokens, uint64_t seed, int mutation, unsigned jitter = 6, bool reset = false) {
  Outcome out;
  Violation v;
  const uint32_t G = uint32_t(sh.units.size());
  uint32_t keys = 0;
  for (uint32_t u : sh.units) keys += u;
  DgConfig c;
  c.n_units = sh.units;
  c.persistent_slots = sh.residents + sh.spare;
  c.bank_slots = sh.bank;
  c.blocks = keys - sh.residents - sh.disk + sh.host_spares;
  c.disk_tier = sh.disk_tier || sh.disk > 0;
  c.disk_prefetch = sh.disk_prefetch;
  c.home_blocks = c.disk_tier ? keys - sh.residents - sh.disk : 0;
  c.disk_staging = c.disk_tier ? sh.staging : 0;
  c.inclusive = sh.inclusive;
  c.disk_queue = sh.queue;
  c.spare_target = sh.spare;
  c.prefetch_reserve = std::min(24u, sh.spare / 4);
  c.prefetch_per_layer = sh.pf_layer;
  c.prefetch_per_token = sh.pf_token;
  c.wb_defer = sh.defer;
  c.wb_on_d2h = sh.on_d2h;
  c.wb_layer_bytes = sh.wb_bytes;
  c.wb_max_outstanding = sh.wb_out;
  c.scored = sh.scored;
  c.boundary_score = sh.boundary_score;
  c.scored_auto = sh.scored_auto;
  Scripted t(keys, c.persistent_slots + 2 * sh.bank, c.blocks + c.disk_staging, G, seed, &v);
  t.set_slow_d2h(sh.slow_d2h);
  t.set_async_reads(sh.async_reads, seed * 31u + 7u);
  t.set_eager(sh.eager);
  Oracle orc(sh.units);
  orc.disk_tier = c.disk_tier;
  std::vector<uint8_t> disk;  // three-tier: the plan's file tier, `sh.disk` random non-resident keys
  std::mt19937_64 vr(seed * 1000003u + 17u);  // scored runs: keep lists and targets (not the scheduler's stream)
  std::vector<uint32_t> keep;
  std::mt19937_64 kr(seed * 2654435761u + 5u);  // resident / skip runs: episodes and skipped layers
  std::vector<std::vector<uint32_t>> held(G);    // per gated layer: its make_resident set (keys), empty = none
  std::vector<std::vector<int32_t>> held_slot(G);
  uint32_t seq_ctr = 0;
  try {
    DgCore core(c, &t);
    core.set_mutation(mutation);
    core.set_pressure(sh.pressure_free, sh.pressure_recent);
    std::vector<uint32_t> res;
    for (uint32_t k = 0; k < sh.residents; ++k) res.push_back((k * 7919u) % keys);
    if (c.disk_tier) {
      disk.assign(keys, 0);
      std::vector<uint8_t> is_res(keys, 0);
      for (uint32_t k : res) is_res[k] = 1;
      std::vector<uint32_t> rest;
      for (uint32_t k = 0; k < keys; ++k)
        if (!is_res[k]) rest.push_back(k);
      std::mt19937_64 dr(seed * 7u + 3u);
      std::shuffle(rest.begin(), rest.end(), dr);
      for (uint32_t i = 0; i < sh.disk && i < rest.size(); ++i) disk[rest[i]] = 1;
      std::vector<uint32_t> order = res;  // the priority list names every key (as the context's)
      for (uint32_t k = 0; k < keys; ++k) order.push_back(k);
      res = order;
    }
    core.place(res, c.disk_tier ? &disk : nullptr);
    if (c.disk_tier && core.file_units() != sh.disk) v.set("place: file homes != the plan's disk units");
    for (uint32_t k = 0; k < keys; ++k) {  // the open path's fill
      if (core.slot_of(k) >= 0) t.fill_slot(core.slot_of(k), int32_t(k));
      if (core.block_of(k) >= 0) t.fill_block(core.block_of(k), int32_t(k));
    }
    t.set_table(core.table());
    core.check();
    auto& rng = t.rng();
    auto pick = [&](uint32_t g) {  // skewed: low ids are hot
      const uint32_t n = sh.units[g];
      const double x = std::uniform_real_distribution<double>(0, 1)(rng);
      return uint32_t(double(n) * x * x * x) % n;
    };
    if (sh.winpf) {
      ts::DgWinpf w;
      w.enable = true;
      w.top_k = sh.top_k;
      w.prior_layers = 2;
      w.prior_cap = 2;
      w.prior_pmin = 0.2f;
      w.l1_cap = 2;
      w.l1_min_votes = 1;
      w.l1_pmin = 0.4f;
      w.revert_unused = sh.revert;
      w.prior_window_us = sh.prior_window_us;
      w.copy_gbps = 1.0;  // 1 us x 1 GB/s = 1000 bytes: the tiny units of this model (8 KiB) need ~8 us each
      core.set_winpf(w);
      ts::DgPrior pr;  // 16 symbols x 2 layers x 4 ranked units, P(used) 0.1..0.9
      pr.layers = 2;
      pr.k = 4;
      for (int32_t sym = 0; sym < 16; ++sym) {
        pr.row.emplace(sym, uint32_t(sym));
        for (uint32_t l = 0; l < 2; ++l)
          for (uint32_t j = 0; j < 4; ++j) {
            pr.unit.push_back(j == 3 && sym % 5 == 0 ? 0xFFFFFFFFu : uint32_t(rng() % sh.units[l]));
            pr.p.push_back(0.1f + 0.2f * float(rng() % 5));
          }
      }
      core.set_prior(std::move(pr));
    }
    uint32_t win_rows = 1;
    auto hint = [&] {  // between tokens: the bonus row, then the drafts (symbols 0..19: 16..19 have no prior)
      win_rows = 1 + uint32_t(rng() % sh.max_rows);
      std::vector<int32_t> sym(win_rows);
      std::vector<float> w(win_rows);
      for (uint32_t r = 0; r < win_rows; ++r) {
        sym[r] = int32_t(rng() % 20);
        w[r] = r == 0 ? 1.0f : float(rng() % 11) / 10.0f;
      }
      core.window_hint(sym.data(), w.data(), 1);
      core.check();
      t.steps(jitter);
      if (win_rows > 1) core.window_hint(sym.data() + 1, w.data() + 1, win_rows - 1);
      core.check();
      if (rng() % 4 == 0) {  // a seed between tokens: its upload must not publish a prior copy before its bytes
        std::vector<uint32_t> ks;
        for (int i = 0; i < 3; ++i) ks.push_back(uint32_t(rng() % keys));
        core.seed(ks);
        core.check();
      }
    };
    if (sh.winpf) hint();
    // every unit at rest holds its bytes; a held (kept) unit is on the device in the slot it was given
    auto verify_homes = [&](const char* when) {
      for (uint32_t k = 0; k < keys && v.what.empty(); ++k) {
        const DgCore::Home h = core.home(k);
        if (h == DgCore::Home::Coming) continue;
        if (h == DgCore::Home::File) {
          if (t.entry(k) != -1) v.set(std::string("a file home is mapped ") + when);
          continue;
        }
        const bool on_slot = h == DgCore::Home::Device || h == DgCore::Home::Evicting;
        const int32_t holds = on_slot ? t.slot_holds(core.slot_of(k)) : t.block_holds(core.block_of(k));
        if (holds != int32_t(k)) v.set("unit " + std::to_string(k) + " lost its bytes " + when);
        if (core.backed(k) && t.block_holds(core.block_of(k)) != int32_t(k))
          v.set("unit " + std::to_string(k) + ": its clean copy lost its bytes " + when);
        if (t.entry(k) != core.table()[k]) v.set(std::string("device table != host mirror ") + when);
      }
      for (uint32_t g = 0; g < G; ++g)
        for (size_t i = 0; i < held[g].size(); ++i) {
          const uint32_t k = held[g][i];
          if (core.home(k) != DgCore::Home::Device || core.slot_of(k) != held_slot[g][i] ||
              t.entry(k) != held_slot[g][i])
            v.set(std::string("a held unit moved or left the device ") + when);
        }
    };
    // between tokens: release a held layer, or make a layer (or part of one) resident; a refusal changes nothing
    auto episode = [&] {
      std::vector<uint32_t> hl;
      for (uint32_t g = 0; g < G; ++g)
        if (!held[g].empty()) hl.push_back(g);
      if (!hl.empty() && kr() % 2 == 0) {
        const uint32_t g = hl[kr() % hl.size()];
        held[g].clear();
        held_slot[g].clear();
        ++g_res.released;
        return;
      }
      const uint32_t g = uint32_t(kr() % G);
      std::vector<uint32_t> ids;
      for (uint32_t u = 0; u < sh.units[g]; ++u)
        if (kr() % 2 == 0 || kr() % 3 != 0) ids.push_back(u);
      if (kr() % 2 == 0) {  // the whole layer
        ids.clear();
        for (uint32_t u = 0; u < sh.units[g]; ++u) ids.push_back(u);
      }
      std::shuffle(ids.begin(), ids.end(), kr);
      std::vector<uint32_t> other;
      for (uint32_t h = 0; h < G; ++h)
        if (h != g) other.insert(other.end(), held[h].begin(), held[h].end());
      std::vector<int64_t> homes0;
      for (uint32_t k = 0; k < keys; ++k) homes0.insert(homes0.end(), {int64_t(core.home(k)), core.slot_of(k), core.block_of(k)});
      const std::vector<int32_t> table0 = core.table();
      const auto st0 = counters(core.stats());
      const ts::DgStats s0 = core.stats();
      const uint32_t fs0 = core.free_slots(), fb0 = core.free_blocks(), in_flight0 = core.reads_in_flight();
      t.clear_issued();
      try {
        core.make_resident(g, ids, other);
      } catch (const ts::Error& e) {
        if (e.status() != SS_E_CAPACITY) throw;
        std::vector<int64_t> homes1;
        for (uint32_t k = 0; k < keys; ++k) homes1.insert(homes1.end(), {int64_t(core.home(k)), core.slot_of(k), core.block_of(k)});
        if (!t.issued().empty() || homes1 != homes0 || core.table() != table0 || counters(core.stats()) != st0 ||
            core.free_slots() != fs0 || core.free_blocks() != fb0 || core.stats().resident_units != s0.resident_units)
          v.set("a refused make_resident changed something");
        core.check();
        ++g_res.refusals;
        return;
      }
      core.check();
      const ts::DgStats& s1 = core.stats();
      // its copies count only in the resident fields (the write-backs it collected were decode's: not compared)
      if (s1.tokens != s0.tokens || s1.lookups != s0.lookups || s1.misses != s0.misses || s1.fetched != s0.fetched ||
          s1.prefetched != s0.prefetched || s1.evicted != s0.evicted || s1.bank_serves != s0.bank_serves ||
          s1.disk_reads - s0.disk_reads != in_flight0 || s1.disk_drops != s0.disk_drops ||
          s1.disk_discards != s0.disk_discards ||
          s1.inclusive_drops != s0.inclusive_drops || s1.resurrected != s0.resurrected)
        v.set("make_resident counted outside its resident fields");
      if (s1.resident_evicted - s0.resident_evicted != core.resident_victims().size())
        v.set("make_resident: resident_evicted != its victims");
      if (s1.resident_bytes - s0.resident_bytes != (s1.resident_units - s0.resident_units) * 8192)
        v.set("make_resident: resident_bytes != its units x the unit bytes");
      g_res.copied += s1.resident_units - s0.resident_units;
      g_res.evicted += core.resident_victims().size();
      for (uint32_t k : core.resident_victims()) {
        if (std::find(other.begin(), other.end(), k) != other.end()) v.set("make_resident evicted a kept unit");
        if (core.gated_of(k) == g && std::find(ids.begin(), ids.end(), k - core.key(g, 0)) != ids.end())
          v.set("make_resident evicted a unit it was asked for");
      }
      held[g].clear();
      held_slot[g].clear();
      for (uint32_t u : ids) {
        const uint32_t k = core.key(g, u);
        held[g].push_back(k);
        held_slot[g].push_back(core.slot_of(k));
      }
      ++g_res.calls;
      verify_homes("after make_resident");
    };
    for (int tok = 1; tok <= tokens && v.what.empty(); ++tok) {
      const uint32_t seq = ++seq_ctr;
      if (reset) {
        core.set_spare_target(uint32_t(rng() % (sh.spare + 1)));
        core.set_prefetch_budget(uint32_t(rng() % 7), uint32_t(rng() % 25));
        core.observe_copy(0.002);
      }
      if (sh.toggle_defer && rng() % 3 == 0) {
        core.set_wb_defer(!core.wb_defer());
        core.check();
      }
      keep.clear();
      if (sh.vary) {
        if (vr() % 4 == 0) {
          core.set_spare_target(uint32_t(vr() % (sh.spare + 1)));
          ++g_cov.target_changes;
        }
        for (uint32_t i = 0, n = uint32_t(vr() % 6); i < n; ++i) keep.push_back(uint32_t(vr() % keys));
      }
      if (sh.resident && kr() % 3 == 0) episode();
      std::vector<uint8_t> runm(G, 1);  // the exported layers: not a held one, nor a skipped one
      std::set<int32_t> held_slots;
      for (uint32_t g = 0; g < G; ++g) {
        if (!held[g].empty() || (sh.skip && kr() % sh.skip == 0)) runm[g] = 0;
        g_res.skipped_layers += held[g].empty() && !runm[g];
        keep.insert(keep.end(), held[g].begin(), held[g].end());
        held_slots.insert(held_slot[g].begin(), held_slot[g].end());
      }
      core.begin_token(seq, &runm);
      orc.begin();
      std::vector<uint32_t> pred;
      for (uint32_t g = 0; g < G && v.what.empty(); ++g) {
        if (!runm[g]) {  // not exported: no service; the held units are read through the table (no wait)
          pred.clear();
          const uint32_t rd0 = t.ready(g);
          for (const auto& hk : held)
            for (uint32_t k : hk) {
              const int32_t s = t.entry(k);
              if (s < 0 || t.slot_holds(s) != int32_t(k)) v.set("a held unit read wrong bytes");
              ++g_res.held_reads;
            }
          t.begin_read(held_slots);
          t.steps(jitter);
          t.end_read();
          if (t.ready(g) != rd0) v.set("a layer not exported got a ready");
          continue;
        }
        core.poll_reads();  // the service's wait for the export: finished prefetch reads get their copies now
        // export: the ids, the miss bits from the published table; a hit's slot must already hold its bytes
        const uint32_t n = sh.winpf ? win_rows * sh.top_k : 1 + uint32_t(rng() % sh.max_ids);
        std::vector<uint32_t> ids(n);
        uint64_t mask = 0;
        for (uint32_t i = 0; i < n; ++i) {
          ids[i] = (i > 0 && rng() % 4 == 0 && !pred.empty()) ? pred[rng() % pred.size()] : pick(g);
          if (ids[i] >= sh.units[g]) ids[i] = pick(g);
          const int32_t s = t.entry(core.key(g, ids[i]));
          if (s < 0) mask |= 1ull << i;
          else if (t.slot_holds(s) != int32_t(core.key(g, ids[i])))
            v.set("export saw a published entry before its bytes (layer " + std::to_string(g) + ")");
        }
        t.steps(jitter);
        core.serve(g, ids.data(), n, mask);
        if (sh.scored) orc.serve(g, ids.data(), n);
        core.check();
        t.steps(jitter);
        // the prediction for g + 1 (from the graph), served after the demand
        pred.clear();
        if (sh.winpf && g + 1 < G && runm[g + 1]) {  // rows x K ranked ids, the device ran ahead now and then
          for (uint32_t i = 0; i < win_rows * sh.predict_k; ++i) pred.push_back(pick(g + 1));
          core.serve_prediction(g + 1, pred.data(), uint32_t(pred.size()), n, rng() % 8 == 0);
          core.check();
        } else if (sh.predict && g + 1 < G && runm[g + 1]) {
          for (uint32_t i = 0; i < sh.predict; ++i) pred.push_back(pick(g + 1));
          core.serve_prediction(g + 1, pred.data(), uint32_t(pred.size()));
          core.check();
        }
        // the deferred write-back slice, after the ready and the prefetches; cut short now and then (the device ran
        // ahead: the next export was already seen). Never called without a deferred record (no draw without defer).
        bool later = false;  // (the service slices write-backs only while a later layer is exported)
        for (uint32_t h = g + 1; h < G; ++h) later = later || runm[h];
        if (later || g + 1 >= G) core.issue_writebacks(g, [&] { return rng() % 4 == 0; });
        core.check();
        core.score_layer(g);  // scored eviction: layer g ranked after its write-back slice (nothing under LRU)
        if (sh.scored)
          for (const auto& cd : core.score_list(g)) g_cov.coming_at_score += core.home(cd.key) == DgCore::Home::Coming;
        // wait: only if a miss; ready must come
        if (mask)
          while (int32_t(t.ready(g) - seq) < 0)
            if (!t.step()) {
              v.set("ready of layer " + std::to_string(g) + " never came");
              break;
            }
        // read: every routed unit through the table; the slots stay read until the layer ends
        std::set<int32_t> reading = held_slots;
        for (uint32_t id : ids) {
          const uint32_t k = core.key(g, id);
          const int32_t s = t.entry(k);
          if (s < 0) v.set("layer " + std::to_string(g) + " read an unpublished unit");
          else if (t.slot_holds(s) != int32_t(k)) v.set("layer " + std::to_string(g) + " read wrong bytes");
          else reading.insert(s);
        }
        t.begin_read(reading);
        t.steps(jitter);
        for (uint32_t id : ids) {
          const uint32_t k = core.key(g, id);
          const int32_t s = t.entry(k);
          if (s < 0 || t.slot_holds(s) != int32_t(k)) v.set("layer " + std::to_string(g) + " lost its unit while reading");
        }
        t.end_read();
        (void)pred;
      }
      // the end edge, then the boundary: the table is uploaded before any write-back is issued (design rev 3.4 §11.4)
      t.clear_issued();
      // the oracle at every boundary: scored runs always (also under a mutation it must catch), LRU runs without a
      // mutation (the recency order; the LRU mutation counts above stay those of the cache invariants and the model)
      Oracle::Expect want;
      const uint32_t rows = sh.auto_rows ? sh.auto_rows : (tok % 2 ? 1u : 6u);
      const uint64_t short0 = core.stats().evict_short, fb0 = core.stats().score_fallbacks;
      bool lists_ok = true;
      const bool probed = sh.scored || mutation == DgCore::kNoMutation;
      if (!sh.scored && probed)
        core.set_boundary_probe([&] {
          want = orc.boundary(core, keep, sh.pressure_recent, true);
          ++g_cov.lru_boundaries;
        });
      if (sh.scored)
        core.set_boundary_probe([&] {
          // a list not tagged with this token (only a skipped-rank fault leaves one): the recency order, counted
          for (uint32_t g = 0; g < G; ++g) lists_ok = lists_ok && core.score_list_epoch(g) == core.epoch();
          want = orc.boundary(core, keep, sh.pressure_recent, !lists_ok || (sh.scored_auto && rows == 1));
          ++g_cov.boundaries;
          const size_t dfr = core.deferred(), part = core.deferred_parts_issued() ? 1 : 0;
          g_cov.unsubmitted += dfr > part;
          g_cov.partial += part;
          g_cov.recorded += core.pending_writebacks() > dfr;
          for (uint32_t k = 0; k < keys; ++k) {
            const bool dev = core.home(k) == DgCore::Home::Device;
            g_cov.wb_pending_skipped += dev && core.wb_pending(k);
            g_cov.kept_skipped += dev && std::find(keep.begin(), keep.end(), k) != keep.end();
          }
        });
      core.end_token(keep, rows);
      out.victims.push_back(core.last_victims());
      if (probed) {
        core.set_boundary_probe(nullptr);
        if (!want.armed || core.last_victims() != want.victims)
          v.set("boundary " + std::to_string(tok) + ": victims differ from the oracle's (" +
                std::to_string(core.last_victims().size()) + " vs " + std::to_string(want.victims.size()) + ")");
        if ((core.stats().evict_short - short0 != 0) != want.short_blocks) v.set("boundary: evict_short");
      }
      if (sh.scored) {
        g_cov.victims += core.last_victims().size();
        g_cov.short_boundaries += want.short_blocks;
        g_cov.pressure_boundaries += sh.pressure_recent != 0;
        if ((core.stats().score_fallbacks != fb0) == lists_ok) v.set("scored boundary: fallback != invalid lists");
        g_cov.fallbacks += !lists_ok;
      }
      {
        const auto& is = t.issued();
        const auto up = std::find(is.begin(), is.end(), Scripted::Upload);
        if (std::find(is.begin(), up, Scripted::D2H) != up || std::find(is.begin(), up, Scripted::D2HPart) != up)
          v.set("a write-back was issued before the boundary's table upload");
      }
      core.check();
      t.steps(jitter);
      if (sh.resident) verify_homes("after the boundary");
      // every home holds its unit's bytes (a write-back in flight: the slot is authoritative)
      for (uint32_t k = 0; k < keys && v.what.empty(); ++k) {
        const DgCore::Home h = core.home(k);
        const bool on_slot = h == DgCore::Home::Device || h == DgCore::Home::Evicting;
        if (h == DgCore::Home::Coming) continue;  // a prior prefetch in flight: its slot fills on kPf
        if (h == DgCore::Home::File) {  // three-tier: nothing to hold, nothing mapped
          if (t.entry(k) != -1) v.set("a file home is mapped after the boundary");
          continue;
        }
        const int32_t holds = on_slot ? t.slot_holds(core.slot_of(k)) : t.block_holds(core.block_of(k));
        if (holds != int32_t(k)) v.set("unit " + std::to_string(k) + " lost its bytes at a boundary");
        if (core.backed(k) && t.block_holds(core.block_of(k)) != int32_t(k))
          v.set("unit " + std::to_string(k) + ": its clean copy lost its bytes");
        if (t.entry(k) != core.table()[k] && h != DgCore::Home::Coming)
          v.set("device table != host mirror after the boundary");
      }
      if (sh.winpf) hint();  // the next window's prior prefetches, before its export reads the table
    }
    if (reset) reset_compare(core, t, c, res, v, seq_ctr, c.disk_tier ? &disk : nullptr);
    if (c.disk_tier) core.drain();  // (the last window's prior reads in flight: copied before the end state is taken)
    t.drain_all();
    out.st = core.stats();
    for (uint32_t k = 0; k < keys; ++k) {
      out.homes.insert(out.homes.end(), {int64_t(core.home(k)), core.slot_of(k), core.block_of(k)});
      out.device_table.push_back(t.entry(k));
    }
    out.table = core.table();
    for (uint32_t s = 0; s < c.persistent_slots + 2 * sh.bank; ++s) out.slots.push_back(t.slot_holds(int32_t(s)));
    out.max_in_flight = t.max_in_flight();
    out.landed_out_of_order = t.landed_out_of_order();
  } catch (const ts::Error& e) {
    out.threw = true;
    v.set(std::string("exception: ") + e.what());
  }
  out.violation = v.what;
  return out;
}

const Shape kWarm{{64, 64, 48, 64}, 120, 24, 8, 24, 8};
const Shape kPressure{{64, 64, 48, 64}, 120, 16, 8, 16, 8, 2, 6};
const Shape kPredict{{64, 64, 48, 64}, 120, 24, 8, 24, 8, 0, 0, 6};
const Shape kPredictPressure{{64, 64, 48, 64}, 100, 16, 8, 16, 8, 1, 6, 6, 3, 12};
const Shape kSlowWriteback{{64, 64, 48, 64}, 120, 16, 8, 16, 8, 2, 6, 6, 4, 48, true};
// winpf: 3-row windows of top_k 2 (6 ids), 3 predicted per row; one with wb_defer and a tight prior budget, one
// with pressure, one without revert.
Shape winpf_shape(bool defer, bool pressure, bool revert, uint32_t window_us) {
  Shape s{{64, 64, 48, 64}, 110, 24, 8, 24, 6};
  s.winpf = true;
  s.max_rows = 3;
  s.top_k = 2;
  s.predict_k = 3;
  s.pf_layer = 4;
  s.pf_token = 48;
  s.defer = defer;
  s.revert = revert;
  s.prior_window_us = window_us;
  if (pressure) {
    s.pressure_free = 2;
    s.pressure_recent = 6;
    s.spare = 16;
    s.host_spares = 16;
  }
  return s;
}
const Shape kWinpf = winpf_shape(false, false, true, 1500);
const Shape kWinpfDefer = winpf_shape(true, false, true, 20);  // ~2 units of prior budget: the ledger binds
const Shape kWinpfPressure = winpf_shape(true, true, true, 1500);
const Shape kWinpfKeep = winpf_shape(false, false, false, 1500);
const Shape* const kShapes[] = {&kWarm, &kPressure, &kPredict, &kPredictPressure, &kSlowWriteback,
                                &kWinpf, &kWinpfDefer, &kWinpfPressure, &kWinpfKeep};

}  // namespace

TEST(dg_clean_runs) {
  for (const Shape* sh : kShapes)
    for (uint64_t seed = 1; seed <= 12; ++seed) {
      Outcome o = run(*sh, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
    }
}

TEST(dg_rare_paths_exercised) {
  ts::DgStats sum{};
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    for (const Shape* sh : {&kPressure, &kPredictPressure, &kSlowWriteback}) {
      Outcome o = run(*sh, 40, seed, DgCore::kNoMutation);
      CHECK(o.violation.empty());
      sum.bank_serves += o.st.bank_serves;
      sum.resurrected += o.st.resurrected;
      sum.collected_mid += o.st.collected_mid;
      sum.covered += o.st.covered;
      sum.covered_waits += o.st.covered_waits;
      sum.prefetched += o.st.prefetched;
      sum.prefetch_used += o.st.prefetch_used;
      sum.evicted += o.st.evicted;
    }
  }
  std::printf("  banks %llu, resurrections %llu, mid-token collects %llu, covered %llu, prefetched %llu (used %llu), "
              "evicted %llu\n",
              (unsigned long long)sum.bank_serves, (unsigned long long)sum.resurrected,
              (unsigned long long)sum.collected_mid, (unsigned long long)sum.covered,
              (unsigned long long)sum.prefetched, (unsigned long long)sum.prefetch_used,
              (unsigned long long)sum.evicted);
  CHECK(sum.bank_serves > 0);
  CHECK(sum.resurrected > 0);
  CHECK(sum.collected_mid > 0);
  CHECK(sum.covered > 0);
  CHECK(sum.covered_waits > 0);  // the demand stream waited on a prefetch's copies (the path kNoCover breaks)
  CHECK(sum.prefetch_used > 0);
  CHECK(sum.evicted > 0);
}

TEST(dg_winpf_paths_exercised) {
  ts::DgStats sum{};
  for (uint64_t seed = 1; seed <= 12; ++seed)
    for (const Shape* sh : {&kWinpf, &kWinpfDefer, &kWinpfPressure, &kWinpfKeep}) {
      Outcome o = run(*sh, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
      for (auto f : {&ts::DgStats::winpf_hints, &ts::DgStats::prior_issued, &ts::DgStats::prior_used,
                     &ts::DgStats::prior_late, &ts::DgStats::prior_reverted, &ts::DgStats::l1_candidates,
                     &ts::DgStats::l1_issued, &ts::DgStats::l1_used, &ts::DgStats::l1_late, &ts::DgStats::l1_reverted,
                     &ts::DgStats::l1_priced_out, &ts::DgStats::l1_skipped_ahead, &ts::DgStats::covered_waits,
                     &ts::DgStats::resurrected, &ts::DgStats::wb_cancelled})
        sum.*f += o.st.*f;
    }
  std::printf("  hints %llu, prior issued %llu used %llu late %llu reverted %llu; l1 candidates %llu issued %llu used %llu "
              "late %llu reverted %llu priced out %llu ahead %llu\n",
              (unsigned long long)sum.winpf_hints, (unsigned long long)sum.prior_issued,
              (unsigned long long)sum.prior_used, (unsigned long long)sum.prior_late,
              (unsigned long long)sum.prior_reverted, (unsigned long long)sum.l1_candidates,
              (unsigned long long)sum.l1_issued, (unsigned long long)sum.l1_used, (unsigned long long)sum.l1_late,
              (unsigned long long)sum.l1_reverted, (unsigned long long)sum.l1_priced_out,
              (unsigned long long)sum.l1_skipped_ahead);
  CHECK(sum.prior_issued > 0);
  CHECK(sum.prior_used > 0);
  CHECK(sum.prior_late > 0);
  CHECK(sum.prior_reverted > 0);
  CHECK(sum.l1_issued > 0);
  CHECK(sum.l1_used > 0);
  CHECK(sum.l1_late > 0);
  CHECK(sum.l1_reverted > 0);
  CHECK(sum.l1_priced_out > 0);
  CHECK(sum.l1_skipped_ahead > 0);
}

// winpf off (the default DgWinpf) is today's core: identical counters on the same seeds as a run without winpf code.
TEST(dg_winpf_off_is_today) {
  for (const Shape* sh : {&kPredict, &kPredictPressure}) {
    Outcome a = run(*sh, 30, 5, DgCore::kNoMutation);
    CHECK(a.violation.empty());
    CHECK_EQ(a.st.winpf_hints + a.st.prior_issued + a.st.l1_issued + a.st.l1_candidates, 0ull);
  }
}

// A window export that is not rows x top_k fails the token (the engine broke its declared layout).
TEST(dg_winpf_layout_refused) {
  Violation v;
  Scripted t(40, 16, 40, 2, 3, &v);
  DgConfig c;
  c.n_units = {20, 20};
  c.persistent_slots = 8;
  c.bank_slots = 8;
  c.blocks = 40;
  c.spare_target = 8;
  DgCore core(c, &t);
  core.place({});
  ts::DgWinpf w;
  w.enable = true;
  w.top_k = 2;
  core.set_winpf(w);
  const int32_t sym[2] = {1, 2};
  core.window_hint(sym, nullptr, 2);
  core.begin_token(1);
  const uint32_t ids[3] = {1, 2, 3}, ok4[4] = {1, 2, 3, 4}, pred[4] = {4, 5, 6, 7};
  bool threw = false;
  try {
    core.serve(0, ids, 3, 0);  // 3 ids in a hinted window: not rows x 2
  } catch (const ts::Error& e) {
    threw = e.status() == SS_E_BACKEND;
  }
  CHECK(threw);
  core.serve(0, ok4, 4, 0);  // 2 rows x 2
  threw = false;
  try {
    core.serve_prediction(1, pred, 3, 4, false);  // 3 predicted ids for 2 rows
  } catch (const ts::Error& e) {
    threw = e.status() == SS_E_BACKEND;
  }
  CHECK(threw);
}

// Each mutation must be caught (model violation or invariant) on some seed of the shapes that exercise its path.
TEST(dg_mutations_caught) {
  const struct {
    int m;
    const char* name;
  } muts[] = {{DgCore::kEntryBeforeCopy, "entry before copy"},
              {DgCore::kNoCover, "no cover wait"},
              {DgCore::kEarlyEventRecycle, "event recycled early"},
              {DgCore::kNoBankUnpublish, "bank alias not unpublished"},
              {DgCore::kEarlySourceFree, "coming source block freed early"},
              {DgCore::kFreeResurrectedSlot, "resurrected slot freed"},
              {DgCore::kEarlyWritebackBlock, "write-back block reused early"},
              {DgCore::kReevictPending, "re-evict with a pending write-back"},
              {DgCore::kRevertKeepsEntry, "winpf revert keeps the table entry"}};
  // (kSeedNoPfSync: random seeds rarely move a unit; dg_winpf_seed_after_hint catches it deterministically)
  for (const auto& mu : muts) {
    int caught = 0, runs = 0;
    for (const Shape* sh : kShapes)
      for (uint64_t seed = 1; seed <= 12; ++seed) {
        ++runs;
        if (!run(*sh, 40, seed, mu.m).violation.empty()) ++caught;
      }
    std::printf("  %-36s caught in %d/%d runs\n", mu.name, caught, runs);
    CHECK(caught > 0);
  }
}

TEST(dg_refusals) {
  Violation v;
  Scripted t(10, 10, 10, 1, 1, &v);
  DgConfig c;
  c.n_units = {10};
  c.persistent_slots = 4;
  c.spare_target = 5;
  bool threw = false;
  try {
    DgCore core(c, &t);
  } catch (const ts::Error& e) {
    threw = e.status() == SS_E_INVALID;
  }
  CHECK(threw);  // the spare target cannot exceed the slots
  c.spare_target = 1;
  c.blocks = 5;  // 10 units - 3 residents need 7 blocks
  DgCore core(c, &t);
  threw = false;
  try {
    core.place({0, 1, 2});
  } catch (const ts::Error& e) {
    threw = e.status() == SS_E_CAPACITY;
  }
  CHECK(threw);
}

// A demand id outside its layer throws before any key is computed; a predicted one is dropped and counted.
TEST(dg_ids_validated) {
  Violation v;
  Scripted t(30, 12, 30, 2, 3, &v);
  DgConfig c;
  c.n_units = {10, 20};
  c.persistent_slots = 4;
  c.bank_slots = 4;
  c.blocks = 30;
  c.spare_target = 4;  // R = 0: every unit starts in RAM
  DgCore core(c, &t);
  core.place({});
  core.check();
  CHECK_EQ(core.on_device(), 0u);
  core.begin_token(1);
  const uint32_t bad[2] = {3, 10};
  bool threw = false;
  try {
    core.serve(0, bad, 2, 3);
  } catch (const ts::Error& e) {
    threw = e.status() == SS_E_INVALID;
  }
  CHECK(threw);
  core.check();  // nothing was fetched for the valid id either
  CHECK_EQ(core.stats().fetched, 0ull);
  const uint32_t pred[3] = {19, 25, 19};
  core.serve_prediction(1, pred, 3);
  CHECK_EQ(core.stats().pred_invalid, 1ull);
  CHECK_EQ(core.stats().prefetched, 1ull);
  core.end_token({});
  core.check();
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }

// ---- design rev 3.4 §11.6-11.8: the replica's counter definitions, seed order, run-time budgets ----
namespace {

// 3 gated layers x 16 units; R persistent residents + D spares (target T), 2 x 4 bank slots, enough blocks.
struct Mini {
  static constexpr uint32_t G = 3, U = 16, K = G * U;
  Violation v;
  ts::DgConfig c;
  std::unique_ptr<Scripted> t;
  std::unique_ptr<DgCore> core;
  // Q: spare blocks (default D, the planner's usual grant; may be below D)
  Mini(uint32_t R, uint32_t D, uint32_t T, const std::vector<uint32_t>& order, uint32_t pf_layer = 4,
       bool reserve_auto = false, int Q = -1, bool defer = false, uint64_t wb_bytes = 0, uint64_t wb_out = 0,
       bool on_d2h = false) {
    c.n_units.assign(G, U);
    c.persistent_slots = R + D;
    c.bank_slots = 4;
    c.blocks = K - R + (Q < 0 ? D : uint32_t(Q));
    c.spare_target = T;
    c.prefetch_reserve_auto = reserve_auto;
    c.prefetch_reserve = reserve_auto ? 0 : 1;
    c.prefetch_per_layer = pf_layer;
    c.prefetch_per_token = 48;
    c.wb_defer = defer;
    c.wb_layer_bytes = wb_bytes;
    c.wb_max_outstanding = wb_out;
    c.wb_on_d2h = on_d2h;
    t = std::make_unique<Scripted>(K, c.persistent_slots + 2 * c.bank_slots, c.blocks, G, 7, &v);
    core = std::make_unique<DgCore>(c, t.get());
    core->place(order);
    for (uint32_t k = 0; k < K; ++k) {
      if (core->slot_of(k) >= 0) t->fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t->fill_block(core->block_of(k), int32_t(k));
    }
    t->set_table(core->table());
  }
  // the export's mask from the published table
  uint64_t mask(uint32_t g, const std::vector<uint32_t>& ids) const {
    uint64_t m = 0;
    for (size_t i = 0; i < ids.size(); ++i)
      if (t->entry(core->key(g, ids[i])) < 0) m |= 1ull << i;
    return m;
  }
  void serve(uint32_t g, const std::vector<uint32_t>& ids) { core->serve(g, ids.data(), uint32_t(ids.size()), mask(g, ids)); }
  // as the service: the layer, then its deferred write-back slice; check() after each
  uint64_t serve_w(uint32_t g, const std::vector<uint32_t>& ids, const std::function<bool()>& ahead = nullptr) {
    const uint64_t mk = mask(g, ids);
    core->serve(g, ids.data(), uint32_t(ids.size()), mk);
    core->check();
    core->issue_writebacks(g, ahead);
    core->check();
    return mk;
  }
  // After t->drain_all(): every home holds its unit's bytes (every part of a block), the device table == the mirror.
  bool homes_ok() const {
    for (uint32_t k = 0; k < K; ++k) {
      const DgCore::Home h = core->home(k);
      const bool on_slot = h == DgCore::Home::Device || h == DgCore::Home::Evicting || h == DgCore::Home::Coming;
      const int32_t holds = on_slot ? t->slot_holds(core->slot_of(k)) : t->block_holds(core->block_of(k));
      if (holds != int32_t(k) || t->entry(k) != core->table()[k]) return false;
    }
    return true;
  }
};

std::vector<uint32_t> keys_in_order(uint32_t n) {
  std::vector<uint32_t> v(n);
  for (uint32_t k = 0; k < n; ++k) v[k] = k;
  return v;
}

}  // namespace

// A repeated masked id is fetched at its first occurrence and covered at the later ones (site.cu, export order).
TEST(dg_counters_occurrence_order) {
  Mini m(8, 8, 8, keys_in_order(8));  // keys 0..7 resident: layer 1 is all absent
  m.core->begin_token(1);
  const ts::DgStats a = m.core->stats();
  m.serve(1, {5, 5, 9});  // 5 twice (masked twice), 9 once
  const ts::DgStats b = m.core->stats();
  CHECK_EQ(b.fetched - a.fetched, 2ull);  // 5 (first), 9
  CHECK_EQ(b.covered - a.covered, 1ull);  // 5 (second: coming)
  CHECK_EQ(b.covered_waits - a.covered_waits, 0ull);  // a demand copy, not a prefetch to wait for
  CHECK_EQ(b.misses - a.misses, 2ull);  // unique ids with a masked occurrence
  CHECK_EQ(b.lookups - a.lookups, 2ull);
  m.core->end_token({});
  m.core->check();
  CHECK(m.v.what.empty());
}

// predicted = the deduplicated valid prediction (whatever the budget lets through); predicted_true = unique ids of
// the next layer found in it; a prediction is for the next layer only; miss_layers counts layers with a miss.
TEST(dg_counters_prediction_and_miss_layers) {
  for (uint32_t budget : {4u, 0u}) {
    Mini m(8, 8, 8, keys_in_order(8), budget);
    m.core->begin_token(1);
    m.serve(0, {0, 1});  // resident: no miss
    m.core->serve_prediction(1, std::vector<uint32_t>{3, 4, 4, 99}.data(), 4);
    CHECK_EQ(m.core->stats().predicted, 2ull);
    CHECK_EQ(m.core->stats().pred_invalid, 1ull);
    m.serve(1, {3, 7, 3});
    CHECK_EQ(m.core->stats().predicted_true, 1ull);  // 3 (unique), not 7
    m.serve(2, {3, 4});  // no prediction was made for layer 2
    CHECK_EQ(m.core->stats().predicted_true, 1ull);
    CHECK_EQ(m.core->stats().miss_layers, 2ull);  // layers 1 and 2
    if (budget == 0) CHECK_EQ(m.core->stats().prefetched, 0ull);
    m.core->serve_prediction(1, std::vector<uint32_t>{6}.data(), 1);  // made last in token 1
    m.core->end_token({});
    m.core->begin_token(2);
    m.serve(1, {6});  // a new token: that prediction is gone (begin_token clears it)
    CHECK_EQ(m.core->stats().predicted_true, 1ull);
    m.core->end_token({});
    m.core->check();
    CHECK(m.v.what.empty());
  }
}

// The seed order picks the residents (the first persistent_slots - T distinct keys) and is the recency order: the
// first evictions are its last entries (cache.cpp:236-256).
// Fewer spare blocks than spare slots (Q < D, as the engine caches run): a boundary that needs more victims than free
// blocks writes back what fits and keeps the rest resident (evict_short), never a victim without a block.
TEST(dg_spare_blocks_below_slots) {
  Mini m(8, 4, 4, keys_in_order(8), 4, false, 1);
  m.core->set_pressure(0, 6);  // the boundary also evicts the 6 most recently used
  m.core->begin_token(1);
  m.serve(0, {0, 1, 2, 3});  // residents only: no promotion frees a block
  m.core->end_token({});
  CHECK_EQ(m.core->stats().evicted, 1ull);
  CHECK_EQ(m.core->stats().evict_short, 1ull);
  CHECK_EQ(m.core->free_blocks(), 0u);
  m.core->check();
}

TEST(dg_seed_order_residents_and_lru) {
  std::vector<uint32_t> order = {40, 3, 17, 40, 29, 8, 33, 12, 5};  // 40 twice: skipped the second time
  for (uint32_t k = 0; k < Mini::K; ++k) order.push_back(k);
  Mini m(8, 8, 8, order);
  const uint32_t want[8] = {40, 3, 17, 29, 8, 33, 12, 5};
  for (uint32_t k : want) CHECK(m.core->slot_of(k) >= 0);
  CHECK_EQ(m.core->on_device(), 8u);
  // The 8 spares fill with 8 misses (most recent now), then a target of 4 free slots: the boundary evicts the four
  // least recent units = the last four seed entries; the fifth from the end stays.
  m.core->set_spare_target(4);
  m.core->begin_token(1);
  m.serve(0, {0, 1, 2, 4, 6, 7, 9, 10});
  m.core->end_token({});
  for (uint32_t k : {5u, 12u, 33u, 8u}) CHECK(m.core->home(k) == DgCore::Home::Evicting);
  CHECK(m.core->home(29) == DgCore::Home::Device);
  CHECK(m.core->home(0) == DgCore::Home::Device);
  m.t->drain_all();
  m.core->check();
  CHECK(m.v.what.empty());
}

// An open-time spare target T < D places R + D - T units (cache.cpp:204); the blocks are not short.
TEST(dg_spare_target_below_d_at_open) {
  Mini m(8, 8, 3, keys_in_order(Mini::K));
  CHECK_EQ(m.core->on_device(), 13u);
  CHECK_EQ(m.core->free_slots(), 3u);
  m.core->check();
}

// The default reserve follows the spare target, min(24, target / 4); an explicit one stays; the setters are bounded.
TEST(dg_budgets_reserve_rule) {
  Mini a(8, 120, 120, keys_in_order(8), 4, true);
  CHECK_EQ(a.core->prefetch_reserve(), 24u);
  a.core->set_spare_target(40);
  CHECK_EQ(a.core->prefetch_reserve(), 10u);
  CHECK_EQ(a.core->spare_target(), 40u);
  Mini b(8, 120, 120, keys_in_order(8), 4, false);
  b.core->set_spare_target(40);
  CHECK_EQ(b.core->prefetch_reserve(), 1u);  // explicit: unchanged
  bool threw = false;
  try {
    b.core->set_spare_target(129);
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
  // a zero prefetch budget: no prefetch copy, the prediction is still counted
  b.core->set_prefetch_budget(0, 0);
  b.core->begin_token(1);
  b.core->serve_prediction(1, std::vector<uint32_t>{1, 2}.data(), 2);
  CHECK_EQ(b.core->stats().prefetched, 0ull);
  CHECK_EQ(b.core->stats().predicted, 2ull);
  b.core->end_token({});
}

// ---- dg_wb_defer: write-backs deferred into the next token (scripted parts of 2048 / 2048 / 4096 bytes) ----
namespace {

using Kinds = std::vector<Scripted::Kind>;
constexpr Scripted::Kind kH2D = Scripted::H2D, kEntry = Scripted::Entry, kReady = Scripted::Ready,
                         kRecord = Scripted::Record, kPart = Scripted::D2HPart, kUpload = Scripted::Upload;
const std::function<bool()> kAhead = [] { return true; };  // the device already exported the next layer

// R = 8 residents (keys 0..7, slots 0..7, key 7 least recent), D = 12 spares, target 8. Token 1 misses 8 units of
// layer 1: 4 spares stay free, and the boundary defers 4 victims, 7, 6, 5, 4 (slots 7..4), nothing copied.
std::unique_ptr<Mini> deferred_four(uint64_t wb_bytes = 0, uint64_t wb_out = 0, bool on_d2h = false) {
  auto m = std::make_unique<Mini>(8, 12, 8, keys_in_order(8), 4, false, -1, true, wb_bytes, wb_out, on_d2h);
  m->core->begin_token(1);
  m->serve_w(1, {0, 1, 2, 3, 4, 5, 6, 7});
  m->t->clear_issued();
  m->core->end_token({});
  m->core->check();
  return m;
}

// Every shape of the model run with the write-backs deferred, on either stream, under tiny / unequal byte budgets and
// a small outstanding cap.
std::vector<Shape> defer_shapes() {
  std::vector<Shape> v;
  for (const Shape* sh : kShapes) {
    Shape a = *sh;
    a.defer = true;
    v.push_back(a);
    a.on_d2h = true;
    v.push_back(a);
    a.on_d2h = false;
    a.wb_bytes = 1;
    v.push_back(a);
    a.wb_bytes = 3000;
    a.wb_out = 2048;
    v.push_back(a);
    a.on_d2h = true;
    a.slow_d2h = true;
    v.push_back(a);
  }
  return v;
}

}  // namespace

TEST(dg_defer_clean_runs_and_rare_paths) {
  ts::DgStats sum{};
  int runs = 0;
  for (const Shape& sh : defer_shapes())
    for (uint64_t seed = 1; seed <= 6; ++seed) {
      Outcome o = run(sh, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
      ++runs;
      sum.evicted += o.st.evicted;
      sum.wb_deferred += o.st.wb_deferred;
      sum.wb_cancelled += o.st.wb_cancelled;
      sum.wb_forced += o.st.wb_forced;
      sum.wb_boundary_flushed += o.st.wb_boundary_flushed;
      sum.resurrected += o.st.resurrected;
      sum.bank_serves += o.st.bank_serves;
      sum.wb_issued_bytes += o.st.wb_issued_bytes;
      sum.wb_cancelled_bytes += o.st.wb_cancelled_bytes;
    }
  std::printf("  %d runs: evicted %llu, deferred %llu, cancelled %llu, forced %llu, boundary-flushed %llu, "
              "resurrected %llu, banks %llu, bytes issued %llu, never copied %llu\n",
              runs, (unsigned long long)sum.evicted, (unsigned long long)sum.wb_deferred,
              (unsigned long long)sum.wb_cancelled, (unsigned long long)sum.wb_forced,
              (unsigned long long)sum.wb_boundary_flushed, (unsigned long long)sum.resurrected,
              (unsigned long long)sum.bank_serves, (unsigned long long)sum.wb_issued_bytes,
              (unsigned long long)sum.wb_cancelled_bytes);
  CHECK_EQ(sum.wb_deferred, sum.evicted);
  CHECK(sum.wb_cancelled > 0);
  CHECK(sum.wb_forced > 0);
  CHECK(sum.wb_boundary_flushed > 0);
  CHECK(sum.resurrected > sum.wb_cancelled);  // partly submitted or recorded write-backs resurrected too
  CHECK(sum.bank_serves > 0);
}

// A deferred slot reused before its write-back's event, and an event recorded before a write-back's last part (collect
// may promote a torn block): both must be caught.
TEST(dg_defer_mutations_caught) {
  const struct {
    int m;
    const char* name;
  } muts[] = {{DgCore::kEarlyDeferredSlot, "deferred slot reused early"},
              {DgCore::kRecordBeforeLastPart, "event before the last part"}};
  for (const auto& mu : muts) {
    int caught = 0, runs = 0;
    for (const Shape& sh : defer_shapes())
      for (uint64_t seed = 1; seed <= 6; ++seed) {
        ++runs;
        if (!run(sh, 40, seed, mu.m).violation.empty()) ++caught;
      }
    std::printf("  %-36s caught in %d/%d runs\n", mu.name, caught, runs);
    CHECK(caught > 0);
  }
}

// Order and budget: one part's credit per served layer, on kDem behind the layer's ready (the 4096-byte part still
// goes, the excess is a debt), victims in order. A boundary carries the older records (up to the larger of its own
// and the last boundary's victim counts, pending counted them) and submits only the excess, after its table upload.
TEST(dg_defer_order_budget_boundary) {
  auto m = deferred_four();
  CHECK_EQ(m->core->deferred(), 4u);
  CHECK_EQ(m->core->stats().wb_deferred, 4ull);
  CHECK_EQ(m->core->stats().evicted, 4ull);
  CHECK(m->t->issued() == Kinds({kUpload}));  // nothing copied at the boundary
  for (uint32_t k : {7u, 6u, 5u, 4u}) CHECK(m->core->home(k) == DgCore::Home::Evicting);
  CHECK_EQ(m->core->free_slots(), 4u);
  m->core->begin_token(2);
  m->t->clear_issued();
  CHECK_EQ(m->serve_w(1, {0}), 0ull);  // key 16: a hit
  CHECK(m->t->issued() == Kinds({kPart}));
  CHECK(m->t->wb_slots() == std::vector<int32_t>({7}));
  m->t->clear_issued();
  CHECK_EQ(m->serve_w(1, {8}), 1ull);  // key 24: a miss into a free slot, then the next part
  CHECK(m->t->issued() == Kinds({kH2D, kEntry, kReady, kPart}));
  CHECK_EQ(m->core->deferred_parts_issued(), 2u);
  m->t->clear_issued();
  m->serve_w(0, {1, 2});  // victim 7's last (larger) part and its event
  CHECK(m->t->issued() == Kinds({kPart, kRecord}));
  CHECK_EQ(m->core->deferred(), 3u);
  CHECK_EQ(m->core->wb_outstanding_bytes(), 8192ull);
  m->t->clear_issued();
  m->serve_w(0, {3});  // the 2048-byte debt eats this layer's credit
  CHECK(m->t->issued().empty());
  m->core->end_token({});  // collects 7; selects one victim (key 0); carries 6, 5, 4 (bound 4)
  m->core->check();
  CHECK(m->t->issued() == Kinds({kUpload}));
  CHECK_EQ(m->core->deferred(), 4u);
  CHECK(m->core->home(0) == DgCore::Home::Evicting);
  CHECK_EQ(m->core->stats().wb_boundary_flushed, 0ull);
  m->core->begin_token(3);
  m->t->clear_issued();
  m->core->end_token({});  // no victim; bound max(0, 1) = 1: the three oldest go after the upload
  m->core->check();
  CHECK(!m->t->issued().empty() && m->t->issued()[0] == kUpload);
  CHECK(m->t->wb_slots() == std::vector<int32_t>({6, 6, 6, 5, 5, 5, 4, 4, 4}));
  CHECK_EQ(m->core->stats().wb_boundary_flushed, 3ull);
  CHECK_EQ(m->core->stats().wb_boundary_flushed_bytes, 3ull * 8192);
  CHECK_EQ(m->core->deferred(), 1u);
  m->core->begin_token(4);
  m->core->end_token({});
  CHECK_EQ(m->core->deferred(), 0u);
  m->t->drain_all();
  m->core->begin_token(5);
  m->core->end_token({});
  m->core->check();
  m->t->drain_all();
  CHECK(m->homes_ok());
  CHECK(m->v.what.empty());
}

// Nothing served: the 4 deferred victims are carried (pending counted them: no new victim), then flushed by a boundary
// whose bound is 0, then collected.
TEST(dg_defer_pending_counts_deferred) {
  auto m = deferred_four();
  m->core->begin_token(2);
  m->core->end_token({});
  m->core->check();
  CHECK_EQ(m->core->deferred(), 4u);
  CHECK_EQ(m->core->stats().wb_boundary_flushed, 0ull);
  CHECK_EQ(m->core->stats().evicted, 4ull);
  m->core->begin_token(3);
  m->core->end_token({});
  m->core->check();
  CHECK_EQ(m->core->deferred(), 0u);
  CHECK_EQ(m->core->stats().wb_boundary_flushed, 4ull);
  m->t->drain_all();
  m->core->begin_token(4);
  m->core->end_token({});
  m->core->check();
  for (uint32_t k : {7u, 6u, 5u, 4u}) CHECK(m->core->home(k) == DgCore::Home::Host);
  CHECK(m->homes_ok());
  CHECK_EQ(m->core->stats().wb_completed_bytes, 4ull * 8192);
}

// Budgets: nothing when the device is ahead (also after a miss layer) and no credit saved; the predicate asked before
// every part; nothing after the last gated layer; an unequal budget of 3000 bytes (two parts, then the debt allows
// one); a 1-byte budget still makes progress; the outstanding cap holds parts back but keeps the credit.
TEST(dg_defer_budgets) {
  {
    auto m = deferred_four();
    m->core->begin_token(2);
    m->t->clear_issued();
    m->serve_w(0, {0}, kAhead);
    CHECK(m->t->issued().empty());
    m->serve_w(1, {8}, kAhead);  // a miss layer: its ready, no part
    CHECK(m->t->issued() == Kinds({kH2D, kEntry, kReady}));
    m->t->clear_issued();
    m->serve_w(0, {1});
    CHECK(m->t->issued() == Kinds({kPart}));  // one part's credit: nothing was saved
    m->t->clear_issued();
    m->core->issue_writebacks(2);  // the last gated layer
    CHECK(m->t->issued().empty());
    m->core->end_token({});
    m->core->check();
  }
  {
    auto m = deferred_four(1u << 20, 1u << 20);
    m->core->begin_token(2);
    m->t->clear_issued();
    int asked = 0;
    m->core->issue_writebacks(0, [&] { return ++asked > 2; });
    CHECK_EQ(m->t->issued().size(), 2u);  // asked before each part: the third saw the device ahead
    CHECK_EQ(asked, 3);
    m->core->end_token({});
    m->core->check();
  }
  {
    auto m = deferred_four(3000, 1u << 20);
    m->core->begin_token(2);
    m->t->clear_issued();
    m->core->issue_writebacks(0);
    CHECK_EQ(m->t->issued().size(), 2u);  // 3000: 2048, then 2048 into a debt of 1096
    m->t->clear_issued();
    m->core->issue_writebacks(0);
    CHECK(m->t->issued() == Kinds({kPart, kRecord}));  // 1904: the 4096-byte part
    m->core->check();
    m->core->end_token({});
    m->core->check();
  }
  {
    auto m = deferred_four(1, 1u << 20);
    m->core->begin_token(2);
    m->t->clear_issued();
    m->core->issue_writebacks(0);
    CHECK(m->t->issued() == Kinds({kPart}));
    m->t->clear_issued();
    for (int i = 0; i < 50; ++i) m->core->issue_writebacks(0);  // repaying 2047 bytes of debt
    CHECK(m->t->issued().empty());
    m->core->end_token({});
    m->core->begin_token(3);
    m->core->end_token({});
    m->core->check();
    CHECK_EQ(m->core->stats().wb_boundary_flushed, 4ull);
    m->t->drain_all();
    CHECK(m->homes_ok());
  }
  {
    auto m = deferred_four(0, 8192);  // one unit in flight at most
    m->core->begin_token(2);
    for (int i = 0; i < 3; ++i) m->core->issue_writebacks(0);  // victim 7 whole: 8192 in flight, a 2048 debt
    m->t->clear_issued();
    m->core->issue_writebacks(0);  // the debt repaid
    m->core->issue_writebacks(0);  // 2048 of credit, held back by the cap (the event has not completed)
    CHECK(m->t->issued().empty());
    m->t->drain_all();
    m->core->issue_writebacks(0);  // collected: one part (the credit was kept, not doubled)
    CHECK(m->t->issued() == Kinds({kPart}));
    m->core->check();
    m->core->end_token({});
    m->core->check();
  }
}

// Cancel before any part: no copy, the block returns, the entry is published on kDem (also for a prefetch); a demand
// of the prefetch-cancelled key before that publication ran is covered and its ready follows the entry on kDem. The
// returned blocks are reused at once by the next victims, and the cancelled keys are evicted again later: every home
// holds its bytes.
TEST(dg_defer_cancel) {
  auto m = deferred_four();
  const uint32_t blocks0 = m->core->free_blocks();
  m->core->begin_token(2);
  m->t->clear_issued();
  CHECK_EQ(m->serve_w(0, {5}), 1ull);
  CHECK(m->t->issued() == Kinds({kEntry, kReady, kPart}));  // no H2D; then victim 7's first part
  CHECK(m->core->home(5) == DgCore::Home::Device);
  CHECK_EQ(m->core->slot_of(5), 5);
  CHECK_EQ(m->core->stats().wb_cancelled, 1ull);
  CHECK_EQ(m->core->stats().resurrected, 1ull);
  CHECK_EQ(m->core->stats().wb_cancelled_bytes, 8192ull);
  CHECK_EQ(m->core->free_blocks(), blocks0 + 1);
  m->t->clear_issued();
  m->core->serve_prediction(0, std::vector<uint32_t>{4}.data(), 1);
  m->core->check();
  CHECK(m->t->issued() == Kinds({kEntry}));  // on kDem
  CHECK_EQ(m->core->stats().wb_cancelled, 2ull);
  CHECK_EQ(m->core->stats().prefetched, 0ull);
  const uint64_t mk = m->mask(0, {4});
  CHECK(mk != 0);  // nothing ran yet: the export still sees -1
  const uint64_t covered0 = m->core->stats().covered;
  m->t->clear_issued();
  m->serve_w(0, {4});
  CHECK_EQ(m->core->stats().covered, covered0 + 1);  // a Device key: no fetch, no wait
  CHECK(!m->t->issued().empty() && m->t->issued()[0] == kReady);  // on kDem, behind the entry above
  m->t->sync(DgTransport::kDem);
  CHECK_EQ(m->t->entry(4), 4);
  CHECK_EQ(m->t->slot_holds(4), 4);
  m->core->end_token({});  // carries 7, 6; selects 3, 2 into the blocks 4 and 5 just returned
  m->core->check();
  CHECK(m->core->home(3) == DgCore::Home::Evicting && m->core->home(2) == DgCore::Home::Evicting);
  m->core->set_pressure(0, 2);  // the next boundary also evicts the two most recent: 4 and 5 again
  m->core->begin_token(3);
  m->core->end_token({});
  m->core->check();
  CHECK(m->core->home(4) == DgCore::Home::Evicting && m->core->home(5) == DgCore::Home::Evicting);
  m->core->begin_token(4);
  m->core->end_token({});
  m->t->drain_all();
  m->core->begin_token(5);
  m->core->end_token({});
  m->core->check();
  m->t->drain_all();
  CHECK(m->homes_ok());
  CHECK(m->v.what.empty());
}

// A partly submitted victim resurrected at every cursor (1, 2 parts; 3 = every part, its record failed): its event
// goes after the submitted parts only, discarded; the key keeps its slot, wb_pending and the block stay until
// collection, the unsent parts are never sent.
TEST(dg_defer_partial_resurrection) {
  for (uint32_t cursor = 1; cursor <= 3; ++cursor) {
    auto m = deferred_four(0, 1u << 20);
    m->core->begin_token(2);
    for (uint32_t i = 1; i < cursor; ++i) m->core->issue_writebacks(0);
    if (cursor == 3) {
      m->t->fail_after(2);  // the last part is accepted, then the event (creation or record) fails
      bool threw = false;
      try {
        m->core->issue_writebacks(0);
      } catch (const ts::Error&) {
        threw = true;
      }
      CHECK(threw);
      m->t->fail_after(0);
    } else {
      m->core->issue_writebacks(0);
    }
    m->core->check();
    CHECK_EQ(m->core->deferred_parts_issued(), cursor);
    const int32_t block7 = m->core->block_of(7);
    m->t->clear_issued();
    m->serve_w(0, {7});
    CHECK(m->t->issued().size() >= 3 && m->t->issued()[0] == kRecord && m->t->issued()[1] == kEntry);
    CHECK(m->core->home(7) == DgCore::Home::Device);
    CHECK_EQ(m->core->slot_of(7), 7);
    CHECK_EQ(m->core->stats().wb_cancelled, 0ull);
    CHECK_EQ(m->core->stats().resurrected, 1ull);
    const uint64_t sent = cursor == 1 ? 2048 : cursor == 2 ? 4096 : 8192;
    CHECK_EQ(m->core->stats().wb_cancelled_bytes, 8192 - sent);
    m->core->end_token({});
    m->core->check();
    m->t->drain_all();
    m->core->begin_token(3);
    m->core->end_token({});
    m->core->check();
    CHECK(m->core->home(7) == DgCore::Home::Device);
    bool returned = false;  // the discarded block went back to the pool (and may already hold a new victim)
    for (uint32_t k = 0; k < Mini::K; ++k) returned = returned || m->core->block_of(k) == block7;
    CHECK(returned || m->core->free_blocks() > 0);
    m->t->drain_all();
    CHECK(m->homes_ok());
    CHECK(m->v.what.empty());
  }
}

// The resurrection's own record fails (its event's creation, then the record itself): the key stays evicting with
// its partly submitted record at the front, nothing leaks; the next demand resurrects it.
TEST(dg_defer_resurrection_record_fails) {
  for (int at = 1; at <= 2; ++at) {
    auto m = deferred_four();
    m->core->begin_token(2);
    m->core->issue_writebacks(0);  // victim 7: one part
    m->t->fail_after(at);
    bool threw = false;
    try {
      m->serve_w(0, {7});
    } catch (const ts::Error& e) {
      threw = std::string(e.what()).find("injected failure") != std::string::npos;
    }
    m->t->fail_after(0);
    CHECK(threw);
    m->core->check();
    CHECK(m->core->home(7) == DgCore::Home::Evicting);
    CHECK_EQ(m->core->deferred(), 4u);
    CHECK_EQ(m->core->deferred_parts_issued(), 1u);
    m->serve_w(0, {7});
    CHECK(m->core->home(7) == DgCore::Home::Device);
    m->core->end_token({});
    m->core->check();
    m->t->drain_all();
    CHECK(m->homes_ok());
  }
}

// Starvation: a demand finds no free slot while victims are deferred: it waits for the oldest live write-back in
// flight, or with none finishes the front and waits, until a persistent slot is free (past a discarded head, which
// returns only a block); with the pressure cap the capped demand still takes a bank slot.
TEST(dg_defer_forced) {
  for (int variant = 0; variant < 4; ++variant) {
    Mini m(8, 8, 8, keys_in_order(8), 4, false, -1, true);
    m.core->begin_token(1);
    m.serve_w(1, {0, 1, 2, 3, 4, 5, 6, 7});  // every spare taken: 8 victims deferred
    m.core->end_token({});
    m.core->check();
    CHECK_EQ(m.core->deferred(), 8u);
    CHECK_EQ(m.core->free_slots(), 0u);
    m.core->begin_token(2);
    int32_t want_slot = 7;
    if (variant == 1) {  // victim 7 partly submitted, then resurrected: a discarded record heads wbs_
      m.core->issue_writebacks(0);
      m.serve_w(0, {7});
      want_slot = 6;
    }
    if (variant == 2) m.core->set_pressure(1, 0);
    if (variant == 3)  // victim 7 whole and recorded, not completed: the demand waits for it, finishes nothing
      for (int i = 0; i < 3; ++i) m.core->issue_writebacks(0);
    m.t->clear_issued();
    m.serve_w(1, variant == 2 ? std::vector<uint32_t>{8, 9} : std::vector<uint32_t>{8});
    CHECK_EQ(m.core->stats().wb_forced, 1ull);
    CHECK_EQ(m.core->slot_of(m.core->key(1, 8)), want_slot);
    if (variant == 0) CHECK(m.t->issued() == Kinds({kPart, kPart, kPart, kRecord, kH2D, kEntry, kReady, kPart}));
    if (variant == 3) {
      CHECK(m.t->issued() == Kinds({kH2D, kEntry, kReady}));
      CHECK_EQ(m.core->deferred(), 7u);
    }
    CHECK_EQ(m.core->stats().bank_serves, variant == 2 ? 1ull : 0ull);
    m.core->end_token({});
    m.core->check();
    m.t->drain_all();
    m.core->begin_token(3);
    m.core->end_token({});
    m.core->check();
    m.t->drain_all();
    CHECK(m.homes_ok());
    CHECK(m.v.what.empty());
  }
}

// keep protects selection (7 and 6 are not chosen), and never cancels an older deferred victim (4, deferred at the
// last boundary and kept now, stays evicting and is written back).
TEST(dg_defer_keep) {
  Mini m(8, 12, 8, keys_in_order(8), 4, false, -1, true);
  m.core->begin_token(1);
  m.serve_w(1, {0, 1, 2, 3, 4, 5, 6, 7});
  m.core->end_token({7, 6});
  m.core->check();
  for (uint32_t k : {7u, 6u}) CHECK(m.core->home(k) == DgCore::Home::Device);
  for (uint32_t k : {5u, 4u, 3u, 2u}) CHECK(m.core->home(k) == DgCore::Home::Evicting);
  m.core->begin_token(2);
  m.core->end_token({4});
  m.core->check();
  CHECK(m.core->home(4) == DgCore::Home::Evicting);
  CHECK_EQ(m.core->deferred(), 4u);
  m.core->begin_token(3);
  m.core->end_token({4});  // flushed (bound 0), still not cancelled
  m.core->check();
  CHECK(m.core->home(4) == DgCore::Home::Evicting);
  m.t->drain_all();
  m.core->begin_token(4);
  m.core->end_token({});
  m.core->check();
  CHECK(m.core->home(4) == DgCore::Home::Host);
  m.t->drain_all();
  CHECK(m.homes_ok());
}

// Fewer spare blocks than slots (Q = 1) under MRU pressure: one victim deferred, the rest stay (evict_short); its
// cancellation returns the only block.
TEST(dg_defer_spare_blocks_below_slots) {
  Mini m(8, 4, 4, keys_in_order(8), 4, false, 1, true);
  m.core->set_pressure(0, 6);
  m.core->begin_token(1);
  m.serve_w(0, {0, 1, 2, 3});
  m.core->end_token({});
  m.core->check();
  CHECK_EQ(m.core->stats().evicted, 1ull);
  CHECK_EQ(m.core->stats().evict_short, 1ull);
  CHECK_EQ(m.core->deferred(), 1u);
  CHECK_EQ(m.core->free_blocks(), 0u);
  m.core->begin_token(2);
  m.serve_w(0, {3});
  CHECK_EQ(m.core->stats().wb_cancelled, 1ull);
  CHECK_EQ(m.core->free_blocks(), 1u);
  m.core->end_token({});
  m.core->check();
}

// A failure at every submission stage (each part, the event's creation, its record) of the in-token slices, the
// boundary and the drain: the core stays consistent (no block of a partly written write-back is promoted or freed),
// then a poisoned close drops the unsubmitted work without one more device call and drains the submitted copies.
TEST(dg_defer_failure_at_every_stage) {
  int failed = 0;
  for (int n = 1; n < 200; ++n) {
    auto m = deferred_four();
    m->core->begin_token(2);
    m->t->fail_after(n);
    bool threw = false;
    try {
      m->serve_w(1, {0});
      m->serve_w(0, {1, 2});
      m->serve_w(1, {8});
      m->core->end_token({});
      m->core->drain();
    } catch (const ts::Error& e) {
      threw = std::string(e.what()).find("injected failure") != std::string::npos;
      CHECK(threw);
    }
    m->t->fail_after(0);
    if (!threw) break;  // past the last submission
    ++failed;
    m->core->check();
    m->t->drain_all();
    for (uint32_t k = 0; k < Mini::K; ++k)  // a host home is never a torn block
      if (m->core->home(k) == DgCore::Home::Host) CHECK_EQ(m->t->block_holds(m->core->block_of(k)), int32_t(k));
    m->t->clear_issued();
    m->core->abandon_writebacks();
    m->t->drain_all();
    CHECK(m->t->issued().empty());
    CHECK_EQ(m->core->deferred(), 0u);
  }
  std::printf("  %d failure points\n", failed);
  CHECK(failed >= 10);
}

// dg_wb_stream (default 1): the same scripted tokens with the deferred write-backs on kDem and on
// kD2h. Pure transport: the same copies (H2D, D2H parts: slot, block, part; table entries; readies) in the same issue
// order, the same credit / debt / outstanding-cap decisions, cancellations, partial resurrections, boundary flushes and
// completed bytes, the same homes; only the stream of the parts and their completion events differs, and on kD2h every
// layer's first part sits behind a wait on an event recorded on kDem after that layer's ready (nothing before it).
// Copies run only at fixed drain points, so completion (collect, the cap) is identical in both runs.
namespace {
struct StreamRun {
  std::vector<std::vector<int64_t>> trace;
  ts::DgStats st;
  std::vector<int> home;
  std::vector<int32_t> slot, block;
  uint64_t outstanding_mid = 0, deferred_mid = 0, parts_mid = 0, outstanding_end = 0;
  bool ok = false;
};
StreamRun wb_stream_script(bool on_d2h, uint64_t wb_bytes, uint64_t wb_out) {
  StreamRun r;
  auto m = deferred_four(wb_bytes, wb_out, on_d2h);
  m->core->begin_token(2);
  m->serve_w(1, {0});     // a hit: credit, victim 7's first part(s)
  m->serve_w(1, {8});     // a miss into a free spare
  m->serve_w(0, {5});     // victim 5 cancelled before any part (no copy, its block returns)
  m->serve_w(0, {1, 2});  // more parts (debt / cap)
  r.outstanding_mid = m->core->wb_outstanding_bytes();
  r.deferred_mid = m->core->deferred();
  r.parts_mid = m->core->deferred_parts_issued();
  m->serve_w(0, {6});     // victim 6: partly submitted or untouched, resurrected
  m->t->drain_all();
  m->serve_w(1, {9, 10});  // collected mid-token, the cap frees up
  m->serve_w(2, {0});      // the last gated layer: no write-back
  m->t->drain_all();
  m->core->end_token({});  // carries / selects victims
  m->core->check();
  m->t->drain_all();
  m->core->begin_token(3);
  m->serve_w(0, {3, 11});
  m->serve_w(1, {12});
  m->t->drain_all();
  m->core->end_token({});  // the older excess flushed after the upload
  m->core->check();
  m->t->drain_all();
  m->core->begin_token(4);
  m->core->end_token({});
  m->t->drain_all();
  m->core->begin_token(5);
  m->core->end_token({});
  m->core->check();
  m->t->drain_all();
  r.trace = m->t->trace();
  r.st = m->core->stats();
  r.outstanding_end = m->core->wb_outstanding_bytes();
  for (uint32_t k = 0; k < Mini::K; ++k) {
    r.home.push_back(int(m->core->home(k)));
    r.slot.push_back(m->core->slot_of(k));
    r.block.push_back(m->core->block_of(k));
  }
  r.ok = m->homes_ok() && m->v.what.empty();
  return r;
}
// The copies and publications of a trace without their stream: {kind, a, b, key}.
std::vector<std::vector<int64_t>> copies_of(const std::vector<std::vector<int64_t>>& tr) {
  std::vector<std::vector<int64_t>> v;
  for (const auto& op : tr) {
    const auto k = Scripted::Kind(op[1]);
    if (k == kH2D || k == kPart || k == Scripted::D2H || k == kEntry || k == kReady || k == kUpload)
      v.push_back({op[1], op[2], op[3], op[4]});
  }
  return v;
}
}  // namespace

TEST(dg_defer_wb_stream_same_parts) {
  const struct {
    uint64_t bytes, out;
  } budgets[] = {{0, 0}, {3000, 1u << 20}, {1, 4096}, {0, 8192}, {12288, 2 * 8192 - 4096 + 1}};
  for (const auto& b : budgets) {
    const StreamRun dem = wb_stream_script(false, b.bytes, b.out);
    const StreamRun d2h = wb_stream_script(true, b.bytes, b.out);
    CHECK(dem.ok);
    CHECK(d2h.ok);
    CHECK(copies_of(dem.trace) == copies_of(d2h.trace));
    CHECK(dem.home == d2h.home);
    CHECK(dem.slot == d2h.slot);
    CHECK(dem.block == d2h.block);
    CHECK_EQ(dem.outstanding_mid, d2h.outstanding_mid);
    CHECK_EQ(dem.deferred_mid, d2h.deferred_mid);
    CHECK_EQ(dem.parts_mid, d2h.parts_mid);
    CHECK_EQ(dem.outstanding_end, d2h.outstanding_end);
    {  // every counter (all uint64_t) but the host's measured wait times
      static_assert(sizeof(ts::DgStats) % sizeof(uint64_t) == 0, "DgStats: uint64_t counters only");
      uint64_t a[sizeof(ts::DgStats) / 8], c[sizeof(ts::DgStats) / 8];
      std::memcpy(a, &dem.st, sizeof a);
      std::memcpy(c, &d2h.st, sizeof c);
      const size_t skip0 = offsetof(ts::DgStats, wb_forced_wait_ns) / 8, skip1 = offsetof(ts::DgStats, score_ns) / 8,
                   skip2 = offsetof(ts::DgStats, score_last_ns) / 8;
      for (size_t i = 0; i < sizeof a / 8; ++i) {
        if (i == skip0 || i == skip1 || i == skip2) continue;
        if (a[i] != c[i]) std::printf("  budget %llu/%llu: DgStats word %zu: %llu vs %llu\n", (unsigned long long)b.bytes,
                                      (unsigned long long)b.out, i, (unsigned long long)a[i], (unsigned long long)c[i]);
        CHECK_EQ(a[i], c[i]);
      }
    }
    CHECK(dem.st.wb_issued_bytes > 0);
    CHECK(dem.st.wb_cancelled > 0);
    CHECK_EQ(dem.st.wb_issued_bytes - dem.st.wb_completed_bytes, dem.outstanding_end);  // collected at boundaries
    // streams: kDem mode issues no wait and every part on kDem; kD2h mode every part on kD2h, an in-token layer's
    // first part right after {Record on kDem (behind everything the layer queued there, its ready included), Wait on
    // kD2h}, a boundary's flushed parts after the table upload (which synchronizes kDem)
    int waits = 0, d2h_parts = 0;
    for (const auto& op : dem.trace) {
      CHECK(Scripted::Kind(op[1]) != Scripted::Wait);
      if (Scripted::Kind(op[1]) == kPart) CHECK_EQ(op[0], int64_t(DgTransport::kDem));
    }
    bool behind_ready = false;  // since the last post-ready wait on kD2h
    for (size_t i = 0; i < d2h.trace.size(); ++i) {
      const auto& op = d2h.trace[i];
      const auto k = Scripted::Kind(op[1]);
      if (k == kRecord && op[0] == int64_t(DgTransport::kDem)) behind_ready = false;  // a new layer ready: wait again
      if (k == Scripted::Wait) {
        CHECK_EQ(op[0], int64_t(DgTransport::kD2h));
        CHECK(i >= 1 && Scripted::Kind(d2h.trace[i - 1][1]) == kRecord && d2h.trace[i - 1][0] == int64_t(DgTransport::kDem));
        behind_ready = true;
        ++waits;
      }
      if (k == kPart) {
        CHECK_EQ(op[0], int64_t(DgTransport::kD2h));
        CHECK(behind_ready);
        ++d2h_parts;
      }
      if (k == kUpload) behind_ready = true;  // a boundary's flush: the demand stream synced at the upload
    }
    CHECK(waits > 0);
    CHECK(d2h_parts > 0);
  }
}

// check_unit's drain between tokens: every deferred write-back submitted and synchronized; bookkeeping stays coherent.
TEST(dg_defer_drain) {
  auto m = deferred_four();
  m->core->drain();
  m->core->check();
  CHECK_EQ(m->core->deferred(), 0u);
  CHECK_EQ(m->core->stats().wb_issued_bytes, 4ull * 8192);
  m->core->begin_token(2);
  m->core->end_token({});
  m->core->check();
  CHECK(m->homes_ok());
}

// ---- winpf: deterministic edges ----
namespace {

// A prior for gated layer 0: symbol s -> units {s, s + 1} with P(used) 0.9 / 0.8.
ts::DgPrior tiny_prior() {
  ts::DgPrior pr;
  pr.layers = 1;
  pr.k = 2;
  for (int32_t s = 0; s < 16; ++s) {
    pr.row.emplace(s, uint32_t(s));
    pr.unit.push_back(uint32_t(s));
    pr.unit.push_back(uint32_t((s + 1) % 16));
    pr.p.push_back(0.9f);
    pr.p.push_back(0.8f);
  }
  return pr;
}

void winpf_on(DgCore& core, uint32_t top_k = 1, uint32_t cap = 1) {
  ts::DgWinpf w;
  w.enable = true;
  w.top_k = top_k;
  w.prior_layers = 1;
  w.prior_cap = cap;
  w.prior_pmin = 0.5f;
  w.prior_window_us = 1000000;  // no budget limit here
  w.copy_gbps = 1.0;
  core.set_winpf(w);
  core.set_prior(tiny_prior());
}

}  // namespace

// A prior prefetch of a deferred victim that has no part submitted cancels its write-back (no copy at all), of a
// partly submitted one discards it (the slot keeps the bytes, the block waits for the parts' event); neither counts as
// a prior copy (a recorded write-back's discard is today's resurrection path, covered by the oracle runs). Every state is checked and every home keeps its bytes.
TEST(dg_winpf_prior_resurrects_victims) {
  for (int state = 0; state < 2; ++state) {  // 0 unsubmitted, 1 partly submitted (carried across the boundary)
    auto m = deferred_four(0, 1u << 20);
    const uint32_t victim = 7;  // the deferred front (keys 7, 6, 5, 4)
    if (state > 0) {
      m->core->begin_token(2);
      m->serve_w(1, {0});  // a hit; its slice submits the front's first part
      m->core->end_token({});  // carried across the boundary (a099af1)
      CHECK(m->core->home(victim) == DgCore::Home::Evicting);
      m->core->check();
    }
    winpf_on(*m->core);
    const uint64_t cancelled0 = m->core->stats().wb_cancelled, res0 = m->core->stats().resurrected;
    m->t->clear_issued();
    const int32_t sym = int32_t(victim);  // the prior names unit 7 (and 8) of gated layer 0
    m->core->window_hint(&sym, nullptr, 1);
    m->core->check();
    CHECK(m->core->home(victim) == DgCore::Home::Device);
    CHECK_EQ(m->core->stats().resurrected, res0 + 1);
    if (state == 0) CHECK_EQ(m->core->stats().wb_cancelled, cancelled0 + 1);
    const auto& is = m->t->issued();
    CHECK(std::count(is.begin(), is.end(), Scripted::H2D) <= 1);  // only unit 8's copy, never the victim's
    m->t->drain_all();
    m->core->begin_token(3);
    m->serve_w(0, {victim});  // the export sees the published slot: a hit
    m->core->end_token({});
    m->core->check();
    m->t->drain_all();
    CHECK(m->homes_ok());
    CHECK(m->v.what.empty());
  }
}

// seed() after a hint: the prior copy is Coming with its entry in the mirror; the seed's whole-table upload happens only
// after the prefetch stream is synced, so the published entry always has its bytes (the mutation skips that sync and
// publishes a slot that does not hold the unit yet).
TEST(dg_winpf_seed_after_hint) {
  for (int mut = 0; mut < 2; ++mut) {
    Mini m(8, 12, 4, keys_in_order(8));
    if (mut) m.core->set_mutation(DgCore::kSeedNoPfSync);
    winpf_on(*m.core, 1, 6);
    const int32_t sym[3] = {8, 10, 12};  // units 8..13 of gated layer 0: on the host (12 free slots, spare target 4)
    m.t->clear_issued();
    m.core->window_hint(sym, nullptr, 3);
    CHECK(m.core->home(9) == DgCore::Home::Coming);
    CHECK_EQ(m.core->stats().prior_issued, 6ull);
    m.core->seed({m.core->key(1, 3)});  // a free slot above the spare target: seeds and uploads the table
    bool published_without_bytes = false;  // (the scheduler may run some prefetch copies while the upload syncs)
    for (uint32_t u = 8; u < 14; ++u) {
      const int32_t s = m.t->entry(u);
      published_without_bytes = published_without_bytes || (s >= 0 && m.t->slot_holds(s) != int32_t(u));
    }
    if (mut) CHECK(published_without_bytes);
    else CHECK(!published_without_bytes);
    m.t->drain_all();
    m.core->begin_token(1);
    m.serve(0, {9});
    m.core->end_token({});
    if (!mut) {
      m.core->check();
      CHECK(m.homes_ok());
      CHECK_EQ(m.core->stats().prior_used, 1ull);
      CHECK_EQ(m.core->stats().prior_reverted, 5ull);
    }
  }
}

// A prior copy still in flight at the export: masked, covered with a wait on its event before ready (late = 1), then
// read exact; an unused one in the same hinted window is reverted to its block at the boundary (no write-back).
TEST(dg_winpf_prior_late_and_revert) {
  Mini m(8, 12, 4, keys_in_order(8));
  winpf_on(*m.core, 1, 2);
  const int32_t sym = 11;  // units 11 and 12
  m.core->window_hint(&sym, nullptr, 1);
  CHECK_EQ(m.core->stats().prior_issued, 2ull);
  m.core->begin_token(1);
  CHECK(m.mask(0, {11}) == 1ull);  // the copy has not run: the export sees -1
  m.serve(0, {11});
  CHECK_EQ(m.core->stats().prior_late, 1ull);
  CHECK_EQ(m.core->stats().covered_waits, 1ull);
  m.t->sync(DgTransport::kDem);  // ready behind the wait: the slot holds the unit
  CHECK_EQ(m.t->slot_holds(m.t->entry(11)), 11);
  const uint64_t ev0 = m.core->stats().evicted;
  m.core->end_token({});
  m.core->check();
  CHECK(m.core->home(12) == DgCore::Home::Host);  // unused: reverted, its block kept
  CHECK_EQ(m.core->stats().prior_reverted, 1ull);
  CHECK_EQ(m.core->stats().prior_used, 1ull);
  CHECK_EQ(m.core->stats().evicted, ev0);
  m.t->drain_all();
  CHECK(m.homes_ok());
  CHECK(m.v.what.empty());
}

// The prior byte budget binds: with a budget of one unit and nothing queued, a 2-unit candidate list stops after one.
TEST(dg_winpf_prior_budget_binds) {
  Mini m(8, 12, 4, keys_in_order(8));
  ts::DgWinpf w;
  w.enable = true;
  w.top_k = 1;
  w.prior_layers = 1;
  w.prior_cap = 4;
  w.prior_pmin = 0.5f;
  w.copy_gbps = 1.0;
  w.prior_window_us = uint32_t((m.t->part_bytes(0) * 2 + m.t->part_bytes(2)) / 1000 + 1);  // one unit's bytes
  m.core->set_winpf(w);
  m.core->set_prior(tiny_prior());
  const int32_t sym = 12;
  m.core->window_hint(&sym, nullptr, 1);
  CHECK_EQ(m.core->stats().prior_issued, 1ull);
  CHECK_EQ(m.core->stats().prior_budget_stops, 1ull);
  m.t->drain_all();
}

// The prior budget counts the write-back bytes already queued: with one unit of write-back submitted (not collected)
// and a budget of two units, one prior copy is admitted and the next stops.
TEST(dg_winpf_budget_counts_queued_writebacks) {
  // write-backs on kD2h (behind a post-ready event), the scripted kD2h held: submitted parts stay queued
  auto m = std::make_unique<Mini>(8, 12, 8, keys_in_order(8), 4, false, -1, true, 0, 1u << 20, true);
  m->t->hold(DgTransport::kD2h, true);
  m->core->begin_token(1);
  m->serve_w(1, {0, 1, 2, 3, 4, 5, 6, 7});
  m->core->end_token({});  // 4 victims deferred
  m->core->begin_token(2);
  for (int i = 0; i < 4; ++i) m->serve_w(1, {0});  // hits: their slices submit parts on kD2h
  m->core->end_token({});
  const uint64_t queued = m->core->wb_outstanding_bytes();
  std::printf("  queued write-back bytes %llu\n", (unsigned long long)queued);
  CHECK(queued >= 8192);
  ts::DgWinpf w;
  w.enable = true;
  w.top_k = 1;
  w.prior_layers = 1;
  w.prior_cap = 4;
  w.prior_pmin = 0.5f;
  w.copy_gbps = 1.0;
  w.prior_window_us = uint32_t((queued + 8192) / 1000 + 1);  // room for exactly one unit next to the queued bytes
  m->core->set_winpf(w);
  m->core->set_prior(tiny_prior());
  const int32_t sym = 12;  // units 12, 13: on the host
  m->core->window_hint(&sym, nullptr, 1);
  CHECK_EQ(m->core->stats().prior_issued, 1ull);
  CHECK_EQ(m->core->stats().prior_budget_stops, 1ull);
  m->core->check();
  m->t->hold(DgTransport::kD2h, false);
  m->t->drain_all();
}

TEST(dg_reset_open_randomized) {
  for (const Shape* sh : kShapes)
    for (uint64_t seed = 1; seed <= 20; ++seed) {
      Outcome o = run(*sh, 1 + int(seed % 19), 600 + seed, DgCore::kNoMutation, 6, true);
      if (!o.violation.empty()) std::printf("  reset seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
    }
  // Minimum spare resources, including a fully occupied persistent arena. A single host spare is sufficient;
  // cycles use the dedicated staging block and do not borrow from the free lists.
  for (bool defer : {false, true})
    for (uint32_t spare : {0u, 1u})
      for (uint64_t seed = 1; seed <= 20; ++seed) {
        Shape sh{{8, 8, 8}, 6, spare, 2, 1, 2};
        sh.defer = defer;
        sh.on_d2h = seed % 2 == 0;
        sh.pressure_recent = 2;
        sh.predict = 2;
        Outcome o = run(sh, 12, seed, DgCore::kNoMutation, 3, true);
        CHECK(o.violation.empty());
      }
  // Filled by the randomized reset suite above: both physical permutations and closed cycles were exercised.
  std::printf("  reset D2D %llu, H2H %llu, staging %llu\n", (unsigned long long)reset_d2d,
              (unsigned long long)reset_h2h, (unsigned long long)reset_staging);
  CHECK(reset_d2d > 0 && reset_h2h > 0 && reset_staging > 0);
}

// persist: deferral switched on and off between tokens (set_wb_defer) keeps every invariant, every home's bytes and the
// reads exact; with a reset at the end the core is back to the open state (open mode restored).
// persist: residency() lists exactly the Device keys, most recent first; a core placed from it puts the same keys on
// the device (the next open's seed).
TEST(dg_residency_order) {
  Violation v, v2;
  DgConfig c;
  c.n_units = {8, 8, 8};
  c.persistent_slots = 14;
  c.spare_target = 4;
  c.blocks = 24;
  Scripted t(24, 14, 24, 3, 1, &v), t2(24, 14, 24, 3, 1, &v2);
  DgCore core(c, &t), again(c, &t2);
  std::vector<uint32_t> order;
  for (uint32_t k = 0; k < 24; ++k) order.push_back((k * 7) % 24);
  core.place(order);
  const auto res = core.residency();
  CHECK_EQ(res.size(), 10u);  // persistent_slots - spare_target
  for (size_t i = 0; i < res.size() && i < order.size(); ++i) CHECK_EQ(res[i], order[i]);  // open: the order itself
  again.place(res);
  CHECK(again.residency() == res);
}

TEST(dg_set_wb_defer_runtime) {
  for (bool start : {false, true})
    for (bool reset : {false, true})
      for (uint64_t seed = 1; seed <= 25; ++seed) {
        Shape sh{{8, 8, 8}, 10, 4, 2, 2, 2};  // max_ids <= bank slots (the core's precondition)
        sh.defer = start;
        sh.toggle_defer = true;
        sh.on_d2h = seed % 2 == 0;
        sh.predict = seed % 3 == 0 ? 2 : 0;
        Outcome o = run(sh, 30, 900 + seed, DgCore::kNoMutation, 6, reset);
        if (!o.violation.empty()) std::printf("  toggle seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
        CHECK(o.violation.empty());
      }
}

TEST(dg_reset_open_transport_failure_poison) {
  Violation v;
  DgConfig c;
  c.n_units = {4};
  c.persistent_slots = 2;
  c.spare_target = 1;
  c.blocks = 4;
  Scripted t(4, 2, 4, 1, 3, &v);
  DgCore core(c, &t);
  core.place({0});
  for (uint32_t k = 0; k < 4; ++k) {
    if (core.slot_of(k) >= 0) t.fill_slot(core.slot_of(k), int32_t(k));
    else t.fill_block(core.block_of(k), int32_t(k));
  }
  t.set_table(core.table());
  core.set_spare_target(0);
  CHECK_EQ(core.seed({1}), 1u);
  t.fail_reset_after(1);
  bool threw = false;
  try { core.reset_open(); } catch (const ts::Error&) { threw = true; }
  CHECK(threw);
  CHECK(core.reset_failed());
  threw = false;
  try { core.begin_token(1); } catch (const ts::Error&) { threw = true; }
  CHECK(threw);
  threw = false;
  try { core.reset_open(); } catch (const ts::Error&) { threw = true; }
  CHECK(threw);
}

TEST(dg_reset_open_resolved_blueprint) {
  Violation v, fv;
  DgConfig c;
  c.n_units = {8};
  c.persistent_slots = 4;
  c.spare_target = 1;
  c.blocks = 8;
  c.prefetch_reserve_auto = true;
  c.prefetch_per_layer = 2;
  c.prefetch_per_token = 7;
  Scripted t(8, 4, 8, 1, 9, &v), ft(8, 4, 8, 1, 9, &fv);
  DgCore core(c, &t), fresh(c, &ft);
  ts::DgWinpf open;
  open.copy_gbps = 19;
  core.set_winpf(open);
  fresh.set_winpf(open);
  const std::vector<uint32_t> order{6, 2, 6, 0, 7};  // duplicates and excess residents resolved at open
  core.place(order);
  fresh.place(order);
  for (uint32_t k = 0; k < 8; ++k) {
    if (core.slot_of(k) >= 0) t.fill_slot(core.slot_of(k), int32_t(k));
    else t.fill_block(core.block_of(k), int32_t(k));
  }
  t.set_table(core.table());
  core.set_spare_target(0);
  core.set_prefetch_budget(0, 0);
  CHECK_EQ(core.seed({1}), 1u);
  auto changed = open;
  changed.enable = true;
  changed.top_k = 1;
  changed.copy_gbps = 51;
  core.set_winpf(changed);
  core.observe_copy(0.001);
  core.reset_open();
  CHECK(core.same_open_state(fresh));
  CHECK_EQ(core.copy_gbps(), 19.0);
  CHECK_EQ(core.slot_of(6), 0);
  CHECK_EQ(core.slot_of(2), 1);
  CHECK_EQ(core.slot_of(0), 2);
  verify_bytes(core, t);
}

TEST(dg_reset_open_failure_at_every_stage) {
  int failures = 0;
  for (int n = 1; n < 200; ++n) {
    Mini m(8, 4, 4, keys_in_order(8), 4, false, 1, true, 1, 4096, true);
    m.core->set_pressure(1, 3);
    for (uint32_t tok = 1; tok <= 6; ++tok) {
      m.core->begin_token(tok);
      for (uint32_t g = 0; g < Mini::G; ++g) {
        m.serve_w(g, {tok % Mini::U, (tok * 3) % Mini::U});
        m.t->sync(DgTransport::kDem);
      }
      m.core->end_token({});
    }
    winpf_on(*m.core);
    const int32_t sym = 15;
    m.core->window_hint(&sym, nullptr, 1);
    m.t->fail_reset_after(n);
    bool threw = false;
    try { m.core->reset_open(); } catch (const ts::Error& e) {
      threw = std::string(e.what()).find("injected failure") != std::string::npos;
      CHECK(threw);
    }
    m.t->fail_after(0);
    if (!threw) {
      verify_bytes(*m.core, *m.t);
      break;
    }
    ++failures;
    CHECK(m.core->reset_failed());
    bool refused = false;
    try { m.core->reset_open(); } catch (const ts::Error&) { refused = true; }
    CHECK(refused);
    m.t->drain_all();
    CHECK(m.v.what.empty());
  }
  CHECK(failures >= 5);
  std::printf("  reset failure points %d\n", failures);
}

TEST(dg_reset_open_unconsumed_prior) {
  const auto order = keys_in_order(8);
  Mini m(8, 4, 4, order);
  winpf_on(*m.core);
  const int32_t sym = 10;
  m.core->window_hint(&sym, nullptr, 1);
  CHECK_EQ(m.core->coming(), 1u);
  CHECK(m.core->home(10) == DgCore::Home::Coming);
  // No synthetic token/end boundary: copies and source/event ownership are reset directly.
  reset_compare(*m.core, *m.t, m.c, order, m.v, 0);
}

// ---- scored eviction (docs/design.md §3.8; design scored.md sections 8-10) ----
namespace {

// Every model shape with scored eviction on, random keep lists and spare targets, deferral (on, toggled), fewer spare
// blocks than spare slots (shortage), and the window-prefetch shapes (prior copies Coming at their layer's ranking).
std::vector<Shape> scored_shapes() {
  std::vector<Shape> v;
  for (const Shape* sh : kShapes) {
    Shape a = *sh;
    a.scored = true;
    a.vary = true;
    v.push_back(a);
    if (!a.winpf) {
      a.defer = true;
      v.push_back(a);
      a.toggle_defer = true;
      a.slow_d2h = true;
      v.push_back(a);
    }
  }
  Shape s = kPredictPressure;  // 3 spare blocks for 16 spare slots: boundaries run short
  s.host_spares = 3;
  s.scored = true;
  s.vary = true;
  v.push_back(s);
  s.defer = true;
  v.push_back(s);
  return v;
}

// A small core for the deterministic cases: gated layers of `units`, the first `residents` keys (or `order`) on the
// device, `spares` spare slots (the target) and as many spare blocks.
struct Solo {
  Violation v;
  DgConfig c;
  std::unique_ptr<Scripted> t;
  std::unique_ptr<DgCore> core;
  uint32_t seq = 0;
  Solo(const std::vector<uint32_t>& units, uint32_t residents, uint32_t spares, bool scored, uint32_t max_ids = 64,
       bool boundary_score = false) {
    uint32_t keys = 0;
    for (uint32_t u : units) keys += u;
    c.n_units = units;
    c.persistent_slots = residents + spares;
    c.bank_slots = max_ids;
    c.blocks = keys - residents + spares;
    c.spare_target = spares;
    c.prefetch_reserve = 0;
    c.scored = scored;
    c.boundary_score = boundary_score;
    c.max_ids = max_ids;
    t = std::make_unique<Scripted>(keys, c.persistent_slots + 2 * c.bank_slots, c.blocks, uint32_t(units.size()), 5, &v);
    core = std::make_unique<DgCore>(c, t.get());
    core->place(keys_in_order(residents));
    for (uint32_t k = 0; k < keys; ++k) {
      if (core->slot_of(k) >= 0) t->fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t->fill_block(core->block_of(k), int32_t(k));
    }
    t->set_table(core->table());
  }
  void begin() { core->begin_token(++seq); }
  void serve(uint32_t g, const std::vector<uint32_t>& ids, bool rank = true) {
    uint64_t m = 0;
    for (size_t i = 0; i < ids.size(); ++i)
      if (t->entry(core->key(g, ids[i])) < 0) m |= 1ull << i;
    core->serve(g, ids.data(), uint32_t(ids.size()), m);
    if (rank) core->score_layer(g);
  }
  void end() {
    core->end_token({});
    t->drain_all();
    core->check();
  }
};

std::string fake_identity(const std::vector<uint32_t>& units, uint64_t salt, std::vector<std::vector<uint64_t>>& h) {
  h.assign(units.size(), {});
  std::vector<ts::DgScoreIdLayer> l;
  for (size_t g = 0; g < units.size(); ++g) {
    for (uint32_t u = 0; u < units[g]; ++u) h[g].push_back(salt * 1000003u + g * 977u + u);
    l.push_back({uint32_t(g), units[g], 8192, &h[g]});
  }
  return ts::dg_score_identity(l);
}

template <class F>
std::string error_of(F f, ss_status want) {
  try {
    f();
  } catch (const ts::Error& e) {
    return e.status() == want ? std::string(e.what()) : std::string("wrong status: ") + e.what();
  }
  return "no error";
}

}  // namespace

// Every boundary of every scored run equals the oracle's brute-force choice (victims and their order, shortage, no
// fallback), across deferred unsubmitted / partly submitted / recorded write-backs, resurrections and collections after
// a layer was ranked, prefetched and prior units Coming at ranking, keep lists, changing targets, shortage, pressure.
TEST(dg_scored_oracle_runs) {
  g_cov = OracleCoverage{};
  ts::DgStats sum{};
  int runs = 0;
  for (const Shape& sh : scored_shapes())
    for (uint64_t seed = 1; seed <= 6; ++seed) {
      Outcome o = run(sh, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
      ++runs;
      sum.resurrected += o.st.resurrected;
      sum.collected_mid += o.st.collected_mid;
      sum.wb_cancelled += o.st.wb_cancelled;
      sum.scored_victims += o.st.scored_victims;
      sum.score_fallbacks += o.st.score_fallbacks;
      sum.bank_serves += o.st.bank_serves;
    }
  const OracleCoverage& c = g_cov;
  std::printf("  %d runs, %llu boundaries, %llu victims (%llu in score order), deferred unsubmitted %llu / partial %llu / "
              "recorded %llu, wb_pending skipped %llu, kept skipped %llu, short %llu, pressure %llu, target changes "
              "%llu, coming at ranking %llu, resurrected %llu, mid-token collects %llu, cancelled %llu\n",
              runs, (unsigned long long)c.boundaries, (unsigned long long)c.victims,
              (unsigned long long)sum.scored_victims, (unsigned long long)c.unsubmitted, (unsigned long long)c.partial,
              (unsigned long long)c.recorded, (unsigned long long)c.wb_pending_skipped,
              (unsigned long long)c.kept_skipped, (unsigned long long)c.short_boundaries,
              (unsigned long long)c.pressure_boundaries, (unsigned long long)c.target_changes,
              (unsigned long long)c.coming_at_score, (unsigned long long)sum.resurrected,
              (unsigned long long)sum.collected_mid, (unsigned long long)sum.wb_cancelled);
  CHECK_EQ(sum.score_fallbacks, 0ull);
  for (uint64_t x : {c.unsubmitted, c.partial, c.recorded, c.wb_pending_skipped, c.kept_skipped, c.short_boundaries,
                     c.pressure_boundaries, c.target_changes, c.coming_at_score, sum.resurrected, sum.collected_mid,
                     sum.wb_cancelled, sum.scored_victims})
    CHECK(x > 0);
}

TEST(dg_scored_bounded_heap_differential_auto) {
  uint64_t boundaries = 0;
  for (const Shape& sh : scored_shapes())
    for (uint64_t seed = 1; seed <= 6; ++seed) {
      const Outcome ref = run(sh, 40, seed, DgCore::kNoMutation);
      Shape fast = sh;
      fast.boundary_score = true;
      const Outcome bounded = run(fast, 40, seed, DgCore::kNoMutation);
      CHECK(ref.violation.empty());
      CHECK(bounded.violation.empty());
      CHECK(ref.victims == bounded.victims);
      CHECK(counters(ref.st) == counters(bounded.st));
      fast.scored_auto = true;
      const Outcome windows = run(fast, 40, seed, DgCore::kNoMutation);
      CHECK(windows.violation.empty());
      CHECK(windows.victims == bounded.victims);
      fast.auto_rows = 1;
      const Outcome r1 = run(fast, 40, seed, DgCore::kNoMutation);
      fast.scored = fast.scored_auto = false;
      const Outcome lru = run(fast, 40, seed, DgCore::kNoMutation);
      CHECK(r1.violation.empty());
      CHECK(lru.violation.empty());
      CHECK(r1.victims == lru.victims);
      CHECK(counters(r1.st) == counters(lru.st));
      CHECK_EQ(r1.st.score_candidates, 0ull);
      CHECK_EQ(r1.st.scored_victims, 0ull);
      fast.scored = fast.scored_auto = true;
      fast.auto_rows = 0;
      const Outcome mixed = run(fast, 40, seed, DgCore::kNoMutation);
      CHECK(mixed.violation.empty());  // each boundary matched to the appropriate oracle on its current state
      boundaries += ref.victims.size() * 6;
    }
  std::printf("  %llu differential / oracle boundaries (public, heap, auto windows, auto R1, LRU, mixed)\n",
              (unsigned long long)boundaries);
}

TEST(dg_runtime_policy_resets_learning) {
  Solo s({16, 16}, 16, 4, true, 8, true);
  for (uint32_t policy : {2u, 0u, 1u, 2u}) {
    s.core->set_policy(policy);
    CHECK_EQ(s.core->policy(), policy);
    CHECK_EQ(s.core->epoch(), 0ull);
    CHECK_EQ(s.core->score_rowsum(0, 1), 0ull);
    for (int i = 0; i < 3; ++i) {
      s.begin();
      s.serve(0, {1, 2, 3});
      s.serve(1, {1, 2, 3});
      s.core->end_token({}, i == 1 ? 6 : 1);
      s.t->drain_all();
      s.core->check();
    }
    if (policy) CHECK(s.core->score_rowsum(0, 1) > 0);
    const auto before = s.core->residency();
    s.core->set_policy(policy);
    CHECK(before == s.core->residency());
    CHECK_EQ(s.core->score_rowsum(0, 1), 0ull);
    CHECK(s.v.what.empty());
  }
  s.core->reset_open();
  CHECK_EQ(s.core->policy(), 1u);
  CHECK_EQ(s.core->epoch(), 0ull);
  Solo lru({8}, 4, 2, false);
  CHECK(error_of([&] { lru.core->set_policy(1); }, SS_E_INVALID).find("provisioned") != std::string::npos);
  CHECK(error_of([&] { s.core->set_policy(3); }, SS_E_INVALID).find("policy") != std::string::npos);
  CHECK(error_of([&] { s.core->end_token({}, 0); }, SS_E_INVALID).find("rows") != std::string::npos);
}

TEST(dg_bounded_profile_and_runtime_reset) {
  const std::vector<uint32_t> units{16, 16};
  std::vector<std::vector<uint64_t>> hashes;
  const auto identity = fake_identity(units, 91, hashes);
  std::vector<std::vector<uint32_t>> cells(2, std::vector<uint32_t>(256));
  for (size_t g = 0; g < cells.size(); ++g)
    for (size_t i = 0; i < cells[g].size(); ++i) cells[g][i] = i % 7 ? uint32_t(i * 7919) : UINT32_MAX;
  const auto profile = ts::dg_profile_write(units, identity, "bounded differential", cells);
  Solo ref(units, 16, 4, true, 8), fast(units, 16, 4, true, 8, true);
  for (Solo* s : {&ref, &fast}) s->core->load_profile(profile.data(), profile.size(), identity);
  for (uint32_t tok = 0; tok < 40; ++tok) {
    for (Solo* s : {&ref, &fast}) {
      s->begin();
      s->serve(0, {tok % 16, (tok + 3) % 16, (tok + 9) % 16});
      s->serve(1, {(tok + 1) % 16, (tok + 7) % 16});
      s->end();
    }
    CHECK(ref.core->last_victims() == fast.core->last_victims());
  }
  fast.core->set_policy(2);
  CHECK_EQ(fast.core->score_cell(0, 0, 0), UINT32_MAX);
  CHECK_EQ(fast.core->epoch(), 0ull);
  for (int i = 0; i < 2; ++i) {
    fast.begin();
    fast.serve(0, {1});
    fast.serve(1, {2});
    fast.core->end_token({}, 1);
    fast.t->drain_all();
  }
  CHECK_EQ(fast.core->score_cell(0, 1, 1), cells[0][17] + 1);
  fast.core->set_policy(0);
  fast.core->set_policy(1);
  CHECK_EQ(fast.core->score_cell(0, 1, 1), cells[0][17]);
  fast.core->check();
  CHECK(fast.v.what.empty());
}

// Faults the oracle (or the cache invariants) must catch: a victim with a write-back in flight, a kept unit evicted,
// a list re-tagged without being re-ranked, ranking from the shared scratch after the prediction overwrote it.
TEST(dg_scored_mutations_caught) {
  const struct {
    int m;
    const char* name;
  } muts[] = {{DgCore::kReevictPending, "scored: re-evict with a pending write-back"},
              {DgCore::kIgnoreKeep, "scored: keep list ignored"},
              {DgCore::kStaleList, "scored: stale list used"},
              {DgCore::kNoUnionSave, "scored: union read after the prediction"}};
  for (const auto& mu : muts) {
    int caught = 0, runs = 0;
    for (const Shape& sh : scored_shapes())
      for (uint64_t seed = 1; seed <= 3; ++seed) {
        ++runs;
        if (!run(sh, 40, seed, mu.m).violation.empty()) ++caught;
      }
    std::printf("  %-44s caught in %d/%d runs\n", mu.name, caught, runs);
    CHECK(caught > 0);
  }
}

// A layer left unranked now and then (an internal fault the guard exists for): those boundaries fall back to the
// recency order and count it, and the oracle, expecting the recency order there, still sees no mismatch.
TEST(dg_scored_skipped_rank_falls_back) {
  g_cov = OracleCoverage{};
  uint64_t fallbacks = 0;
  for (const Shape& sh : scored_shapes())
    for (uint64_t seed = 1; seed <= 2; ++seed) {
      Outcome o = run(sh, 40, seed, DgCore::kSkipRank);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
      fallbacks += o.st.score_fallbacks;
    }
  std::printf("  %llu fallbacks, all matched by the oracle\n", (unsigned long long)fallbacks);
  CHECK(fallbacks > 0);
  CHECK_EQ(fallbacks, g_cov.fallbacks);
}

// The oracle also checks every boundary of the LRU runs (no mutation): exact victims in the recency order, shortage.
TEST(dg_lru_oracle_runs) {
  g_cov = OracleCoverage{};
  for (const Shape* sh : kShapes)
    for (uint64_t seed = 1; seed <= 4; ++seed) {
      Shape a = *sh;
      a.defer = seed % 2 == 0;
      Outcome o = run(a, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
    }
  std::printf("  %llu LRU boundaries matched\n", (unsigned long long)g_cov.lru_boundaries);
  CHECK(g_cov.lru_boundaries > 0);
}

// ss_dg_reset_open with scored eviction: the learning state goes back to the open blueprint (empty tables, epoch 0,
// the open stamps) with everything else; the reset core then decides exactly as a fresh one.
TEST(dg_scored_reset_open_randomized) {
  int runs = 0;
  for (const Shape& sh : scored_shapes())
    for (uint64_t seed = 1; seed <= 2; ++seed) {
      Outcome o = run(sh, 12 + int(seed) * 7, seed, DgCore::kNoMutation, 6, true);
      if (!o.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
      ++runs;
    }
  std::printf("  %d reset histories\n", runs);
}

// Invalid score metadata after a successful token (a layer left unranked): that boundary takes the recency order into
// the same acceptance loop and counts a fallback; the next token is scored again.
TEST(dg_scored_fallback_on_missing_list) {
  Solo s({16, 16, 16}, 16, 4, true, 8);
  s.begin();
  s.serve(0, {0, 1});
  s.serve(1, {0, 1, 2, 3, 4});  // 4 into the spares, one bank serve
  s.serve(2, {0}, false);       // served, never ranked
  std::vector<std::pair<uint64_t, uint32_t>> dev;
  for (uint32_t k = 0; k < s.core->keys(); ++k)
    if (s.core->home(k) == DgCore::Home::Device || s.core->home(k) == DgCore::Home::Coming)
      dev.push_back({s.core->lru_stamp(k), k});
  std::sort(dev.begin(), dev.end());
  s.end();
  CHECK_EQ(s.core->stats().score_fallbacks, 1ull);
  CHECK_EQ(s.core->stats().scored_victims, 0ull);
  std::vector<uint32_t> want;
  for (size_t i = 0; i < 4; ++i) want.push_back(dev[i].second);
  CHECK(s.core->last_victims() == want);
  s.begin();
  for (uint32_t g = 0; g < 3; ++g) s.serve(g, {5, 6, 7});
  s.end();
  CHECK_EQ(s.core->stats().score_fallbacks, 1ull);
  CHECK(s.core->stats().scored_victims > 0);
  CHECK(s.v.what.empty());
}

// The epoch starts at 1 and moves once per token; a demand use sets the unit's epoch, a prefetch does not; the list
// holds the device and coming units of the layer in (score, epoch, stamp) order with the exact integer scores; the
// transition counts pair the previous token's union with this one; the ring holds 16 unions.
TEST(dg_scored_epochs_history_order) {
  Solo s({8, 8}, 8, 4, true, 8);
  CHECK_EQ(s.core->epoch(), 0ull);
  s.begin();
  CHECK_EQ(s.core->epoch(), 1ull);
  s.serve(0, {3, 1, 3});
  s.core->serve_prediction(1, std::vector<uint32_t>{5}.data(), 1);
  s.serve(1, {2});
  CHECK_EQ(s.core->demand_epoch(1), 1ull);
  CHECK_EQ(s.core->demand_epoch(3), 1ull);
  CHECK_EQ(s.core->demand_epoch(0), 0ull);
  CHECK_EQ(s.core->demand_epoch(8 + 5), 0ull);  // prefetched, not used
  std::vector<uint32_t> keys;
  for (const auto& c : s.core->score_list(0)) keys.push_back(c.key);
  CHECK(keys == std::vector<uint32_t>({7, 6, 5, 4, 2, 0, 1, 3}));  // unused (epoch 0, place order), then 1, 3
  const auto& l0 = s.core->score_list(0);
  CHECK_EQ(l0.front().s, 100ull * 32768);                // P = 0 (empty table), F = 0, Rec = 1/2
  CHECK_EQ(l0.back().s, 200ull * 4096 + 100ull * 65536);  // F = 1/16, Rec = 1
  bool coming = false;
  for (const auto& c : s.core->score_list(1)) coming = coming || c.key == 8 + 5;
  CHECK(coming);  // a Coming prefetch is a candidate (eligibility is the boundary's)
  CHECK_EQ(s.core->score_list_epoch(1), 1ull);
  s.end();
  s.begin();
  s.serve(0, {1});
  s.serve(1, {2});
  CHECK_EQ(s.core->score_cell(0, 1, 1), 1u);
  CHECK_EQ(s.core->score_cell(0, 3, 1), 1u);
  CHECK_EQ(s.core->score_rowsum(0, 3), 1ull);
  CHECK_EQ(s.core->score_cell(1, 2, 2), 1u);
  s.end();
  for (int i = 0; i < 18; ++i) {
    s.begin();
    s.serve(0, {0});
    s.serve(1, {2});
    s.end();  // check(): ring counts == their recount
  }
  // key 0: F = 16/16, Rec = 1, P = |cur| N / D = 1 (every row-0 count went to 0)
  for (const auto& c : s.core->score_list(0))
    if (c.key == 0) CHECK_EQ(c.s, 700ull * 65536 + 200ull * 65536 + 100ull * 65536);
  CHECK(s.v.what.empty());
}

// The profile format end to end: written, loaded (cells, exact row sums, provenance), restored by reset_open; every
// malformed or mismatched file refused with SS_E_INVALID before any cell is read.
TEST(dg_scored_profile_roundtrip_and_refusals) {
  const std::vector<uint32_t> units = {8, 12};
  std::vector<std::vector<uint64_t>> h;
  const std::string id = fake_identity(units, 1, h);
  std::mt19937_64 r(5);
  std::vector<std::vector<uint32_t>> cells(2);
  for (size_t g = 0; g < 2; ++g)
    for (uint32_t i = 0; i < units[g] * units[g]; ++i)
      cells[g].push_back(i % 7 == 0 ? 0xFFFFFFFFu - uint32_t(r() % 3) : uint32_t(r() % 100000));
  const std::string prof = ts::dg_profile_write(units, id, "fit: unit test", cells);
  Solo s(units, 8, 4, true, 8);
  s.core->load_profile(prof.data(), prof.size(), id);
  CHECK_EQ(s.core->profile_provenance(), std::string("fit: unit test"));
  bool same = true;
  for (uint32_t g = 0; g < 2; ++g)
    for (uint32_t a = 0; a < units[g]; ++a) {
      uint64_t sum = 0;
      for (uint32_t e = 0; e < units[g]; ++e) {
        same = same && s.core->score_cell(g, a, e) == cells[g][a * units[g] + e];
        sum += cells[g][a * units[g] + e];
      }
      same = same && s.core->score_rowsum(g, a) == sum;
    }
  CHECK(same);
  s.core->check();
  for (int i = 0; i < 3; ++i) {  // learning moves the table; reset_open puts the profile's counts back
    s.begin();
    s.serve(0, {1, 2});
    s.serve(1, {3, 4});
    s.end();
  }
  CHECK(s.core->score_cell(0, 1, 1) != cells[0][1 * 8 + 1] || s.core->score_cell(0, 1, 1) == 0xFFFFFFFFu);
  s.core->reset_open();
  CHECK_EQ(s.core->score_cell(0, 1, 1), cells[0][1 * 8 + 1]);
  CHECK_EQ(s.core->score_cell(1, 3, 4), cells[1][3 * 12 + 4]);
  CHECK_EQ(s.core->epoch(), 0ull);
  auto refused = [&](const std::string& bytes, const std::string& ident, const char* needle) {
    Solo x(units, 8, 4, true, 8);
    const std::string e = error_of([&] { x.core->load_profile(bytes.data(), bytes.size(), ident); }, SS_E_INVALID);
    if (e.find(needle) == std::string::npos) std::printf("  wanted '%s', got '%s'\n", needle, e.c_str());
    CHECK(e.find(needle) != std::string::npos);
  };
  refused(prof.substr(0, 20), id, "truncated header");
  refused(prof + "x", id, "size");
  refused(prof.substr(0, prof.size() - 1), id, "size");
  std::string b = prof;
  b[0] = 'X';
  refused(b, id, "bad magic");
  b = prof;
  b[8] = 2;
  refused(b, id, "unsupported version");
  std::vector<std::vector<uint64_t>> h1;
  refused(ts::dg_profile_write({8}, fake_identity({8}, 1, h1), "", {cells[0]}), id, "n_gated");
  std::vector<std::vector<uint32_t>> c11 = {cells[0], std::vector<uint32_t>(121, 1)};
  refused(ts::dg_profile_write({8, 11}, id, "", c11), id, "units of gated layer 1");
  std::vector<std::vector<uint64_t>> h2;
  refused(prof, fake_identity(units, 2, h2), "identity mismatch");
  b = prof;
  b[b.size() - 40] ^= 1;  // a cell
  refused(b, id, "checksum mismatch");
  b = prof;
  const size_t at_prov = 8 + 4 + 4 + 4 * 2 + 32;
  const unsigned char lo = 0x88;  // prov_len 0x1388 = 5000
  b[at_prov] = char(lo);
  b[at_prov + 1] = char(0x13);
  refused(b, id, "provenance longer than 4096");
  std::vector<ts::DgScoreIdLayer> nohash = {{0, 8, 8192, nullptr}};
  CHECK(error_of([&] { (void)ts::dg_score_identity(nohash); }, SS_E_INVALID).find("per-unit hashes") !=
        std::string::npos);
  Solo lru(units, 8, 4, false, 8);
  CHECK(error_of([&] { lru.core->load_profile(prof.data(), prof.size(), id); }, SS_E_INVALID).find("policy SCORED") !=
        std::string::npos);
  CHECK(error_of([&] { (void)ts::dg_score_cells({65537}); }, SS_E_INVALID).find("65536") != std::string::npos);
  DgConfig bad = s.c;
  bad.w_rec = 200;  // 700 + 200 + 200
  Violation bv;
  Scripted bt(20, 20, 20, 2, 1, &bv);
  CHECK(error_of([&] { DgCore x(bad, &bt); }, SS_E_INVALID).find("sum to 1000") != std::string::npos);
}

// The integer contract at its edges: a cell at 2^32 - 2 takes one increment and then saturates while its row sum keeps
// equal to the cells; a 64-id union over fully saturated rows (the largest intermediate) scores exactly.
TEST(dg_scored_saturation_and_max_union) {
  {
    const std::vector<uint32_t> units = {8};
    std::vector<std::vector<uint64_t>> h;
    const std::string id = fake_identity(units, 3, h);
    std::vector<std::vector<uint32_t>> cells(1, std::vector<uint32_t>(64, 0));
    cells[0][0 * 8 + 1] = 0xFFFFFFFEu;
    const std::string prof = ts::dg_profile_write(units, id, "", cells);
    Solo s(units, 8, 2, true, 8);
    s.core->load_profile(prof.data(), prof.size(), id);
    for (int i = 0; i < 4; ++i) {
      s.begin();
      s.serve(0, i % 2 ? std::vector<uint32_t>{1, 2} : std::vector<uint32_t>{0});
      s.end();  // check(): every row sum == the sum of its cells
    }
    CHECK_EQ(s.core->score_cell(0, 0, 1), 0xFFFFFFFFu);
    CHECK_EQ(s.core->score_cell(0, 0, 2), 2u);
    CHECK_EQ(s.core->score_rowsum(0, 0), 0xFFFFFFFFull + 2);
    CHECK_EQ(s.core->stats().score_saturated, 1ull);
  }
  {
    const std::vector<uint32_t> units = {64};
    std::vector<std::vector<uint64_t>> h;
    const std::string id = fake_identity(units, 4, h);
    std::vector<std::vector<uint32_t>> cells(1, std::vector<uint32_t>(64 * 64, 0xFFFFFFFFu));
    const std::string prof = ts::dg_profile_write(units, id, "", cells);
    Solo s(units, 64, 4, true, 64);
    s.core->load_profile(prof.data(), prof.size(), id);
    std::vector<uint32_t> all = keys_in_order(64);
    static_assert(((64ull * 64ull * 0xFFFFFFFFull) << 16) < (1ull << 60), "the largest intermediate");
    for (int i = 0; i < 2; ++i) {
      s.begin();
      s.serve(0, all);
      if (i == 0) s.end();
    }
    const auto& l = s.core->score_list(0);
    CHECK_EQ(l.size(), size_t(64));
    bool ok = true;  // P = ((64 x 64 M) << 16) / (64 x 64 M) = 1 << 16, F = 2/16, Rec = 1; ties by stamp (use order)
    for (size_t i = 0; i < l.size(); ++i)
      ok = ok && l[i].key == i && l[i].s == 700ull * 65536 + 200ull * 8192 + 100ull * 65536;
    CHECK(ok);
    CHECK_EQ(s.core->stats().score_saturated, 64ull * 64);
    s.end();
  }
}

// The stamp counter is checked before every phase that pushes (a token, a seed, a window hint): past the limit the
// phase fails before touching anything (the service's failure poisons the context); at the limit it still runs.
TEST(dg_scored_stamp_overflow_refused) {
  Solo s({8, 8}, 8, 4, true, 8);
  s.core->set_stamp_counter(DgCore::kStampLimit + 1);
  CHECK(error_of([&] { s.core->begin_token(1); }, SS_E_BACKEND).find("stamp counter") != std::string::npos);
  CHECK(error_of([&] { s.core->seed({9}); }, SS_E_BACKEND).find("stamp counter") != std::string::npos);
  CHECK_EQ(s.core->epoch(), 0ull);
  CHECK(s.core->home(9) == DgCore::Home::Host);
  s.core->set_stamp_counter(DgCore::kStampLimit);
  s.begin();
  s.serve(0, {1, 2});
  s.serve(1, {1});
  s.end();
  CHECK(s.core->stamp_counter() > DgCore::kStampLimit);
  CHECK(error_of([&] { s.core->begin_token(9); }, SS_E_BACKEND).find("stamp counter") != std::string::npos);
  CHECK(s.v.what.empty());
}

// ---- replay: the real core against a C++ port of the reference simulator's integer path ----
namespace {

// A route trace: sequences of tokens, each token L layers x K ids (flat, uint16).
struct Trace {
  uint32_t L = 0, E = 0, K = 0;
  std::vector<std::string> name;
  std::vector<std::vector<uint16_t>> ids;  // [seq][(t * L + l) * K + i]
  size_t tokens(size_t s) const { return ids[s].size() / (size_t(L) * K); }
};

// The W1 text format: "# seq <name> rows <n>" starts a sequence; a row is "l:id,id,..." for every layer.
bool load_trace(const char* path, uint32_t E, Trace& tr) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  tr = Trace{};
  tr.E = E;
  std::string line;
  for (int ch;;) {
    line.clear();
    while ((ch = std::fgetc(f)) != EOF && ch != '\n') line.push_back(char(ch));
    if (line.empty() && ch == EOF) break;
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (line[0] == '#') {
      char a[64], b[64], name[512];
      if (std::sscanf(line.c_str(), "%63s %63s %511s", a, b, name) == 3) tr.name.push_back(name);
      tr.ids.emplace_back();
      continue;
    }
    std::vector<std::vector<uint16_t>> row;
    size_t p = 0;
    while (p < line.size()) {
      while (p < line.size() && line[p] == ' ') ++p;
      if (p >= line.size()) break;
      const size_t c = line.find(':', p);
      const uint32_t l = uint32_t(std::strtoul(line.c_str() + p, nullptr, 10));
      if (row.size() <= l) row.resize(l + 1);
      p = c + 1;
      while (p < line.size() && line[p] != ' ') {
        char* end = nullptr;
        row[l].push_back(uint16_t(std::strtoul(line.c_str() + p, &end, 10)));
        p = size_t(end - line.c_str());
        if (p < line.size() && line[p] == ',') ++p;
      }
    }
    if (!tr.L) {
      tr.L = uint32_t(row.size());
      tr.K = uint32_t(row[0].size());
    }
    for (const auto& r : row) {
      if (r.size() != tr.K || row.size() != tr.L) {
        std::fclose(f);
        return false;
      }
      tr.ids.back().insert(tr.ids.back().end(), r.begin(), r.end());
    }
  }
  std::fclose(f);
  return !tr.ids.empty();
}

// The simulator's fold mapping: fold f = the sequences with index % 5 == f; the seed order is the other folds'
// per-key frequency (key = layer x E + unit), descending, ties by key (numpy's stable argsort of -counts).
std::vector<uint32_t> hot_order(const Trace& tr, int fold) {
  std::vector<uint64_t> cnt(size_t(tr.L) * tr.E, 0);
  for (size_t s = 0; s < tr.ids.size(); ++s) {
    if (int(s % 5) == fold) continue;
    const auto& v = tr.ids[s];
    for (size_t i = 0; i < v.size(); ++i) {
      const uint32_t l = uint32_t(i / tr.K % tr.L);
      ++cnt[size_t(l) * tr.E + v[i]];
    }
  }
  std::vector<uint32_t> order(cnt.size());
  for (uint32_t k = 0; k < order.size(); ++k) order[k] = k;
  std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return cnt[a] > cnt[b]; });
  return order;
}

std::vector<uint32_t> union_of(const Trace& tr, size_t s, size_t t, uint32_t l) {
  const uint16_t* p = tr.ids[s].data() + (t * tr.L + l) * tr.K;
  std::vector<uint32_t> u(p, p + tr.K);
  std::sort(u.begin(), u.end());
  u.erase(std::unique(u.begin(), u.end()), u.end());
  return u;
}

struct ReplayOut {
  std::vector<uint32_t> miss;
  std::vector<std::vector<uint32_t>> vict;
  uint64_t misses = 0;
};

// scoresim.py v3's step function, INT=1 HIST=context SEEDAGE=0 PROTECT=0 ONLINE=1 (an empty transition table): every
// demand miss is admitted into a free slot if one is left (else served transiently); at the step's end the victims
// (lowest (score, demand epoch, LRU stamp), or the lowest LRU stamp) bring free + in flight back to the spares; they
// are free again at the next step.
ReplayOut port_run(const Trace& tr, int fold, const std::vector<uint32_t>& order, uint32_t active, uint32_t spares,
                   bool scored) {
  const uint32_t L = tr.L, E = tr.E, NK = L * E;
  ReplayOut out;
  std::vector<uint8_t> resident(NK, 0);
  std::vector<int64_t> epoch(NK, 0), lru(NK, 0);
  for (uint32_t i = 0; i < active; ++i) {
    resident[order[i]] = 1;
    lru[order[i]] = -int64_t(i);
  }
  int64_t clk = 0, tok = 0, free = spares;
  std::deque<std::pair<int64_t, int64_t>> pending;
  std::vector<uint32_t> T(size_t(L) * E * E, 0);
  std::vector<uint64_t> rs(size_t(L) * E, 0);
  std::vector<uint16_t> cnt(size_t(L) * E, 0);
  std::vector<std::deque<std::vector<uint32_t>>> ring(L);
  std::vector<std::vector<uint32_t>> prev(L);
  std::vector<uint8_t> has_prev(L, 0);
  struct C {
    uint64_t s;
    int64_t ep, st;
    uint32_t k;
  };
  std::vector<C> cand;
  for (size_t s = 0; s < tr.ids.size(); ++s) {
    if (int(s % 5) != fold) continue;
    for (size_t t = 0; t < tr.tokens(s); ++t) {
      ++tok;
      while (!pending.empty() && pending.front().first <= tok) {
        free += pending.front().second;
        pending.pop_front();
      }
      uint32_t m = 0;
      for (uint32_t l = 0; l < L; ++l) {
        const std::vector<uint32_t> cur = union_of(tr, s, t, l);
        for (uint32_t e : cur) {
          const uint32_t k = l * E + e;
          if (!resident[k]) {
            ++m;
            if (free > 0) {
              --free;
              resident[k] = 1;
            }
          }
          epoch[k] = tok;
          lru[k] = ++clk;
        }
        if (scored && has_prev[l])
          for (uint32_t r : prev[l])
            for (uint32_t e : cur) {
              uint32_t& cell = T[(size_t(l) * E + r) * E + e];
              if (cell < 0xFFFFFFFFu) {
                ++cell;
                ++rs[size_t(l) * E + r];
              }
            }
        prev[l] = cur;
        has_prev[l] = 1;
        ring[l].push_back(cur);
        for (uint32_t e : cur) ++cnt[size_t(l) * E + e];
        if (ring[l].size() > 16) {
          for (uint32_t e : ring[l].front()) --cnt[size_t(l) * E + e];
          ring[l].pop_front();
        }
      }
      out.miss.push_back(m);
      out.misses += m;
      int64_t inflight = 0;
      for (const auto& pd : pending) inflight += pd.second;
      const int64_t need = int64_t(spares) - free - inflight;
      out.vict.emplace_back();
      if (need <= 0) continue;
      cand.clear();
      for (uint32_t k = 0; k < NK; ++k) {
        if (!resident[k]) continue;
        uint64_t S = 0;
        if (scored) {
          const uint32_t l = k / E, e = k % E;
          const std::vector<uint32_t>& cur = prev[l];
          uint64_t N = 0, D = 0;
          for (uint32_t r : cur) {
            N += T[(size_t(l) * E + r) * E + e];
            D += rs[size_t(l) * E + r];
          }
          const uint64_t P = ((uint64_t(cur.size()) * N) << 16) / (D ? D : 1);
          const uint64_t F = (uint64_t(cnt[k]) << 16) / 16;
          const uint64_t R = (uint64_t(1) << 16) / uint64_t(1 + tok - epoch[k]);
          S = 700 * P + 200 * F + 100 * R;
        }
        cand.push_back({S, scored ? epoch[k] : 0, lru[k], k});
      }
      const size_t n = std::min<size_t>(size_t(need), cand.size());
      std::partial_sort(cand.begin(), cand.begin() + ptrdiff_t(n), cand.end(), [](const C& a, const C& b) {
        return a.s != b.s ? a.s < b.s : a.ep != b.ep ? a.ep < b.ep : a.st < b.st;
      });
      for (size_t i = 0; i < n; ++i) {
        resident[cand[i].k] = 0;
        out.vict.back().push_back(cand[i].k);
      }
      pending.push_back({tok + 1, int64_t(n)});
    }
  }
  return out;
}

// The same fold through the real core (prefetch off, every write-back completed and collected before the next token,
// as the simulator's next-step completion): ids in ascending unique order (the simulator admits in that order).
ReplayOut core_run(const Trace& tr, int fold, const std::vector<uint32_t>& order, uint32_t active, uint32_t spares,
                   bool scored) {
  const uint32_t L = tr.L, E = tr.E, NK = L * E;
  DgConfig c;
  c.n_units.assign(L, E);
  c.persistent_slots = active + spares;
  c.bank_slots = tr.K;
  c.blocks = NK - active + spares;
  c.spare_target = spares;
  c.prefetch_reserve = 0;
  c.prefetch_per_layer = 0;
  c.prefetch_per_token = 0;
  c.scored = scored;
  c.max_ids = tr.K;
  Violation v;
  Scripted t(NK, c.persistent_slots + 2 * c.bank_slots, c.blocks, L, 1, &v);
  DgCore core(c, &t);
  core.place(order);
  t.set_table(core.table());
  ReplayOut out;
  uint32_t seq = 0;
  uint64_t m0 = 0;
  for (size_t s = 0; s < tr.ids.size(); ++s) {
    if (int(s % 5) != fold) continue;
    for (size_t tk = 0; tk < tr.tokens(s); ++tk) {
      core.seed({});  // collects the completed write-backs (the simulator frees victims at the next step)
      core.begin_token(++seq);
      for (uint32_t l = 0; l < L; ++l) {
        const std::vector<uint32_t> u = union_of(tr, s, tk, l);
        uint64_t mask = 0;
        for (size_t i = 0; i < u.size(); ++i)
          if (t.entry(core.key(l, u[i])) < 0) mask |= 1ull << i;
        core.serve(l, u.data(), uint32_t(u.size()), mask);
        core.score_layer(l);
      }
      core.end_token({});
      t.drain_all();
      const uint64_t m = core.stats().misses;
      out.miss.push_back(uint32_t(m - m0));
      out.misses += m - m0;
      m0 = m;
      out.vict.push_back(core.last_victims());
    }
  }
  if (!v.what.empty()) std::printf("  core replay: %s\n", v.what.c_str());
  return out;
}

struct ReplaySummary {
  uint64_t tokens = 0, misses_port[2] = {0, 0}, misses_core[2] = {0, 0}, mismatched[2] = {0, 0}, victims[2] = {0, 0};
  uint64_t first_bad[2] = {0, 0};
};

ReplaySummary replay(const Trace& tr, uint32_t active, uint32_t spares) {
  ReplaySummary r;
  for (int f = 0; f < 5; ++f) {
    const std::vector<uint32_t> order = hot_order(tr, f);
    for (int arm = 0; arm < 2; ++arm) {
      const ReplayOut a = port_run(tr, f, order, active, spares, arm == 1);
      const ReplayOut b = core_run(tr, f, order, active, spares, arm == 1);
      if (arm == 0) r.tokens += a.miss.size();
      r.misses_port[arm] += a.misses;
      r.misses_core[arm] += b.misses;
      for (size_t i = 0; i < a.miss.size(); ++i) {
        r.victims[arm] += a.vict[i].size();
        if (i >= b.miss.size() || a.miss[i] != b.miss[i] || a.vict[i] != b.vict[i]) {
          if (!r.mismatched[arm]) r.first_bad[arm] = uint64_t(f) << 32 | i;
          ++r.mismatched[arm];
        }
      }
    }
  }
  return r;
}

void print_replay(const ReplaySummary& r) {
  for (int arm = 0; arm < 2; ++arm)
    std::printf("  %-6s port %.4f misses/token, core %.4f, %llu victims, %llu mismatched tokens\n",
                arm ? "scored" : "lru", double(r.misses_port[arm]) / double(r.tokens),
                double(r.misses_core[arm]) / double(r.tokens), (unsigned long long)r.victims[arm],
                (unsigned long long)r.mismatched[arm]);
}

// A small synthetic trace with per-layer transitions (a unit tends to be followed by its successors), so scored and
// LRU choose differently.
Trace synthetic_trace() {
  Trace tr;
  tr.L = 4;
  tr.E = 40;
  tr.K = 3;
  std::mt19937_64 r(11);
  for (int s = 0; s < 10; ++s) {
    tr.name.push_back("syn-" + std::to_string(s));
    tr.ids.emplace_back();
    std::vector<std::vector<uint32_t>> last(tr.L);
    for (int t = 0; t < 30; ++t)
      for (uint32_t l = 0; l < tr.L; ++l) {
        std::vector<uint32_t> row;
        while (row.size() < tr.K) {
          uint32_t e;
          if (!last[l].empty() && r() % 10 < 6) e = (last[l][r() % last[l].size()] * 7 + 3 + l) % tr.E;
          else {
            const double x = std::uniform_real_distribution<double>(0, 1)(r);
            e = uint32_t(double(tr.E) * x * x) % tr.E;
          }
          if (std::find(row.begin(), row.end(), e) == row.end()) row.push_back(e);
        }
        last[l] = row;
        for (uint32_t e : row) tr.ids.back().push_back(uint16_t(e));
      }
  }
  return tr;
}

}  // namespace

// The real core replays a route trace exactly as the reference simulator's integer path decides it: per token the
// same misses and the same victims in the same order, for both policies (5 folds, the simulator's seed order).
TEST(dg_scored_replay_synthetic) {
  const Trace tr = synthetic_trace();
  const ReplaySummary r = replay(tr, 70, 8);
  print_replay(r);
  CHECK_EQ(r.mismatched[0], 0ull);
  CHECK_EQ(r.mismatched[1], 0ull);
  CHECK(r.misses_port[0] != r.misses_port[1]);  // the policies really differ on this trace
  CHECK(r.victims[1] > 0);
}

// The same replay of a W1 route file (TS_W1_ROUTES; TS_W1_ACTIVE / TS_W1_SPARES, default 8576 / 96, 384 units per
// layer): skipped when the variable is not set. scoresim.py v3 (INT=1 HIST=context SEEDAGE=0 ONLINE=1) reads 8.140
// (LRU) and 7.652 (scored) misses per token on w1-routes-r0.txt.
TEST(dg_scored_replay_w1) {
  const char* path = std::getenv("TS_W1_ROUTES");
  if (!path || !*path) {
    std::printf("  skipped (set TS_W1_ROUTES to a W1 route file)\n");
    return;
  }
  const char* a = std::getenv("TS_W1_ACTIVE");
  const char* sp = std::getenv("TS_W1_SPARES");
  const uint32_t active = a ? uint32_t(std::strtoul(a, nullptr, 10)) : 8576;
  const uint32_t spares = sp ? uint32_t(std::strtoul(sp, nullptr, 10)) : 96;
  Trace tr;
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(load_trace(path, 384, tr));
  if (tr.ids.empty()) return;
  size_t toks = 0;
  for (size_t s = 0; s < tr.ids.size(); ++s) toks += tr.tokens(s);
  std::printf("  %zu sequences, %zu tokens, %u layers x %u ids, active %u + spares %u\n", tr.ids.size(), toks, tr.L,
              tr.K, active, spares);
  const ReplaySummary r = replay(tr, active, spares);
  print_replay(r);
  std::printf("  %.1f s\n",
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  CHECK_EQ(r.mismatched[0], 0ull);
  CHECK_EQ(r.mismatched[1], 0ull);
  if (active == 8576 && spares == 96 && toks == 49430) {  // w1-routes-r0.txt: scoresim v3's exact counts
    CHECK_EQ(r.misses_port[0], 402375ull);                // 8.1403 misses / token
    CHECK_EQ(r.misses_core[0], 402375ull);
    CHECK_EQ(r.misses_port[1], 378245ull);  // 7.6521
    CHECK_EQ(r.misses_core[1], 378245ull);
  }
}

// ---- three-tier placement (ss.h SS_DG_DISK_TIER): file homes, pinned-unit drops, the inclusive share ----
namespace {

Shape disk_shape(uint32_t disk, uint32_t host_spares) {
  Shape s{{64, 64, 48, 64}, 100, 24, 8, host_spares, 8};
  s.disk = disk;
  return s;
}
const Shape kDisk = disk_shape(60, 8);
const Shape kDiskTiny = [] {  // 10 pinned homes, 2 spare blocks: victims are dropped to their files too
  Shape s = disk_shape(130, 2);
  s.pressure_free = 2;
  s.pressure_recent = 6;
  return s;
}();
const Shape kDiskPredict = [] {  // predictions read file units too (disk_prefetch), pressure, staging of 2
  Shape s = disk_shape(60, 8);
  s.predict = 6;
  s.disk_prefetch = true;
  s.pressure_free = 1;
  s.pressure_recent = 6;
  s.staging = 2;
  return s;
}();
const Shape kDiskDefer = [] {
  Shape s = disk_shape(80, 6);
  s.defer = true;
  s.slow_d2h = true;
  s.predict = 4;
  return s;
}();
const Shape kDiskWinpf = [] {
  Shape s = winpf_shape(true, true, true, 1500);
  s.disk = 50;
  s.disk_prefetch = true;
  return s;
}();
const Shape kDiskScored = [] {
  Shape s = disk_shape(70, 8);
  s.scored = true;
  s.boundary_score = true;
  s.scored_auto = true;
  s.auto_rows = 0;
  s.vary = true;
  s.predict = 4;
  return s;
}();
const Shape kInclusive = [] {  // two-tier with clean copies (out of 24 spare blocks)
  Shape s{{64, 64, 48, 64}, 120, 24, 8, 24, 8};
  s.inclusive = 12;
  s.predict = 4;
  return s;
}();
const Shape kDiskInclusive = [] {
  Shape s = disk_shape(60, 20);
  s.inclusive = 12;
  s.predict = 4;
  s.defer = true;
  return s;
}();
const Shape kDiskZero = [] {  // no pinned homes, 1 spare block, slow write-backs: victims dropped to their files
  Shape s = disk_shape(140, 1);
  s.slow_d2h = true;
  s.defer = true;
  s.pressure_free = 2;
  s.pressure_recent = 6;
  return s;
}();
const Shape* const kTierShapes[] = {&kDisk, &kDiskTiny, &kDiskPredict, &kDiskDefer, &kDiskWinpf,
                                    &kDiskScored, &kInclusive, &kDiskInclusive, &kDiskZero};

}  // namespace

// Each shape three ways: the default queue depth (16) with the synchronous transport read, the depth 16 with reads
// that land late and out of order, and depth 1 (the synchronous reads).
Shape with_reads(const Shape& base, int how) {
  Shape s = base;
  s.async_reads = how == 1;
  s.queue = how == 2 ? 1 : 16;
  return s;
}

TEST(dg_three_tier_runs) {
  ts::DgStats sum{};
  uint64_t reads = 0, overlapped = 0, max_in_flight = 0, ooo = 0, carried = 0, carried_bytes = 0, read_bytes = 0;
  for (const Shape* base : kTierShapes)
    for (int how = 0; how < 3; ++how)
    for (uint64_t seed = 1; seed <= 12; ++seed) {
      const Shape sh = with_reads(*base, how);
      Outcome o = run(sh, 40, seed, DgCore::kNoMutation);
      if (!o.violation.empty())
        std::printf("  seed %llu (reads %d): %s\n", (unsigned long long)seed, how, o.violation.c_str());
      CHECK(o.violation.empty());
      if (how == 2) CHECK_EQ(o.st.disk_overlapped + o.st.disk_carried, 0ull);
      overlapped += o.st.disk_overlapped;
      carried += o.st.disk_carried;
      carried_bytes += o.st.disk_carried_bytes;
      read_bytes += o.st.disk_read_bytes;
      max_in_flight = std::max(max_in_flight, o.max_in_flight);
      ooo += o.landed_out_of_order;
      for (auto f : {&ts::DgStats::disk_reads, &ts::DgStats::disk_read_bytes, &ts::DgStats::disk_stage_waits,
                     &ts::DgStats::disk_drops, &ts::DgStats::disk_discards, &ts::DgStats::inclusive_kept,
                     &ts::DgStats::inclusive_drops, &ts::DgStats::bank_serves, &ts::DgStats::resurrected,
                     &ts::DgStats::prefetched, &ts::DgStats::evict_short, &ts::DgStats::l1_reverted})
        sum.*f += o.st.*f;
      reads += o.st.disk_reads;
    }
  std::printf("  disk reads %llu (stage waits %llu), pinned drops %llu, victim discards %llu, clean copies kept %llu "
              "dropped %llu, banks %llu, resurrections %llu, prefetched %llu, short %llu, l1 reverted %llu\n",
              (unsigned long long)sum.disk_reads, (unsigned long long)sum.disk_stage_waits,
              (unsigned long long)sum.disk_drops, (unsigned long long)sum.disk_discards,
              (unsigned long long)sum.inclusive_kept, (unsigned long long)sum.inclusive_drops,
              (unsigned long long)sum.bank_serves, (unsigned long long)sum.resurrected,
              (unsigned long long)sum.prefetched, (unsigned long long)sum.evict_short,
              (unsigned long long)sum.l1_reverted);
  std::printf("  overlapped reads %llu, most in flight %llu, landed out of order %llu; prefetch reads carried across "
              "calls %llu (%llu of %llu bytes read)\n",
              (unsigned long long)overlapped, (unsigned long long)max_in_flight, (unsigned long long)ooo,
              (unsigned long long)carried, (unsigned long long)carried_bytes, (unsigned long long)read_bytes);
  CHECK(overlapped > 0);
  CHECK(carried > 0);
  CHECK(max_in_flight > 1);
  CHECK(ooo > 0);
  CHECK(sum.disk_reads > 0);
  CHECK(sum.disk_stage_waits > 0);
  CHECK(sum.disk_drops > 0);
  CHECK(sum.disk_discards > 0);
  CHECK(sum.inclusive_kept > 0);
  CHECK(sum.inclusive_drops > 0);
  CHECK(sum.bank_serves > 0);
  CHECK(sum.l1_reverted > 0);
}

// Disk units arrive on demand only unless disk_prefetch: a prediction of a file home issues nothing.
TEST(dg_three_tier_demand_only) {
  Shape sh = kDiskPredict;
  sh.disk_prefetch = false;
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    Outcome o = run(sh, 30, seed, DgCore::kNoMutation);
    CHECK(o.violation.empty());
  }
  Violation v;
  Scripted t(40, 12 + 2 * 4, 30 + 2, 2, 5, &v);
  DgConfig c;
  c.n_units = {20, 20};
  c.persistent_slots = 12;
  c.bank_slots = 4;
  c.blocks = 30;
  c.spare_target = 4;
  c.disk_tier = true;
  c.home_blocks = 28;
  c.disk_staging = 2;
  DgCore core(c, &t);
  std::vector<uint8_t> disk(40, 0);
  disk[25] = disk[26] = 1;  // gated layer 1, units 5 and 6
  std::vector<uint32_t> order;
  for (uint32_t k = 0; k < 40; ++k) order.push_back(k);
  core.place(order, &disk);
  // 40 keys - 8 device (12 slots - 4 spares) - 28 pinned homes = 4 file homes: the 2 disk keys, then 2 by priority
  CHECK_EQ(core.file_units(), 4u);
  CHECK(core.home(25) == DgCore::Home::File && core.home(26) == DgCore::Home::File);
  core.begin_token(1);
  const uint32_t id = 0;
  core.serve(0, &id, 1, 1);
  const uint32_t pred[2] = {5, 6};
  core.serve_prediction(1, pred, 2);
  CHECK_EQ(core.stats().prefetched, 0ull);
  CHECK_EQ(core.stats().disk_reads, 0ull);
  core.serve(1, pred, 2, 3);  // the demand reads both
  CHECK_EQ(core.stats().disk_reads, 2ull);
  core.end_token({});
  core.check();
  CHECK(core.home(25) == DgCore::Home::Device && core.block_of(25) == -1);
  t.drain_all();
  CHECK(v.what.empty());
}

TEST(dg_three_tier_mutations_caught) {
  const struct {
    int m;
    const char* name;
  } muts[] = {{DgCore::kEntryBeforeCopy, "entry before copy"},
              {DgCore::kNoCover, "no cover wait"},
              {DgCore::kNoBankUnpublish, "bank alias not unpublished"},
              {DgCore::kEarlyWritebackBlock, "write-back block reused early"},
              {DgCore::kStageNoWait, "staging buffer reused before its copy"}};
  for (const auto& mu : muts) {
    int caught = 0, runs = 0;
    for (const Shape* base : kTierShapes)
      for (int how = 0; how < 3; ++how)
      for (uint64_t seed = 1; seed <= 12; ++seed) {
        ++runs;
        if (!run(with_reads(*base, how), 40, seed, mu.m).violation.empty()) ++caught;
      }
    std::printf("  %-38s caught in %d/%d runs\n", mu.name, caught, runs);
    CHECK(caught > 0);
  }
}

TEST(dg_three_tier_reset_open) {
  for (const Shape* base : kTierShapes)
    for (uint64_t seed = 1; seed <= 10; ++seed) {
      Outcome o = run(with_reads(*base, int(seed % 3)), 1 + int(seed % 19), 700 + seed, DgCore::kNoMutation, 6, true);
      if (!o.violation.empty()) std::printf("  reset seed %llu: %s\n", (unsigned long long)seed, o.violation.c_str());
      CHECK(o.violation.empty());
    }
}

// The disk tier switched on with no file unit and blocks for every victim is the two-tier cache: the same victims,
// copies and counters on the same seeds (the engine turns it on only when its plan has disk units).
TEST(dg_three_tier_full_coverage_is_today) {
  for (const Shape* base : {&kWarm, &kPredict, &kSlowWriteback}) {
    Shape a = *base;
    a.host_spares = 200;  // never short of a block
    Shape b = a;
    b.disk_tier = true;
    for (uint64_t seed = 1; seed <= 6; ++seed) {
      Outcome x = run(a, 30, seed, DgCore::kNoMutation), y = run(b, 30, seed, DgCore::kNoMutation);
      CHECK(x.violation.empty() && y.violation.empty());
      CHECK(x.victims == y.victims);
      CHECK(counters(x.st) == counters(y.st));
      CHECK_EQ(y.st.disk_reads + y.st.disk_drops + y.st.disk_discards, 0ull);
    }
  }
}

// Overlapped file misses are exact: on a timing-free device, depth 16 with reads landing out of order and depth 1 (the
// synchronous reads) end with the same homes, table, device table and slot contents, the same victims at every
// boundary (each also checked against the oracle inside run) and the same counters, except what the export observed:
// a prefetch whose read is still in flight is not published yet, so the export sees a miss that its cover serves (no
// copy more, the same slot and entry).
ts::DgStats placement(ts::DgStats s) {
  s.lookups = s.misses = s.covered = s.covered_waits = s.miss_layers = s.prior_late = s.l1_late = 0;
  return s;
}

TEST(dg_three_tier_queue_depth_is_exact) {
  uint64_t overlapped = 0, compared = 0;
  for (const Shape* base : kTierShapes)
    for (uint64_t seed = 1; seed <= 6; ++seed) {
      Shape a = *base;
      a.eager = true;
      a.queue = 1;
      Shape b = a;
      b.queue = 16;
      b.async_reads = true;
      const Outcome x = run(a, 30, seed, DgCore::kNoMutation, 0), y = run(b, 30, seed, DgCore::kNoMutation, 0);
      if (!x.violation.empty() || !y.violation.empty())
        std::printf("  seed %llu: %s / %s\n", (unsigned long long)seed, x.violation.c_str(), y.violation.c_str());
      CHECK(x.violation.empty() && y.violation.empty());
      CHECK(x.victims == y.victims);
      CHECK(counters(placement(x.st)) == counters(placement(y.st)));
      CHECK(x.homes == y.homes);
      CHECK(x.table == y.table);
      CHECK(x.device_table == y.device_table);
      CHECK(x.slots == y.slots);
      for (auto f : {&ts::DgStats::disk_reads, &ts::DgStats::disk_read_bytes, &ts::DgStats::disk_drops,
                     &ts::DgStats::disk_discards, &ts::DgStats::inclusive_kept, &ts::DgStats::inclusive_drops})
        CHECK_EQ(x.st.*f, y.st.*f);
      CHECK_EQ(x.st.disk_overlapped, 0ull);
      overlapped += y.st.disk_overlapped;
      ++compared;
    }
  std::printf("  %llu runs compared, %llu reads overlapped at depth 16\n", (unsigned long long)compared,
              (unsigned long long)overlapped);
  CHECK(overlapped > 0);
}

// A failed file read surfaces where its read is ended: at the drain (here the layer's ready) with the reads in flight,
// in the fetch itself with synchronous reads. Either way the failed key is back in its file, its slot free, the other
// misses complete, and the core stays consistent and usable.
namespace {
struct ReadErrorRig {
  Violation v;
  Scripted t{40, 12 + 2 * 4, 30 + 4, 2, 5, &v};
  DgConfig c;
  std::unique_ptr<DgCore> core;
  explicit ReadErrorRig(uint32_t queue, bool async, bool disk_prefetch = false, uint32_t ndisk = 4) {
    c.n_units = {20, 20};
    c.persistent_slots = 12;
    c.bank_slots = 4;
    c.blocks = 30;
    c.spare_target = 4;
    c.disk_tier = true;
    c.home_blocks = 32 - ndisk;
    c.disk_staging = 4;
    c.disk_queue = queue;
    c.disk_prefetch = disk_prefetch;
    core = std::make_unique<DgCore>(c, &t);
    std::vector<uint8_t> disk(40, 0);
    for (uint32_t k = 25; k < 25 + ndisk; ++k) disk[k] = 1;  // gated layer 1, units 5..
    std::vector<uint32_t> order;
    for (uint32_t k = 0; k < 40; ++k) order.push_back(k);
    core->place(order, &disk);
    for (uint32_t k = 0; k < 40; ++k) {
      if (core->slot_of(k) >= 0) t.fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t.fill_block(core->block_of(k), int32_t(k));
    }
    t.set_table(core->table());
    t.set_async_reads(async, 77);
    core->begin_token(1);
  }
};
}  // namespace

TEST(dg_three_tier_read_error) {
  {  // depth 16, reads in flight: the second of three fails, at the drain before the layer's ready
    ReadErrorRig r(16, true);
    DgCore& core = *r.core;
    CHECK_EQ(core.file_units(), 4u);
    const uint32_t free0 = core.free_slots();
    r.t.fail_read(2);
    CHECK(core.fetch(1, 5, false));
    CHECK(core.fetch(1, 6, false));
    CHECK(core.fetch(1, 7, false));
    CHECK_EQ(core.reads_in_flight(), 3u);
    CHECK_EQ(r.t.reads_in_flight(), 3u);
    core.check();
    bool threw = false;
    try {
      core.signal(1);
    } catch (const ts::Error& e) {
      threw = e.status() == SS_E_IO;
    }
    CHECK(threw);
    CHECK_EQ(core.reads_in_flight(), 0u);
    CHECK_EQ(r.t.reads_in_flight(), 0u);
    core.check();
    CHECK(core.home(26) == DgCore::Home::File && core.table()[26] == -1);
    CHECK_EQ(core.free_slots(), free0 - 2);
    CHECK_EQ(core.file_units(), 2u);
    CHECK_EQ(core.stats().fetched, 2ull);
    CHECK_EQ(core.stats().disk_reads, 2ull);
    r.t.sync(DgTransport::kDem);
    for (uint32_t k : {25u, 27u}) {
      CHECK(core.home(k) == DgCore::Home::Coming);
      CHECK_EQ(r.t.entry(k), core.slot_of(k));
      CHECK_EQ(r.t.slot_holds(core.slot_of(k)), int32_t(k));
    }
    CHECK_EQ(r.t.entry(26), -1);
    core.signal(1);  // usable: the ready goes now; the failed unit is fetched again from its file
    const uint32_t id = 6;
    core.serve(1, &id, 1, 1);
    r.t.sync(DgTransport::kDem);
    CHECK_EQ(r.t.slot_holds(core.slot_of(26)), 26);
    core.end_token({});
    core.check();
    r.t.drain_all();
    for (uint32_t k = 0; k < 40; ++k) CHECK_EQ(r.t.entry(k), core.table()[k]);
    CHECK(r.v.what.empty());
  }
  {  // depth 16, the error from serve_demand: its fetches begun every read, the drain before the ready throws
    ReadErrorRig r(16, true);
    r.t.fail_read(1);
    const uint32_t ids[3] = {5, 6, 7};
    bool threw = false;
    try {
      r.core->serve_demand(1, ids, 3, 7);
    } catch (const ts::Error& e) {
      threw = e.status() == SS_E_IO;
    }
    CHECK(threw);
    r.core->check();
    CHECK(r.core->home(25) == DgCore::Home::File);
    CHECK(r.core->home(26) == DgCore::Home::Coming && r.core->home(27) == DgCore::Home::Coming);
    r.t.drain_all();
    CHECK(r.v.what.empty());
  }
  {  // depth 1, synchronous reads: the failing fetch throws itself and gives its slot back
    ReadErrorRig r(1, false);
    DgCore& core = *r.core;
    const uint32_t free0 = core.free_slots();
    CHECK(core.fetch(1, 5, false));
    r.t.fail_read(1);
    bool threw = false;
    try {
      core.fetch(1, 6, false);
    } catch (const ts::Error& e) {
      threw = e.status() == SS_E_IO;
    }
    CHECK(threw);
    core.check();
    CHECK(core.home(26) == DgCore::Home::File);
    CHECK_EQ(core.free_slots(), free0 - 1);
    CHECK_EQ(core.reads_in_flight(), 0u);
    CHECK(core.fetch(1, 6, false));
    core.signal(1);
    core.end_token({});
    core.check();
    r.t.drain_all();
    CHECK(r.v.what.empty());
  }
}

// Prefetch file reads stay in flight across calls: issued by the prediction for layer 1, they are still in flight
// through layer 0's write-back slice, ranking and use, and end at layer 1's demand: a covered key's at its cover, the
// others' before the ready; every copy and entry of layer 1 is enqueued before its ready. A fetch of a key whose read
// is in flight ends it; the boundary ends the rest.
TEST(dg_three_tier_prefetch_reads_carried) {
  ReadErrorRig r(16, true, true, 8);
  DgCore& core = *r.core;
  const uint32_t id0 = 0;
  core.serve(0, &id0, 1, r.t.entry(0) < 0 ? 1 : 0);
  const uint32_t pred[3] = {5, 6, 7};
  core.serve_prediction(1, pred, 3);
  CHECK_EQ(core.stats().prefetched, 3ull);
  CHECK_EQ(core.reads_in_flight(), 3u);
  core.issue_writebacks(0);
  core.score_layer(0);
  core.use(0, 0, false);
  core.check();
  CHECK_EQ(core.reads_in_flight(), 3u);  // still in flight: nothing needed them
  for (uint32_t k : {25u, 26u, 27u}) CHECK_EQ(r.t.entry(k), -1);  // not published before their copies
  r.t.clear_issued();
  const uint32_t ids[2] = {6, 8};  // 6 was prefetched (covered), 8 is a demand miss
  core.serve(1, ids, 2, 3);
  CHECK_EQ(core.reads_in_flight(), 0u);
  CHECK_EQ(core.stats().disk_carried, 3ull);
  CHECK_EQ(core.stats().disk_carried_bytes, 3 * (2048ull + 2048 + 4096));
  CHECK_EQ(core.stats().covered_waits, 1ull);
  // the ready of layer 1 after every copy and entry of its units
  const auto& tr = r.t.trace();
  size_t ready = tr.size();
  for (size_t i = 0; i < tr.size(); ++i)
    if (tr[i][1] == Scripted::Ready && tr[i][2] == 1) ready = i;
  CHECK(ready < tr.size());
  for (uint32_t k : {25u, 26u, 27u, 28u}) {
    bool copied = false, entry = false;
    for (size_t i = 0; i < ready; ++i) {
      copied = copied || (tr[i][1] == Scripted::H2D && tr[i][2] == core.slot_of(k));
      entry = entry || (tr[i][1] == Scripted::Entry && tr[i][4] == int64_t(k));
    }
    CHECK(copied && entry);
  }
  r.t.sync(DgTransport::kDem);
  r.t.sync(DgTransport::kPf);
  for (uint32_t k : {25u, 26u, 27u, 28u}) CHECK_EQ(r.t.slot_holds(core.slot_of(k)), int32_t(k));
  core.end_token({});
  core.check();
  r.t.drain_all();  // the boundary's write-backs land: their slots are free for the next prefetches
  // a fetch of a key whose read is in flight ends it (and only up to it)
  core.begin_token(2);
  core.serve(0, &id0, 1, r.t.entry(0) < 0 ? 1 : 0);
  const uint32_t pred2[2] = {9, 10};
  core.serve_prediction(1, pred2, 2);
  CHECK_EQ(core.reads_in_flight(), 2u);
  CHECK(!core.fetch(1, 9, true));
  CHECK_EQ(core.reads_in_flight(), 1u);
  core.end_token({});  // the boundary ends the last one
  CHECK_EQ(core.reads_in_flight(), 0u);
  core.check();
  r.t.drain_all();
  for (uint32_t k = 0; k < 40; ++k) CHECK_EQ(r.t.entry(k), core.table()[k]);
  CHECK(r.v.what.empty());
  // scripted runs with predictions / window priors reading file units, reads landing late and out of order
  for (const Shape* base : {&kDiskPredict, &kDiskWinpf}) {
    const Outcome o = run(with_reads(*base, 1), 40, 1, DgCore::kNoMutation);
    CHECK(o.violation.empty());
    CHECK(o.st.disk_carried > 0);
    std::printf("  %s, 40 tokens: %llu file reads (%llu bytes; %llu prefetched units), %llu begun with another in "
                "flight, %llu prefetch reads carried across calls (%llu bytes)\n",
                base == &kDiskPredict ? "predictions" : "window priors", (unsigned long long)o.st.disk_reads,
                (unsigned long long)o.st.disk_read_bytes, (unsigned long long)o.st.prefetched,
                (unsigned long long)o.st.disk_overlapped, (unsigned long long)o.st.disk_carried,
                (unsigned long long)o.st.disk_carried_bytes);
  }
}

// Pressure mode (as production launches it): every demand miss collects, a demand without a free slot forces a
// deferred write-back, past the per-token free share misses go to the banks. None of these ends the reads in flight
// any more (they touch no key, slot, staging buffer or event a read holds), so the misses of a layer still overlap.
// On a timing-free device depth 1 and depth 16 (reads landing late and out of order) end identical; on the random
// device every run holds its invariants with reads overlapping.
namespace {
Shape pressure_shape(bool defer) {
  Shape s{{64, 64, 48, 64}, 40, 16, 8, 8, 8};  // a small arena: 40 residents + 16 spares for 240 units
  s.disk = 150;
  s.staging = 8;
  s.pressure_free = 8;
  s.pressure_recent = 8;
  s.defer = defer;
  s.slow_d2h = defer;
  s.predict = defer ? 0 : 4;
  s.disk_prefetch = !defer;
  return s;
}
}  // namespace

TEST(dg_three_tier_pressure_overlap) {
  uint64_t overlapped = 0, most = 0, collects = 0, forced = 0, banks = 0, misses = 0, compared = 0;
  for (bool defer : {true, false})
    for (uint64_t seed = 1; seed <= 8; ++seed) {
      Shape a = pressure_shape(defer);
      a.eager = true;
      a.queue = 1;
      Shape b = a;
      b.queue = 16;
      b.async_reads = true;
      const Outcome x = run(a, 30, seed, DgCore::kNoMutation, 0), y = run(b, 30, seed, DgCore::kNoMutation, 0);
      if (!x.violation.empty() || !y.violation.empty())
        std::printf("  seed %llu: %s / %s\n", (unsigned long long)seed, x.violation.c_str(), y.violation.c_str());
      CHECK(x.violation.empty() && y.violation.empty());
      CHECK(x.victims == y.victims);
      CHECK(counters(placement(x.st)) == counters(placement(y.st)));
      CHECK(x.homes == y.homes);
      CHECK(x.table == y.table);
      CHECK(x.device_table == y.device_table);
      CHECK(x.slots == y.slots);
      for (auto f : {&ts::DgStats::disk_reads, &ts::DgStats::disk_read_bytes, &ts::DgStats::disk_drops,
                     &ts::DgStats::disk_discards, &ts::DgStats::wb_forced, &ts::DgStats::collected_mid})
        CHECK_EQ(x.st.*f, y.st.*f);
      CHECK_EQ(x.st.disk_overlapped, 0ull);
      CHECK_EQ(x.st.disk_max_in_flight, 1ull);
      ++compared;
      // the random device, the same shape at depth 16 with late reads
      Shape r = pressure_shape(defer);
      r.async_reads = true;
      const Outcome z = run(r, 40, seed, DgCore::kNoMutation);
      if (!z.violation.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, z.violation.c_str());
      CHECK(z.violation.empty());
      for (const Outcome* o : {&y, &z}) {
        overlapped += o->st.disk_overlapped;
        most = std::max(most, o->st.disk_max_in_flight);
        collects += o->st.collected_mid;
        forced += o->st.wb_forced;
        banks += o->st.bank_serves;
        misses += o->st.fetched;
      }
    }
  std::printf("  pressure: %llu runs compared; depth 16: %llu demand fetches, %llu reads begun with another in flight, "
              "at most %llu in flight; mid-token collects %llu, forced write-backs %llu, bank serves %llu\n",
              (unsigned long long)compared, (unsigned long long)misses, (unsigned long long)overlapped,
              (unsigned long long)most, (unsigned long long)collects, (unsigned long long)forced,
              (unsigned long long)banks);
  CHECK(overlapped > 0);
  CHECK(most > 1);
  CHECK(forced > 0);
  CHECK(banks > 0);
}

// ---- kept-resident layers (ss_dg_make_resident) and steps that leave gated layers unexported ----
namespace {

// A small core under the scripted device: gated layers of `units`, the first `residents` keys on the device (or the
// given priority order), `spares` spare slots (the target), `host_spares` spare blocks. Three-tier when `disk` names
// file-homed keys (home_blocks pinned homes); `inclusive` clean copies.
struct ResRig {
  Violation v;
  DgConfig c;
  std::unique_ptr<Scripted> t;
  std::unique_ptr<DgCore> core;
  uint32_t seq = 0, keys = 0, G = 0;
  std::mt19937_64 rng{11};
  ResRig(const std::vector<uint32_t>& units, uint32_t residents, uint32_t spares, uint32_t host_spares,
         bool scored = false, const std::vector<uint8_t>* disk = nullptr, uint32_t home_blocks = 0,
         uint64_t seed = 7, uint32_t inclusive = 0, const std::function<void(DgConfig&)>& tweak = nullptr) {
    for (uint32_t u : units) keys += u;
    G = uint32_t(units.size());
    c.n_units = units;
    c.persistent_slots = residents + spares;
    c.bank_slots = 8;
    c.spare_target = spares;
    c.prefetch_reserve = 0;
    c.scored = scored;
    c.max_ids = 8;
    c.inclusive = inclusive;
    if (disk) {
      c.disk_tier = true;
      c.home_blocks = home_blocks;
      c.blocks = home_blocks + host_spares;
      c.disk_staging = 2;
      c.disk_queue = 2;
    } else {
      c.blocks = keys - residents + host_spares;
    }
    if (tweak) tweak(c);
    t = std::make_unique<Scripted>(keys, c.persistent_slots + 2 * c.bank_slots, c.blocks + c.disk_staging, G, seed, &v);
    core = std::make_unique<DgCore>(c, t.get());
    std::vector<uint32_t> order;
    for (uint32_t k = 0; k < (disk ? keys : residents); ++k) order.push_back(k);
    core->place(order, disk);
    for (uint32_t k = 0; k < keys; ++k) {
      if (core->slot_of(k) >= 0) t->fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t->fill_block(core->block_of(k), int32_t(k));
    }
    t->set_table(core->table());
    core->check();
  }
  std::vector<uint32_t> layer(uint32_t g) const {
    std::vector<uint32_t> out;
    for (uint32_t u = 0; u < core->units(g); ++u) out.push_back(u);
    return out;
  }
  std::vector<uint32_t> keys_of(uint32_t g, const std::vector<uint32_t>& ids) const {
    std::vector<uint32_t> out;
    for (uint32_t u : ids) out.push_back(core->key(g, u));
    return out;
  }
  // One token: the layers with run[g] export ids[g] (miss bits from the device table), are served (ranked when
  // scored), waited for and read; every key of `held` is read in every layer, its slot under the read throughout.
  // Returns the units the token read (the bytes of each slot).
  std::vector<int32_t> token(const std::vector<std::vector<uint32_t>>& ids, const std::vector<uint8_t>& run,
                             const std::vector<uint32_t>& keep, const std::vector<uint32_t>& held = {}) {
    ++seq;
    core->begin_token(seq, &run);
    std::vector<int32_t> out;
    for (uint32_t g = 0; g < G; ++g) {
      std::set<int32_t> reading;
      std::vector<std::pair<int32_t, int32_t>> read;  // (slot, the unit it must hold while read)
      for (uint32_t k : held) {
        const int32_t s = t->entry(k);
        if (s < 0 || t->slot_holds(s) != int32_t(k)) v.set("a held unit read wrong bytes");
        else reading.insert(s), read.push_back({s, int32_t(k)});
        out.push_back(s < 0 ? -1 : t->slot_holds(s));
      }
      if (run[g]) {
        uint64_t m = 0;
        for (size_t i = 0; i < ids[g].size(); ++i)
          if (t->entry(core->key(g, ids[g][i])) < 0) m |= 1ull << i;
        t->steps(4);
        core->serve(g, ids[g].data(), uint32_t(ids[g].size()), m);
        core->score_layer(g);
        if (m)
          while (int32_t(t->ready(g) - seq) < 0)
            if (!t->step()) {
              v.set("ready never came");
              break;
            }
        for (uint32_t id : ids[g]) {
          const uint32_t k = core->key(g, id);
          const int32_t s = t->entry(k);
          if (s < 0 || t->slot_holds(s) != int32_t(k)) v.set("layer " + std::to_string(g) + " read wrong bytes");
          else reading.insert(s), read.push_back({s, int32_t(k)});
          out.push_back(s < 0 ? -1 : t->slot_holds(s));
        }
      }
      t->begin_read(reading);
      t->steps(4);
      for (const auto& sk : read)
        if (t->slot_holds(sk.first) != sk.second) v.set("a slot changed while read");
      t->end_read();
    }
    core->end_token(keep);
    core->check();
    t->steps(8);
    return out;
  }
  std::vector<std::vector<uint32_t>> routes(uint32_t n) {
    std::vector<std::vector<uint32_t>> r(G);
    for (uint32_t g = 0; g < G; ++g)
      for (uint32_t i = 0; i < n; ++i) r[g].push_back(uint32_t(rng() % core->units(g)));
    return r;
  }
  // every unit at rest holds its bytes in its home, the device table equals the mirror
  void verify() {
    t->drain_all();
    core->check();
    for (uint32_t k = 0; k < keys; ++k) {
      CHECK_EQ(t->entry(k), core->table()[k]);
      const DgCore::Home h = core->home(k);
      if (h == DgCore::Home::Device || h == DgCore::Home::Evicting) CHECK_EQ(t->slot_holds(core->slot_of(k)), int32_t(k));
      else if (h == DgCore::Home::Host) CHECK_EQ(t->block_holds(core->block_of(k)), int32_t(k));
      if (core->backed(k)) CHECK_EQ(t->block_holds(core->block_of(k)), int32_t(k));
    }
  }
  std::vector<int64_t> state() const {
    std::vector<int64_t> s;
    for (uint32_t k = 0; k < keys; ++k)
      s.insert(s.end(), {int64_t(core->home(k)), core->slot_of(k), core->block_of(k), int64_t(core->lru_stamp(k))});
    s.insert(s.end(), core->table().begin(), core->table().end());
    s.insert(s.end(), {int64_t(core->free_slots()), int64_t(core->free_blocks()), int64_t(core->on_device())});
    for (uint64_t x : counters(core->stats())) s.push_back(int64_t(x));
    s.push_back(int64_t(core->stats().resident_units));
    return s;
  }
};

std::vector<uint8_t> all_run(uint32_t G) { return std::vector<uint8_t>(G, 1); }

// The counters of decode's own work (make_resident may only complete the write-backs and reads already in flight:
// writebacks_done / wb_*_bytes / disk_read* are not among them).
std::vector<uint64_t> decode_counts(const ts::DgStats& s) {
  return {s.tokens, s.lookups, s.misses, s.covered, s.fetched, s.prefetched, s.prefetch_used, s.resurrected,
          s.bank_serves, s.collected_mid, s.evicted, s.evict_short, s.stalled, s.pred_invalid, s.covered_waits,
          s.predicted, s.predicted_true, s.miss_layers, s.wb_deferred, s.wb_cancelled, s.wb_forced,
          s.wb_boundary_flushed, s.disk_drops, s.disk_discards, s.inclusive_kept, s.inclusive_drops, s.disk_stage_waits,
          s.score_layers, s.score_candidates, s.scored_victims, s.score_fallbacks};
}

Shape with_resident(const Shape& base, bool resident, uint32_t skip) {
  Shape s = base;
  s.resident = resident;
  s.skip = skip;
  return s;
}

}  // namespace

// A whole layer made resident where it fits: every unit on the device with its file bytes, kept, its entries
// published; a step that exports no layer reads the kept slots through the table; after the release the next decode
// tokens read exactly what a cache that never made the layer resident reads (only the placement differs).
TEST(dg_resident_full_layer) {
  ResRig a({16, 16, 16}, 20, 8, 8), b({16, 16, 16}, 20, 8, 8);
  for (int tok = 0; tok < 6; ++tok) {  // warm both the same way
    const auto r = a.routes(4);
    b.rng = a.rng;
    CHECK(a.token(r, all_run(3), {}) == b.token(r, all_run(3), {}));
  }
  a.verify();
  const uint32_t g = 1;
  uint32_t on_dev = 0;
  for (uint32_t u = 0; u < 16; ++u) on_dev += a.core->home(a.core->key(g, u)) == DgCore::Home::Device;
  const auto before = decode_counts(a.core->stats());
  const DgCore::ResidentResult res = a.core->make_resident(g, a.layer(g), {});
  CHECK_EQ(res.copied, 16u - on_dev);
  CHECK_EQ(a.core->stats().resident_units, uint64_t(16 - on_dev));
  CHECK_EQ(a.core->stats().resident_bytes, uint64_t(16 - on_dev) * 8192u);
  CHECK_EQ(a.core->stats().resident_evicted, uint64_t(res.evicted));
  CHECK(decode_counts(a.core->stats()) == before);  // nothing counted as a miss, fetch, prefetch or eviction
  std::vector<int32_t> slots;
  for (uint32_t u = 0; u < 16; ++u) {
    const uint32_t k = a.core->key(g, u);
    CHECK(a.core->home(k) == DgCore::Home::Device);
    CHECK_EQ(a.t->entry(k), a.core->slot_of(k));  // published when the call returned (no stream left to run)
    CHECK_EQ(a.t->slot_holds(a.core->slot_of(k)), int32_t(k));
    slots.push_back(a.core->slot_of(k));
  }
  a.verify();
  // the prefill step: no layer exported, the kept units read through the table; the boundary keeps them in place
  const auto kept = a.keys_of(g, a.layer(g));
  const uint64_t lookups = a.core->stats().lookups;
  const uint32_t ready1 = a.t->ready(g);
  const auto out = a.token(a.routes(1), {0, 0, 0}, kept, kept);
  CHECK_EQ(out.size(), size_t(3 * 16));
  CHECK_EQ(a.core->stats().lookups, lookups);
  CHECK_EQ(a.t->ready(g), ready1);
  for (uint32_t u = 0; u < 16; ++u) CHECK_EQ(a.core->slot_of(a.core->key(g, u)), slots[u]);
  a.token(a.routes(1), {0, 0, 0}, kept, kept);  // held for several steps: still in place
  for (uint32_t u = 0; u < 16; ++u) CHECK_EQ(a.core->slot_of(a.core->key(g, u)), slots[u]);
  a.verify();
  CHECK(a.v.what.empty());
  // released: the same decode tokens read the same bytes as the cache that never did any of this
  for (int tok = 0; tok < 12; ++tok) {
    const auto r = a.routes(5);
    b.rng = a.rng;
    const auto x = a.token(r, all_run(3), {}), y = b.token(r, all_run(3), {});
    CHECK(x == y);
  }
  a.verify();
  b.verify();
  if (!a.v.what.empty() || !b.v.what.empty()) std::printf("  %s / %s\n", a.v.what.c_str(), b.v.what.c_str());
  CHECK(a.v.what.empty() && b.v.what.empty());
}

// Several layers held at once; a call over the capacity is refused before anything changes (homes, slots, blocks,
// recency, table, counters, no device operation); replacing a layer's set counts it once.
TEST(dg_resident_several_layers_and_capacity) {
  ResRig r({16, 16, 16}, 32, 8, 8);  // 40 persistent slots, target 8: 32 units may be kept
  for (int tok = 0; tok < 4; ++tok) r.token(r.routes(4), all_run(3), {});
  r.core->make_resident(0, r.layer(0), {});
  const auto k0 = r.keys_of(0, r.layer(0));
  r.core->make_resident(1, r.layer(1), k0);
  auto k1 = r.keys_of(1, r.layer(1));
  std::vector<uint32_t> both = k0;
  both.insert(both.end(), k1.begin(), k1.end());
  for (uint32_t k : both) CHECK(r.core->home(k) == DgCore::Home::Device);
  r.verify();
  r.t->drain_all();
  auto refused = [&](uint32_t g, const std::vector<uint32_t>& ids, const std::vector<uint32_t>& other) {
    const auto s0 = r.state();
    r.t->clear_issued();
    bool cap = false;
    try {
      r.core->make_resident(g, ids, other);
    } catch (const ts::Error& e) {
      cap = e.status() == SS_E_CAPACITY;
    }
    CHECK(cap);
    CHECK(r.t->issued().empty());
    CHECK(r.state() == s0);
    r.core->check();
  };
  refused(2, r.layer(2), both);  // 16 + 32 + 8 > 40
  refused(2, {5}, both);         // 1 + 32 + 8 > 40
  // invalid arguments are refused too, before anything changes
  for (const std::vector<uint32_t>& bad : {std::vector<uint32_t>{3, 3}, std::vector<uint32_t>{16}}) {
    bool inv = false;
    try {
      r.core->make_resident(2, bad, {});
    } catch (const ts::Error& e) {
      inv = e.status() == SS_E_INVALID;
    }
    CHECK(inv);
  }
  // layer 1 again with fewer units: its old set does not count (replaced); then layer 2's part fits
  r.core->make_resident(1, {0, 1, 2, 3}, k0);
  k1 = r.keys_of(1, {0, 1, 2, 3});
  std::vector<uint32_t> other = k0;
  other.insert(other.end(), k1.begin(), k1.end());
  std::vector<uint32_t> part;
  for (uint32_t u = 0; u < 12; ++u) part.push_back(u);
  r.core->make_resident(2, part, other);
  const auto k2 = r.keys_of(2, part);
  std::vector<uint32_t> all = other;
  all.insert(all.end(), k2.begin(), k2.end());
  for (uint32_t k : all) CHECK(r.core->home(k) == DgCore::Home::Device);
  // three layers held: a step exports none of them and reads only the kept units
  r.token(r.routes(1), {0, 0, 0}, all, all);
  r.verify();
  for (uint32_t k : all) CHECK(r.core->home(k) == DgCore::Home::Device);
  // two-tier with no free slot and no free block to start the write-backs: refused, nothing changed
  ResRig z({8, 8}, 8, 0, 0);
  const auto s0 = z.state();
  bool cap = false;
  try {
    z.core->make_resident(1, z.layer(1), {});
  } catch (const ts::Error& e) {
    cap = e.status() == SS_E_CAPACITY;
  }
  CHECK(cap);
  CHECK(z.state() == s0);
  CHECK(r.v.what.empty() && z.v.what.empty());
}

// LRU: the victims are the least recent device units that no layer keeps and the call does not want, written back
// into free blocks (two-tier) with every byte where its home says; kept units of other layers stay.
TEST(dg_resident_eviction_lru) {
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    ResRig r({24, 24, 24}, 30, 6, 6, false, nullptr, 0, seed);
    r.rng.seed(seed);
    for (int tok = 0; tok < 10; ++tok) r.token(r.routes(5), all_run(3), {});
    const std::vector<uint32_t> keep0 = r.keys_of(0, {0, 1, 2, 3});  // 24 + 4 + 6 <= 36
    std::vector<uint8_t> want(r.keys, 0), other(r.keys, 0);
    for (uint32_t k : keep0) other[k] = 1;
    for (uint32_t k : r.keys_of(2, r.layer(2))) want[k] = 1;
    // the expected victims: after the drain every write-back is collected (its slot free); then the recency tail
    size_t need = 0, free_after = r.core->free_slots();
    for (uint32_t k = 0; k < r.keys; ++k) {
      free_after += r.core->home(k) == DgCore::Home::Evicting;
      need += want[k] && r.core->home(k) != DgCore::Home::Device;
    }
    std::vector<uint32_t> expect, rec = r.core->residency();
    for (size_t i = rec.size(); i-- > 0 && expect.size() + free_after < need;)
      if (!want[rec[i]] && !other[rec[i]]) expect.push_back(rec[i]);
    r.core->make_resident(2, r.layer(2), keep0);
    CHECK(r.core->resident_victims() == expect);
    CHECK(!expect.empty());
    for (uint32_t k : keep0)
      if (r.core->home(k) == DgCore::Home::Device) CHECK(std::find(expect.begin(), expect.end(), k) == expect.end());
    for (uint32_t k : expect) CHECK(r.core->home(k) == DgCore::Home::Host);  // written back
    r.verify();
    // dirty victims come back through a demand with their bytes
    std::vector<std::vector<uint32_t>> back(3);
    for (uint32_t k : expect) {
      const uint32_t g = r.core->gated_of(k);
      if (back[g].size() < 4) back[g].push_back(k - r.core->key(g, 0));
    }
    for (auto& b : back)
      if (b.empty()) b.push_back(0);
    auto keep = r.keys_of(2, r.layer(2));
    keep.insert(keep.end(), keep0.begin(), keep0.end());
    r.token(back, {1, 1, 0}, keep, r.keys_of(2, r.layer(2)));
    r.verify();
    CHECK(r.v.what.empty());
  }
}

// Scored: the victims are the eligible units in the boundary's order (score, demand epoch, recency stamp), from the
// oracle's own history.
TEST(dg_resident_eviction_scored) {
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    ResRig r({24, 24, 24}, 30, 6, 6, true, nullptr, 0, seed);
    r.rng.seed(seed * 3);
    Oracle orc({24, 24, 24});
    for (int tok = 0; tok < 12; ++tok) {
      const auto rt = r.routes(5);
      orc.begin();
      for (uint32_t g = 0; g < 3; ++g) orc.serve(g, rt[g].data(), uint32_t(rt[g].size()));
      r.token(rt, all_run(3), {});
    }
    const std::vector<uint32_t> keep0 = r.keys_of(0, {1, 3, 5});
    std::vector<uint8_t> want(r.keys, 0), other(r.keys, 0);
    for (uint32_t k : keep0) other[k] = 1;
    for (uint32_t k : r.keys_of(1, r.layer(1))) want[k] = 1;
    size_t need = 0, free_after = r.core->free_slots();
    struct C {
      uint64_t s, ep, st;
      uint32_t k;
    };
    std::vector<C> all;
    for (uint32_t k = 0; k < r.keys; ++k) {
      free_after += r.core->home(k) == DgCore::Home::Evicting;
      need += want[k] && r.core->home(k) != DgCore::Home::Device;
      if (r.core->home(k) == DgCore::Home::Device && !want[k] && !other[k])
        all.push_back({orc.score(k), orc.epoch_of[k], r.core->lru_stamp(k), k});
    }
    std::sort(all.begin(), all.end(), [](const C& a, const C& b) {
      return a.s != b.s ? a.s < b.s : a.ep != b.ep ? a.ep < b.ep : a.st < b.st;
    });
    std::vector<uint32_t> expect;
    for (size_t i = 0; i < all.size() && expect.size() + free_after < need; ++i) expect.push_back(all[i].k);
    const uint64_t layers = r.core->stats().score_layers;
    r.core->make_resident(1, r.layer(1), keep0);
    CHECK(r.core->resident_victims() == expect);
    CHECK(!expect.empty());
    CHECK_EQ(r.core->stats().score_layers, layers);  // ranked outside the boundary's counters
    r.verify();
    CHECK(r.v.what.empty());
  }
}

// Three-tier: units copied in from their pinned blocks and their files (through the staging buffers, reads in flight
// together); victims written back into blocks freed by those copies or by pinned units dropped to their files, else
// dropped to their files themselves; a clean copy goes back to its block. A failed read leaves the cache consistent.
TEST(dg_resident_three_tier) {
  uint64_t written_back = 0, to_file = 0, reads = 0, io_failures = 0;
  for (uint64_t seed = 1; seed <= 8; ++seed)
    for (int fail = 0; fail < 2; ++fail) {
      // 72 keys: 24 resident, 4 pinned homes (+ 0 or 1 spare block), the rest file-homed
      std::vector<uint8_t> disk(72, 0);
      for (uint32_t k = 24 + 4; k < 72; ++k) disk[k] = 1;
      ResRig r({24, 24, 24}, 24, 4, seed % 2, false, &disk, 4, seed);
      r.t->set_async_reads(seed % 3 != 0, seed);
      r.rng.seed(seed);
      for (int tok = 0; tok < 6; ++tok) r.token(r.routes(3), all_run(3), {});
      r.verify();
      // a failure case makes a layer with file units resident (layers 1 and 2): its read must fail, observed
      const uint32_t g = fail ? 1 + uint32_t(seed % 2) : uint32_t(seed % 3);
      const uint64_t reads0 = r.t->reads(), drops0 = r.core->stats().disk_drops;
      if (fail) {
        r.t->fail_read(1);
        bool io = false;
        try {
          r.core->make_resident(g, r.layer(g), {});
        } catch (const ts::Error& e) {
          io = e.status() == SS_E_IO;
        }
        CHECK(io);
        CHECK(!r.t->read_failure_pending());  // the injected read failure fired
        CHECK(!r.core->poisoned());           // a read failure: the device is fine, the cache re-established
        r.t->fail_read(0);
        r.verify();  // consistent: what moved before the failure holds its bytes
        ++io_failures;
      }
      r.core->make_resident(g, r.layer(g), {});
      reads += r.t->reads() - reads0;
      CHECK_EQ(r.core->stats().disk_drops, drops0);  // counted only in the resident fields
      for (uint32_t u = 0; u < 24; ++u) {
        const uint32_t k = r.core->key(g, u);
        CHECK(r.core->home(k) == DgCore::Home::Device);
        CHECK_EQ(r.t->slot_holds(r.core->slot_of(k)), int32_t(k));
      }
      for (uint32_t k : r.core->resident_victims()) {
        written_back += r.core->home(k) == DgCore::Home::Host;
        to_file += r.core->home(k) == DgCore::Home::File;
      }
      r.verify();
      const auto kept = r.keys_of(g, r.layer(g));
      std::vector<uint8_t> run = all_run(3);
      run[g] = 0;
      for (int tok = 0; tok < 4; ++tok) r.token(r.routes(3), run, kept, kept);  // other layers decode around it
      r.verify();
      for (int tok = 0; tok < 4; ++tok) r.token(r.routes(3), all_run(3), {});  // released
      r.verify();
      if (!r.v.what.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, r.v.what.c_str());
      CHECK(r.v.what.empty());
    }
  std::printf("  file reads %llu, victims written back %llu, victims dropped to their files %llu\n",
              (unsigned long long)reads, (unsigned long long)written_back, (unsigned long long)to_file);
  CHECK(reads > 0);
  CHECK(written_back > 0);
  CHECK_EQ(io_failures, 8ull);
  CHECK(to_file > 0);
}

// A clean copy (inclusive share) goes back to its block without a write-back.
TEST(dg_resident_clean_copy_victims) {
  ResRig r({16, 16, 16}, 20, 8, 12, false, nullptr, 0, 3, 8);
  for (uint32_t tok = 0; tok < 3; ++tok) {  // misses of layer 2, admitted with their clean copies
    std::vector<std::vector<uint32_t>> ids = {{0}, {0}, {3 * tok, 3 * tok + 1, 3 * tok + 2}};
    r.token(ids, {0, 0, 1}, {});
  }
  for (int tok = 0; tok < 2; ++tok) r.token({r.layer(0), {}, {}}, {1, 0, 0}, {});  // layer 0 the most recent
  CHECK(r.core->backed_units() > 0);
  std::vector<uint8_t> was_backed(r.keys, 0);
  for (uint32_t k = 0; k < r.keys; ++k) was_backed[k] = r.core->backed(k);
  r.core->make_resident(1, r.layer(1), {});
  uint32_t clean = 0;
  for (uint32_t k : r.core->resident_victims()) {
    CHECK(r.core->home(k) == DgCore::Home::Host);
    clean += was_backed[k];
  }
  CHECK(clean > 0);
  r.verify();
  CHECK(r.v.what.empty());
}

// rows_per_layer[h] == 0: the layer is not served (no lookups, misses, ready, recency), its score history is untouched
// (cells, row sums), the boundary still ranks it (no fallback), and its units age as unrouted ones.
TEST(dg_skip_layers_untouched) {
  for (bool scored : {false, true}) {
    ResRig r({16, 16, 16}, 20, 6, 6, scored);
    for (int tok = 0; tok < 5; ++tok) r.token(r.routes(4), all_run(3), {});
    const std::vector<ts::DgLayerCounts> lc0 = r.core->layer_counts();
    std::vector<uint32_t> cells0;
    std::vector<uint64_t> sums0;
    for (uint32_t a = 0; scored && a < 16; ++a) {
      sums0.push_back(r.core->score_rowsum(1, a));
      for (uint32_t b = 0; b < 16; ++b) cells0.push_back(r.core->score_cell(1, a, b));
    }
    const uint32_t ready0 = r.t->ready(1);
    const uint64_t fb0 = r.core->stats().score_fallbacks;
    for (int tok = 0; tok < 6; ++tok) r.token(r.routes(4), {1, 0, 1}, {});
    const auto& lc1 = r.core->layer_counts();
    CHECK_EQ(lc1[1].lookups, lc0[1].lookups);
    CHECK_EQ(lc1[1].misses, lc0[1].misses);
    CHECK_EQ(lc1[1].fetched, lc0[1].fetched);
    CHECK(lc1[0].lookups > lc0[0].lookups && lc1[2].lookups > lc0[2].lookups);
    CHECK_EQ(r.t->ready(1), ready0);
    for (uint32_t a = 0; scored && a < 16; ++a) {
      CHECK_EQ(r.core->score_rowsum(1, a), sums0[a]);
      for (uint32_t b = 0; b < 16; ++b) CHECK_EQ(r.core->score_cell(1, a, b), cells0[size_t(a) * 16 + b]);
    }
    CHECK_EQ(r.core->stats().score_fallbacks, fb0);
    // a step exporting nothing at all, then every layer again
    r.token(r.routes(1), {0, 0, 0}, {});
    for (int tok = 0; tok < 4; ++tok) r.token(r.routes(4), all_run(3), {});
    CHECK_EQ(r.core->stats().score_fallbacks, fb0);
    r.verify();
    CHECK(r.v.what.empty());
  }
}

// The randomized model with resident episodes and skipped layers on every shape: the device model reads exact bytes
// at every layer (held units in every layer), the oracles' victims at every boundary, held units never move.
TEST(dg_resident_runs) {
  g_res = ResidentCoverage{};
  std::vector<Shape> shapes;
  for (const Shape* sh : kShapes) shapes.push_back(*sh);
  for (const Shape* sh : kTierShapes) shapes.push_back(with_reads(*sh, 1));
  for (const Shape& sh : scored_shapes()) shapes.push_back(sh);
  int runs = 0;
  for (const Shape& base : shapes)
    for (int mode = 0; mode < 2; ++mode)
      for (uint64_t seed = 1; seed <= 3; ++seed) {
        const Shape sh = with_resident(base, mode == 0, mode == 0 ? 5 : 3);
        Outcome o = run(sh, 30, seed, DgCore::kNoMutation);
        ++runs;
        if (!o.violation.empty())
          std::printf("  shape %zu mode %d seed %llu: %s\n", size_t(&base - shapes.data()), mode,
                      (unsigned long long)seed, o.violation.c_str());
        CHECK(o.violation.empty());
      }
  std::printf("  %d runs: make_resident %llu (refused %llu, released %llu), units copied in %llu, victims %llu; "
              "layers skipped %llu, held units read %llu\n",
              runs, (unsigned long long)g_res.calls, (unsigned long long)g_res.refusals,
              (unsigned long long)g_res.released, (unsigned long long)g_res.copied, (unsigned long long)g_res.evicted,
              (unsigned long long)g_res.skipped_layers, (unsigned long long)g_res.held_reads);
  CHECK(g_res.calls > 0 && g_res.refusals > 0 && g_res.released > 0);
  CHECK(g_res.copied > 0 && g_res.evicted > 0 && g_res.skipped_layers > 0 && g_res.held_reads > 0);
}

// The boundary ignoring the keep list (a held unit evicted, its slot reused) is caught by the held reads.
TEST(dg_resident_ignored_keep_caught) {
  int caught = 0;
  for (uint64_t seed = 1; seed <= 6; ++seed)
    caught += !run(with_resident(kPressure, true, 0), 30, seed, DgCore::kIgnoreKeep).violation.empty();
  std::printf("  ignored keep caught in %d/6 runs\n", caught);
  CHECK(caught > 0);
}

// Device failures inside make_resident: an H2D copy, a write-back, an entry publication, an event record, an event or
// stream synchronize, the table upload, each injected at every call of its kind in turn (until the call has no more of
// them). Every injected failure must be observed (the call fails with SS_E_BACKEND only through it), the core is then
// poisoned (no token starts) and its host records stay one consistent ownership: no source block freed while a unit
// still names it as its home, no slot lost, no event lost (check()).
TEST(dg_resident_device_failures) {
  const char* names[Scripted::kFaultKinds] = {"h2d", "d2h", "entry", "record", "event_sync", "sync", "upload"};
  uint64_t fired[Scripted::kFaultKinds] = {};
  uint64_t attempts = 0;
  for (int tier = 0; tier < 2; ++tier)
    for (int k = 0; k < Scripted::kFaultKinds; ++k)
      for (int n = 1; n <= 500; ++n) {
        // two-tier: pinned sources and dirty victims; three-tier: file reads through staging events, drops to files
        std::vector<uint8_t> disk(72, 0);
        for (uint32_t key = 24 + 4; key < 72; ++key) disk[key] = 1;
        std::unique_ptr<ResRig> r = tier ? std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1u, false,
                                                                    &disk, 4, 5)
                                         : std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6, false,
                                                                    nullptr, 0, 5);
        if (tier) r->t->set_async_reads(true, 5);
        r->rng.seed(5);
        for (int tok = 0; tok < 6; ++tok) r->token(r->routes(3), all_run(3), {});
        r->verify();
        r->t->fail_kind(Scripted::FaultKind(k), n);
        bool threw = false;
        ss_status st = SS_OK;
        try {
          r->core->make_resident(2, r->layer(2), {});
        } catch (const ts::Error& e) {
          threw = true;
          st = e.status();
        }
        const bool f = r->t->kind_fired();
        r->t->fail_kind(Scripted::kFH2D, 0);
        if (!threw) {  // past the call's last operation of this kind (none at all: a kind the other tier covers)
          CHECK(!f);
          break;
        }
        ++attempts;
        CHECK(f);  // it failed through the injection only
        CHECK_EQ(int(st), int(SS_E_BACKEND));
        CHECK(r->core->poisoned());
        r->core->check();
        bool refused = false;
        try {
          r->core->begin_token(r->seq + 1);
        } catch (const ts::Error& e) {
          refused = e.status() == SS_E_BACKEND;
        }
        CHECK(refused);
        ++fired[k];
      }
  std::printf("  %llu injected device failures:", (unsigned long long)attempts);
  for (int k = 0; k < Scripted::kFaultKinds; ++k) std::printf(" %s %llu", names[k], (unsigned long long)fired[k]);
  std::printf("\n");
  for (int k = 0; k < Scripted::kFaultKinds; ++k) CHECK(fired[k] > 0);
}

// A failed replacement of g's set (a file read failing after evictions under slot pressure) leaves g's current set as
// it was: its units were protected during the call (the context passes every keep, g's own included), so the next step,
// which does not export g, reads every one of them in its slot. The capacity counts that set too.
TEST(dg_resident_failed_replacement_keeps_old_set) {
  uint64_t evicted = 0;
  for (uint64_t seed = 1; seed <= 6; ++seed) {
    std::vector<uint8_t> disk(72, 0);
    for (uint32_t k = 24 + 4; k < 72; ++k) disk[k] = 1;
    ResRig r({24, 24, 24}, 24, 4, 1, seed % 2 == 0, &disk, 4, seed);
    r.t->set_async_reads(seed % 3 != 0, seed);
    r.rng.seed(seed);
    for (int tok = 0; tok < 4; ++tok) r.token(r.routes(3), all_run(3), {});
    std::vector<uint32_t> a, b;
    for (uint32_t u = 0; u < 12; ++u) a.push_back(u), b.push_back(u + 12);
    r.core->make_resident(1, a, {});
    const auto ka = r.keys_of(1, a);
    std::vector<int32_t> slots;
    for (uint32_t k : ka) slots.push_back(r.core->slot_of(k));
    for (int tok = 0; tok < 6; ++tok) r.token(r.routes(3), {1, 0, 1}, ka, ka);  // held: g's set ages under decode
    // the capacity counts g's current set: b (12) + a (12, protected) = 24 = 28 - 4 fits; with one more kept unit of
    // another layer it does not: refused, nothing changed
    {
      std::vector<uint32_t> other = ka;
      other.push_back(r.core->key(0, 0));
      const auto st0 = r.state();
      r.t->clear_issued();
      bool cap = false;
      try {
        r.core->make_resident(1, b, other);
      } catch (const ts::Error& e) {
        cap = e.status() == SS_E_CAPACITY;
      }
      CHECK(cap);
      CHECK(r.t->issued().empty());
      CHECK(r.state() == st0);
    }
    // the failing replacement: b (file units) with a protected; a read fails after the evictions began
    const uint64_t ev0 = r.core->stats().resident_evicted;
    r.t->fail_read(3);
    bool io = false;
    try {
      r.core->make_resident(1, b, ka);
    } catch (const ts::Error& e) {
      io = e.status() == SS_E_IO;
    }
    CHECK(io);
    CHECK(!r.t->read_failure_pending());
    CHECK(!r.core->poisoned());
    r.t->fail_read(0);
    evicted += r.core->stats().resident_evicted - ev0;
    for (size_t i = 0; i < ka.size(); ++i) {
      CHECK(r.core->home(ka[i]) == DgCore::Home::Device);
      CHECK_EQ(r.core->slot_of(ka[i]), slots[i]);
      CHECK_EQ(r.t->entry(ka[i]), slots[i]);
    }
    r.token(r.routes(3), {1, 0, 1}, ka, ka);  // at once: g not exported, its old set read through the table
    r.verify();
    if (!r.v.what.empty()) std::printf("  seed %llu: %s\n", (unsigned long long)seed, r.v.what.c_str());
    CHECK(r.v.what.empty());
  }
  std::printf("  victims evicted before the failed reads: %llu\n", (unsigned long long)evicted);
  CHECK(evicted > 0);
}

// ---- placement rounds (ss_dg_round) and no-admission misses (dg_no_admit) ----
namespace {

ts::DgRoundIn round_in(uint32_t max_units = 96, uint64_t budget = 0, double hysteresis = 1.25, double decay = 0.85) {
  ts::DgRoundIn in;
  in.max_units = max_units;
  in.budget_bytes = budget;
  in.hysteresis = hysteresis;
  in.decay = decay;
  return in;
}

void seed_counts(DgCore& core, const std::vector<std::pair<uint32_t, float>>& kv) {
  std::vector<uint32_t> keys;
  std::vector<float> counts;
  for (const auto& p : kv) keys.push_back(p.first), counts.push_back(p.second);
  core.seed_round_counts(keys, counts);
}

// Every key's placement: home, slot, block.
std::vector<int64_t> homes(const DgCore& core) {
  std::vector<int64_t> s;
  for (uint32_t k = 0; k < core.keys(); ++k)
    s.insert(s.end(), {int64_t(core.home(k)), core.slot_of(k), core.block_of(k)});
  return s;
}

// Two layers of 16: keys 0..7 on the device (key 0 most recent), 4 spares, 4 spare blocks; counts: device keys 0, 1
// at 9 and 2..7 at 1, pinned keys 20..27 at 10.
std::unique_ptr<ResRig> seeded_rig(const std::function<void(DgConfig&)>& tweak = nullptr, uint32_t inclusive = 0) {
  auto r = std::make_unique<ResRig>(std::vector<uint32_t>{16, 16}, 8, 4, inclusive ? inclusive + 4 : 4, false, nullptr,
                                    0, 7, inclusive, tweak);
  std::vector<std::pair<uint32_t, float>> kv = {{0, 9.f}, {1, 9.f}};
  for (uint32_t k = 2; k < 8; ++k) kv.push_back({k, 1.f});
  for (uint32_t k = 20; k < 28; ++k) kv.push_back({k, 10.f});
  seed_counts(*r->core, kv);
  return r;
}

// Routes with a hot set: per gated layer `n_hot` ids from hot[g], then `n_cold` random ones.
std::vector<std::vector<uint32_t>> skewed(ResRig& r, const std::vector<std::vector<uint32_t>>& hot, uint32_t n_hot,
                                          uint32_t n_cold) {
  std::vector<std::vector<uint32_t>> ids(r.G);
  for (uint32_t g = 0; g < r.G; ++g) {
    for (uint32_t i = 0; i < n_hot; ++i) ids[g].push_back(hot[g][r.rng() % hot[g].size()]);
    for (uint32_t i = 0; i < n_cold; ++i) ids[g].push_back(uint32_t(r.rng() % r.core->units(g)));
  }
  return ids;
}

std::vector<uint8_t> disk_flags(uint32_t keys, uint32_t pinned_from, uint32_t file_from) {
  std::vector<uint8_t> d(keys, 0);
  for (uint32_t k = std::max(pinned_from, file_from); k < keys; ++k) d[k] = 1;
  return d;
}

}  // namespace

// The rank: the hottest pinned units replace the coldest device units when their count is above hysteresis x theirs
// (the device units at 9 stay against candidates at 10: within 1.25x), every moved unit's bytes in its new home, the
// table published; decay alone (no exports) moves nothing; max_units and budget_bytes bound the moves; kept units
// never move; a device below its share takes the hottest into its free slots first.
TEST(dg_round_rank_hysteresis_limits) {
  {
    auto r = seeded_rig();
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    CHECK_EQ(res.promoted_to_device, 6u);
    CHECK_EQ(res.demoted_from_device, 6u);
    CHECK_EQ(res.promoted_to_pinned + res.dropped_to_file + res.cancelled, 0u);
    CHECK_EQ(res.bytes_h2d, 6u * 8192u);
    CHECK_EQ(res.bytes_written_back, 6u * 8192u);
    CHECK_EQ(res.bytes_read, 0u);
    for (uint32_t k = 20; k < 26; ++k) CHECK(r->core->home(k) == DgCore::Home::Device);
    for (uint32_t k = 26; k < 28; ++k) CHECK(r->core->home(k) == DgCore::Home::Host);
    for (uint32_t k = 0; k < 2; ++k) CHECK(r->core->home(k) == DgCore::Home::Device);
    for (uint32_t k = 2; k < 8; ++k) CHECK(r->core->home(k) == DgCore::Home::Host);
    for (uint32_t k = 0; k < r->keys; ++k) CHECK_EQ(r->t->entry(k), r->core->table()[k]);  // uploaded at return
    r->verify();
    const ts::DgStats& s = r->core->stats();
    CHECK_EQ(s.round_count, 1u);
    CHECK_EQ(s.round_units, 12u);
    CHECK_EQ(s.round_bytes, 12u * 8192u);
    CHECK_EQ(s.evicted + s.fetched + s.writebacks_done + s.resident_units, 0u);  // counted only as a round
    // no exports since: every count decays alike, nothing churns
    const auto before = homes(*r->core);
    const ts::DgRoundOut res2 = r->core->round(round_in(), {});
    CHECK_EQ(res2.promoted_to_device + res2.demoted_from_device, 0u);
    CHECK(homes(*r->core) == before);
    CHECK(std::fabs(r->core->round_count(20) - 10.0 * 0.85 * 0.85) < 1e-9);
    CHECK(r->v.what.empty());
  }
  {  // max_units: the two hottest (ties by key) replace the two least recent of the coldest
    auto r = seeded_rig();
    const ts::DgRoundOut res = r->core->round(round_in(2), {});
    CHECK_EQ(res.promoted_to_device, 2u);
    CHECK(r->core->home(20) == DgCore::Home::Device && r->core->home(21) == DgCore::Home::Device);
    CHECK(r->core->home(22) == DgCore::Home::Host);
    CHECK(r->core->home(7) == DgCore::Home::Host && r->core->home(6) == DgCore::Home::Host);
    CHECK(r->core->home(5) == DgCore::Home::Device);
    r->verify();
  }
  {  // budget: a swap costs its H2D + its write-back (16 KiB); 40000 bytes take two
    auto r = seeded_rig();
    const ts::DgRoundOut res = r->core->round(round_in(96, 40000), {});
    CHECK_EQ(res.promoted_to_device, 2u);
    CHECK(res.bytes_h2d + res.bytes_written_back <= 40000u);
    r->verify();
  }
  {  // kept units never move (a cold device unit, a hot pinned one); the kept count leaves the device's share
    auto r = seeded_rig();
    const int32_t slot7 = r->core->slot_of(7), block20 = r->core->block_of(20);
    const ts::DgRoundOut res = r->core->round(round_in(), {7, 20});
    CHECK_EQ(res.promoted_to_device, 5u);
    CHECK_EQ(r->core->slot_of(7), slot7);
    CHECK(r->core->home(20) == DgCore::Home::Host && r->core->block_of(20) == block20);
    r->verify();
  }
  {  // a device below its share (spare target 4 -> 2): two free slots first, then the swaps
    auto r = seeded_rig();
    r->core->set_spare_target(2);
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    CHECK_EQ(res.promoted_to_device, 8u);
    CHECK_EQ(res.demoted_from_device, 6u);
    CHECK_EQ(r->core->free_slots(), 2u);
    r->verify();
  }
  {  // clean copies (inclusive): a demoted device unit goes back to its block, no write
    auto r = seeded_rig(nullptr, 8);
    for (uint32_t tok = 0; tok < 3; ++tok) {  // misses of layer 1 admitted with their clean copies
      std::vector<std::vector<uint32_t>> ids = {{0}, {8 + tok}};
      r->token(ids, {0, 1}, {});
    }
    uint32_t backed = 0;
    for (uint32_t k = 0; k < r->keys; ++k) backed += r->core->backed(k);
    CHECK(backed > 0);
    // the backed units coldest (their export since: 1), the other units of the open placement warmer (5 -> 4.25)
    for (uint32_t k = 0; k < 8; ++k) seed_counts(*r->core, {{k, 5.f}});
    for (uint32_t k = 24; k < 27; ++k) seed_counts(*r->core, {{k, 0.f}});
    const uint64_t wb0 = r->core->stats().round_bytes;
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    CHECK(res.demoted_from_device > 0);
    CHECK(res.bytes_written_back < uint64_t(res.demoted_from_device) * 8192u);  // the clean ones wrote nothing
    CHECK_EQ(r->core->stats().round_bytes - wb0, res.bytes_h2d + res.bytes_written_back);
    r->verify();
    CHECK(r->v.what.empty());
  }
}

// Three-tier: (a) pinned -> device with write-backs into the free block and then the promoted units' blocks; (b) the
// hottest file units read into the blocks of the coldest pinned units (dropped to their files), the reads overlapped.
TEST(dg_round_three_tier) {
  for (int async = 0; async < 2; ++async) {
    const auto disk = disk_flags(72, 24, 28);
    ResRig r({24, 24, 24}, 24, 4, 1, false, &disk, 4, 3);
    r.t->set_async_reads(async != 0, 3);
    std::vector<std::pair<uint32_t, float>> kv;
    for (uint32_t k = 0; k < 24; ++k) kv.push_back({k, 1.f});
    kv.push_back({24, 15.f});
    kv.push_back({25, 15.f});
    for (uint32_t k = 40; k < 44; ++k) kv.push_back({k, 20.f});
    seed_counts(*r.core, kv);
    const uint64_t reads0 = r.t->reads();
    const ts::DgRoundOut res = r.core->round(round_in(), {});
    CHECK_EQ(res.promoted_to_device, 2u);
    CHECK_EQ(res.demoted_from_device, 2u);
    CHECK_EQ(res.promoted_to_pinned, 4u);
    CHECK_EQ(res.dropped_to_file, 4u);  // 26, 27 (count 0), then the two demoted units (count 1)
    CHECK_EQ(res.bytes_read, 4u * 8192u);
    CHECK_EQ(r.t->reads() - reads0, 4u);
    CHECK(r.core->home(24) == DgCore::Home::Device && r.core->home(25) == DgCore::Home::Device);
    CHECK(r.core->home(26) == DgCore::Home::File && r.core->home(27) == DgCore::Home::File);
    for (uint32_t k = 40; k < 44; ++k) CHECK(r.core->home(k) == DgCore::Home::Host);
    CHECK_EQ(r.core->stats().disk_reads, 0u);  // counted only as a round
    r.verify();
    // the next round promotes the new pinned units to the device (their 20 against the device's 1)
    const ts::DgRoundOut res2 = r.core->round(round_in(), {});
    CHECK_EQ(res2.promoted_to_device, 4u);
    for (uint32_t k = 40; k < 44; ++k) CHECK(r.core->home(k) == DgCore::Home::Device);
    r.verify();
    for (int tok = 0; tok < 4; ++tok) r.token(r.routes(3), all_run(3), {});
    r.verify();
    CHECK(r.v.what.empty());
  }
}

// The oracle: a cache with rounds between its tokens (random counts, limits and keep lists; deferred, recorded and
// in-flight write-backs, both write-back streams, clean copies, three-tier with overlapped reads) reads exactly what a
// cache without rounds reads: only placement differs. Kept units never move in a round.
TEST(dg_round_outputs_equal_without_rounds) {
  uint64_t promoted = 0, demoted = 0, pinned = 0, rounds = 0;
  for (uint64_t seed = 1; seed <= 12; ++seed) {
    const bool three = seed % 2 == 0;
    const auto tweak = [seed](DgConfig& c) {
      c.wb_defer = seed % 3 != 0;
      c.wb_on_d2h = seed % 4 < 2;
    };
    const auto disk = disk_flags(72, 24, 28);
    const uint32_t incl = !three && seed % 3 == 1 ? 3u : 0u;
    std::unique_ptr<ResRig> a, b;
    if (three) {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 2, seed % 5 == 0, &disk, 4, seed, 0u, tweak);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 2, seed % 5 == 0, &disk, 4, seed, 0u, tweak);
      a->t->set_async_reads(seed % 3 != 1, seed);
      b->t->set_async_reads(seed % 3 != 1, seed);
    } else {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6 + incl, seed % 5 == 0, nullptr, 0, seed,
                                   incl, tweak);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6 + incl, seed % 5 == 0, nullptr, 0, seed,
                                   incl, tweak);
    }
    std::vector<std::vector<uint32_t>> hot(3);
    for (uint32_t g = 0; g < 3; ++g)
      for (uint32_t i = 0; i < 6; ++i) hot[g].push_back(uint32_t((7 * i + 5 * g + seed) % 24));
    std::mt19937_64 rr(seed * 31);
    for (int tok = 0; tok < 36; ++tok) {
      std::vector<uint32_t> keep;
      if (tok % 7 == 3) keep = {a->core->key(0, uint32_t(rr() % 24)), a->core->key(2, uint32_t(rr() % 24))};
      const auto ids = skewed(*a, hot, 3, 2);
      b->rng = a->rng;
      const auto x = a->token(ids, all_run(3), keep), y = b->token(ids, all_run(3), keep);
      CHECK(x == y);
      if (tok % 3 == 2) {
        if (tok % 9 == 2) {  // random counts: forces moves the exports alone would not
          for (uint32_t k = 0; k < a->keys; ++k)
            if (rr() % 3 == 0) seed_counts(*a->core, {{k, float(rr() % 40)}});
        }
        std::vector<int64_t> before;
        for (uint32_t k : keep) before.insert(before.end(), {int64_t(a->core->home(k)), a->core->slot_of(k), a->core->block_of(k)});
        const ts::DgRoundOut res =
            a->core->round(round_in(uint32_t(rr() % 12), rr() % 2 ? 0 : 8192u * (1 + rr() % 10),
                                    1.0 + double(rr() % 4) / 4, double(rr() % 5) / 4), keep);
        std::vector<int64_t> after;
        for (uint32_t k : keep) after.insert(after.end(), {int64_t(a->core->home(k)), a->core->slot_of(k), a->core->block_of(k)});
        CHECK(before == after);
        promoted += res.promoted_to_device;
        demoted += res.demoted_from_device;
        pinned += res.promoted_to_pinned;
        ++rounds;
        a->verify();
      }
    }
    a->verify();
    b->verify();
    if (!a->v.what.empty() || !b->v.what.empty())
      std::printf("  seed %llu: %s / %s\n", (unsigned long long)seed, a->v.what.c_str(), b->v.what.c_str());
    CHECK(a->v.what.empty() && b->v.what.empty());
  }
  std::printf("  %llu rounds: promoted to the device %llu, demoted %llu, file -> pinned %llu\n",
              (unsigned long long)rounds, (unsigned long long)promoted, (unsigned long long)demoted,
              (unsigned long long)pinned);
  CHECK(promoted > 0 && demoted > 0 && pinned > 0);
}

// Device failures inside a round (an H2D, a write-back, an entry, an event record, an event / stream synchronize, the
// upload; deferred write-backs left by the last boundary give the drain records and waits), each injected at every
// call of its kind in turn: observed, SS_E_BACKEND, the core poisoned, its host records one consistent ownership. A
// file read failure: SS_E_IO, consistent (verify), not poisoned, and the next round completes.
TEST(dg_round_failures) {
  const char* names[Scripted::kFaultKinds] = {"h2d", "d2h", "entry", "record", "event_sync", "sync", "upload"};
  uint64_t fired[Scripted::kFaultKinds] = {};
  uint64_t attempts = 0;
  const auto disk = disk_flags(72, 24, 28);
  auto make = [&](int tier) {
    const auto tweak = [](DgConfig& c) {
      c.wb_defer = true;
      c.wb_on_d2h = true;
    };
    std::unique_ptr<ResRig> r = tier ? std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1u, false,
                                                                &disk, 4, 5, 0u, tweak)
                                     : std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6, false,
                                                                nullptr, 0, 5, 0u, tweak);
    if (tier) r->t->set_async_reads(true, 5);
    r->rng.seed(5);
    for (int tok = 0; tok < 6; ++tok) r->token(r->routes(3), all_run(3), {});
    for (uint32_t k = 0; k < r->keys; ++k)  // the units off the device hot, the device cold
      seed_counts(*r->core, {{k, r->core->home(k) == DgCore::Home::Device ? 1.f : float(10 + k % 7)}});
    return r;
  };
  for (int tier = 0; tier < 2; ++tier)
    for (int k = 0; k < Scripted::kFaultKinds; ++k)
      for (int n = 1; n <= 500; ++n) {
        auto r = make(tier);
        r->t->fail_kind(Scripted::FaultKind(k), n);
        bool threw = false;
        ss_status st = SS_OK;
        try {
          r->core->round(round_in(), {});
        } catch (const ts::Error& e) {
          threw = true;
          st = e.status();
        }
        const bool f = r->t->kind_fired();
        r->t->fail_kind(Scripted::kFH2D, 0);
        if (!threw) {
          CHECK(!f);
          break;
        }
        ++attempts;
        CHECK(f);
        CHECK_EQ(int(st), int(SS_E_BACKEND));
        CHECK(r->core->poisoned());
        r->core->check();
        bool refused = false;
        try {
          r->core->begin_token(r->seq + 1);
        } catch (const ts::Error& e) {
          refused = e.status() == SS_E_BACKEND;
        }
        CHECK(refused);
        ++fired[k];
      }
  std::printf("  %llu injected device failures:", (unsigned long long)attempts);
  for (int k = 0; k < Scripted::kFaultKinds; ++k) std::printf(" %s %llu", names[k], (unsigned long long)fired[k]);
  std::printf("\n");
  for (int k = 0; k < Scripted::kFaultKinds; ++k) CHECK(fired[k] > 0);
  // file read failures: the first read, a later one with others in flight
  uint64_t io = 0;
  for (int which = 1; which <= 3; ++which) {
    auto r = make(1);
    r->t->set_async_reads(which != 1, 9);
    r->t->fail_read(which);
    bool got = false;
    try {
      r->core->round(round_in(), {});
    } catch (const ts::Error& e) {
      got = e.status() == SS_E_IO;
    }
    CHECK(got);
    CHECK(!r->t->read_failure_pending());
    CHECK(!r->core->poisoned());
    r->t->fail_read(0);
    r->verify();
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    CHECK(res.promoted_to_pinned > 0 || res.promoted_to_device > 0);
    r->verify();
    for (int tok = 0; tok < 3; ++tok) r->token(r->routes(3), all_run(3), {});
    r->verify();
    CHECK(r->v.what.empty());
    io += got;
  }
  CHECK_EQ(io, 3u);
}

// ss_dg_round_cancel from another thread: at a unit boundary (the round stops there), during a copy (the copy is
// dropped: its source keeps its home, its entry is -1 again), during the file reads (the reads in flight are ended and
// dropped). Every case: cancelled = 1, consistent, nothing half moved; a later round completes.
TEST(dg_round_cancel_from_thread) {
  auto cancel_now = [](DgCore* core) {
    std::thread th([core] { core->cancel_round(); });
    th.join();
  };
  {  // at a unit boundary
    auto r = seeded_rig();
    DgCore* core = r->core.get();
    r->core->set_round_probe([&](uint32_t moved) {
      if (moved == 2) cancel_now(core);
    });
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    CHECK_EQ(res.cancelled, 1u);
    CHECK_EQ(res.promoted_to_device, 2u);
    CHECK_EQ(res.demoted_from_device, 2u);
    CHECK_EQ(r->core->stats().round_cancelled, 1u);
    r->verify();
    r->core->set_round_probe(nullptr);
    const ts::DgRoundOut res2 = r->core->round(round_in(), {});
    CHECK_EQ(res2.cancelled, 0u);
    CHECK_EQ(res2.promoted_to_device, 4u);
    for (uint32_t k = 20; k < 26; ++k) CHECK(r->core->home(k) == DgCore::Home::Device);
    r->verify();
  }
  {  // during the third promotion's copy
    auto r = seeded_rig();
    DgCore* core = r->core.get();
    int h2d = 0;
    r->t->set_on_h2d([&] {
      if (++h2d == 3) cancel_now(core);
    });
    const int32_t block22 = r->core->block_of(22);
    const uint32_t free0 = r->core->free_slots();
    const ts::DgRoundOut res = r->core->round(round_in(), {});
    r->t->set_on_h2d(nullptr);
    CHECK_EQ(res.cancelled, 1u);
    CHECK_EQ(res.promoted_to_device, 2u);
    CHECK_EQ(res.demoted_from_device, 3u);  // the third victim's write-back was complete: its slot is free now
    CHECK(r->core->home(22) == DgCore::Home::Host && r->core->block_of(22) == block22);
    CHECK_EQ(r->t->entry(22), -1);
    CHECK_EQ(r->core->free_slots(), free0 + 1);
    r->verify();
    const ts::DgRoundOut res2 = r->core->round(round_in(), {});
    CHECK_EQ(res2.cancelled, 0u);
    CHECK(r->core->home(22) == DgCore::Home::Device);
    r->verify();
    CHECK(r->v.what.empty());
  }
  for (int async = 0; async < 2; ++async) {  // three-tier, during the file reads
    const auto disk = disk_flags(72, 24, 28);
    ResRig r({24, 24, 24}, 24, 4, 1, false, &disk, 4, 3);
    r.t->set_async_reads(async != 0, 3);
    std::vector<std::pair<uint32_t, float>> kv;
    for (uint32_t k = 40; k < 46; ++k) kv.push_back({k, 20.f});
    seed_counts(*r.core, kv);
    DgCore* core = r.core.get();
    r.core->set_round_probe([&](uint32_t moved) {
      if (moved == 3) cancel_now(core);
    });
    const ts::DgRoundOut res = r.core->round(round_in(), {});
    CHECK_EQ(res.cancelled, 1u);
    CHECK(res.promoted_to_pinned <= 3u);
    CHECK_EQ(res.dropped_to_file, 3u);
    r.verify();
    for (uint32_t k = 40; k < 46; ++k)
      CHECK(r.core->home(k) == DgCore::Home::File || r.core->home(k) == DgCore::Home::Host);
    r.core->set_round_probe(nullptr);
    const ts::DgRoundOut res2 = r.core->round(round_in(), {});
    CHECK_EQ(res2.cancelled, 0u);
    CHECK(res2.promoted_to_pinned > 0);
    r.verify();
    for (int tok = 0; tok < 3; ++tok) r.token(r.routes(3), all_run(3), {});
    r.verify();
    CHECK(r.v.what.empty());
  }
}

// The counts: every export of a unit since the last round (unique ids of an exported layer, not of an unexported one),
// s = decay x s + hits; seeded counts drive the first round's ranking; bad seeds change nothing.
TEST(dg_round_counts_and_seeds) {
  ResRig r({16, 16}, 8, 4, 4);
  r.token({{9, 9, 10}, {3}}, {1, 1}, {});
  r.token({{9}, {4}}, {1, 0}, {});  // layer 1 not exported
  CHECK_EQ(r.core->round_hits(9), 2u);
  CHECK_EQ(r.core->round_hits(10), 1u);
  CHECK_EQ(r.core->round_hits(16 + 3), 1u);
  CHECK_EQ(r.core->round_hits(16 + 4), 0u);
  r.core->round(round_in(0), {});  // max_units 0: only the counts
  CHECK(r.core->round_count(9) == 2.0 && r.core->round_count(10) == 1.0);
  CHECK_EQ(r.core->round_hits(9), 0u);
  r.token({{9}, {3}}, {1, 1}, {});
  r.core->round(round_in(0, 0, 1.25, 0.5), {});
  CHECK(r.core->round_count(9) == 2.0);  // 0.5 x 2 + 1
  // seeds: two opposite profiles promote opposite ends of the pinned units
  for (int dir = 0; dir < 2; ++dir) {
    ResRig q({16, 16}, 8, 4, 4);
    std::vector<std::pair<uint32_t, float>> kv;
    for (uint32_t k = 8; k < 32; ++k) kv.push_back({k, float(dir ? k : 40 - k)});
    seed_counts(*q.core, kv);
    q.core->round(round_in(3), {});
    for (uint32_t i = 0; i < 3; ++i) CHECK(q.core->home(dir ? 31 - i : 8 + i) == DgCore::Home::Device);
    q.verify();
  }
  const double c9 = r.core->round_count(9);
  for (const auto& bad : std::vector<std::pair<uint32_t, float>>{{9, -1.f}, {9, NAN}, {999, 1.f}}) {
    bool refused = false;
    try {
      seed_counts(*r.core, {{10, 5.f}, bad});
    } catch (const ts::Error& e) {
      refused = e.status() == SS_E_INVALID;
    }
    CHECK(refused);
    CHECK(r.core->round_count(9) == c9 && r.core->round_count(10) != 5.0);  // nothing set
  }
  bool refused = false;
  try {
    ts::DgRoundIn in = round_in();
    in.decay = 1.5;
    r.core->round(in, {});
  } catch (const ts::Error& e) {
    refused = e.status() == SS_E_INVALID;
  }
  CHECK(refused);
  CHECK(r.v.what.empty());
}

// dg_no_admit: every demand miss is served through a bank slot (the bytes the cache without it reads), released at
// the boundary with every home unchanged (no admission, no prefetch, no eviction); too few bank slots are refused at
// construction; with rounds the placement converges to the hot set and the misses fall.
TEST(dg_no_admit_bank_serves) {
  const auto no_admit = [](DgConfig& c) { c.no_admit = true; };
  for (int three = 0; three < 2; ++three) {
    const auto disk = disk_flags(72, 24, 28);
    std::unique_ptr<ResRig> a, b;
    if (three) {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1, false, &disk, 4, 2, 0u, no_admit);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1, false, &disk, 4, 2);
      a->t->set_async_reads(true, 2);
      b->t->set_async_reads(true, 2);
    } else {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6, false, nullptr, 0, 2, 0u, no_admit);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 30, 6, 6, false, nullptr, 0, 2);
    }
    const auto open = homes(*a->core);
    const uint64_t reads0 = a->t->reads();
    for (int tok = 0; tok < 16; ++tok) {
      const auto ids = a->routes(6);
      b->rng = a->rng;
      CHECK(a->token(ids, all_run(3), {}) == b->token(ids, all_run(3), {}));
      CHECK(homes(*a->core) == open);  // (a pinned home's recency order may change, never a home)
    }
    const ts::DgStats& s = a->core->stats();
    CHECK(s.misses > 0);
    CHECK_EQ(s.bank_serves, s.fetched);
    CHECK_EQ(s.prefetched + s.evicted + s.resurrected, 0u);
    if (three) CHECK(a->t->reads() > reads0);  // file units served through staging -> bank
    // a prefetch does nothing
    a->core->begin_token(++a->seq);
    CHECK(!a->core->fetch(0, 23, true));
    a->core->end_token({});
    a->verify();
    b->verify();
    CHECK(a->v.what.empty() && b->v.what.empty());
  }
  {  // refused: fewer bank slots per bank than a layer's distinct misses
    bool refused = false;
    std::string what;
    try {
      ResRig r({16, 16}, 8, 4, 4, false, nullptr, 0, 7, 0u, [](DgConfig& c) {
        c.no_admit = true;
        c.bank_slots = 4;
      });
    } catch (const ts::Error& e) {
      refused = e.status() == SS_E_CAPACITY;
      what = e.what();
    }
    CHECK(refused);
    CHECK(what.find("needs 8 bank slots") != std::string::npos);
  }
  // with rounds: the hot set moves onto the device (three-tier: file -> pinned -> device) and the misses fall
  for (int three = 0; three < 2; ++three) {
    const auto disk = disk_flags(72, 24, 28);
    std::unique_ptr<ResRig> a, b;
    if (three) {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1, false, &disk, 4, 4, 0u, no_admit);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 24, 4, 1, false, &disk, 4, 4);
    } else {
      a = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 12, 4, 4, false, nullptr, 0, 4, 0u, no_admit);
      b = std::make_unique<ResRig>(std::vector<uint32_t>{24, 24, 24}, 12, 4, 4, false, nullptr, 0, 4);
    }
    std::vector<std::vector<uint32_t>> hot(3);  // four units per layer, none on the device at open
    for (uint32_t g = 0; g < 3; ++g)
      for (uint32_t i = 0; i < 4; ++i) hot[g].push_back(20 - i);
    uint64_t early = 0, late = 0;
    for (int tok = 0; tok < 40; ++tok) {
      const auto ids = skewed(*a, hot, 4, tok % 4 == 0 ? 1 : 0);
      b->rng = a->rng;
      const uint64_t m0 = a->core->stats().misses;
      CHECK(a->token(ids, all_run(3), {}) == b->token(ids, all_run(3), {}));
      const uint64_t m = a->core->stats().misses - m0;
      if (tok < 4) early += m;
      if (tok >= 32 && tok % 4 != 0) late += m;
      if (tok % 2 == 1) a->core->round(round_in(), {});
      a->verify();
    }
    for (uint32_t g = 0; g < 3; ++g)
      for (uint32_t u : hot[g]) CHECK(a->core->home(a->core->key(g, u)) == DgCore::Home::Device);
    std::printf("  %s: misses in the first 4 tokens %llu, in the hot-only tokens of the last 8 %llu\n",
                three ? "three-tier" : "two-tier", (unsigned long long)early, (unsigned long long)late);
    CHECK(early > 0);
    CHECK_EQ(late, 0u);
    b->verify();
    CHECK(a->v.what.empty() && b->v.what.empty());
  }
}
