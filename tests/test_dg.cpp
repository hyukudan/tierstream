// Device-gated cache core (src/dg_core.cpp) against a scripted device: every copy, table publication, ready write,
// event record / wait and write-back is a pending operation of its stream, and a randomized scheduler completes them
// in any order the stream and event rules allow. A device model plays the engine's graph (export -> wait -> read
// -> next layer, the end edge, the boundary) and checks, at every read, that the slot it resolves holds the unit's
// bytes and that no copy lands in a slot while a layer reads it. check() runs after every service step. Each
// mutation of the core must be caught.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "check.hpp"
#include "dg_core.hpp"
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
  enum Kind { H2D, D2H, Entry, Ready, Record, Wait, Upload, D2HPart, D2D, H2H };
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
  void h2d(Stream s, int32_t slot, int32_t block) override { if (move_faults_) fault("h2d"); push(s, {H2D, slot, block}); }
  void d2d(int32_t src, int32_t dst) override { fault("d2d"); push(kDem, {D2D, src, dst}); sync(kDem); }
  void h2h(int32_t src, int32_t dst) override { fault("h2h"); push(kDem, {H2H, src, dst}); sync(kDem); }
  void d2h(int32_t slot, int32_t block) override { if (move_faults_) fault("d2h"); push(kD2h, {D2H, slot, block}); }
  void d2h_part(Stream s, int32_t slot, int32_t block, uint32_t part) override {
    if (part >= parts_) v_->set("a write-back part out of range");
    fault("d2h_part");
    Op o{D2HPart, slot, block};
    o.key = part;
    push(s, std::move(o));
  }
  uint32_t parts() const override { return parts_; }
  // unequal parts, as the real gate / up / down (the last one larger)
  uint64_t part_bytes(uint32_t part) const override { return part + 1 == parts_ ? 4096 : 2048; }
  void write_entry(Stream s, uint32_t key, int32_t v) override {
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
    while (!done(ev))
      if (!step()) {
        v_->set("event_sync: deadlock");
        return;
      }
  }
  void sync(Stream s) override {
    while (!q_[s].empty())
      if (!step()) {
        v_->set("sync: deadlock");
        return;
      }
  }
  void upload_table(const int32_t* table, uint32_t n) override {
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
    int cand[3], n = 0;
    for (int s = 0; s < 3; ++s)
      if (runnable(s)) cand[n++] = s;
    if (!n) return false;
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

 private:
  void fault(const char* what) {
    if (fail_at_ > 0 && --fail_at_ == 0) ts::fail(SS_E_BACKEND, std::string("injected failure: ") + what);
  }
  void push(Stream s, Op o) {
    issued_.push_back(o.kind);
    trace_.push_back({int64_t(s), int64_t(o.kind), o.a, o.b, int64_t(o.key)});
    for (int32_t value : o.table) trace_.back().push_back(value);
    if (o.kind == D2H || o.kind == D2HPart) wb_slots_.push_back(o.a);
    q_[s].push_back(std::move(o));
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
};

struct Outcome {
  std::string violation;
  ts::DgStats st;
  bool threw = false;
};

std::vector<uint64_t> counters(const ts::DgStats& s) { return {s.tokens, s.lookups, s.misses, s.covered, s.fetched, s.prefetched, s.prefetch_used, s.resurrected, s.bank_serves, s.collected_mid, s.evicted, s.evict_short, s.writebacks_done, s.stalled, s.pred_invalid, s.covered_waits, s.predicted, s.predicted_true, s.miss_layers, s.wb_deferred, s.wb_cancelled, s.wb_forced, s.wb_boundary_flushed, s.wb_issued_bytes, s.wb_completed_bytes, s.wb_cancelled_bytes, s.wb_boundary_flushed_bytes, s.winpf_hints, s.winpf_hint_rows, s.prior_issued, s.prior_used, s.prior_late, s.prior_reverted, s.l1_candidates, s.l1_issued, s.l1_used, s.l1_late, s.l1_reverted, s.l1_priced_out, s.l1_skipped_ahead, s.prior_budget_stops}; }

void verify_bytes(DgCore& core, Scripted& t) {
  core.check();
  for (uint32_t k = 0; k < core.keys(); ++k) {
    CHECK_EQ(t.entry(k), core.table()[k]);
    if (core.home(k) == DgCore::Home::Device) CHECK_EQ(t.slot_holds(core.slot_of(k)), int32_t(k));
    else {
      CHECK(core.home(k) == DgCore::Home::Host);
      CHECK_EQ(t.block_holds(core.block_of(k)), int32_t(k));
    }
  }
}

uint64_t reset_d2d = 0, reset_h2h = 0, reset_staging = 0;

void reset_compare(DgCore& core, Scripted& t, const DgConfig& c, const std::vector<uint32_t>& order,
                   Violation& v, uint32_t seq) {
  Violation fv;
  Scripted ft(core.keys(), c.persistent_slots + 2 * c.bank_slots, c.blocks, core.gated(), 901, &fv);
  DgCore fresh(c, &ft);
  fresh.place(order);
  fresh.set_prior(ts::DgPrior(core.prior()));
  for (uint32_t k = 0; k < fresh.keys(); ++k) {
    if (fresh.slot_of(k) >= 0) ft.fill_slot(fresh.slot_of(k), int32_t(k));
    else ft.fill_block(fresh.block_of(k), int32_t(k));
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
          op != Scripted::D2D && op != Scripted::H2H);

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
  c.blocks = keys - sh.residents + sh.host_spares;
  c.spare_target = sh.spare;
  c.prefetch_reserve = std::min(24u, sh.spare / 4);
  c.prefetch_per_layer = sh.pf_layer;
  c.prefetch_per_token = sh.pf_token;
  c.wb_defer = sh.defer;
  c.wb_on_d2h = sh.on_d2h;
  c.wb_layer_bytes = sh.wb_bytes;
  c.wb_max_outstanding = sh.wb_out;
  Scripted t(keys, c.persistent_slots + 2 * sh.bank, c.blocks, G, seed, &v);
  t.set_slow_d2h(sh.slow_d2h);
  try {
    DgCore core(c, &t);
    core.set_mutation(mutation);
    core.set_pressure(sh.pressure_free, sh.pressure_recent);
    std::vector<uint32_t> res;
    for (uint32_t k = 0; k < sh.residents; ++k) res.push_back((k * 7919u) % keys);
    core.place(res);
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
    for (int tok = 1; tok <= tokens && v.what.empty(); ++tok) {
      const uint32_t seq = uint32_t(tok);
      if (reset) {
        core.set_spare_target(uint32_t(rng() % (sh.spare + 1)));
        core.set_prefetch_budget(uint32_t(rng() % 7), uint32_t(rng() % 25));
        core.observe_copy(0.002);
      }
      if (sh.toggle_defer && rng() % 3 == 0) {
        core.set_wb_defer(!core.wb_defer());
        core.check();
      }
      core.begin_token(seq);
      std::vector<uint32_t> pred;
      for (uint32_t g = 0; g < G && v.what.empty(); ++g) {
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
        core.check();
        t.steps(jitter);
        // the prediction for g + 1 (from the graph), served after the demand
        pred.clear();
        if (sh.winpf && g + 1 < G) {  // rows x K ranked ids, the device ran ahead now and then
          for (uint32_t i = 0; i < win_rows * sh.predict_k; ++i) pred.push_back(pick(g + 1));
          core.serve_prediction(g + 1, pred.data(), uint32_t(pred.size()), n, rng() % 8 == 0);
          core.check();
        } else if (sh.predict && g + 1 < G) {
          for (uint32_t i = 0; i < sh.predict; ++i) pred.push_back(pick(g + 1));
          core.serve_prediction(g + 1, pred.data(), uint32_t(pred.size()));
          core.check();
        }
        // the deferred write-back slice, after the ready and the prefetches; cut short now and then (the device ran
        // ahead: the next export was already seen). Never called without a deferred record (no draw without defer).
        core.issue_writebacks(g, [&] { return rng() % 4 == 0; });
        core.check();
        // wait: only if a miss; ready must come
        if (mask)
          while (int32_t(t.ready(g) - seq) < 0)
            if (!t.step()) {
              v.set("ready of layer " + std::to_string(g) + " never came");
              break;
            }
        // read: every routed unit through the table; the slots stay read until the layer ends
        std::set<int32_t> reading;
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
      core.end_token({});
      {
        const auto& is = t.issued();
        const auto up = std::find(is.begin(), is.end(), Scripted::Upload);
        if (std::find(is.begin(), up, Scripted::D2H) != up || std::find(is.begin(), up, Scripted::D2HPart) != up)
          v.set("a write-back was issued before the boundary's table upload");
      }
      core.check();
      t.steps(jitter);
      // every home holds its unit's bytes (a write-back in flight: the slot is authoritative)
      for (uint32_t k = 0; k < keys && v.what.empty(); ++k) {
        const DgCore::Home h = core.home(k);
        const bool on_slot = h == DgCore::Home::Device || h == DgCore::Home::Evicting;
        if (h == DgCore::Home::Coming) continue;  // a prior prefetch in flight: its slot fills on kPf
        const int32_t holds = on_slot ? t.slot_holds(core.slot_of(k)) : t.block_holds(core.block_of(k));
        if (holds != int32_t(k)) v.set("unit " + std::to_string(k) + " lost its bytes at a boundary");
        if (t.entry(k) != core.table()[k] && h != DgCore::Home::Coming)
          v.set("device table != host mirror after the boundary");
      }
      if (sh.winpf) hint();  // the next window's prior prefetches, before its export reads the table
    }
    if (reset) reset_compare(core, t, c, res, v, uint32_t(tokens));
    t.drain_all();
    out.st = core.stats();
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
std::unique_ptr<Mini> deferred_four(uint64_t wb_bytes = 0, uint64_t wb_out = 0) {
  auto m = std::make_unique<Mini>(8, 12, 8, keys_in_order(8), 4, false, -1, true, wb_bytes, wb_out);
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

// ---- winpf: deterministic edges (review finding 6) ----
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
