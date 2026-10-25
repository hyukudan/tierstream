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
  enum Kind { H2D, D2H, Entry, Ready, Record, Wait, Upload };
  struct Op {
    Kind kind;
    int32_t a = 0, b = 0;
    uint32_t key = 0;
    Ev* ev = nullptr;
    uint64_t target = 0;
    std::vector<int32_t> table;
  };

  Scripted(uint32_t keys, uint32_t slots, uint32_t blocks, uint32_t gated, uint64_t seed, Violation* v)
      : table_(keys, -1), slot_(slots, -1), block_(blocks, -1), ready_(gated, 0), rng_(seed), v_(v) {}

  // ---- DgTransport ----
  void h2d(Stream s, int32_t slot, int32_t block) override { push(s, {H2D, slot, block}); }
  void d2h(int32_t slot, int32_t block) override { push(kD2h, {D2H, slot, block}); }
  void write_entry(Stream s, uint32_t key, int32_t v) override {
    Op o{Entry, v};
    o.key = key;
    push(s, std::move(o));
  }
  void signal_ready(uint32_t g, uint32_t seq) override { push(kDem, {Ready, int32_t(g), int32_t(seq)}); }
  void* event_create() override {
    evs_.push_back(std::make_unique<Ev>());
    return evs_.back().get();
  }
  void event_destroy(void*) override {}
  void record(void* ev, Stream s) override {
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
    Op o{Upload};
    o.table.assign(table, table + n);
    push(kDem, std::move(o));
    sync(kDem);
  }

  // ---- scheduler ----
  bool runnable(int s) const {
    if (q_[s].empty()) return false;
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
  int32_t block_holds(int32_t b) const { return block_[size_t(b)]; }
  void fill_slot(int32_t s, int32_t key) { slot_[size_t(s)] = key; }
  void fill_block(int32_t b, int32_t key) { block_[size_t(b)] = key; }
  void set_table(const std::vector<int32_t>& t) { table_ = t; }
  void set_slow_d2h(bool on) { slow_d2h_ = on; }
  void begin_read(const std::set<int32_t>& slots) { reading_ = slots; }
  // The kinds of the operations in the order they were issued (not run), since the last clear.
  const std::vector<Kind>& issued() const { return issued_; }
  void clear_issued() { issued_.clear(); }
  void end_read() { reading_.clear(); }
  std::mt19937_64& rng() { return rng_; }

 private:
  void push(Stream s, Op o) {
    issued_.push_back(o.kind);
    q_[s].push_back(std::move(o));
  }
  void run(const Op& o) {
    switch (o.kind) {
      case H2D:
        if (reading_.count(o.a)) v_->set("a copy landed in slot " + std::to_string(o.a) + " while a layer read it");
        slot_[size_t(o.a)] = block_[size_t(o.b)];
        break;
      case D2H:
        block_[size_t(o.b)] = slot_[size_t(o.a)];
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
  std::set<int32_t> reading_;
  std::vector<Kind> issued_;
  bool slow_d2h_ = false;
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
};

struct Outcome {
  std::string violation;
  ts::DgStats st;
  bool threw = false;
};

// Runs `tokens` tokens of random routes (a skewed popularity so the cache warms) through the core and the model.
Outcome run(const Shape& sh, int tokens, uint64_t seed, int mutation, unsigned jitter = 6) {
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
    for (int tok = 1; tok <= tokens && v.what.empty(); ++tok) {
      const uint32_t seq = uint32_t(tok);
      core.begin_token(seq);
      std::vector<uint32_t> pred;
      for (uint32_t g = 0; g < G && v.what.empty(); ++g) {
        // export: the ids, the miss bits from the published table; a hit's slot must already hold its bytes
        const uint32_t n = 1 + uint32_t(rng() % sh.max_ids);
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
        if (sh.predict && g + 1 < G) {
          for (uint32_t i = 0; i < sh.predict; ++i) pred.push_back(pick(g + 1));
          core.serve_prediction(g + 1, pred.data(), uint32_t(pred.size()));
          core.check();
        }
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
        if (std::find(is.begin(), up, Scripted::D2H) != up)
          v.set("a write-back was issued before the boundary's table upload");
      }
      core.check();
      t.steps(jitter);
      // every home holds its unit's bytes (a write-back in flight: the slot is authoritative)
      for (uint32_t k = 0; k < keys && v.what.empty(); ++k) {
        const DgCore::Home h = core.home(k);
        const bool on_slot = h == DgCore::Home::Device || h == DgCore::Home::Evicting;
        const int32_t holds = on_slot ? t.slot_holds(core.slot_of(k)) : t.block_holds(core.block_of(k));
        if (holds != int32_t(k)) v.set("unit " + std::to_string(k) + " lost its bytes at a boundary");
        if (t.entry(k) != core.table()[k]) v.set("device table != host mirror after the boundary");
      }
    }
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
const Shape* const kShapes[] = {&kWarm, &kPressure, &kPredict, &kPredictPressure, &kSlowWriteback};

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
  CHECK(sum.prefetch_used > 0);
  CHECK(sum.evicted > 0);
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
              {DgCore::kReevictPending, "re-evict with a pending write-back"}};
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
  Mini(uint32_t R, uint32_t D, uint32_t T, const std::vector<uint32_t>& order, uint32_t pf_layer = 4,
       bool reserve_auto = false) {
    c.n_units.assign(G, U);
    c.persistent_slots = R + D;
    c.bank_slots = 4;
    c.blocks = K - R + D;
    c.spare_target = T;
    c.prefetch_reserve_auto = reserve_auto;
    c.prefetch_reserve = reserve_auto ? 0 : 1;
    c.prefetch_per_layer = pf_layer;
    c.prefetch_per_token = 48;
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
    m.core->end_token({});
    m.core->begin_token(2);
    m.serve(1, {3});  // a new token: the old prediction is gone
    CHECK_EQ(m.core->stats().predicted_true, 1ull);
    m.core->end_token({});
    m.core->check();
    CHECK(m.v.what.empty());
  }
}

// The seed order picks the residents (the first persistent_slots - T distinct keys) and is the recency order: the
// first evictions are its last entries (cache.cpp:236-256).
TEST(dg_seed_order_residents_and_lru) {
  std::vector<uint32_t> order = {40, 3, 17, 40, 29, 8, 33, 12, 5};  // 40 twice: skipped the second time
  for (uint32_t k = 0; k < Mini::K; ++k) order.push_back(k);
  Mini m(8, 4, 4, order);
  const uint32_t want[8] = {40, 3, 17, 29, 8, 33, 12, 5};
  for (uint32_t k : want) CHECK(m.core->slot_of(k) >= 0);
  CHECK_EQ(m.core->on_device(), 8u);
  // target 7: the miss below takes one of the 4 spares (3 free left), so the boundary evicts the four least recent =
  // the last four seed entries, and the fifth from the end stays
  m.core->set_spare_target(7);
  m.core->begin_token(1);
  m.serve(0, {0});  // a miss into one spare (most recent now)
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
