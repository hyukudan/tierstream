// Device-gated service thread (src/dg_service.cpp) with a real thread: the test thread plays the device (the export,
// prediction and wait kernels and the routed reads of a captured step) against plain memory, the service runs the
// core through an immediate transport. Checks the token handoff (Served, then the engine owns the core), the bytes
// every read sees, and every failure path: a missing export or prediction, a desync, a bad id or count, a stop while
// idle or waiting. Under TSan the handoffs are checked too.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "check.hpp"
#include "dg_core.hpp"
#include "dg_proto.hpp"
#include "dg_service.hpp"
#include "dg_steps.hpp"
#include "errors.hpp"

using ts::DgCore;
using ts::DgService;
using ts::DgTransport;

namespace {

// Copies and publications happen at once, on the service thread; the "device" (the test thread) reads the table and
// the ready words with acquire loads.
class Immediate final : public DgTransport {
 public:
  void d2d(int32_t, int32_t) override {}
  void h2h(int32_t, int32_t) override {}
  Immediate(uint32_t keys, uint32_t slots, uint32_t blocks, uint32_t* ready)
      : table_(keys), slot_(slots), block_(blocks, -1), ready_(ready) {
    for (auto& t : table_) t.store(-1);
    for (auto& s : slot_) s.store(-1);
  }
  void h2d(Stream, int32_t slot, int32_t block) override {
    if (fail_h2d) ts::fail(SS_E_BACKEND, "injected: an H2D copy failed");
    slot_[size_t(slot)].store(block_[size_t(block)], std::memory_order_relaxed);
  }
  bool fail_h2d = false;  // test: the next H2D copies fail (a device failure)
  void d2h(int32_t slot, int32_t block) override { block_[size_t(block)] = slot_[size_t(slot)].load(std::memory_order_relaxed); }
  void d2h_part(Stream, int32_t slot, int32_t block, uint32_t) override { d2h(slot, block); }  // one part per unit
  // three-tier: the unit's file bytes land in the block at once (the staging buffers follow the pinned blocks)
  void read_unit(uint32_t key, int32_t block) override {
    block_[size_t(block)] = int32_t(key);
    ++file_reads;
  }
  uint64_t file_reads = 0;
  uint32_t parts() const override { return 1; }
  uint64_t part_bytes(uint32_t) const override { return 8192; }
  void write_entry(Stream, uint32_t key, int32_t v) override { table_[key].store(v, std::memory_order_release); }
  void signal_ready(uint32_t g, uint32_t seq) override { ts::dg_store_release(&ready_[g], seq); }
  void signal_ready_part(uint32_t part, uint32_t seq) override { ts::dg_store_release(&lm_ready_[part], seq); }
  void set_lm_ready(uint32_t* p) { lm_ready_ = p; }
  void* event_create() override { return new int(0); }
  void event_destroy(void* ev) override { delete static_cast<int*>(ev); }
  void record(void*, Stream) override {}
  void wait(Stream, void*) override {}
  bool done(void*) override { return true; }
  void event_sync(void*) override {}
  void sync(Stream) override {}
  void upload_table(const int32_t* t, uint32_t n) override {
    for (uint32_t k = 0; k < n; ++k) table_[k].store(t[k], std::memory_order_release);
  }
  void upload_table_ranges(const int32_t* t, const uint32_t* r, uint32_t n) override {
    for (uint32_t i = 0; i < n; ++i)
      for (uint32_t k = r[2 * i]; k < r[2 * i] + r[2 * i + 1]; ++k) table_[k].store(t[k], std::memory_order_release);
  }
  int32_t entry(uint32_t k) const { return table_[k].load(std::memory_order_acquire); }
  int32_t slot_holds(int32_t s) const { return slot_[size_t(s)].load(std::memory_order_relaxed); }
  void fill_slot(int32_t s, int32_t k) { slot_[size_t(s)].store(k); }
  void fill_block(int32_t b, int32_t k) { block_[size_t(b)] = k; }

 private:
  std::vector<std::atomic<int32_t>> table_, slot_;
  std::vector<int32_t> block_;
  uint32_t* ready_;
  uint32_t* lm_ready_ = nullptr;
};

// A gated model of G layers x U units, its mapped area in plain memory, the core and the service.
struct Rig {
  static constexpr uint32_t G = 4, U = 32;
  std::vector<ts::DgRecord> rec{G};
  std::vector<ts::DgPredRecord> prec{G};
  std::vector<uint32_t> flag = std::vector<uint32_t>(G, 0), pflag = std::vector<uint32_t>(G, 0),
                        ready = std::vector<uint32_t>(G, 0);
  std::vector<ts::DgRecord> lm_rec{ts::kDgMaxParts};
  std::vector<uint32_t> lm_flag = std::vector<uint32_t>(ts::kDgMaxParts, 0),
                        lm_ready = std::vector<uint32_t>(ts::kDgMaxParts, 0);
  uint32_t done = 0, err_code = 0, err_aux = 0, host_abort = 0, began = 0;
  std::unique_ptr<Immediate> t;
  std::unique_ptr<DgCore> core;
  std::unique_ptr<DgService> svc;
  uint32_t seq = 0;

  // with_began: the service waits for the begin kernel's word before any flag (design rev 3.4 §11.2).
  // wb_defer: the boundary's write-backs are issued by the service after each served layer.
  explicit Rig(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000), uint32_t max_ids = 8,
               bool with_began = false, bool wb_defer = false, bool scored = false, bool wb_on_d2h = false,
               uint32_t spare = 12, uint32_t disk = 0) {
    ts::DgConfig c;
    c.n_units.assign(G, U);
    c.persistent_slots = 48 + spare;
    c.bank_slots = max_ids;
    c.blocks = G * U - 48 + 12;
    if (disk) {  // three-tier: 40 pinned homes, 4 spare blocks, the other 40 non-device keys in their files
      c.disk_tier = true;
      c.home_blocks = 40;
      c.blocks = 44;
      c.disk_staging = 2;
      c.disk_queue = disk == 1 ? 1 : 2;  // synchronous reads / reads in flight
      c.inclusive = disk == 3 ? 2 : 0;   // the inclusive share (clean copies)
    }
    c.spare_target = spare;
    c.prefetch_reserve = spare < 3 ? spare : 3;
    c.wb_defer = wb_defer;
    c.wb_on_d2h = wb_on_d2h;  // dg_wb_stream
    c.scored = scored;
    c.max_ids = max_ids;
    t = std::make_unique<Immediate>(G * U, c.persistent_slots + 2 * c.bank_slots, c.blocks + c.disk_staging,
                                    ready.data());
    t->set_lm_ready(lm_ready.data());
    core = std::make_unique<DgCore>(c, t.get());
    std::vector<uint32_t> res;
    for (uint32_t k = 0; k < 48; ++k) res.push_back((k * 37) % (G * U));
    core->place(res);
    for (uint32_t k = 0; k < G * U; ++k) {
      if (core->slot_of(k) >= 0) t->fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t->fill_block(core->block_of(k), int32_t(k));
    }
    t->upload_table(core->table().data(), G * U);
    ts::DgMapped m;
    m.rec = rec.data();
    m.prec = prec.data();
    m.flag = flag.data();
    m.pflag = pflag.data();
    m.ready = ready.data();
    m.done = &done;
    m.err_code = &err_code;
    m.err_aux = &err_aux;
    m.host_abort = &host_abort;
    if (with_began) m.began = &began;
    m.lm_rec = lm_rec.data();
    m.lm_flag = lm_flag.data();
    m.lm_ready = lm_ready.data();
    ts::DgServiceConfig sc;
    sc.max_ids = max_ids;
    sc.timeout = timeout;
    sc.diag = true;
    svc = std::make_unique<DgService>(core.get(), m, sc);
    svc->start();
  }
  ~Rig() { svc->stop_join(); }

  // The export kernel of gated layer g.
  uint64_t do_export(uint32_t g, const std::vector<uint32_t>& ids, uint32_t n_pred, uint32_t as_seq) {
    ts::DgRecord& r = rec[g];
    uint64_t mask = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      r.ids[i] = ids[i];
      if (ids[i] < U && t->entry(core->key(g, ids[i])) < 0) mask |= 1ull << i;
    }
    r.n_ids = uint32_t(ids.size());
    r.n_pred_promised = n_pred;
    r.mask_lo = uint32_t(mask);
    r.mask_hi = uint32_t(mask >> 32);
    r.seq = as_seq;
    ts::dg_store_release(&flag[g], as_seq);
    return mask;
  }
  // The part export kernel of a layer token.
  uint64_t do_export_part(uint32_t g, uint32_t k, const std::vector<uint32_t>& ids, uint32_t as_seq) {
    ts::DgRecord& r = lm_rec[k];
    uint64_t mask = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      r.ids[i] = ids[i];
      if (ids[i] < U && t->entry(core->key(g, ids[i])) < 0) mask |= 1ull << i;
    }
    r.n_ids = uint32_t(ids.size());
    r.n_pred_promised = 0;
    r.mask_lo = uint32_t(mask);
    r.mask_hi = uint32_t(mask >> 32);
    r.seq = as_seq;
    r.pad[0] = g;
    r.pad[1] = k;
    ts::dg_store_release(&lm_flag[k], as_seq);
    return mask;
  }
  bool do_wait_part(uint32_t k) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (int32_t(ts::dg_load_acquire(&lm_ready[k]) - seq) < 0) {
      if (ts::dg_load_acquire(&host_abort)) return false;
      if (std::chrono::steady_clock::now() > end) return false;
      std::this_thread::yield();
    }
    return true;
  }
  // The layer end kernel (dg_end_layer_flags, as end_layer_kernel): a flag already at the token's seq is a stray full
  // export (kDgDesync, err_aux = that layer; the first error wins), then every export flag takes seq, then done.
  void do_end_layer() {
    ts::dg_end_layer_flags(
        G, seq, [&](uint32_t g) { return ts::dg_load_acquire(&flag[g]); },
        [&](uint32_t g, uint32_t v) { ts::dg_store_release(&flag[g], v); },
        [&](uint32_t g) {
          if (err_code == ts::kDgOk) {
            err_code = ts::kDgDesync;
            err_aux = g;
          }
        });
    ts::dg_store_release(&done, seq);
  }
  void do_predict(uint32_t g, const std::vector<uint32_t>& p) {
    for (size_t i = 0; i < p.size(); ++i) prec[g].pred[i] = p[i];
    prec[g].n_pred = uint32_t(p.size());
    ts::dg_store_release(&pflag[g], seq);
  }
  // The wait kernel: false on host_abort or the deadline.
  bool do_wait(uint32_t g) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (int32_t(ts::dg_load_acquire(&ready[g]) - seq) < 0) {
      if (ts::dg_load_acquire(&host_abort)) return false;
      if (std::chrono::steady_clock::now() > end) return false;
      std::this_thread::yield();
    }
    return true;
  }
  // The routed reads: every id resolves to a slot holding its unit.
  bool do_read(uint32_t g, const std::vector<uint32_t>& ids) {
    for (uint32_t id : ids) {
      const int32_t s = t->entry(core->key(g, id));
      if (s < 0 || t->slot_holds(s) != int32_t(core->key(g, id))) return false;
    }
    return true;
  }
};

std::vector<uint32_t> routes(std::mt19937& rng, uint32_t n) {
  std::vector<uint32_t> v(n);
  for (auto& x : v) {
    const double u = std::uniform_real_distribution<double>(0, 1)(rng);
    x = uint32_t(double(Rig::U) * u * u) % Rig::U;
  }
  return v;
}

}  // namespace

TEST(dg_service_tokens) {
  Rig r;
  std::mt19937 rng(5);
  int bad_reads = 0;
  for (int tok = 0; tok < 200; ++tok) {
    r.seq = uint32_t(tok + 1);
    r.svc->begin(r.seq);
    for (uint32_t g = 0; g < Rig::G; ++g) {
      std::vector<uint32_t> ids = routes(rng, 1 + uint32_t(rng() % 8));
      const bool predict = g + 1 < Rig::G && tok % 3 != 0;
      const uint64_t mask = r.do_export(g, ids, predict ? 6 : 0, r.seq);
      if (predict) r.do_predict(g, routes(rng, 6));
      if (mask) CHECK(r.do_wait(g));
      if (!r.do_read(g, ids)) ++bad_reads;
    }
    ts::dg_store_release(&r.done, r.seq);
    CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
    r.core->end_token({});  // the engine thread owns the core again
    r.core->check();
  }
  CHECK_EQ(bad_reads, 0);
  const ts::DgStats& st = r.core->stats();
  std::printf("  tokens %llu, misses %llu, covered %llu, banks %llu, prefetched %llu (used %llu), evicted %llu\n",
              (unsigned long long)st.tokens, (unsigned long long)st.misses, (unsigned long long)st.covered,
              (unsigned long long)st.bank_serves, (unsigned long long)st.prefetched,
              (unsigned long long)st.prefetch_used, (unsigned long long)st.evicted);
  CHECK_EQ(st.tokens, 200ull);
  CHECK(st.misses > 0 && st.prefetched > 0 && st.evicted > 0);
  CHECK_EQ(r.svc->timeline().size(), size_t(Rig::G));
}

// dg_wb_defer: the service issues the deferred write-backs after the layers it serves; every read stays exact, and
// a boundary issues what the token left (the engine thread then owns the deferred queue). On the demand stream
// (dg_wb_stream = 0) and on the write-back stream (1, the default).
namespace {
ts::DgStats service_wb_defer_run(bool on_d2h) {
  Rig r(std::chrono::milliseconds(2000), 8, false, true, false, on_d2h);
  std::mt19937 rng(6);
  int bad_reads = 0;
  for (int tok = 0; tok < 200; ++tok) {
    r.seq = uint32_t(tok + 1);
    r.svc->begin(r.seq);
    for (uint32_t g = 0; g < Rig::G; ++g) {
      std::vector<uint32_t> ids = routes(rng, 1 + uint32_t(rng() % 8));
      const bool predict = g + 1 < Rig::G && tok % 3 != 0;
      const uint64_t mask = r.do_export(g, ids, predict ? 6 : 0, r.seq);
      if (predict) r.do_predict(g, routes(rng, 6));
      if (mask) CHECK(r.do_wait(g));
      if (!r.do_read(g, ids)) ++bad_reads;
    }
    ts::dg_store_release(&r.done, r.seq);
    CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
    r.core->check();  // the token's issued parts and cancels, before the boundary
    r.core->end_token({});
    r.core->check();
  }
  CHECK_EQ(bad_reads, 0);
  const ts::DgStats& st = r.core->stats();
  std::printf("  evicted %llu: deferred %llu, cancelled %llu, forced %llu, flushed at a boundary %llu\n",
              (unsigned long long)st.evicted, (unsigned long long)st.wb_deferred, (unsigned long long)st.wb_cancelled,
              (unsigned long long)st.wb_forced, (unsigned long long)st.wb_boundary_flushed);
  CHECK(st.evicted > 0);
  CHECK_EQ(st.wb_deferred, st.evicted);
  CHECK(st.wb_deferred > st.wb_boundary_flushed);  // most write-backs went out inside a token
  return st;
}
}  // namespace

TEST(dg_service_tokens_wb_defer) {
  // the service thread races the exports (the "device ahead" stop), so the counts are per run; the scripted
  // dg_defer_wb_stream_same_parts (test_dg) compares the two streams decision for decision
  const ts::DgStats dem = service_wb_defer_run(false);
  const ts::DgStats d2h = service_wb_defer_run(true);
  CHECK(dem.wb_issued_bytes > 0 && d2h.wb_issued_bytes > 0);
}

// Scored eviction: a layer whose next layer already exported (the device ran ahead) is ranked after that layer's
// service, at the latest before the token is published; every list is this token's when the engine gets the core, and
// equals a ranking made afterwards (a served layer's candidates, epochs and stamps are final). Half the tokens export
// every layer up front.
TEST(dg_service_scored_deferred_ranking) {
  Rig r(std::chrono::milliseconds(2000), 8, false, true, true);
  std::mt19937 rng(9);
  int bad_reads = 0, stale = 0, differ = 0;
  for (int tok = 0; tok < 120; ++tok) {
    r.seq = uint32_t(tok + 1);
    r.svc->begin(r.seq);
    std::vector<std::vector<uint32_t>> ids(Rig::G);
    std::vector<uint64_t> mask(Rig::G, 0);
    const bool ahead = tok % 2 == 1;
    for (uint32_t g = 0; g < Rig::G; ++g) {
      ids[g] = routes(rng, 1 + uint32_t(rng() % 8));
      if (ahead) continue;
      mask[g] = r.do_export(g, ids[g], 0, r.seq);
      if (mask[g]) CHECK(r.do_wait(g));
      if (!r.do_read(g, ids[g])) ++bad_reads;
    }
    if (ahead) {  // every export first (each layer's mask from the table as it stood), then the waits and reads
      for (uint32_t g = 0; g < Rig::G; ++g) mask[g] = r.do_export(g, ids[g], 0, r.seq);
      for (uint32_t g = 0; g < Rig::G; ++g) {
        if (mask[g]) CHECK(r.do_wait(g));
        if (!r.do_read(g, ids[g])) ++bad_reads;
      }
    }
    ts::dg_store_release(&r.done, r.seq);
    CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
    for (uint32_t g = 0; g < Rig::G; ++g) {
      stale += r.core->score_list_epoch(g) != r.core->epoch();
      const std::vector<DgCore::ScoreCand> was = r.core->score_list(g);
      r.core->score_layer(g);
      const auto& now = r.core->score_list(g);
      bool same = was.size() == now.size();
      for (size_t i = 0; same && i < was.size(); ++i)
        same = was[i].key == now[i].key && was[i].s == now[i].s && was[i].stamp == now[i].stamp;
      differ += !same;
    }
    r.core->end_token({});
    r.core->check();
  }
  const ts::DgStats& st = r.core->stats();
  std::printf("  ranked layers %llu, deferred %llu, terminal ranking %llu ns, fallbacks %llu, scored victims %llu\n",
              (unsigned long long)st.score_layers, (unsigned long long)st.score_deferred,
              (unsigned long long)st.score_last_ns, (unsigned long long)st.score_fallbacks,
              (unsigned long long)st.scored_victims);
  CHECK_EQ(bad_reads, 0);
  CHECK_EQ(stale, 0);
  CHECK_EQ(differ, 0);
  CHECK(st.score_deferred > 0);
  CHECK_EQ(st.score_fallbacks, 0ull);
  CHECK_EQ(st.score_layers, 2ull * 120 * Rig::G);  // the service's and the re-ranking above
  CHECK(st.score_last_ns > 0 && st.scored_victims > 0);
}

static void expect_failure(Rig& r, const char* needle);

// Layer tokens between full tokens: every part's ids read their units, a unit is copied at most once per layer token
// (later parts cover it), the boundary leaves a consistent core, and full tokens continue unchanged afterwards.
static void layer_token_rounds(Rig& r, std::mt19937& rng, int rounds, bool all_exports_first, int& bad_reads) {
  for (int round = 0; round < rounds; ++round) {
    for (uint32_t g = 0; g < Rig::G; ++g) {  // one layer-major block: every gated layer as its own layer token
      ++r.seq;
      const uint32_t parts = 1 + uint32_t(rng() % 6);
      r.core->set_layer_token(g);
      r.svc->begin_layer(r.seq, g, parts);
      std::vector<std::vector<uint32_t>> ids(parts);
      std::vector<uint64_t> masks(parts);
      std::vector<uint32_t> distinct;
      for (uint32_t k = 0; k < parts; ++k) {
        ids[k] = routes(rng, 1 + uint32_t(rng() % 8));
        distinct.insert(distinct.end(), ids[k].begin(), ids[k].end());
      }
      std::sort(distinct.begin(), distinct.end());
      distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
      const uint64_t fetched0 = r.core->stats().fetched;
      if (all_exports_first) {  // every part exported before any wait (the layer-major prompt's router pass)
        for (uint32_t k = 0; k < parts; ++k) masks[k] = r.do_export_part(g, k, ids[k], r.seq);
        for (uint32_t k = 0; k < parts; ++k) {
          if (masks[k]) CHECK(r.do_wait_part(k));
          if (!r.do_read(g, ids[k])) ++bad_reads;
        }
      } else {
        for (uint32_t k = 0; k < parts; ++k) {
          masks[k] = r.do_export_part(g, k, ids[k], r.seq);
          if (masks[k]) CHECK(r.do_wait_part(k));
          if (!r.do_read(g, ids[k])) ++bad_reads;
        }
      }
      r.do_end_layer();
      CHECK_EQ(r.err_code, uint32_t(ts::kDgOk));  // no flag held the token's seq before the end's stamp
      CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
      CHECK(r.core->stats().fetched - fetched0 <= distinct.size());  // one copy per unit per layer token
      CHECK_EQ(r.core->layer_token(), g);
      r.core->end_token({}, 96);
      CHECK_EQ(r.core->layer_token(), 0xffffffffu);
      r.core->check();
    }
    // a full token right after the layer-major block: the stamped flags keep the strict sequence check
    ++r.seq;
    r.svc->begin(r.seq);
    for (uint32_t g = 0; g < Rig::G; ++g) {
      std::vector<uint32_t> ids = routes(rng, 1 + uint32_t(rng() % 8));
      if (r.do_export(g, ids, 0, r.seq)) CHECK(r.do_wait(g));
      if (!r.do_read(g, ids)) ++bad_reads;
    }
    ts::dg_store_release(&r.done, r.seq);
    CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
    r.core->end_token({});
    r.core->check();
  }
}

TEST(dg_service_layer_tokens) {
  for (const bool first : {true, false})
    for (const bool defer : {false, true})
      for (const bool scored : {false, true}) {
        Rig r(std::chrono::milliseconds(2000), 8, false, defer, scored);
        std::mt19937 rng(11 + (first ? 1 : 0) + (defer ? 2 : 0) + (scored ? 4 : 0));
        int bad_reads = 0;
        layer_token_rounds(r, rng, 40, first, bad_reads);
        CHECK_EQ(bad_reads, 0);
        const ts::DgStats& st = r.core->stats();
        CHECK_EQ(st.lm_tokens, 40ull * Rig::G);
        CHECK(st.lm_parts >= st.lm_tokens);
        CHECK(st.misses > 0 && st.evicted > 0);
        CHECK_EQ(st.score_fallbacks, 0ull);  // a layer token's boundary is the recency walk by design
        CHECK_EQ(r.svc->timeline().size(), size_t(Rig::G));
      }
}

// Layer tokens on the three-tier placement: random rounds with file-homed units (reads synchronous, in flight, with
// an inclusive share), wb_defer on and off; every read exact, the core consistent at every boundary.
TEST(dg_service_layer_tokens_three_tier) {
  for (uint32_t disk = 1; disk <= 3; ++disk)
    for (const bool defer : {false, true}) {
      Rig r(std::chrono::milliseconds(2000), 8, false, defer, false, false, 12, disk);
      std::mt19937 rng(101 + disk * 2 + (defer ? 1 : 0));
      int bad_reads = 0;
      layer_token_rounds(r, rng, 25, disk != 2, bad_reads);
      CHECK_EQ(bad_reads, 0);
      CHECK_EQ(r.core->stats().lm_tokens, 25ull * Rig::G);
      CHECK(r.core->stats().disk_reads > 0);
    }
}

// Two free slots and one bank of 8: a whole layer through one layer token needs more slots than that, so the
// demand writes back units of other layers inside the token; every read stays exact and the core consistent.
// clean: dg_clean_evict on (three-tier): a mid-token victim without a free block drops to its file, never a pinned one
static void layer_token_mid_eviction(uint32_t disk, bool clean = false) {
  Rig r(std::chrono::milliseconds(2000), 8, false, false, false, false, 2, disk);
  if (clean) r.core->set_clean_evict(true);
  uint64_t mid_drops = 0, mid_clean = 0;  // inside the layer tokens only (the boundary's own are not counted here)
  int bad_reads = 0;
  for (uint32_t g = 0; g < Rig::G; ++g) {
    ++r.seq;
    const uint32_t parts = Rig::U / 8;
    r.core->set_layer_token(g);
    const uint64_t drops0 = r.core->stats().disk_drops, clean0 = r.core->stats().clean_drops;
    const uint64_t wbn0 = r.core->stats().writebacks_done, wbb0 = r.core->stats().wb_completed_bytes;
    r.svc->begin_layer(r.seq, g, parts);
    std::vector<std::vector<uint32_t>> ids(parts);
    std::vector<uint64_t> masks(parts);
    for (uint32_t k = 0; k < parts; ++k) {
      for (uint32_t i = 0; i < 8; ++i) ids[k].push_back(k * 8 + i);
      masks[k] = r.do_export_part(g, k, ids[k], r.seq);
    }
    for (uint32_t k = 0; k < parts; ++k) {
      if (masks[k]) CHECK(r.do_wait_part(k));
      if (!r.do_read(g, ids[k])) ++bad_reads;
    }
    r.do_end_layer();
    const DgService::State s = r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000));
    if (s != DgService::kServed) std::printf("  layer %u: %s\n", g, r.svc->error().c_str());
    CHECK_EQ(int(s), int(DgService::kServed));
    if (s != DgService::kServed) {
      r.svc->stop_join();
      return;
    }
    mid_drops += r.core->stats().disk_drops - drops0;  // (the service is done: the core is the engine thread's)
    mid_clean += r.core->stats().clean_drops - clean0;
    // every completed write-back inside the token (collected or the mid-token's synchronous one) counts its unit too
    CHECK_EQ(r.core->stats().wb_completed_bytes - wbb0, (r.core->stats().writebacks_done - wbn0) * 8192u);
    r.core->end_token({}, 32);
    r.core->check();
  }
  CHECK_EQ(bad_reads, 0);
  const ts::DgStats& st = r.core->stats();
  std::printf("  disk %u: layer tokens %llu, fetched %llu, bank %llu, mid evictions %llu, early admits %llu, disk reads"
              " %llu, drops %llu, discards %llu, inclusive kept %llu / drops %llu\n",
              disk, (unsigned long long)st.lm_tokens, (unsigned long long)st.fetched,
              (unsigned long long)st.bank_serves, (unsigned long long)st.lm_mid_evictions,
              (unsigned long long)st.lm_early_admits, (unsigned long long)st.disk_reads,
              (unsigned long long)st.disk_drops, (unsigned long long)st.disk_discards,
              (unsigned long long)st.inclusive_kept, (unsigned long long)st.inclusive_drops);
  CHECK(st.lm_mid_evictions > 0);
  if (disk) CHECK(st.disk_reads > 0);
  std::printf("  clean_evict %d: pinned drops inside the layer tokens %llu, clean drops there %llu (all %llu)\n",
              int(clean), (unsigned long long)mid_drops, (unsigned long long)mid_clean,
              (unsigned long long)st.clean_drops);
  if (clean) {  // a mid-token victim never pushes a pinned unit out; its drop to its file is a clean drop
    CHECK_EQ(mid_drops, 0ull);
    if (disk != 3) CHECK(mid_clean > 0);
    CHECK_EQ(st.clean_drop_bytes, st.clean_drops * 8192u);  // one part of 8192 bytes per unit
  } else {
    CHECK_EQ(st.clean_drops, 0ull);
  }
}

TEST(dg_service_layer_token_mid_eviction) { layer_token_mid_eviction(0); }

// The same under the three-tier placement (units on the file tier): demands read file-homed units through the staging
// buffers (synchronously, in flight, with an inclusive share), and the mid-token evictions take the boundary's ways
// down (a clean copy, a write-back into a free block, a pinned unit dropped to its file, the victim to its file).
TEST(dg_service_layer_token_mid_eviction_three_tier) {
  for (uint32_t disk = 1; disk <= 3; ++disk) layer_token_mid_eviction(disk);
}

// dg_clean_evict honoured by the mid-token eviction (as by the boundary's accept): no pinned unit is dropped for a
// block inside a layer token, the victims that found none go to their files and count in clean_drops / bytes.
TEST(dg_service_layer_token_mid_eviction_clean_evict) {
  for (uint32_t disk = 1; disk <= 3; ++disk) layer_token_mid_eviction(disk, true);
}

TEST(dg_service_layer_token_missing_part) {
  Rig r(std::chrono::milliseconds(200));
  r.seq = 1;
  r.core->set_layer_token(2);
  r.svc->begin_layer(1, 2, 3);
  r.do_export_part(2, 0, {1, 2}, 1);
  r.do_export_part(2, 1, {3}, 1);  // part 2 never exports
  expect_failure(r, "part flag of gated layer 2 never came");
}

TEST(dg_service_layer_token_wrong_part) {
  Rig r(std::chrono::milliseconds(500));
  r.seq = 1;
  r.core->set_layer_token(1);
  r.svc->begin_layer(1, 1, 2);
  r.do_export_part(3, 0, {1}, 1);  // exported for another gated layer
  expect_failure(r, "was exported for gated layer 3");
}

// A full export of a gated layer inside a layer token (an engine bug: the stray export's readers may read slots the
// token's mid-token evictions reuse) is caught by the layer end kernel's own check (dg_end_layer_flags): kDgDesync,
// err_aux = that layer, before every flag is stamped with the token's seq (after which the next ss_begin's flag check
// could not see it). Another gated layer, and the token's own layer (it exports through its part words only).
TEST(dg_service_layer_token_stray_export) {
  for (const uint32_t stray : {0u, 2u, 3u}) {
    Rig r;
    r.seq = 1;
    r.core->set_layer_token(2);
    r.svc->begin_layer(1, 2, 1);
    if (r.do_export_part(2, 0, {1, 2, 5}, 1)) CHECK(r.do_wait_part(0));
    CHECK(r.do_read(2, {1, 2, 5}));
    r.do_export(stray, {3}, 0, 1);  // the stray full export
    r.do_end_layer();
    CHECK_EQ(r.err_code, uint32_t(ts::kDgDesync));
    CHECK_EQ(r.err_aux, stray);
    for (uint32_t g = 0; g < Rig::G; ++g) CHECK_EQ(r.flag[g], 1u);  // the stamp alone would have hidden it
    CHECK_EQ(int(r.svc->wait_terminal(1, std::chrono::milliseconds(5000))), int(DgService::kServed));
    r.core->end_token({}, 96);
    r.core->check();
  }
  // two stray exports: the first gated layer the end meets is reported (the first error wins)
  Rig r;
  r.seq = 1;
  r.core->set_layer_token(1);
  r.svc->begin_layer(1, 1, 1);
  if (r.do_export_part(1, 0, {4}, 1)) CHECK(r.do_wait_part(0));
  r.do_export(3, {0}, 0, 1);
  r.do_export(2, {0}, 0, 1);
  r.do_end_layer();
  CHECK_EQ(r.err_code, uint32_t(ts::kDgDesync));
  CHECK_EQ(r.err_aux, 2u);
  CHECK_EQ(int(r.svc->wait_terminal(1, std::chrono::milliseconds(5000))), int(DgService::kServed));
}

// The service fails a token, sets host_abort, returns, and reports why; the engine sees Failed.
static void expect_failure(Rig& r, const char* needle) {
  const DgService::State s = r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000));
  CHECK_EQ(int(s), int(DgService::kFailed));
  CHECK_EQ(ts::dg_load_acquire(&r.host_abort), 1u);
  const std::string e = r.svc->error();
  if (e.find(needle) == std::string::npos) std::printf("  error: %s (wanted '%s')\n", e.c_str(), needle);
  CHECK(e.find(needle) != std::string::npos);
  r.svc->stop_join();
  r.core->check();  // the service returned without breaking the core's bookkeeping
}

TEST(dg_service_missing_export) {
  Rig r(std::chrono::milliseconds(200));
  r.seq = 1;
  r.svc->begin(1);
  r.do_export(0, {1, 2}, 0, 1);  // layer 1 never exports
  expect_failure(r, "export flag of gated layer 1 never came");
}

// The prediction never comes and every later route hits: no device wait times out, the token still fails.
TEST(dg_service_missing_prediction) {
  Rig r(std::chrono::milliseconds(200));
  r.seq = 1;
  r.svc->begin(1);
  r.do_export(0, {r.core->table()[0] >= 0 ? 0u : 1u}, 4, 1);  // promises 4 predictions, never sends them
  expect_failure(r, "prediction flag of gated layer 0 never came");
}

TEST(dg_service_desync) {
  Rig r;
  r.seq = 1;
  r.svc->begin(1);
  r.do_export(0, {1}, 0, 2);  // a step ran twice: the flag holds the next token
  expect_failure(r, "sequence desync");
}

TEST(dg_service_record_for_another_token) {
  Rig r;
  r.seq = 1;
  r.svc->begin(1);
  r.rec[0].seq = 7;
  r.rec[0].n_ids = 1;
  r.rec[0].ids[0] = 1;
  ts::dg_store_release(&r.flag[0], 1);
  expect_failure(r, "the record of gated layer 0 is for token 7");
}

TEST(dg_service_bad_id) {
  Rig r;
  r.seq = 1;
  r.svc->begin(1);
  r.do_export(0, {3, Rig::U + 5}, 0, 1);
  expect_failure(r, "exported id");
}

TEST(dg_service_bad_counts) {
  {
    Rig r;
    r.seq = 1;
    r.svc->begin(1);
    r.do_export(0, {1, 2, 3, 4, 5, 6, 7, 8, 9}, 0, 1);  // 9 > max_ids 8
    expect_failure(r, "exported 9 ids");
  }
  {
    Rig r;
    r.seq = 1;
    r.svc->begin(1);
    r.do_export(0, {1}, 3, 1);
    r.do_predict(0, {1, 2});  // 2 sent, 3 promised
    expect_failure(r, "predicted 2 ids, its export promised 3");
  }
}

TEST(dg_service_stop) {
  {
    Rig r;  // idle: no token begun
    const auto t0 = std::chrono::steady_clock::now();
    r.svc->stop_join();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
  }
  {
    Rig r(std::chrono::milliseconds(60000));  // waiting for an export that never comes: stop must not wait 60 s
    r.seq = 1;
    r.svc->begin(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto t0 = std::chrono::steady_clock::now();
    r.svc->stop_join();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(1));
    CHECK_EQ(ts::dg_load_acquire(&r.host_abort), 1u);
    r.core->check();
  }
}

// A demand miss stays unserved (the service is stuck on an earlier prediction): the device's wait ends on host_abort.
TEST(dg_service_host_abort_ends_device_wait) {
  Rig r(std::chrono::milliseconds(300));
  r.seq = 1;
  r.svc->begin(1);
  r.do_export(0, {1}, 2, 1);  // promises a prediction it never sends
  // layer 1: find a unit that is not resident, so the device must wait
  uint32_t miss = 0;
  while (miss < Rig::U && r.core->table()[r.core->key(1, miss)] >= 0) ++miss;
  CHECK(miss < Rig::U);
  r.do_export(1, {miss}, 0, 1);
  const auto t0 = std::chrono::steady_clock::now();
  CHECK(!r.do_wait(1));  // ends on host_abort, well before the 5 s device deadline
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(3));
  expect_failure(r, "prediction flag of gated layer 0 never came");
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }

// design rev 3.4 §11.2: the begin kernel may come long after ss_begin (a graph captured and instantiated in between);
// no flag deadline runs before it. The export after it is served as usual.
TEST(dg_service_slow_launch_is_not_a_failure) {
  Rig r(std::chrono::milliseconds(150), 8, true);
  r.seq = 1;
  r.svc->begin(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(450));  // 3x the flag deadline
  CHECK_EQ(int(r.svc->wait_terminal(1, std::chrono::milliseconds(1))), int(DgService::kIdle));
  ts::dg_store_release(&r.began, 1u);
  for (uint32_t g = 0; g < Rig::G; ++g) {
    const uint64_t mask = r.do_export(g, {g, g + 1}, 0, 1);
    if (mask) CHECK(r.do_wait(g));
  }
  ts::dg_store_release(&r.done, 1u);
  CHECK_EQ(int(r.svc->wait_terminal(1, std::chrono::milliseconds(5000))), int(DgService::kServed));
  // after the begin kernel, the flag deadline applies again
  r.core->end_token({});
  r.seq = 2;
  r.svc->begin(2);
  ts::dg_store_release(&r.began, 2u);
  expect_failure(r, "export flag of gated layer 0 never came");
}

TEST(dg_service_began_past_is_a_desync) {
  Rig r(std::chrono::milliseconds(2000), 8, true);
  r.seq = 1;
  r.svc->begin(1);
  ts::dg_store_release(&r.began, 2u);  // a later step began: one ran outside the pair
  expect_failure(r, "a gated step began with seq 2 while serving token 1");
}

TEST(dg_service_stop_while_waiting_for_began) {
  Rig r(std::chrono::milliseconds(100), 8, true);
  r.seq = 1;
  r.svc->begin(1);
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  const auto t0 = std::chrono::steady_clock::now();
  r.svc->stop_join();
  CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(2));
  CHECK_EQ(int(r.svc->wait_terminal(1, std::chrono::milliseconds(1))), int(DgService::kStopped));
}

// seen_us of layer 0 starts at the previous token's end (site.cu), so host time between tokens shows in it.
TEST(dg_service_seen_us_origin) {
  Rig r(std::chrono::milliseconds(2000), 8, true);
  double seen0_tok1 = 0;
  for (uint32_t tok = 1; tok <= 2; ++tok) {
    // token 1: the origin is the service's start (no previous token); token 2: the previous token's end
    std::this_thread::sleep_for(std::chrono::milliseconds(60));  // the engine before the token
    r.seq = tok;
    r.svc->begin(tok);
    ts::dg_store_release(&r.began, tok);
    for (uint32_t g = 0; g < Rig::G; ++g) {
      const uint64_t mask = r.do_export(g, {g}, 0, tok);
      if (mask) CHECK(r.do_wait(g));
    }
    ts::dg_store_release(&r.done, tok);
    CHECK_EQ(int(r.svc->wait_terminal(tok, std::chrono::milliseconds(5000))), int(DgService::kServed));
    if (tok == 1) seen0_tok1 = r.svc->timeline()[0].seen_us;
    r.core->end_token({});
  }
  if (seen0_tok1 < 30000) std::printf("  token 1 seen_us[0] = %.0f us (wanted >= 30000)\n", seen0_tok1);
  CHECK(seen0_tok1 >= 30000);  // 60 ms after the Rig started the service (its thread start may lag)
  const double seen0 = r.svc->timeline()[0].seen_us;
  if (seen0 < 60000) std::printf("  seen_us[0] = %.0f us (wanted >= 60000)\n", seen0);
  CHECK(seen0 >= 60000);
}

// Steps that leave gated layers unexported (rows_per_layer 0, or a held make_resident set): the service never waits
// for them, their flags keep the last token that exported them (a later export is checked against that one), a
// prediction for an unexported layer is ignored, the core does not serve them. Every read stays exact; scored tokens
// keep every list current (no fallback).
TEST(dg_service_skipped_layers) {
  for (bool scored : {false, true}) {
    Rig r(std::chrono::milliseconds(2000), 8, false, scored, scored);
    std::mt19937 rng(17);
    int bad_reads = 0, skipped = 0;
    for (int tok = 0; tok < 150; ++tok) {
      r.seq = uint32_t(tok + 1);
      std::vector<uint8_t> run(Rig::G, 1);
      for (auto& x : run) x = rng() % 3 != 0;
      if (tok % 25 == 7) std::fill(run.begin(), run.end(), uint8_t(0));  // a step exporting nothing
      std::vector<uint32_t> flag0(r.flag), ready0(r.ready);
      std::vector<uint64_t> lookups0;
      for (const auto& l : r.core->layer_counts()) lookups0.push_back(l.lookups);
      r.svc->begin(r.seq, &run);
      for (uint32_t g = 0; g < Rig::G; ++g) {
        if (!run[g]) {
          ++skipped;
          continue;
        }
        std::vector<uint32_t> ids = routes(rng, 1 + uint32_t(rng() % 8));
        const bool predict = g + 1 < Rig::G && tok % 2 == 0;  // promised also for an unexported next layer
        const uint64_t mask = r.do_export(g, ids, predict ? 4 : 0, r.seq);
        if (predict && run[g + 1]) r.do_predict(g, routes(rng, 4));
        if (mask) CHECK(r.do_wait(g));
        if (!r.do_read(g, ids)) ++bad_reads;
      }
      ts::dg_store_release(&r.done, r.seq);
      CHECK_EQ(int(r.svc->wait_terminal(r.seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
      for (uint32_t g = 0; g < Rig::G; ++g)
        if (!run[g]) {
          CHECK_EQ(r.flag[g], flag0[g]);
          CHECK_EQ(r.ready[g], ready0[g]);
          CHECK_EQ(r.core->layer_counts()[g].lookups, lookups0[g]);
        }
      r.core->end_token({});
      r.core->check();
    }
    CHECK_EQ(bad_reads, 0);
    CHECK(skipped > 0);
    CHECK_EQ(r.core->stats().score_fallbacks, 0ull);
    CHECK_EQ(r.core->stats().tokens, 150ull);
  }
}

// After a skipped token, an export flag holding neither the last exporting token nor this one is still a desync.
TEST(dg_service_skipped_layer_desync) {
  Rig r;
  for (uint32_t tok = 1; tok <= 2; ++tok) {
    r.seq = tok;
    std::vector<uint8_t> run(Rig::G, 1);
    if (tok == 2) run[1] = 0;
    r.svc->begin(tok, &run);
    for (uint32_t g = 0; g < Rig::G; ++g) {
      if (!run[g]) continue;
      const uint64_t mask = r.do_export(g, {g}, 0, tok);
      if (mask) CHECK(r.do_wait(g));
    }
    ts::dg_store_release(&r.done, tok);
    CHECK_EQ(int(r.svc->wait_terminal(tok, std::chrono::milliseconds(5000))), int(DgService::kServed));
    r.core->end_token({});
  }
  r.seq = 3;
  r.svc->begin(3);
  r.do_export(0, {1}, 0, 3);
  ts::dg_store_release(&r.flag[1], 2u);  // layer 1 last exported in token 1: a flag of token 2 is a stray step's
  expect_failure(r, "sequence desync: export flag of gated layer 1");
}

// ---- the context path of kept-resident layers on the CPU: DgSteps (the context's keep / resident / export-flag
// bookkeeping, as dg_ctx.cpp uses it) + the service thread + the core, the test thread playing the GLM engine's steps
// and reading every weight through a host port of the ss_dg_slot rules ----
namespace {

struct CtxRig {
  static constexpr uint32_t G = 4, U = 288;
  std::vector<ts::DgRecord> rec{G};
  std::vector<ts::DgPredRecord> prec{G};
  std::vector<uint32_t> flag = std::vector<uint32_t>(G, 0), pflag = std::vector<uint32_t>(G, 0),
                        ready = std::vector<uint32_t>(G, 0);
  std::vector<ts::DgRecord> lm_rec{ts::kDgMaxParts};
  std::vector<uint32_t> lm_flag = std::vector<uint32_t>(ts::kDgMaxParts, 0),
                        lm_ready = std::vector<uint32_t>(ts::kDgMaxParts, 0);
  uint32_t done = 0, err_code = 0, err_aux = 0, host_abort = 0, began = 0;
  std::unique_ptr<Immediate> t;
  std::unique_ptr<DgCore> core;
  std::unique_ptr<DgService> svc;
  ts::DgSteps steps;
  std::vector<int> layer_of{0, 1, 2, 3};
  uint32_t seq = 0, guard = 0;
  uint64_t reads = 0, bad = 0, absent = 0;
  std::mt19937 rng{23};

  // The configurations: LRU, plain SCORED (the public reference lists), and production as dg_ctx.cpp opens it:
  // SCORED_AUTO with the bounded boundary selection, three-tier with mixed sources (of the 516 non-resident units
  // every other one pinned, the rest file-only; 32 spare blocks, 4 staging buffers, 4 reads in flight).
  enum Mode { kLru, kScored, kProduction };
  Mode mode;
  uint32_t window = 1;  // rows of the next boundary (ss_dg_end_rows): AUTO ranks by score only for windows > 1
  // disk_mode (LRU / SCORED only): 0 = two-tier; 1 / 2 = three-tier (300 pinned homes, 16 spare blocks, the other 216
  // non-device keys in their files; reads synchronous / in flight)
  explicit CtxRig(bool scored, uint32_t disk_mode = 0) : CtxRig(scored ? kScored : kLru, disk_mode) {}
  explicit CtxRig(Mode md, uint32_t disk_mode = 0) : mode(md) {
    const bool three = md == kProduction;
    ts::DgConfig c;
    c.n_units.assign(G, U);
    c.persistent_slots = 636 + 64;  // 636 residents + 64 spares: two whole layers fit kept at once
    c.bank_slots = 8;
    c.spare_target = 64;
    c.prefetch_reserve = 8;
    c.scored = md != kLru;
    c.scored_auto = three;
    c.boundary_score = three;
    c.max_ids = 8;
    std::vector<uint32_t> res;
    for (uint32_t k = 0; k < 636; ++k) res.push_back((k * 389) % (G * U));
    std::vector<uint8_t> disk(G * U, 0), is_res(G * U, 0);
    for (uint32_t k : res) is_res[k] = 1;
    uint32_t pinned = 0, n_disk = 0;
    for (uint32_t k = 0, i = 0; k < G * U; ++k)
      if (!is_res[k]) {
        if (i++ % 2) {
          disk[k] = 1;
          ++n_disk;
        } else {
          ++pinned;
        }
      }
    if (three) {
      c.disk_tier = true;
      c.home_blocks = pinned;
      c.blocks = pinned + 32;
      c.disk_staging = 4;
      c.disk_queue = 4;
      for (uint32_t k = 0; k < G * U; ++k) res.push_back(k);  // the priority list names every key (the context's)
    } else if (disk_mode) {
      c.disk_tier = true;
      c.home_blocks = 300;
      c.blocks = 316;
      c.disk_staging = 2;
      c.disk_queue = disk_mode == 1 ? 1 : 2;
    } else {
      c.blocks = G * U - 636 + 64;
    }
    guard = c.persistent_slots + 2 * c.bank_slots;
    t = std::make_unique<Immediate>(G * U, guard + 1, c.blocks + c.disk_staging, ready.data());
    t->set_lm_ready(lm_ready.data());
    core = std::make_unique<DgCore>(c, t.get());
    core->place(res, three ? &disk : nullptr);
    if (three) CHECK_EQ(core->file_units(), n_disk);
    for (uint32_t k = 0; k < G * U; ++k) {
      if (core->slot_of(k) >= 0) t->fill_slot(core->slot_of(k), int32_t(k));
      if (core->block_of(k) >= 0) t->fill_block(core->block_of(k), int32_t(k));
    }
    t->upload_table(core->table().data(), G * U);
    ts::DgMapped m;
    m.rec = rec.data();
    m.prec = prec.data();
    m.flag = flag.data();
    m.pflag = pflag.data();
    m.ready = ready.data();
    m.done = &done;
    m.err_code = &err_code;
    m.err_aux = &err_aux;
    m.host_abort = &host_abort;
    m.began = &began;
    m.lm_rec = lm_rec.data();
    m.lm_flag = lm_flag.data();
    m.lm_ready = lm_ready.data();
    ts::DgServiceConfig sc;
    sc.max_ids = 8;
    svc = std::make_unique<DgService>(core.get(), m, sc);
    svc->start();
    steps.init(G);
  }
  ~CtxRig() { svc->stop_join(); }

  // the ss_dg_slot rules on the host: an id out of range or an absent entry reads the guard slot (absent is reported)
  int32_t slot(uint32_t g, uint32_t id) {
    if (id >= U) return int32_t(guard);
    const int32_t s = t->entry(core->key(g, id));
    if (s >= 0) return s;
    ++absent;
    return int32_t(guard);
  }
  void read(uint32_t g, uint32_t id) {
    ++reads;
    const int32_t s = slot(g, id);
    if (s == int32_t(guard) || t->slot_holds(s) != int32_t(core->key(g, id))) ++bad;
  }
  // ss_dg_make_resident / ss_dg_keep(g, NULL, 0) as the context runs them
  void make_resident(uint32_t g, const std::vector<uint32_t>& ids) {
    core->make_resident(g, ids, steps.protect());
    std::vector<uint32_t> keys;
    for (uint32_t u : ids) keys.push_back(core->key(g, u));
    steps.hold(g, std::move(keys));
  }
  void release(uint32_t g) { steps.keep(g, {}); }
  // One step as the GLM engine runs it: ss_begin(rows) (the context plans the exported layers and checks the flags),
  // the begin kernel; per gated layer with rows that holds no resident set: export, wait (on a miss), read the routed
  // ids; a held layer: no export, no wait, `held_reads` ids of its kept set read through the table; the end; ss_end.
  void step(const std::vector<uint32_t>& rows, size_t held_reads = SIZE_MAX, int stray_export = -1) {
    const std::vector<uint8_t>& run = steps.plan(rows.data(), layer_of);
    for (uint32_t g = 0; g < G; ++g) CHECK_EQ(run[g], uint8_t(rows[g] > 0 && !steps.resident(g)));
    CHECK(steps.flags_clean(flag.data(), ts::dg_load_acquire));
    ++seq;
    steps.begin(seq);
    svc->begin(seq, &run);
    ts::dg_store_release(&began, seq);
    for (uint32_t g = 0; g < G; ++g) {
      if (!run[g]) {
        const auto& kept = steps.kept(g);
        for (size_t i = 0; i < kept.size() && i < held_reads; ++i) {
          const size_t j = held_reads == SIZE_MAX ? i : size_t(rng()) % kept.size();
          read(g, kept[j] - core->key(g, 0));
        }
        if (int(g) == stray_export) {  // an engine bug: it exported a layer the step did not announce
          ts::DgRecord& r = rec[g];
          r.n_ids = 1;
          r.ids[0] = 0;
          r.seq = seq;
          ts::dg_store_release(&flag[g], seq);
        }
        continue;
      }
      std::vector<uint32_t> ids(1 + rng() % 8);
      for (auto& id : ids) id = uint32_t(rng()) % U;
      ts::DgRecord& r = rec[g];
      uint64_t mask = 0;
      for (size_t i = 0; i < ids.size(); ++i) {
        r.ids[i] = ids[i];
        if (t->entry(core->key(g, ids[i])) < 0) mask |= 1ull << i;
      }
      r.n_ids = uint32_t(ids.size());
      r.n_pred_promised = 0;
      r.mask_lo = uint32_t(mask);
      r.mask_hi = uint32_t(mask >> 32);
      r.seq = seq;
      ts::dg_store_release(&flag[g], seq);
      if (mask) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (int32_t(ts::dg_load_acquire(&ready[g]) - seq) < 0 && std::chrono::steady_clock::now() < end)
          std::this_thread::yield();
      }
      for (uint32_t id : ids) read(g, id);
    }
    ts::dg_store_release(&done, seq);
    CHECK_EQ(int(svc->wait_terminal(seq, std::chrono::milliseconds(5000))), int(DgService::kServed));
    core->end_token(steps.all_kept(), window);
    core->check();
    window = window == 1 ? 6 : 1;  // R1 and R6 windows alternate (AUTO: the LRU walk, then the score order)
  }
  // One layer token as the context runs it (ss_dg_begin_layer): plan g only, the desync check, the core told every
  // layer's keep, the service handed the token; the device exports every part first (the router pass), then waits
  // and reads each part's ids, then the end kernel (every export flag stamped). ss_end: the boundary with every keep.
  // Returns false (the service's error in `why`) when the service failed the token.
  // stray_export >= 0: the engine also exports that gated layer in full inside the token (before the end kernel).
  bool layer_token(uint32_t g, const std::vector<uint32_t>& ids, std::string* why = nullptr, int stray_export = -1) {
    CHECK(steps.plan_layer(g));
    CHECK(steps.flags_clean(flag.data(), ts::dg_load_acquire));
    core->set_layer_token(g, steps.all_kept());
    ++seq;
    steps.begin(seq);
    const uint32_t parts = uint32_t((ids.size() + 7) / 8);
    svc->begin_layer(seq, g, parts);
    ts::dg_store_release(&began, seq);
    std::vector<uint64_t> masks(parts);
    for (uint32_t k = 0; k < parts; ++k) {
      ts::DgRecord& r = lm_rec[k];
      uint64_t mask = 0;
      uint32_t n = 0;
      for (size_t i = size_t(k) * 8; i < ids.size() && n < 8; ++i, ++n) {
        r.ids[n] = ids[i];
        if (t->entry(core->key(g, ids[i])) < 0) mask |= 1ull << n;
      }
      r.n_ids = n;
      r.n_pred_promised = 0;
      r.mask_lo = uint32_t(mask);
      r.mask_hi = uint32_t(mask >> 32);
      r.seq = seq;
      r.pad[0] = g;
      r.pad[1] = k;
      masks[k] = mask;
      ts::dg_store_release(&lm_flag[k], seq);
    }
    bool aborted = false;
    for (uint32_t k = 0; k < parts && !aborted; ++k) {
      if (masks[k]) {
        const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (int32_t(ts::dg_load_acquire(&lm_ready[k]) - seq) < 0) {
          if (ts::dg_load_acquire(&host_abort) || std::chrono::steady_clock::now() > end) {
            aborted = true;
            break;
          }
          std::this_thread::yield();
        }
      }
      if (aborted) break;
      for (size_t i = size_t(k) * 8; i < ids.size() && i < size_t(k) * 8 + 8; ++i) read(g, ids[i]);
    }
    if (stray_export >= 0) {  // an engine bug: a full export of gated layer `stray_export` inside the layer token
      ts::DgRecord& r = rec[uint32_t(stray_export)];
      r.n_ids = 1;
      r.ids[0] = 0;
      r.seq = seq;
      ts::dg_store_release(&flag[uint32_t(stray_export)], seq);
    }
    ts::dg_end_layer_flags(  // the end kernel (end_layer_kernel's own function)
        G, seq, [&](uint32_t h) { return ts::dg_load_acquire(&flag[h]); },
        [&](uint32_t h, uint32_t v) { ts::dg_store_release(&flag[h], v); },
        [&](uint32_t h) {
          if (err_code == ts::kDgOk) {
            err_code = ts::kDgDesync;
            err_aux = h;
          }
        });
    ts::dg_store_release(&done, seq);
    const DgService::State st = svc->wait_terminal(seq, std::chrono::milliseconds(5000));
    if (st != DgService::kServed) {
      if (why) *why = svc->error();
      return false;
    }
    CHECK(!aborted);
    core->end_token(steps.all_kept(), 32);
    core->check();
    return true;
  }
  // The keys of gated layer g's units on the device now, their slots.
  std::vector<std::pair<uint32_t, int32_t>> on_device(uint32_t g) const {
    std::vector<std::pair<uint32_t, int32_t>> v;
    for (uint32_t u = 0; u < U; ++u) {
      const uint32_t k = core->key(g, u);
      if (core->home(k) == DgCore::Home::Device) v.push_back({k, core->slot_of(k)});
    }
    return v;
  }
  // Every key of `held` still on the device in its slot (published: the table names that slot).
  bool still_held(const std::vector<std::pair<uint32_t, int32_t>>& held) const {
    for (const auto& kv : held)
      if (core->home(kv.first) != DgCore::Home::Device || core->slot_of(kv.first) != kv.second ||
          t->entry(kv.first) != kv.second)
        return false;
    return true;
  }
  std::vector<uint32_t> all() const {
    std::vector<uint32_t> v(U);
    for (uint32_t u = 0; u < U; ++u) v[u] = u;
    return v;
  }
  std::vector<int32_t> slots_of(uint32_t g) const {
    std::vector<int32_t> v;
    for (uint32_t k : steps.kept(g)) v.push_back(core->slot_of(k));
    return v;
  }
};

}  // namespace

// Per layer of a 4 x 288 model: make_resident(g, all 288) -> one step with rows only for g (g held: nothing exported)
// reading all 288 through the table -> release -> an all-layer decode step. Then two whole layers held at once through
// 40 decode steps of the other layers (their slots stable, sampled reads every step, all at the end), released, decode.
// Every read is exact, nothing is absent, every step is served.
TEST(dg_ctx_path_make_resident_prefill) {
  for (CtxRig::Mode mode : {CtxRig::kLru, CtxRig::kScored, CtxRig::kProduction}) {
    CtxRig r(mode);
    const uint64_t file0 = r.t->file_reads;
    const std::vector<uint32_t> every(CtxRig::G, 3);
    for (int i = 0; i < 4; ++i) r.step(every);
    for (uint32_t g = 0; g < CtxRig::G; ++g) {
      r.make_resident(g, r.all());
      for (uint32_t k : r.steps.kept(g)) CHECK(r.core->home(k) == DgCore::Home::Device);
      std::vector<uint32_t> rows(CtxRig::G, 0);
      rows[g] = 512;  // the prefill rows of g: g holds its set, so nothing is exported
      const uint32_t flag_g = r.flag[g];
      r.step(rows);
      CHECK_EQ(r.flag[g], flag_g);
      r.release(g);
      r.step(every);  // all-layer decode: g exported again
      CHECK_EQ(r.flag[g], r.seq);
    }
    // two whole layers held at once, prolonged
    r.make_resident(0, r.all());
    r.make_resident(2, r.all());
    const auto s0 = r.slots_of(0), s2 = r.slots_of(2);
    for (int tok = 0; tok < 40; ++tok) {
      r.step(every, 16);  // layers 1 and 3 decode; 0 and 2 read 16 sampled kept ids each
      CHECK(r.slots_of(0) == s0 && r.slots_of(2) == s2);
    }
    r.step(every);  // every kept id of both
    r.release(0);
    r.release(2);
    for (int i = 0; i < 6; ++i) r.step(every);
    const char* name = mode == CtxRig::kLru ? "lru" : mode == CtxRig::kScored ? "scored" : "auto+bounded+three-tier";
    std::printf("  %s: %llu reads, %llu wrong, %llu absent; resident units %llu, victims %llu, file reads %llu, "
                "scored victims %llu\n",
                name, (unsigned long long)r.reads, (unsigned long long)r.bad, (unsigned long long)r.absent,
                (unsigned long long)r.core->stats().resident_units,
                (unsigned long long)r.core->stats().resident_evicted, (unsigned long long)(r.t->file_reads - file0),
                (unsigned long long)r.core->stats().scored_victims);
    if (mode == CtxRig::kProduction) {
      CHECK(r.t->file_reads > file0);  // units came from their files (and others from pinned blocks)
      CHECK(r.core->stats().scored_victims > 0);
    }
    CHECK(r.reads > 6 * 288);
    CHECK_EQ(r.bad, 0ull);
    CHECK_EQ(r.absent, 0ull);
    CHECK(r.core->stats().resident_evicted > 0);
    CHECK_EQ(r.core->stats().score_fallbacks, 0ull);
  }
}

// The context's rules around held layers: the capacity counts every keep (g's own included); ss_dg_keep replaces a
// held set (g exported again); a step that exports a held layer anyway (an engine bug) shows at the next ss_begin (its
// flag no longer holds its last exporting sequence).
TEST(dg_ctx_path_held_layer_rules) {
  CtxRig r(CtxRig::kLru);
  const std::vector<uint32_t> every(CtxRig::G, 3);
  r.step(every);
  r.make_resident(1, r.all());
  r.make_resident(3, r.all());  // 576 kept + 64 spares <= 700
  bool cap = false;
  try {
    r.core->make_resident(0, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                              47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61},
                          r.steps.protect());  // 62 + 576 + 64 > 700
  } catch (const ts::Error& e) {
    cap = e.status() == SS_E_CAPACITY;
  }
  CHECK(cap);
  r.steps.keep(1, {r.core->key(1, 5)});  // a plain keep of 1 replaces the held set: exported again
  CHECK(!r.steps.resident(1));
  r.step(every);
  CHECK_EQ(r.flag[1], r.seq);
  r.release(1);
  r.release(3);
  r.make_resident(2, r.all());
  r.step(every, 4, 2);  // the engine exports held layer 2 anyway
  CHECK(!r.steps.flags_clean(r.flag.data(), ts::dg_load_acquire));
  CHECK_EQ(r.bad, 0ull);
}

// ---- layer tokens with kept units (ss_dg_keep) and held sets (ss_dg_make_resident), the context path on the CPU ----
namespace {
std::vector<uint32_t> layer_ids(std::mt19937& rng, uint32_t n) {
  std::vector<uint32_t> v(CtxRig::U);
  for (uint32_t u = 0; u < CtxRig::U; ++u) v[u] = u;
  std::shuffle(v.begin(), v.end(), rng);
  v.resize(n);
  return v;
}
std::vector<uint32_t> keys_of(const std::vector<std::pair<uint32_t, int32_t>>& held) {
  std::vector<uint32_t> v;
  for (const auto& kv : held) v.push_back(kv.first);
  return v;
}
size_t left_device(const CtxRig& r, const std::vector<std::pair<uint32_t, int32_t>>& held) {
  size_t n = 0;
  for (const auto& kv : held) n += r.core->home(kv.first) != DgCore::Home::Device ? 1u : 0u;
  return n;
}
}  // namespace

// Layer 0 held whole (make_resident) and 48 device units of layer 2 kept (ss_dg_keep) through layer tokens of layers 1
// and 3 that each route 256 of the 288 units (32 parts): the demands outgrow the free slots, so the tokens evict units
// of other layers mid-token, but never a kept or held one (each stays in its slot, its entry published), every read is
// exact. Then layer 2's keep is released mid-run: its units become victims again. Two-tier and three-tier (reads
// synchronous and in flight).
TEST(dg_ctx_path_layer_tokens_keep) {
  for (uint32_t disk = 0; disk <= 2; ++disk) {
    CtxRig r(false, disk);
    std::mt19937 rng(41 + disk);
    const std::vector<uint32_t> every(CtxRig::G, 3);
    for (int i = 0; i < 3; ++i) r.step(every);
    r.make_resident(0, r.all());
    const auto held0 = r.on_device(0);
    CHECK_EQ(held0.size(), size_t(CtxRig::U));
    auto kept2 = r.on_device(2);
    if (kept2.size() > 48) kept2.resize(48);
    CHECK(kept2.size() >= 24);
    r.steps.keep(2, keys_of(kept2));
    CHECK(!r.steps.plan_layer(0));  // a held layer is never exported: no layer token for it
    const uint64_t mid0 = r.core->stats().lm_mid_evictions;
    for (int round = 0; round < 3; ++round)
      for (uint32_t g : {1u, 3u}) {
        CHECK(r.layer_token(g, layer_ids(rng, 256)));
        CHECK(r.still_held(held0));
        CHECK(r.still_held(kept2));
      }
    const uint64_t mid1 = r.core->stats().lm_mid_evictions;
    CHECK(mid1 > mid0);
    r.step(every, 16);  // a decode step: 0 held (sampled kept reads), 2 exported with its keep
    CHECK(r.still_held(held0));
    CHECK(r.still_held(kept2));
    // the keep of layer 2 released between layer tokens: its units are victims like any other
    r.release(2);
    for (int round = 0; round < 2; ++round)
      for (uint32_t g : {1u, 3u}) {
        CHECK(r.layer_token(g, layer_ids(rng, 256)));
        CHECK(r.still_held(held0));
      }
    const size_t gone2 = left_device(r, kept2);
    CHECK(gone2 > 0);
    r.step(every);  // every kept id of layer 0 read
    r.release(0);
    for (int i = 0; i < 3; ++i) r.step(every);
    std::printf("  disk %u: %llu reads, %llu wrong, %llu absent; mid evictions %llu (with keeps %llu), released keep "
                "units evicted %zu",
                disk, (unsigned long long)r.reads, (unsigned long long)r.bad, (unsigned long long)r.absent,
                (unsigned long long)r.core->stats().lm_mid_evictions, (unsigned long long)(mid1 - mid0), gone2);
    std::putchar(10);
    CHECK_EQ(r.bad, 0ull);
    CHECK_EQ(r.absent, 0ull);
  }
}

// The test sees the rule: with the boundary's keep mutation (kIgnoreKeep) the same layer tokens evict kept or held
// units mid-token.
TEST(dg_ctx_path_layer_tokens_keep_mutation) {
  CtxRig r(false);
  std::mt19937 rng(41);
  const std::vector<uint32_t> every(CtxRig::G, 3);
  for (int i = 0; i < 3; ++i) r.step(every);
  r.make_resident(0, r.all());
  const auto held0 = r.on_device(0);
  auto kept2 = r.on_device(2);
  if (kept2.size() > 48) kept2.resize(48);
  r.steps.keep(2, keys_of(kept2));
  r.core->set_mutation(DgCore::kIgnoreKeep);
  CHECK(r.layer_token(1, layer_ids(rng, 256)));
  CHECK(!r.still_held(held0) || !r.still_held(kept2));
}

// No evictable unit: layers 0 and 2 held whole (576 kept units of 700 slots) and a layer token of layer 1 routing 256
// units. The token evicts layer 3's units, then finds only kept units and its own layer: the demand fails as a full
// token's does (no free slot, no bank slot, no write-back to wait for), the context's error path; nothing kept moved,
// the core's ownership consistent.
TEST(dg_ctx_path_layer_token_no_victim) {
  for (uint32_t disk = 0; disk <= 1; ++disk) {
    CtxRig r(false, disk);
    std::mt19937 rng(43);
    const std::vector<uint32_t> every(CtxRig::G, 3);
    for (int i = 0; i < 3; ++i) r.step(every);
    r.make_resident(0, r.all());
    r.make_resident(2, r.all());
    const auto held0 = r.on_device(0), held2 = r.on_device(2);
    std::string why;
    const bool served = r.layer_token(1, layer_ids(rng, 256), &why);
    if (!served) std::printf("  disk %u: %s", disk, why.c_str());
    std::putchar(10);
    CHECK(!served);
    CHECK(why.find("no free slot, no bank slot, no write-back to wait for") != std::string::npos);
    CHECK(r.still_held(held0));
    CHECK(r.still_held(held2));
    CHECK_EQ(r.bad, 0ull);
    r.svc->stop_join();
    r.core->check();
  }
}

// The context path: a stray full export of another gated layer inside a layer token whose demand forces mid-token
// evictions (two-tier and three-tier). The token itself is served (the service reads part words only) and the host's
// flag check before the next ss_begin passes (the end stamped every flag), so the end kernel's kDgDesync (err_aux =
// the stray layer; ss_end then fails the token and poisons the context) is the one place it shows. A clean layer token
// leaves the error word empty.
TEST(dg_ctx_path_layer_token_stray_export) {
  for (uint32_t disk = 0; disk <= 2; ++disk) {
    CtxRig r(false, disk);
    std::mt19937 rng(47 + disk);
    const std::vector<uint32_t> every(CtxRig::G, 3);
    for (int i = 0; i < 3; ++i) r.step(every);
    CHECK(r.layer_token(1, layer_ids(rng, 256)));
    CHECK_EQ(r.err_code, uint32_t(ts::kDgOk));
    const uint64_t mid0 = r.core->stats().lm_mid_evictions;
    CHECK(r.layer_token(2, layer_ids(rng, 256), nullptr, 0));  // layer 0 exported in full inside layer 2's token
    CHECK(r.core->stats().lm_mid_evictions > mid0);           // its readers' slots were in play
    CHECK_EQ(r.err_code, uint32_t(ts::kDgDesync));
    CHECK_EQ(r.err_aux, 0u);
    CHECK(r.steps.flags_clean(r.flag.data(), ts::dg_load_acquire));  // the host's check alone would not see it
    CHECK_EQ(r.bad, 0ull);
  }
}

// ss_dg_make_resident around layer tokens: a refused call (SS_E_CAPACITY) changes no keep and the next layer token
// still protects them; a call that fails on the device (an injected H2D failure) poisons the core: no layer token
// opens (set_layer_token refuses, nothing changed), the protected keep still in its slots.
TEST(dg_ctx_path_layer_tokens_make_resident) {
  CtxRig r(false);
  std::mt19937 rng(47);
  const std::vector<uint32_t> every(CtxRig::G, 3);
  for (int i = 0; i < 3; ++i) r.step(every);
  r.make_resident(0, r.all());
  r.make_resident(2, r.all());
  const auto held0 = r.on_device(0), held2 = r.on_device(2);
  bool cap = false;
  try {
    r.core->make_resident(1, r.all(), r.steps.protect());  // 288 + 576 + 64 > 700
  } catch (const ts::Error& e) {
    cap = e.status() == SS_E_CAPACITY;
  }
  CHECK(cap);
  r.release(2);
  CHECK(r.layer_token(3, layer_ids(rng, 200)));
  CHECK(r.layer_token(1, layer_ids(rng, 256)));
  CHECK(r.still_held(held0));
  // a device failure inside make_resident of layer 2 (its units copied in from their pinned blocks)
  r.t->fail_h2d = true;
  bool backend = false;
  try {
    r.core->make_resident(2, r.all(), r.steps.protect());
  } catch (const ts::Error& e) {
    backend = e.status() == SS_E_BACKEND;
  }
  r.t->fail_h2d = false;
  CHECK(backend);
  CHECK(r.core->poisoned());
  r.core->check();
  bool refused = false;
  try {
    r.core->set_layer_token(1, r.steps.all_kept());
  } catch (const ts::Error& e) {
    refused = e.status() == SS_E_BACKEND;
  }
  CHECK(refused);
  CHECK_EQ(r.core->layer_token(), 0xffffffffu);
  for (const auto& kv : held0) CHECK(r.core->home(kv.first) == DgCore::Home::Device);
  CHECK_EQ(r.bad, 0ull);
}
