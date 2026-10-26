// Device-gated service thread (src/dg_service.cpp) with a real thread: the test thread plays the device (the export,
// prediction and wait kernels and the routed reads of a captured step) against plain memory, the service runs the
// core through an immediate transport. Checks the token handoff (Served, then the engine owns the core), the bytes
// every read sees, and every failure path: a missing export or prediction, a desync, a bad id or count, a stop while
// idle or waiting. Under TSan the handoffs are checked too.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "dg_core.hpp"
#include "dg_proto.hpp"
#include "dg_service.hpp"
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
  void h2d(Stream, int32_t slot, int32_t block) override { slot_[size_t(slot)].store(block_[size_t(block)], std::memory_order_relaxed); }
  void d2h(int32_t slot, int32_t block) override { block_[size_t(block)] = slot_[size_t(slot)].load(std::memory_order_relaxed); }
  void d2h_part(Stream, int32_t slot, int32_t block, uint32_t) override { d2h(slot, block); }  // one part per unit
  uint32_t parts() const override { return 1; }
  uint64_t part_bytes(uint32_t) const override { return 8192; }
  void write_entry(Stream, uint32_t key, int32_t v) override { table_[key].store(v, std::memory_order_release); }
  void signal_ready(uint32_t g, uint32_t seq) override { ts::dg_store_release(&ready_[g], seq); }
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
  int32_t entry(uint32_t k) const { return table_[k].load(std::memory_order_acquire); }
  int32_t slot_holds(int32_t s) const { return slot_[size_t(s)].load(std::memory_order_relaxed); }
  void fill_slot(int32_t s, int32_t k) { slot_[size_t(s)].store(k); }
  void fill_block(int32_t b, int32_t k) { block_[size_t(b)] = k; }

 private:
  std::vector<std::atomic<int32_t>> table_, slot_;
  std::vector<int32_t> block_;
  uint32_t* ready_;
};

// A gated model of G layers x U units, its mapped area in plain memory, the core and the service.
struct Rig {
  static constexpr uint32_t G = 4, U = 32;
  std::vector<ts::DgRecord> rec{G};
  std::vector<ts::DgPredRecord> prec{G};
  std::vector<uint32_t> flag = std::vector<uint32_t>(G, 0), pflag = std::vector<uint32_t>(G, 0),
                        ready = std::vector<uint32_t>(G, 0);
  uint32_t done = 0, err_code = 0, err_aux = 0, host_abort = 0, began = 0;
  std::unique_ptr<Immediate> t;
  std::unique_ptr<DgCore> core;
  std::unique_ptr<DgService> svc;
  uint32_t seq = 0;

  // with_began: the service waits for the begin kernel's word before any flag (design rev 3.4 §11.2).
  // wb_defer: the boundary's write-backs are issued by the service after each served layer.
  explicit Rig(std::chrono::milliseconds timeout = std::chrono::milliseconds(2000), uint32_t max_ids = 8,
               bool with_began = false, bool wb_defer = false, bool scored = false) {
    ts::DgConfig c;
    c.n_units.assign(G, U);
    c.persistent_slots = 48 + 12;
    c.bank_slots = max_ids;
    c.blocks = G * U - 48 + 12;
    c.spare_target = 12;
    c.prefetch_reserve = 3;
    c.wb_defer = wb_defer;
    c.scored = scored;
    c.max_ids = max_ids;
    t = std::make_unique<Immediate>(G * U, c.persistent_slots + 2 * c.bank_slots, c.blocks, ready.data());
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
// a boundary issues what the token left (the engine thread then owns the deferred queue).
TEST(dg_service_tokens_wb_defer) {
  Rig r(std::chrono::milliseconds(2000), 8, false, true);
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
