// Exclusive tier mode on the CPU backend (the reference): every unit lives in a slot XOR a pinned block XOR only its
// file at each forward boundary, every table entry holds the unit's exact bytes whatever the history of promotions
// and write-backs, and a model covered by slots + blocks never reads its files after open.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <random>
#include <set>
#include <string>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

using sstest::Fixture;

// Test hook (capi.cpp, not in ss.h): the context's ownership check between forwards.
extern "C" ss_status ss_test_check_exclusive(ss_ctx* c);
extern "C" ss_status ss_test_ra_quiesce(ss_ctx* c);
extern "C" ss_status ss_test_hold_readers(ss_ctx* c, int hold);
extern "C" ss_status ss_test_inject(ss_ctx* c, int what);

namespace {

struct Run {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  ~Run() {
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
};

// Host strides of the fixture's units: layers 1/4 (11,000 B), layer 2 (8,192 B), layer 3 (1 MiB + 17 B).
constexpr uint64_t H1 = 12288, H2 = 8192, H3 = 1052672;

void budget(ss_budget& b, uint64_t host, uint32_t spares, int mode = SS_TIER_MODE_EXCLUSIVE) {
  ss_budget_defaults(&b);
  b.margin_gib = 1.0 / 1024;
  b.margin_fraction = 0;
  b.device_bytes = uint64_t(256) << 20;
  b.ring_slots = 32;  // deferred misses are served from the ring: room for a whole layer's routes
  b.host_pinned_bytes = host;
  b.tier_mode = mode;
  b.host_spare_units = spares;
}

std::string tiers_json(std::initializer_list<const char*> tiers) {
  std::string js = "{\"tiers\":[";
  bool first = true;
  for (const char* t : tiers) {
    js += first ? "\"" : ",\"";
    js += t;
    js += "\"";
    first = false;
  }
  return js + "]}";
}

bool open_run(Run& r, const Fixture& fx, std::initializer_list<const char*> tiers, ss_policy policy = SS_POLICY_LRU,
              int spec_persistent = 0, uint32_t discard = 0, uint32_t readahead = 0) {
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return false;
  std::string js = tiers_json(tiers);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return false;
  ss_options_defaults(&r.o);
  r.o.policy = policy;
  r.o.verify_hash = 1;
  r.o.pin_planned = 0;
  r.o.staging_bytes = 64 * 1024;
  r.o.spec_persistent = spec_persistent;
  r.o.exclusive_discard = discard;
  r.o.readahead_units = readahead;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  return r.c != nullptr;
}

bool same(const ss_layer_view& v, const Fixture& fx, int layer, int unit) {
  const auto& want = fx.expected[size_t(layer)][size_t(unit)];
  const void* a = v.table_host ? reinterpret_cast<const void*>(uintptr_t(v.table_host[unit])) : nullptr;
  if (!a) std::printf("  unit (%d, %d): not in the table\n", layer, unit);
  else if (std::memcmp(a, want.data(), want.size()) != 0) std::printf("  unit (%d, %d): bytes differ\n", layer, unit);
  return a && std::memcmp(a, want.data(), want.size()) == 0;
}

std::vector<int32_t> random_route(std::mt19937& g, int n_units, int top_k, int rows) {
  std::vector<int32_t> ids;
  std::uniform_int_distribution<int> hot(0, 3), any(0, n_units - 1), coin(0, 2);
  for (int r = 0; r < rows; ++r) {
    std::set<int> row;
    while (int(row.size()) < top_k) row.insert(coin(g) ? hot(g) : any(g));
    ids.insert(ids.end(), row.begin(), row.end());
  }
  return ids;
}

ss_exclusive_stats xstats(ss_ctx* c) {
  ss_exclusive_stats x{};
  CHECK_OK(ss_stats_exclusive(c, &x));
  return x;
}

// Decode loop over layers 1, 2, 4 (+ the blob layer 3), every routed unit's bytes checked, the ownership invariant
// checked after every forward.
void loop(Run& r, const Fixture& fx, int steps, bool speculative, uint32_t seed) {
  std::mt19937 g(seed);
  for (int step = 0; step < steps; ++step) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    for (int L : {1, 2, 4}) {
      const int E = L == 2 ? Fixture::kE2 : Fixture::kE1, k = L == 2 ? 2 : 4;
      auto ids = random_route(g, E, k, 1 + step % 3);
      if (speculative && L == 1) {
        int32_t pred[2] = {int32_t(step % 8), int32_t((step * 5) % 8)};
        CHECK_OK(ss_prefetch(r.c, 2, pred, 2, SS_PREFETCH_SPECULATIVE));
      }
      if (step % 5 == 3 && L == 4) {  // certain prefetch first: its admission may evict a unit routed just after
        int32_t pre[2] = {int32_t(4 + step % 12), int32_t(4 + (step * 7) % 12)};
        CHECK_OK(ss_prefetch(r.c, 4, pre, 2, SS_PREFETCH_CERTAIN));
      }
      CHECK_OK(ss_layer_routes(r.c, L, ids.data(), int(ids.size())));
      ss_layer_view v{};
      CHECK_OK(ss_acquire(r.c, L, nullptr, &v));
      std::set<int> distinct(ids.begin(), ids.end());
      for (int u : distinct) CHECK(same(v, fx, L, u));
      if (step % 9 == 4) CHECK_OK(ss_check_unit(r.c, L, *distinct.rbegin()));
      CHECK_OK(ss_release(r.c, L, nullptr));
    }
    CHECK_OK(ss_layer_all(r.c, 3));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 3, nullptr, &v));
    CHECK(same(v, fx, 3, 0));
    CHECK_OK(ss_release(r.c, 3, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK_OK(ss_test_check_exclusive(r.c));
  }
}

// Every unit resident or pinned (none on disk): slots + blocks cover the model.
const std::initializer_list<const char*> kCovered = {"", "RRRRPPPPPPPPPPPP", "RRRRPPPP", "P", "RRRRPPPPPPPPPPPP"};

void covered(const char* tag, ss_policy policy, bool speculative, int spec_persistent, uint32_t spares) {
  Fixture fx(tag);
  Run r;
  budget(r.b, 4 << 20, spares);
  if (!open_run(r, fx, kCovered, policy, spec_persistent)) return;
  CHECK_OK(ss_test_check_exclusive(r.c));
  ss_stats_reset(r.c);
  loop(r, fx, 60, speculative, 77);
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK_EQ(t.bytes_disk, uint64_t(0));  // covered: the files are read only at open
  CHECK(t.bytes_pinned > 0 && t.evicted > 0);
  ss_exclusive_stats x = xstats(r.c);
  CHECK(x.writebacks > 0 && x.promoted > 0);
  CHECK_EQ(x.writebacks, t.evicted);  // every eviction found a block (admission capped them at the free blocks)
  if (spares == 1) CHECK(x.deferred > 0);
  else CHECK_EQ(x.deferred, uint64_t(0));
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"exclusive\":{"));
  ss_free(js);
}

}  // namespace

TEST(exclusive_covered_lru) { covered("x-lru", SS_POLICY_LRU, false, 0, 0); }
TEST(exclusive_covered_scored_spec) { covered("x-scored", SS_POLICY_SCORED, true, 0, 0); }
TEST(exclusive_covered_spec_persistent) { covered("x-specp", SS_POLICY_LRU, true, 1, 0); }
TEST(exclusive_covered_one_spare_defers) { covered("x-one", SS_POLICY_LRU, true, 0, 1); }

TEST(exclusive_plan_billing_and_json) {
  Fixture fx("x-plan");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return;
  // Auto spares: min(64, units of the size) = 8 (layer 2), 32 (layers 1 + 4), 1 (layer 3).
  const uint64_t units = 24 * H1 + 4 * H2 + H3, spares = 8 * H2 + 32 * H1 + 1 * H3;
  budget(r.b, units + spares, 0);
  std::string js = tiers_json(kCovered);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return;
  ss_plan_totals t;
  ss_plan_get_totals(r.p, &t);
  CHECK_EQ(t.pinned_bytes, units + spares);
  char* out = nullptr;
  CHECK_OK(ss_plan_to_json(r.p, &out, nullptr));
  CHECK(out && std::strstr(out, "\"tier_mode\":\"exclusive\""));
  CHECK(out && std::strstr(out, "{\"asked\":32,\"block_bytes\":12288,\"blocks\":32}"));
  ss_free(out);
  ss_plan_free(r.p);
  r.p = nullptr;
  // Coverage first (review finding #2): a host budget that holds exactly the pinned units gives no spare, never a unit
  // moved to disk to make room for one.
  budget(r.b, units, 1);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return;
  ss_plan_get_totals(r.p, &t);
  CHECK_EQ(t.pinned_bytes, units);
  CHECK_EQ(t.n_disk, uint64_t(0));
  CHECK_OK(ss_plan_to_json(r.p, &out, nullptr));
  CHECK(out && std::strstr(out, "{\"asked\":1,\"block_bytes\":12288,\"blocks\":0}"));
  ss_free(out);
  ss_plan_free(r.p);
  r.p = nullptr;
  // An inclusive plan's JSON is today's: no tier_mode / spare_blocks keys, no spare bytes.
  budget(r.b, units + spares, 0, SS_TIER_MODE_INCLUSIVE);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return;
  ss_plan_get_totals(r.p, &t);
  CHECK_EQ(t.pinned_bytes, units);
  CHECK_OK(ss_plan_to_json(r.p, &out, nullptr));
  CHECK(out && !std::strstr(out, "tier_mode") && !std::strstr(out, "spare_blocks"));
  ss_free(out);
}

TEST(exclusive_refusals) {
  Fixture fx("x-refuse");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return;
  ss_plan* p = nullptr;
  budget(r.b, 4 << 20, 0);
  // Tier B is the second copy exclusive removes.
  std::string b = tiers_json({"", "BBBBPPPPPPPPPPPP", "RRRRPPPP", "P", "RRRRPPPPPPPPPPPP"});
  CHECK_EQ(int(ss_plan_from_json(r.d, &r.b, b.data(), b.size(), &p)), int(SS_E_INVALID));
  r.b.pinned_backing = 1;
  CHECK_EQ(int(ss_plan_make(r.d, &r.b, nullptr, 0, &p)), int(SS_E_INVALID));
  CHECK(std::strstr(ss_last_error(), "exclusive") != nullptr);
  r.b.pinned_backing = 0;
  r.b.tier_mode = 7;
  CHECK_EQ(int(ss_plan_make(r.d, &r.b, nullptr, 0, &p)), int(SS_E_INVALID));
  budget(r.b, 4 << 20, 0);
  std::string js = tiers_json(kCovered);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  ss_options_defaults(&r.o);
  r.o.fill_residents = 0;  // an evicted resident's slot is its only copy: it must be filled
  CHECK_EQ(int(ss_open(r.d, &r.b, r.p, &r.o, &r.c)), int(SS_E_INVALID));
  ss_options_defaults(&r.o);
  r.o.backend = SS_BACKEND_HIP;
  CHECK_EQ(int(ss_open(r.d, &r.b, r.p, &r.o, &r.c)), int(SS_E_UNSUPPORTED));
  CHECK(std::strstr(ss_last_error(), "discrete device") != nullptr);
  CHECK(r.c == nullptr);
}

// Review finding #2, execution: residents 0-1 of layer 2, pinned 2-7, ONE spare. Routing 2 and 3 together has one block for
// one victim: one is admitted (its victim written back), the other is served from the ring with its victim kept.
// Nothing is dropped to the file, so the next forwards (routing the victims) still read no file.
TEST(exclusive_one_spare_keeps_coverage) {
  Fixture fx("x-cover");
  Run r;
  budget(r.b, 4 << 20, 1);
  if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRPPPPPP", "P", "PPPPPPPPPPPPPPPP"})) return;
  ss_stats_reset(r.c);
  auto fwd = [&](std::vector<int32_t> ids) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_layer_routes(r.c, 2, ids.data(), int(ids.size())));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
    for (int32_t u : ids) CHECK(same(v, fx, 2, u));
    CHECK_OK(ss_release(r.c, 2, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK_OK(ss_test_check_exclusive(r.c));
  };
  fwd({2, 3});
  ss_exclusive_stats x = xstats(r.c);
  CHECK_EQ(x.writebacks, uint64_t(1));
  CHECK_EQ(x.deferred, uint64_t(1));
  fwd({0, 1});  // one of them was written back: read from its block, never the file
  fwd({0, 1, 2, 3});
  fwd({4, 5, 6, 7});
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK_EQ(t.bytes_disk, uint64_t(0));
  // No spare at all and full slots: the cache cannot adapt (every miss from the ring), but stays covered and exact.
  Fixture fz("x-zero");
  Run z;
  const uint64_t units = 32 * H1 + 6 * H2 + H3;
  budget(z.b, units, 1);  // spares asked, none fits
  if (!open_run(z, fz, {"", "PPPPPPPPPPPPPPPP", "RRPPPPPP", "P", "PPPPPPPPPPPPPPPP"})) return;
  CHECK_OK(ss_begin(z.c, nullptr, 0));
  int32_t ids[2] = {5, 6};
  CHECK_OK(ss_layer_routes(z.c, 2, ids, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(z.c, 2, nullptr, &v));
  CHECK(same(v, fz, 2, 5) && same(v, fz, 2, 6));
  CHECK_OK(ss_release(z.c, 2, nullptr));
  CHECK_OK(ss_end(z.c));
  CHECK_OK(ss_test_check_exclusive(z.c));
  x = xstats(z.c);
  CHECK_EQ(x.deferred, uint64_t(2));
  CHECK_EQ(x.writebacks, uint64_t(0));
}

// exclusive_discard, pinned budget full (no spare fits). Layer 2: residents 0-1, disk 2-3, pinned 4-7. Option off: the
// misses are deferred to the ring and the resident set never changes. On: the victims past the free blocks are
// discarded to their files and the misses take their slots; a later fetch of a discarded unit reads its file. Every
// table entry exact, the ownership invariant after every forward, the blocks keep their count.
namespace {
struct DiscardRun {
  Fixture fx;
  Run r;
  explicit DiscardRun(const char* tag, uint32_t discard, uint32_t readahead = 0) : fx(tag) {
    const uint64_t units = 32 * H1 + 4 * H2 + H3;
    budget(r.b, units, 1);  // a spare asked, none fits
    open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRDDPPPP", "P", "PPPPPPPPPPPPPPPP"}, SS_POLICY_LRU, 0, discard, readahead);
    if (r.c) ss_stats_reset(r.c);
  }
  // One forward routing `ids` on layer 2 (host read-ahead of `ahead` first, finished before the routes).
  void fwd(std::vector<int32_t> ids, std::vector<int32_t> ahead = {}) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    if (!ahead.empty()) {
      CHECK_OK(ss_host_prefetch(r.c, 2, ahead.data(), int(ahead.size())));
      CHECK_OK(ss_test_ra_quiesce(r.c));
    }
    CHECK_OK(ss_layer_routes(r.c, 2, ids.data(), int(ids.size())));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
    for (int32_t u : ids) CHECK(same(v, fx, 2, u));
    CHECK_OK(ss_check_unit(r.c, 2, ids[0]));
    CHECK_OK(ss_release(r.c, 2, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK_OK(ss_test_check_exclusive(r.c));
  }
  ss_layer_stats total() {
    ss_layer_stats t{};
    CHECK_OK(ss_stats_total(r.c, &t));
    return t;
  }
};
}  // namespace

TEST(exclusive_discard_off_keeps_todays_behaviour) {
  DiscardRun d("x-disc-off", 0);
  if (!d.r.c) return;
  for (int k = 0; k < 2; ++k) d.fwd({2, 5});  // both deferred both times: the set never adapts
  ss_layer_stats t = d.total();
  CHECK_EQ(t.admitted, uint64_t(0));
  CHECK_EQ(t.transient, uint64_t(4));
  CHECK_EQ(t.evicted, uint64_t(0));
  CHECK_EQ(t.bytes_disk, uint64_t(2 * 8192));
  CHECK_EQ(t.bytes_pinned, uint64_t(2 * 8192));
  ss_exclusive_stats x = xstats(d.r.c);
  CHECK_EQ(x.deferred, uint64_t(4));
  CHECK_EQ(x.writebacks, uint64_t(0));
  CHECK_EQ(x.discards, uint64_t(0));
  CHECK_EQ(x.free_blocks, uint64_t(0));
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(d.r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"discards\":0,"));
  ss_free(js);
}

TEST(exclusive_discard_full_budget_adapts) {
  DiscardRun d("x-disc-on", 1);
  if (!d.r.c) return;
  // 2 (disk) and 5 (pinned) are admitted; no block is free: the victims 0, 1 are discarded.
  d.fwd({2, 5});
  ss_layer_stats t = d.total();
  CHECK_EQ(t.admitted, uint64_t(2));
  CHECK_EQ(t.transient, uint64_t(0));
  CHECK_EQ(t.evicted, uint64_t(2));
  CHECK_EQ(t.bytes_disk, uint64_t(8192));
  CHECK_EQ(t.bytes_pinned, uint64_t(8192));
  ss_exclusive_stats x = xstats(d.r.c);
  CHECK_EQ(x.discards, uint64_t(2));
  CHECK_EQ(x.writebacks, uint64_t(0));
  CHECK_EQ(x.writeback_bytes, uint64_t(0));
  CHECK_EQ(x.deferred, uint64_t(0));
  CHECK_EQ(x.promoted, uint64_t(1));     // 5's block is free now (reusable from the next forward)
  CHECK_EQ(x.free_blocks, uint64_t(1));
  // The set adapted: 2 and 5 are hits now.
  d.fwd({2, 5});
  t = d.total();
  CHECK_EQ(t.hits, uint64_t(2));
  CHECK_EQ(t.bytes_disk, uint64_t(8192));
  // 0 and 1 are file-homed: read from their files. Victims 2, 5: one free block (5's) takes a write-back, the other
  // victim is discarded.
  d.fwd({0, 1});
  t = d.total();
  CHECK_EQ(t.bytes_disk, uint64_t(3 * 8192));
  x = xstats(d.r.c);
  CHECK_EQ(x.writebacks, uint64_t(1));
  CHECK_EQ(x.writeback_bytes, uint64_t(8192));
  CHECK_EQ(x.discards, uint64_t(3));
  CHECK_EQ(x.deferred, uint64_t(0));
  CHECK_EQ(x.free_blocks, uint64_t(0));  // layer 2's 4 blocks (its pinned units at open), all owned again
  CHECK_EQ(t.evicted, x.writebacks + x.discards);
  // Whatever lives where, the next forwards are exact (written-back from its block, discarded from its file).
  d.fwd({2, 5});
  d.fwd({3, 4, 6});  // two slots: two misses admitted (their victims leave), the third from the ring
  d.fwd({7, 0});
  t = d.total();
  x = xstats(d.r.c);
  CHECK_EQ(t.evicted, x.writebacks + x.discards);
  CHECK_EQ(t.transient, uint64_t(1));
  CHECK_EQ(x.deferred, uint64_t(0));
}

// exclusive_discard over the random decode loop (slots + blocks exactly cover the model, no spare): every demand
// admission proceeds (speculative ones keep the cap), evictions are write-backs or discards, every byte exact, the
// invariant after every forward.
TEST(exclusive_discard_loop) {
  for (int spec : {0, 1}) {
    Fixture fx(spec ? "x-disc-loop-spec" : "x-disc-loop");
    Run r;
    const uint64_t units = 24 * H1 + 4 * H2 + H3;
    budget(r.b, units, 1);
    if (!open_run(r, fx, kCovered, SS_POLICY_LRU, spec, 1)) return;
    ss_stats_reset(r.c);
    loop(r, fx, 60, spec != 0, 91);
    ss_layer_stats t;
    CHECK_OK(ss_stats_total(r.c, &t));
    ss_exclusive_stats x = xstats(r.c);
    CHECK(x.discards > 0);
    if (!spec) CHECK_EQ(x.deferred, uint64_t(0));  // speculative admissions keep the cap: only they may defer
    CHECK_EQ(t.evicted, x.writebacks + x.discards);
    CHECK(t.bytes_disk > 0);  // discarded units come back from their files
  }
}

// A discarded unit is file-homed: host read-ahead takes it, and its admission lands from the buffer (the landing's
// victim is discarded too: no write-back to wait for).
TEST(exclusive_discard_then_readahead) {
  DiscardRun d("x-disc-ra", 1, 2);
  if (!d.r.c) return;
  d.fwd({2, 3});  // victims 0, 1 discarded
  d.fwd({0, 1}, {0, 1});
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(d.r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"landed\":2,"));
  ss_free(js);
  ss_layer_stats t = d.total();
  CHECK_EQ(t.bytes_disk, uint64_t(4 * 8192));  // 2, 3 by ordinary reads; 0, 1 once each by their read-ahead
  ss_exclusive_stats x = xstats(d.r.c);
  CHECK_EQ(x.discards, uint64_t(4));
  CHECK_EQ(x.writebacks, uint64_t(0));
}

// exclusive_discard refused with a descriptor whose tiers are "pinned" (its units must not live on disk); a speculative
// admission keeps the cap under discard (a guess defers, never discards).
TEST(exclusive_discard_refusal_and_speculative_cap) {
  Fixture fx("x-disc-refuse");
  std::string pinned_only = fx.json;
  const std::string from = "\"tiers\":\"pinned|disk\"";
  const size_t at = pinned_only.find(from);
  CHECK(at != std::string::npos);
  if (at == std::string::npos) return;
  pinned_only.replace(at, from.size(), "\"tiers\":\"pinned\"");
  Run r;
  CHECK_OK(ss_desc_from_json(pinned_only.data(), pinned_only.size(), &r.d));
  if (!r.d) return;
  budget(r.b, 4 << 20, 0);
  std::string js = tiers_json(kCovered);
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return;
  ss_options_defaults(&r.o);
  r.o.pin_planned = 0;
  r.o.exclusive_discard = 1;
  CHECK_EQ(int(ss_open(r.d, &r.b, r.p, &r.o, &r.c)), int(SS_E_INVALID));
  CHECK(std::strstr(ss_last_error(), "exclusive_discard") != nullptr);
  CHECK(r.c == nullptr);
  r.o.exclusive_discard = 0;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  // Speculative cap: full budget (no free block), spec_persistent, discard on. A speculative prefetch of 5 defers (no
  // victim discarded, 5 goes to the ring); the demand routes then admit 6, discarding a victim.
  Fixture fs("x-disc-spec");
  Run s;
  budget(s.b, 32 * H1 + 4 * H2 + H3, 1);
  if (!open_run(s, fs, {"", "PPPPPPPPPPPPPPPP", "RRDDPPPP", "P", "PPPPPPPPPPPPPPPP"}, SS_POLICY_LRU, 1, 1)) return;
  CHECK_OK(ss_begin(s.c, nullptr, 0));
  int32_t five[1] = {5};
  CHECK_OK(ss_prefetch(s.c, 2, five, 1, SS_PREFETCH_SPECULATIVE));
  ss_exclusive_stats x = xstats(s.c);
  CHECK_EQ(x.deferred, uint64_t(1));
  CHECK_EQ(x.discards, uint64_t(0));
  int32_t ids[2] = {5, 6};
  CHECK_OK(ss_layer_routes(s.c, 2, ids, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(s.c, 2, nullptr, &v));
  CHECK(same(v, fs, 2, 5) && same(v, fs, 2, 6));
  CHECK_OK(ss_release(s.c, 2, nullptr));
  CHECK_OK(ss_end(s.c));
  CHECK_OK(ss_test_check_exclusive(s.c));
  x = xstats(s.c);
  CHECK_EQ(x.deferred, uint64_t(1));
  CHECK_EQ(x.discards, uint64_t(1));
}

// A speculative persistent admission (spec_persistent) of a unit with a Ready read-ahead entry abandons it and reads
// the file (speculation stays off the buffers, as in the ring): landed 0, unused 1, the file read twice.
TEST(exclusive_spec_persistent_abandons_readahead) {
  Fixture fx("x-spec-ra");
  Run r;
  budget(r.b, 4 << 20, 0);
  if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRDDPPPP", "P", "PPPPPPPPPPPPPPPP"}, SS_POLICY_LRU, 1, 0, 2)) return;
  ss_stats_reset(r.c);
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t two[1] = {2};
  CHECK_OK(ss_host_prefetch(r.c, 2, two, 1));
  CHECK_OK(ss_test_ra_quiesce(r.c));  // Ready
  CHECK_OK(ss_prefetch(r.c, 2, two, 1, SS_PREFETCH_SPECULATIVE));  // admitted (0 written back), the entry abandoned
  CHECK_OK(ss_layer_routes(r.c, 2, two, 1));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 2));
  CHECK_OK(ss_check_unit(r.c, 2, 2));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  CHECK_OK(ss_test_check_exclusive(r.c));
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"landed\":0,") && std::strstr(js, "\"unused\":1}"));
  ss_free(js);
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK_EQ(t.spec_used, uint64_t(1));
  CHECK_EQ(t.bytes_disk, uint64_t(2 * 8192));  // the read-ahead read and the ordinary read
  CHECK_EQ(xstats(r.c).writebacks, uint64_t(1));
}

// Failures of a persistent landing (exclusive, into a written-back victim's slot): the read fails with the landing
// parked on it (5, readers held), the landing copy fails before any copy (3) or after its first piece (4, the buffer
// quarantined). The forward fails, the unit is dropped (file-homed), the victim lives in its block, the next forward
// is exact.
TEST(exclusive_persistent_landing_failures) {
  for (int inj : {5, 3, 4}) {
    Fixture fx(std::string("x-land-fail-") + std::to_string(inj));
    Run r;
    budget(r.b, 4 << 20, 0);
    if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRDDPPPP", "P", "PPPPPPPPPPPPPPPP"}, SS_POLICY_LRU, 0, 0, 2)) return;
    ss_stats_reset(r.c);
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    int32_t two[1] = {2};
    if (inj == 5) {  // parked: the read is still queued when the slot is given
      CHECK_OK(ss_test_hold_readers(r.c, 1));
      CHECK_OK(ss_host_prefetch(r.c, 2, two, 1));
      CHECK_OK(ss_test_inject(r.c, 5));
      CHECK_OK(ss_layer_routes(r.c, 2, two, 1));
      CHECK_OK(ss_test_hold_readers(r.c, 0));
    } else {  // Ready
      CHECK_OK(ss_host_prefetch(r.c, 2, two, 1));
      CHECK_OK(ss_test_ra_quiesce(r.c));
      CHECK_OK(ss_test_inject(r.c, inj));
      CHECK_OK(ss_layer_routes(r.c, 2, two, 1));
    }
    ss_layer_view v{};
    CHECK_EQ(int(ss_acquire(r.c, 2, nullptr, &v)), int(SS_E_IO));
    CHECK(ss_end(r.c) != SS_OK);
    CHECK_OK(ss_test_check_exclusive(r.c));
    CHECK_EQ(xstats(r.c).writebacks, uint64_t(1));
    // 2 was dropped (not resident): fetched again; 0 comes back from its block.
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    int32_t ids[2] = {0, 2};
    CHECK_OK(ss_layer_routes(r.c, 2, ids, 2));
    CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
    CHECK(same(v, fx, 2, 0) && same(v, fx, 2, 2));
    CHECK_OK(ss_check_unit(r.c, 2, 2));
    CHECK_OK(ss_release(r.c, 2, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK_OK(ss_test_check_exclusive(r.c));
    ss_layer_stats t;
    CHECK_OK(ss_stats_total(r.c, &t));
    CHECK_EQ(t.hits, uint64_t(0));  // 2 did not stay resident after its failed landing
  }
}

// ABI: an old caller's ss_options (struct_size up to dg_flush_ready) with garbage after it: exclusive_discard reads as
// 0, so a full pinned budget defers as before.
TEST(exclusive_discard_old_options_read_as_off) {
  static_assert(offsetof(ss_options, exclusive_discard) == offsetof(ss_options, dg_prior) + sizeof(const void*),
                "exclusive_discard is the first field after the window-prefetch prior: the old struct ends there");
  const uint32_t old_size = uint32_t(offsetof(ss_options, exclusive_discard));
  Fixture fx("x-disc-abi");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return;
  budget(r.b, 32 * H1 + 4 * H2 + H3, 1);
  std::string js = tiers_json({"", "PPPPPPPPPPPPPPPP", "RRDDPPPP", "P", "PPPPPPPPPPPPPPPP"});
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return;
  unsigned char raw[sizeof(ss_options)];
  std::memset(raw, 0xA5, sizeof raw);
  ss_options_defaults_sized(reinterpret_cast<ss_options*>(raw), old_size);
  ss_options* old = reinterpret_cast<ss_options*>(raw);
  CHECK_EQ(old->struct_size, old_size);
  CHECK_EQ(old->exclusive_discard, 0xA5A5A5A5u);  // untouched: past the caller's struct
  old->verify_hash = 1;
  old->pin_planned = 0;
  old->staging_bytes = 64 * 1024;
  CHECK_OK(ss_open(r.d, &r.b, r.p, old, &r.c));
  if (!r.c) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t ids[2] = {2, 5};
  CHECK_OK(ss_layer_routes(r.c, 2, ids, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 2) && same(v, fx, 2, 5));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  ss_exclusive_stats x = xstats(r.c);
  CHECK_EQ(x.deferred, uint64_t(2));
  CHECK_EQ(x.discards, uint64_t(0));
}

// Certain prefetch of layer 2 evicts resident 0 (LRU, lowest id), then layer 2's routes route 0: it is read back from
// the block its write-back of this same forward targets.
TEST(exclusive_evicted_then_routed_same_forward) {
  Fixture fx("x-same");
  Run r;
  budget(r.b, 4 << 20, 0);
  if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRRRPPPP", "P", "PPPPPPPPPPPPPPPP"})) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t pre[1] = {5};
  CHECK_OK(ss_prefetch(r.c, 2, pre, 1, SS_PREFETCH_CERTAIN));
  int32_t ids[2] = {0, 5};
  CHECK_OK(ss_layer_routes(r.c, 2, ids, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 0) && same(v, fx, 2, 5));
  CHECK_OK(ss_check_unit(r.c, 2, 0));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  CHECK_OK(ss_test_check_exclusive(r.c));
  ss_exclusive_stats x = xstats(r.c);
  CHECK_EQ(x.writebacks, uint64_t(2));  // 0 for 5, then another resident for 0
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK_EQ(t.bytes_disk, uint64_t(0));
}

// Disk units in exclusive mode: admitted from the file, written back into a free block when evicted (they migrate to
// the pinned tier); a failed file read leaves the unit file-homed and the context consistent; seeds free blocks.
TEST(exclusive_disk_units_seed_and_failed_read) {
  Fixture fx("x-disk");
  Run r;
  budget(r.b, 4 << 20, 0);
  if (!open_run(r, fx, {"", "RRRRDDDDDDDDDDDD", "RRRRPPPP", "P", "RRRRPPPPPPPPPPPP"})) return;
  int32_t seed_ids[2] = {4, 5};
  CHECK_OK(ss_seed(r.c, 2, seed_ids, 2));  // no free slot (4 residents, no dynamic slots): nothing placed
  CHECK_OK(ss_test_check_exclusive(r.c));
  auto fwd1 = [&](std::vector<int32_t> ids, ss_status want_acquire) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_layer_routes(r.c, 1, ids.data(), int(ids.size())));
    ss_layer_view v{};
    CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(want_acquire));
    if (want_acquire == SS_OK) {
      for (int32_t u : ids) CHECK(same(v, fx, 1, u));
      CHECK_OK(ss_release(r.c, 1, nullptr));
      CHECK_OK(ss_end(r.c));
    } else {
      CHECK(ss_end(r.c) != SS_OK);
    }
    CHECK_OK(ss_test_check_exclusive(r.c));
  };
  fwd1({6, 7}, SS_OK);        // from the files; residents 0, 1 (LRU, lowest ids) written back into spare blocks
  fwd1({0, 1, 2, 3}, SS_OK);  // 0, 1 back from their blocks; the victims 6, 7 are written back: migrated to blocks
  ss_layer_stats before;
  CHECK_OK(ss_stats_total(r.c, &before));
  fwd1({6, 7}, SS_OK);    // now from blocks, not files
  ss_layer_stats after;
  CHECK_OK(ss_stats_total(r.c, &after));
  CHECK_EQ(after.bytes_disk, before.bytes_disk);
  CHECK(after.bytes_pinned > before.bytes_pinned);
  fwd1({3, 8}, SS_OK);
  fwd1({9, 3}, SS_OK);
  fwd1({10, 11, 12, 13}, SS_OK);
  fwd1({3}, SS_OK);
  // A disk unit with a bad hash: the read fails, the unit stays file-homed, the context stays consistent.
  Fixture fb("x-diskbad", /*corrupt_hash=*/true);
  Run q;
  budget(q.b, 4 << 20, 0);
  if (!open_run(q, fb, {"", "RRDDDDDDDDDDDDDD", "RRRRPPPP", "P", "RRRDDDDDDDDDDDDD"})) return;
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  int32_t bad[2] = {3, 4};
  CHECK_OK(ss_layer_routes(q.c, 1, bad, 2));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(q.c, 1, nullptr, &v)), int(SS_E_HASH));
  CHECK(ss_end(q.c) != SS_OK);
  CHECK_OK(ss_test_check_exclusive(q.c));
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  int32_t good[1] = {4};
  CHECK_OK(ss_layer_routes(q.c, 1, good, 1));
  CHECK_OK(ss_acquire(q.c, 1, nullptr, &v));
  CHECK(same(v, fb, 1, 4));
  CHECK_OK(ss_release(q.c, 1, nullptr));
  CHECK_OK(ss_end(q.c));
  CHECK_OK(ss_test_check_exclusive(q.c));
}

// Every admission's deferrals are counted, also a prefetch admission that admitted nothing (review finding #2).
TEST(exclusive_prefetch_deferrals_counted) {
  for (int spec : {0, 1}) {
    Fixture fx(spec ? "x-pdefs" : "x-pdef");
    Run r;
    budget(r.b, 4 << 20, 1);
    if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRRRPPPP", "P", "PPPPPPPPPPPPPPPP"}, SS_POLICY_LRU, spec)) return;
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    int32_t pre[2] = {5, 6};
    CHECK_OK(ss_prefetch(r.c, 2, pre, 2, spec ? SS_PREFETCH_SPECULATIVE : SS_PREFETCH_CERTAIN));
    ss_exclusive_stats x = xstats(r.c);
    if (spec) {
      CHECK_EQ(x.writebacks, uint64_t(0));  // the one spare is the demand reserve
      CHECK_EQ(x.deferred, uint64_t(2));
    } else {
      CHECK_EQ(x.writebacks, uint64_t(1));
      CHECK_EQ(x.deferred, uint64_t(1));
    }
    int32_t ids[2] = {5, 6};
    CHECK_OK(ss_layer_routes(r.c, 2, ids, 2));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
    CHECK(same(v, fx, 2, 5) && same(v, fx, 2, 6));
    CHECK_OK(ss_release(r.c, 2, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK_OK(ss_test_check_exclusive(r.c));
  }
}

// Seeds with free dynamic slots: a seeded pinned unit leaves its block at once.
TEST(exclusive_seed_frees_blocks) {
  Fixture fx("x-seed");
  Run r;
  budget(r.b, 4 << 20, 0);
  r.b.dynamic_slots_per_layer = 2;
  if (!open_run(r, fx, {"", "PPPPPPPPPPPPPPPP", "RRRRPPPP", "P", "PPPPPPPPPPPPPPPP"})) return;
  int32_t seed_ids[3] = {5, 9, 9};
  CHECK_OK(ss_seed(r.c, 1, seed_ids, 3));
  CHECK_OK(ss_test_check_exclusive(r.c));
  ss_exclusive_stats x = xstats(r.c);
  CHECK_EQ(x.promoted, uint64_t(2));
  loop(r, fx, 12, false, 5);
}

// ABI: an old caller's ss_budget (struct_size 64 on 64-bit, before reserved0) with garbage in its padding stays
// inclusive (review finding #1).
TEST(exclusive_old_budget_padding_is_ignored) {
  static_assert(sizeof(void*) != 8 || offsetof(ss_budget, reserved0) == 60, "old struct end");
  static_assert(sizeof(void*) != 8 || offsetof(ss_budget, tier_mode) == 64, "first new field past the old size");
  Fixture fx("x-abi");
  ss_desc* d = nullptr;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &d));
  if (!d) return;
  unsigned char raw[sizeof(ss_budget)];
  std::memset(raw, 0xA5, sizeof raw);
  ss_budget_defaults_sized(reinterpret_cast<ss_budget*>(raw), uint32_t(offsetof(ss_budget, reserved0) + 4));
  ss_budget* old = reinterpret_cast<ss_budget*>(raw);
  old->device_bytes = uint64_t(256) << 20;
  old->host_pinned_bytes = 4 << 20;
  old->margin_gib = 1.0 / 1024;
  old->reserved0 = 1;  // padding in the old layout: must not read as EXCLUSIVE
  ss_plan* p = nullptr;
  CHECK_OK(ss_plan_make(d, old, nullptr, 0, &p));
  char* out = nullptr;
  if (p) CHECK_OK(ss_plan_to_json(p, &out, nullptr));
  CHECK(out && !std::strstr(out, "tier_mode"));
  ss_free(out);
  ss_plan_free(p);
  ss_desc_free(d);
}

int main() { return sstest::run_all(); }
