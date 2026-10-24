// CPU runtime tests (host-memory backend, unified like Strix Halo): every unit the table points to holds exactly the
// engine bytes the descriptor names, whatever the tier (resident, pinned, disk), the policy, the prefetch kind and
// the read path (in place / staging / chunked).
#include <algorithm>
#include <cstring>
#include <random>
#include <set>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

using sstest::Fixture;

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

// Unit strides (unit_align 256) of layers 1/4, 2 and 3.
constexpr uint64_t S1 = 11008, S3 = 1048832;

// Generous device budget (the placement is given explicitly); margin 1 MiB.
void budget(ss_budget& b, uint32_t ring, uint64_t pinned, uint32_t dynamic = 0) {
  ss_budget_defaults(&b);
  b.margin_gib = 1.0 / 1024;
  b.margin_fraction = 0;
  b.device_bytes = uint64_t(256) << 20;
  b.ring_slots = ring;
  b.host_pinned_bytes = pinned;
  b.dynamic_slots_per_layer = dynamic;
}

// Placement as plan-v1 tier letters per layer (R resident, P pinned, D disk), through ss_plan_from_json.
void plan(Run& r, std::initializer_list<const char*> tiers) {
  std::string js = "{\"tiers\":[";
  bool first = true;
  for (const char* t : tiers) {
    js += first ? "\"" : ",\"";
    js += t;
    js += "\"";
    first = false;
  }
  js += "]}";
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
}

bool same(const ss_layer_view& v, const Fixture& fx, int layer, int unit) {
  const auto& want = fx.expected[size_t(layer)][size_t(unit)];
  if (!v.table_host) return false;
  const void* a = reinterpret_cast<const void*>(uintptr_t(v.table_host[unit]));
  if (!a) std::printf("  unit (%d, %d): not in the table\n", layer, unit);
  else if (std::memcmp(a, want.data(), want.size()) != 0) std::printf("  unit (%d, %d): bytes differ\n", layer, unit);
  return a && std::memcmp(a, want.data(), want.size()) == 0;
}

std::vector<int32_t> random_route(std::mt19937& g, int n_units, int top_k, int rows) {
  std::vector<int32_t> ids;
  std::uniform_int_distribution<int> hot(0, 3), any(0, n_units - 1), coin(0, 2);
  for (int r = 0; r < rows; ++r) {
    std::set<int> row;
    while (int(row.size()) < top_k) row.insert(coin(g) ? hot(g) : any(g));  // skewed: 0-3 are hot
    ids.insert(ids.end(), row.begin(), row.end());
  }
  return ids;
}

void check_stats_balance(ss_ctx* c, int layer) {
  ss_layer_stats s;
  CHECK_OK(ss_stats_layer(c, layer, &s));
  // Every routed unit was a hit, an admission, a transient read, or issued earlier by a prefetch.
  CHECK_EQ(s.hits + s.admitted + s.transient + s.prefetched + s.spec_used, s.routed);
}

void decode_loop(const Fixture& fx, ss_policy policy, bool speculative, int direct, uint64_t staging) {
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return;
  // Residents 0-3 of layers 1, 2 and 4, some pinned, the rest from the files; 2 dynamic slots per layer; 6 ring
  // slots per unit size.
  budget(r.b, 6, 1 << 20, 2);
  plan(r, {"", "RRRRPPPPPPDDDDDD", "RRRRPPDD", "D", "RRRRDDDDDDDDDDDD"});
  if (!r.p) return;
  ss_options_defaults(&r.o);
  r.o.policy = policy;
  r.o.verify_hash = 1;
  r.o.pin_planned = 0;
  r.o.direct_io = direct;
  r.o.staging_bytes = staging;
  r.o.group_units = 2;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  std::mt19937 g(1234);
  for (int step = 0; step < 40; ++step) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    for (int L : {1, 2, 4}) {
      int E = L == 2 ? Fixture::kE2 : Fixture::kE1, k = L == 2 ? 2 : 4;
      auto ids = random_route(g, E, k, 1 + step % 3);
      if (speculative && L == 1) {
        // Lookahead for layer 2 issued before layer 1 runs: its true route half the time, noise otherwise.
        int32_t pred[2] = {int32_t(step % 8), int32_t((step * 5) % 8)};
        CHECK_OK(ss_prefetch(r.c, 2, pred, 2, SS_PREFETCH_SPECULATIVE));
      }
      CHECK_OK(ss_layer_routes(r.c, L, ids.data(), int(ids.size())));
      ss_layer_view v{};
      if (step % 2) {
        CHECK_OK(ss_acquire(r.c, L, nullptr, &v));
      } else {
        CHECK_OK(ss_acquire_groups(r.c, L, nullptr, &v));
        for (int gi = 0; gi < v.n_groups; ++gi) CHECK_OK(ss_group_wait(r.c, L, gi, nullptr));
      }
      std::set<int> distinct(ids.begin(), ids.end());
      int grouped = 0;
      for (int gi = 0; gi < v.n_groups; ++gi) grouped += v.groups[gi].n_units;
      CHECK_EQ(grouped, int(distinct.size()));
      for (int u : distinct) CHECK(same(v, fx, L, u));
      if (step == 7) CHECK_OK(ss_check_unit(r.c, L, *distinct.begin()));
      CHECK_OK(ss_release(r.c, L, nullptr));
    }
    CHECK_OK(ss_layer_all(r.c, 3));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 3, nullptr, &v));
    CHECK(same(v, fx, 3, 0));
    CHECK_OK(ss_release(r.c, 3, nullptr));
    CHECK_OK(ss_end(r.c));
  }
  for (int L = 1; L <= 4; ++L) check_stats_balance(r.c, L);
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK(t.hits > 0);
  CHECK(t.bytes_disk > 0);
  CHECK(t.bytes_pinned > 0);
  if (speculative) CHECK(t.spec_issued > 0 && t.spec_used + t.spec_dropped <= t.spec_issued);
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"schema\":\"strix-stream/stats-v1\""));
  ss_free(js);
}

}  // namespace

TEST(decode_lru_direct_staging) { decode_loop(Fixture("lru"), SS_POLICY_LRU, false, 1, 64 * 1024); }
TEST(decode_scored_spec) { decode_loop(Fixture("scored"), SS_POLICY_SCORED, true, 1, 64 * 1024); }
TEST(decode_lru_spec_buffered_big_staging) { decode_loop(Fixture("buf"), SS_POLICY_LRU, true, 0, 4 << 20); }

TEST(prefill_layer_major_static_residents) {
  // Prefill shape: planner residents pinned, no dynamic slots, every miss from a ring; whole-layer prefetch of later
  // layers while the current one runs. Layers 1 and 4 share the 11,008-B ring (12 slots for 2 x 8 streamed units):
  // layer 4's prefetch waits for slots until layer 1 releases. rows_per_layer skips layer 2 (a cone).
  Fixture fx("prefill");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  budget(r.b, 12, 0);
  plan(r, {"", "RRRRRRRRDDDDDDDD", "RRRRRRRR", "D", "RRRRRRRRDDDDDDDD"});
  ss_plan_totals tot{};
  ss_plan_get_totals(r.p, &tot);
  CHECK_EQ(tot.n_pinned, uint64_t(0));
  CHECK_EQ(tot.n_resident, uint64_t(24));
  CHECK_EQ(tot.ring_bytes, uint64_t(12) * (S1 + S3));
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  uint32_t rows[5] = {512, 512, 0, 512, 512};
  for (int pass = 0; pass < 3; ++pass) {
    CHECK_OK(ss_begin(r.c, rows, 5));
    CHECK_OK(ss_prefetch_layer(r.c, 1));
    CHECK_OK(ss_prefetch_layer(r.c, 3));
    CHECK_OK(ss_prefetch_layer(r.c, 4));  // 4 of its 8 wait for ring slots
    CHECK_OK(ss_layer_all(r.c, 1));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
    for (int u = 0; u < Fixture::kE1; ++u) CHECK(same(v, fx, 1, u));
    CHECK_EQ(v.n_groups, 2);
    CHECK_EQ(v.groups[0].n_units, 8);
    CHECK_EQ(int(ss_layer_routes(r.c, 2, nullptr, 0)), int(SS_E_INVALID));  // skipped layer
    CHECK_OK(ss_release(r.c, 1, nullptr));
    CHECK_OK(ss_layer_all(r.c, 3));
    CHECK_OK(ss_acquire(r.c, 3, nullptr, &v));
    CHECK(same(v, fx, 3, 0));
    CHECK_OK(ss_release(r.c, 3, nullptr));
    CHECK_OK(ss_layer_all(r.c, 4));
    CHECK_OK(ss_acquire(r.c, 4, nullptr, &v));
    for (int u = 0; u < Fixture::kE1; ++u) CHECK(same(v, fx, 4, u));
    CHECK_OK(ss_release(r.c, 4, nullptr));
    CHECK_OK(ss_end(r.c));
  }
  for (int L : {1, 4}) {
    ss_layer_stats s;
    CHECK_OK(ss_stats_layer(r.c, L, &s));
    CHECK_EQ(s.admitted, uint64_t(0));
    CHECK_EQ(s.hits, uint64_t(3 * 8));
    CHECK_EQ(s.prefetched, uint64_t(3 * 8));
    CHECK_EQ(s.bytes_disk, uint64_t(3 * 8 * 11000));
    check_stats_balance(r.c, L);
  }
}

TEST(ring_too_small_is_an_error_not_a_hang) {
  Fixture fx("ring");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  budget(r.b, 4, 0);
  plan(r, {"", "RRDDDDDDDDDDDDDD", "DDDDDDDD", "D", "DDDDDDDDDDDDDDDD"});
  ss_options_defaults(&r.o);
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_layer_all(r.c, 1));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_CAPACITY));
  CHECK(std::strstr(ss_last_error(), "ring") != nullptr);
  CHECK_OK(ss_end(r.c));
}

TEST(hash_mismatch_is_reported) {
  Fixture fx("hash", /*corrupt_hash=*/true);
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  budget(r.b, 8, 0);  // nothing resident: unit 3 of layer 1 is read at its first use
  plan(r, {"", "DDDDDDDDDDDDDDDD", "DDDDDDDD", "D", "DDDDDDDDDDDDDDDD"});
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t ids[2] = {2, 3};
  CHECK_OK(ss_layer_routes(r.c, 1, ids, 2));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_HASH));
  CHECK(ss_end(r.c) != SS_OK);  // the failed read is reported again at the end of the forward
  // The next forward works for the healthy units.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_layer_routes(r.c, 1, ids, 1));
  CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
  CHECK(same(v, fx, 1, 2));
  CHECK_OK(ss_release(r.c, 1, nullptr));
  CHECK_OK(ss_end(r.c));
}

TEST(seed_and_prefetch_between_forwards) {
  Fixture fx("seed");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  budget(r.b, 4, 0, 4);
  plan(r, {"", "DDDDDDDDDDDDDDDD", "DDDDDDDD", "D", "DDDDDDDDDDDDDDDD"});
  ss_options_defaults(&r.o);
  r.o.pin_planned = 0;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  int32_t seed[3] = {5, 9, 9};
  CHECK_OK(ss_seed(r.c, 1, seed, 3));
  // Idle preload of the next forward's layer 2 (certain), then the forward itself.
  int32_t next[2] = {1, 6};
  CHECK_OK(ss_prefetch(r.c, 2, next, 2, SS_PREFETCH_CERTAIN));
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t r1[2] = {5, 9};
  CHECK_OK(ss_layer_routes(r.c, 1, r1, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
  CHECK(same(v, fx, 1, 5) && same(v, fx, 1, 9));
  CHECK_EQ(v.groups[0].n_units, 2);  // both were resident before the forward
  CHECK_OK(ss_release(r.c, 1, nullptr));
  CHECK_OK(ss_layer_routes(r.c, 2, next, 2));
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 1) && same(v, fx, 2, 6));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  ss_layer_stats s;
  CHECK_OK(ss_stats_layer(r.c, 1, &s));
  CHECK_EQ(s.hits, uint64_t(2));
}

TEST(call_order_errors) {
  Fixture fx("order");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  budget(r.b, 0, 0);  // no plan given: ss_open plans from the budget (everything fits)
  CHECK_OK(ss_open(r.d, &r.b, nullptr, nullptr, &r.c));
  if (!r.c) return;
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_INVALID));  // outside a forward
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_EQ(int(ss_begin(r.c, nullptr, 0)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_INVALID));  // no routes
  int32_t ids[1] = {99};
  CHECK_EQ(int(ss_layer_routes(r.c, 1, ids, 1)), int(SS_E_INVALID));  // out of range
  CHECK_EQ(int(ss_layer_routes(r.c, 0, ids, 0)), int(SS_E_INVALID));  // dense layer has no units
  CHECK_OK(ss_layer_all(r.c, 2));
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_INVALID));  // acquired, not released
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
}

int main() { return sstest::run_all(); }
