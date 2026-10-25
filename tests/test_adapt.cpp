// Adaptive memory plan (include/tierstream/ts_adapt.h): usage table (fixed point, ranking, canonical file, merge,
// online updater), tiers from the ranking, trace parsing and the tier simulator, the solver against the placement
// contract's frozen goldens (SV1 and S1, made by wave4/placement/golden/solver_ref.py), the solver's budget split, and
// on the CPU backend: the file (DISK) tier against the pinned tier, bit for bit, with host lookahead prefetch of the
// next layer, and fail-closed reads.
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "adapt_sim.hpp"
#include "adapt_solve.hpp"
#include "adapt_usage.hpp"
#include "check.hpp"
#include "errors.hpp"
#include "fixture.hpp"
#include "json.hpp"
#include "tierstream/ts_adapt.h"

using namespace ts::adapt;
using sstest::Fixture;

namespace {

std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}

}  // namespace

TEST(adapt_fixed_point) {
  CHECK_EQ(mul_q32(1000000, uint64_t(1) << 31), uint64_t(500000));
  CHECK_EQ(pow_q32(uint64_t(1) << 31, 3), uint64_t(1) << 29);
  CHECK_EQ(pow_q32(uint64_t(1) << 32, 12345), uint64_t(1) << 32);
  CHECK_EQ(decay_q32(1), uint64_t(1) << 31);
  CHECK_EQ(decay_q32(0), uint64_t(1) << 32);
  // 2^(-1/256) in Q32
  const uint64_t f = decay_q32(256);
  CHECK(f > 4283353000ull && f < 4283354000ull);
  // 256 steps of the factor halve (within the per-step rounding)
  const uint64_t half = pow_q32(f, 256);
  CHECK(half > (uint64_t(1) << 31) - 300 && half < (uint64_t(1) << 31) + 300);
  CHECK_EQ(rdiv(5, 2), uint64_t(3));
  CHECK_EQ(rdiv(4, 3), uint64_t(1));
  CHECK_EQ(rdiv(0, 7), uint64_t(0));
  CHECK_EQ(mul_q32(0xFFFFFFFFull, 0xFFFFFFFFull), uint64_t(0xFFFFFFFEull));
}

TEST(adapt_usage_counts_recency_rank) {
  Usage u("m", {4, 3}, 1);  // half life 1 token
  const int32_t a[] = {0, 1, 1, -1}, b[] = {2};
  u.observe(0, a, 4);  // unit 1 twice (two rows), -1 ignored
  u.observe(1, b, 1);
  u.tick(1);
  CHECK_EQ(u.count(0), uint64_t(1));
  CHECK_EQ(u.count(1), uint64_t(2));
  CHECK_EQ(u.recency(1), uint64_t(1000000));  // 2 x 10^6 halved once
  CHECK_EQ(u.use(1), uint64_t(2000000));
  const int32_t c[] = {3};
  u.observe(0, c, 1);
  u.tick(1);
  CHECK_EQ(u.recency(1), uint64_t(500000));
  CHECK_EQ(u.recency(3), uint64_t(500000));
  CHECK_EQ(u.use(1), uint64_t(1000000));  // 2 selections over 2 tokens
  CHECK_EQ(u.use(0), uint64_t(500000));
  // rank: use desc (1: 1.0, then 0 / 3 / unit 2 of layer 1 = global 6 at 0.5), recency desc (3 more recent than 0 and
  // 6), then index (0 before 6)
  const std::vector<int64_t> r = u.rank();
  const std::vector<int64_t> want = {1, 3, 0, 6, 2, 4, 5};
  CHECK(r == want);
  bool threw = false;
  try {
    const int32_t bad[] = {3};
    u.observe(1, bad, 1);
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(adapt_usage_file_roundtrip_and_validation) {
  Usage u("deepseek", {8, 8, 8}, 64);
  std::mt19937 rng(5);
  for (int t = 0; t < 500; ++t) {
    for (int L = 0; L < 3; ++L) {
      int32_t ids[2] = {int32_t(rng() % 8), int32_t(rng() % 3)};
      u.observe(L, ids, 2);
    }
    u.tick(1);
  }
  u.sources().push_back({"routes-trace", "x.txt", "ab", "train-", 500});
  const std::string t1 = u.to_text();
  Usage v = Usage::from_text(t1);
  CHECK_EQ(v.to_text(), t1);  // canonical: byte-identical
  CHECK_EQ(v.sha256(), u.sha256());
  CHECK(v.rank() == u.rank());
  // derived "use" edited by hand: refused
  ts::Json j = ts::json_parse(t1);
  j["layers"].arr[1]["use"].arr[2] = ts::Json::number(uint64_t(1));
  bool threw = false;
  try {
    Usage::from_json(j);
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
  ts::Json k = ts::json_parse(t1);
  k["schema"] = ts::Json::string("tierstream/usage-v2");
  threw = false;
  try {
    Usage::from_json(k);
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
  // dense layers (0 units) round-trip
  Usage z("m", {0, 3, 0}, 8);
  const int32_t zz[] = {2};
  z.observe(1, zz, 1);
  z.tick();
  CHECK_EQ(Usage::from_text(z.to_text()).to_text(), z.to_text());
  CHECK_EQ(z.total_units(), int64_t(3));
  // merge(a, b) has the counts and tokens of one table that saw both
  Usage a("m", {5}, 0), b("m", {5}, 0), ab("m", {5}, 0);
  const int32_t x[] = {1, 2}, y[] = {2, 4};
  a.observe(0, x, 2);
  a.tick();
  ab.observe(0, x, 2);
  ab.tick();
  b.observe(0, y, 2);
  b.tick();
  ab.observe(0, y, 2);
  ab.tick();
  a.merge(b);
  CHECK_EQ(a.tokens(), ab.tokens());
  for (int g = 0; g < 5; ++g) CHECK_EQ(a.count(g), ab.count(g));
  CHECK(a.rank() == ab.rank());
}

TEST(adapt_tiers_from_rank) {
  const std::vector<int64_t> order = {3, 0, 4, 1, 2};
  const std::vector<uint8_t> t = tiers_from_rank(order, 5, 2, 2);
  const std::vector<uint8_t> want = {SS_TIER_RESIDENT, SS_TIER_PINNED, SS_TIER_DISK, SS_TIER_RESIDENT, SS_TIER_PINNED};
  CHECK(t == want);
  bool threw = false;
  try {
    tiers_from_rank({0, 0, 1, 2, 3}, 5, 1, 1);
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
  // C ABI
  ts_usage* u = nullptr;
  const uint32_t units[] = {3, 0, 2};
  CHECK_OK(ts_usage_new("m", units, 3, 0, &u));
  const int32_t hot[] = {0, 0, 1};
  CHECK_OK(ts_usage_observe(u, 2, hot, 3));
  CHECK_OK(ts_usage_tick(u, 1));
  CHECK_EQ(ts_usage_total_units(u), int64_t(5));
  int64_t ord[5];
  CHECK_OK(ts_usage_rank(u, ord, 5));
  CHECK_EQ(ord[0], int64_t(3));  // layer 2 unit 0 picked twice... global 3 = (2, 0)
  uint8_t tiers[5];
  CHECK_OK(ts_usage_tiers(u, 1, 1, tiers, 5));
  CHECK_EQ(int(tiers[3]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(tiers[4]), int(SS_TIER_PINNED));
  CHECK_EQ(int(tiers[0]), int(SS_TIER_DISK));
  CHECK(ts_usage_tiers(u, 3, 3, tiers, 5) != SS_OK);
  char hex[65];
  CHECK_OK(ts_usage_sha256(u, hex));
  CHECK_EQ(std::strlen(hex), size_t(64));
  ts_usage_free(u);
}

TEST(adapt_trace_and_filters) {
  const std::string text =
      "# seq train-1:code:x rows 2\n0:1,2 1:0,3\n0:1,0 1:3,3\n# seq val-2:en:y rows 1\r\n0:2,2 1:1,0\r\n";
  Trace t = trace_parse(text, 4);
  CHECK_EQ(t.n_layers, 2);
  CHECK_EQ(t.top_k, 2);
  CHECK_EQ(t.n_units, 4);
  CHECK_EQ(t.seqs.size(), size_t(2));
  CHECK_EQ(t.n_tokens(), uint64_t(3));
  CHECK_EQ(t.seqs[1].first, uint32_t(2));
  CHECK_EQ(t.at(1, 1)[1], 3);
  CHECK(seq_match("train-1:code:x", "train-"));
  CHECK(!seq_match("train-1:code:x", "val-"));
  CHECK(seq_match("val-2:en:y", "val-&:en:,:es:"));
  CHECK(!seq_match("val-2:en:y", "val-&:code:"));
  CHECK(seq_match("val-2:en:y", "!:code:"));
  Usage u = usage_from_trace(t, "train-", "m", 0);
  CHECK_EQ(u.tokens(), uint64_t(2));
  CHECK_EQ(u.count(1), uint64_t(2));           // layer 0 unit 1
  CHECK_EQ(u.count(4 + 3), uint64_t(3));       // layer 1 unit 3: one + two rows of the same token
  CHECK_EQ(u.sources().size(), size_t(1));
  CHECK_EQ(u.sources()[0].tokens, uint64_t(2));
  bool threw = false;
  try {
    trace_parse("0:1,2\n");
  } catch (const ts::Error&) {
    threw = true;
  }
  CHECK(threw);
}

TEST(adapt_simulator_hand_case) {
  // One layer, 4 units, 2 resident + 1 spare, unit 3 is a FILE home. Worked by hand in the design notes.
  Trace t = trace_parse("# seq a rows 3\n0:0,1\n0:2,3\n0:0,1\n", 4);
  Usage u = usage_from_trace(t, "", "m", 0);
  const std::vector<int64_t> order = u.rank();
  CHECK((order == std::vector<int64_t>{0, 1, 2, 3}));
  SimConfig c;
  c.active = 2;
  c.spares = 1;
  c.file_units = 1;
  SimResult r = simulate(t, "", order, c);
  CHECK_EQ(r.windows, uint64_t(3));
  CHECK_EQ(r.lookups, uint64_t(6));
  CHECK_EQ(r.miss_pinned, uint64_t(2));
  CHECK_EQ(r.miss_file, uint64_t(1));
  CHECK_EQ(r.bank, uint64_t(1));
  CHECK_EQ(r.evicted, uint64_t(2));
  CHECK_EQ(r.discarded, uint64_t(0));
  CHECK_EQ(r.file_uses, uint64_t(1));
  // R = 3: one window, union {0,1,2,3}: misses 2 (pinned) and 3 (file); one spare -> 3 is banked
  c.rows = 3;
  r = simulate(t, "", order, c);
  CHECK_EQ(r.windows, uint64_t(1));
  CHECK_EQ(r.lookups, uint64_t(4));
  CHECK_EQ(r.miss_pinned + r.miss_file, uint64_t(2));
  // A FILE victim is discarded and its slot is free at once (no write-back).
  Trace t2 = trace_parse("# seq a rows 4\n0:3\n0:0\n0:1\n0:3\n", 4);
  SimConfig c2;
  c2.active = 1;
  c2.spares = 1;
  c2.file_units = 1;  // ranking {0,1,2,3}: unit 3 file
  r = simulate(t2, "", order, c2);
  CHECK_EQ(r.miss_file, uint64_t(2));
  CHECK(r.discarded >= 1);
  // miss-curve-v1 in the contract's shape, ranked and cross-validated
  Trace t3 = trace_parse("# seq a:code: rows 3\n0:0,1\n0:2,3\n0:0,1\n# seq b:en: rows 2\n0:1,2\n0:3,0\n", 4);
  Usage u3 = usage_from_trace(t3, "", "m", 0);
  SimConfig cb;
  cb.spares = 1;
  for (int folds : {0, 2}) {
    ts::Json mc = miss_curve(t3, {{"code", ":code:"}, {"all", ""}}, u3, {1, 2, 3}, cb, folds);
    CHECK_EQ(mc.find("schema")->text, std::string("tierstream/miss-curve-v1"));
    const ts::Json* k = mc.find("curves")->find("code/R1");
    CHECK(k && k->is_array() && k->arr.size() == 3 && k->arr[0].arr.size() == 2);
    CHECK(mc.find("detail")->find("all/R1") != nullptr);
  }
  SimResult xv = simulate_xval(t3, "", cb, 2, 0);
  CHECK_EQ(xv.tokens, uint64_t(5));
  // all units fit: no misses after warm start
  SimConfig c3;
  c3.active = 4;
  c3.spares = 0;
  r = simulate(t, "", order, c3);
  CHECK_EQ(r.miss_pinned + r.miss_file, uint64_t(0));
}

TEST(adapt_solver_goldens) {
  // SV1 / SV1b = the placement contract's frozen goldens (wave4/placement/golden/sv1, solver_ref.py sha256
  // 5722f6e2...: applicability / calib / sequences rules, incomplete calibration -> unpredicted), byte for byte.
  for (const char* name : {"sv1", "sv1b"}) {
    const std::string dir = "tests/data/adapt/";
    const std::string fx = slurp(dir + name + "-fixture.json"), want = slurp(dir + name + "-expected.json");
    CHECK(!fx.empty() && !want.empty());
    if (fx.empty()) continue;
    const std::string got = ts::json_write(solve_ref(ts::json_parse(fx)));
    CHECK_EQ(got, want);
  }
}

TEST(adapt_solver_budget_split) {
  // 3 layers x 4 units of 1000 bytes; budget leaves room for 5 units at 8K, 2 at 64K (state grows), none at 1M.
  Usage u("m", {4, 4, 4}, 0);
  for (int t = 0; t < 10; ++t) {
    for (int L = 0; L < 3; ++L) {
      int32_t ids[1] = {int32_t((t * 7 + L) % 4)};
      if (L == 1) ids[0] = 2;
      u.observe(L, ids, 1);
    }
    u.tick();
  }
  ts::Json in = ts::json_parse(R"({"schema":"tierstream/adapt-solve-v1",
    "model":{"unit_bytes":1000,"units":12,"host_stride":1024},
    "device":{"budget_bytes":20000,"margin_bytes":1000,"lines":[{"id":"dense","bytes":10000}]},
    "host":{"pinned_bytes":6000,"aux_pinned_bytes":500,"spare_blocks":1},
    "modes":{"plain":{"R":1},"spec":{"R":6,"lines":[{"id":"drafter","bytes":2500}]}},
    "state":{"plain":{"per_sequence":{"8192":3500,"65536":6900}},"spec":{"linear":{"base":100,"per_token":1}}},
    "choices":[{"id":"a","mode":"plain","ctx":8192},{"id":"b","mode":"plain","ctx":65536},
               {"id":"c","mode":"spec","ctx":1048576},{"id":"d","mode":"plain","ctx":8192,"arena_units":3},
               {"id":"e","mode":"spec","ctx":2000}]})");
  in["usage"] = u.to_json();
  ts::Json out = solve(in, "");
  const auto& rows = out.find("rows")->arr;
  auto num = [](const ts::Json& r, const char* k) { return std::stoull(r.find(k)->text); };
  CHECK_EQ(rows.size(), size_t(5));
  // a: 20000 - 1000 - 10000 - 3500 = 5500 -> 5 active; host (6000 - 500 - 1024) / 1024 = 4; file 3
  CHECK_EQ(num(rows[0], "active_units"), 5ull);
  CHECK_EQ(num(rows[0], "host_units"), 4ull);
  CHECK_EQ(num(rows[0], "file_units"), 3ull);
  CHECK_EQ(rows[0].find("state_tag")->text, std::string("table"));
  // b: 2100 -> 2 active, host 4, file 6
  CHECK_EQ(num(rows[1], "active_units"), 2ull);
  CHECK_EQ(num(rows[1], "file_units"), 6ull);
  // c: 1048676 + 2500 > room: refused with the overflow
  CHECK_EQ(rows[2].find("status")->text, std::string("PL_E_BUDGET"));
  CHECK_EQ(rows[2].find("state_tag")->text, std::string("linear-model"));
  // d: operator cap on the arena
  CHECK_EQ(num(rows[3], "active_units"), 3ull);
  CHECK_EQ(num(rows[3], "file_units"), 5ull);
  // e: 20000 - 1000 - 10000 - 2500 - 2100 = 4400 -> 4
  CHECK_EQ(num(rows[4], "active_units"), 4ull);
  // the tiers of row a follow the ranking: layer 1 unit 2 (picked every token) is resident
  const std::vector<uint8_t> t = tiers_from_rank(u.rank(), 12, 5, 4);
  CHECK_EQ(int(t[4 + 2]), int(SS_TIER_RESIDENT));
  // the C ABI gives the same document
  const std::string txt = ts::json_write(in);
  char* js = nullptr;
  CHECK_OK(ts_adapt_solve(txt.data(), txt.size(), nullptr, &js, nullptr));
  CHECK(js && std::string(js) == ts::json_write(out));
  ts_free(js);
}

// ---------------------------------------------------------------------------------------------------------------
// CPU backend: the file tier is bit-identical to the pinned tier; online usage; fail-closed reads.

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

int units_of(int L) { return L == 2 ? Fixture::kE2 : L == 3 ? 1 : L == 0 ? 0 : Fixture::kE1; }

bool open_tiers(Run& r, const Fixture& fx, const std::vector<uint8_t>& tiers, uint32_t ra, int dyn) {
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return false;
  ss_budget_defaults(&r.b);
  r.b.margin_gib = 1.0 / 1024;
  r.b.margin_fraction = 0;
  r.b.device_bytes = uint64_t(256) << 20;
  r.b.ring_slots = 16;
  r.b.host_pinned_bytes = 4 << 20;
  r.b.dynamic_slots_per_layer = uint32_t(dyn);
  CHECK_OK(ss_plan_from_tiers(r.d, &r.b, tiers.data(), tiers.size(), &r.p));
  if (!r.p) return false;
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  r.o.pin_planned = 0;
  r.o.readahead_units = ra;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  return r.c != nullptr;
}

const void* unit_ptr(const ss_layer_view& v, int unit) {
  return v.table_host ? reinterpret_cast<const void*>(uintptr_t(v.table_host[unit])) : nullptr;
}

std::vector<std::vector<int32_t>> random_routes(int tokens, uint32_t seed) {
  // Skewed routing (a hot set per layer), layers 1, 2, 4 (top-4, top-2, top-4).
  std::mt19937 rng(seed);
  std::vector<std::vector<int32_t>> out;
  for (int t = 0; t < tokens; ++t)
    for (int L : {1, 2, 4}) {
      const int n = units_of(L), k = L == 2 ? 2 : 4;
      std::vector<int32_t> ids;
      while (int(ids.size()) < k) {
        int32_t e = int32_t((rng() % 3 == 0) ? rng() % n : rng() % 4);  // 2/3 of picks in units 0-3
        bool dup = false;
        for (int32_t x : ids) dup |= x == e;
        if (!dup) ids.push_back(e);
      }
      out.push_back(ids);
    }
  return out;
}

}  // namespace

TEST(adapt_file_tier_bitwise_vs_pinned) {
  Fixture fa("adapt-pinned"), fb("adapt-file");
  const std::vector<std::vector<int32_t>> routes = random_routes(120, 11);
  // Usage table from the first 60 tokens (a "training" trace); the run uses all 120.
  Usage usage("synthetic", {0, 16, 8, 1, 16}, 32);
  for (size_t i = 0; i < 60 * 3; ++i) {
    const int L = i % 3 == 0 ? 1 : i % 3 == 1 ? 2 : 4;
    usage.observe(L, routes[i].data(), int(routes[i].size()));
    if (i % 3 == 2) usage.tick();
  }
  const int64_t n = usage.total_units();  // 41
  // A: every unit pinned (the all-resident reference of the host tier); B: 6 resident, 12 pinned, the tail on disk.
  std::vector<uint8_t> ta(static_cast<size_t>(n), uint8_t(SS_TIER_PINNED));
  const std::vector<uint8_t> tb = tiers_from_rank(usage.rank(), n, 6, 12);
  int disk_units = 0;
  for (uint8_t x : tb) disk_units += x == SS_TIER_DISK;
  CHECK_EQ(disk_units, 23);
  Run a, b;
  if (!open_tiers(a, fa, ta, 0, 2) || !open_tiers(b, fb, tb, 4, 2)) return;
  // Online usage on B: must count exactly what the run routes.
  ts_usage* online = nullptr;
  CHECK_OK(ts_usage_new_for(b.d, 32, &online));
  CHECK_OK(ts_usage_attach(b.c, online));
  const uint32_t rows[5] = {0, 1, 1, 0, 1};
  uint64_t compared = 0, fed = 0;
  for (size_t t = 0; t < 120; ++t) {
    for (Run* r : {&a, &b}) CHECK_OK(ss_begin(r->c, rows, 5));
    ss_layer_view va[5]{}, vb[5]{};
    for (int li = 0; li < 3; ++li) {
      const int L = li == 0 ? 1 : li == 1 ? 2 : 4;
      const std::vector<int32_t>& ids = routes[t * 3 + size_t(li)];
      fed += ids.size();
      for (Run* r : {&a, &b}) CHECK_OK(ss_layer_routes(r->c, L, ids.data(), int(ids.size())));
      // lookahead: the next layer's routes (the router hint) read ahead from NVMe into pinned buffers on B
      if (li < 2) {
        const int Ln = li == 0 ? 2 : 4;
        const std::vector<int32_t>& nx = routes[t * 3 + size_t(li) + 1];
        CHECK_OK(ss_host_prefetch(b.c, Ln, nx.data(), int(nx.size())));
      }
      CHECK_OK(ss_acquire(a.c, L, nullptr, &va[L]));
      CHECK_OK(ss_acquire(b.c, L, nullptr, &vb[L]));
      for (int32_t u : ids) {
        const void* pa = unit_ptr(va[L], u);
        const void* pb = unit_ptr(vb[L], u);
        const auto& want = fa.expected[size_t(L)][size_t(u)];
        CHECK(pa && pb);
        if (!pa || !pb) continue;
        CHECK(std::memcmp(pa, want.data(), want.size()) == 0);
        CHECK(std::memcmp(pb, want.data(), want.size()) == 0);
        CHECK(std::memcmp(pa, pb, want.size()) == 0);
        ++compared;
      }
      for (Run* r : {&a, &b}) CHECK_OK(ss_release(r->c, L, nullptr));
    }
    for (Run* r : {&a, &b}) CHECK_OK(ss_end(r->c));
  }
  CHECK_EQ(compared, fed);
  ss_layer_stats sa, sb;
  CHECK_OK(ss_stats_total(a.c, &sa));
  CHECK_OK(ss_stats_total(b.c, &sb));
  CHECK_EQ(sa.bytes_disk, uint64_t(0));  // A never reads a file after open
  CHECK(sb.bytes_disk > 0);              // B serves its tail from the file tier
  CHECK_EQ(sa.routed, sb.routed);
  CHECK_EQ(ts_usage_tokens(online), uint64_t(120));
  uint64_t counted = 0;
  for (int64_t g = 0; g < ts_usage_total_units(online); ++g) {
    uint64_t c = 0;
    CHECK_OK(ts_usage_unit(online, g, nullptr, nullptr, &c));
    counted += c;
  }
  CHECK_EQ(counted, fed);
  // detach: no more counting
  CHECK_OK(ts_usage_attach(b.c, nullptr));
  CHECK_OK(ss_begin(b.c, rows, 5));
  const int32_t one[] = {0};
  CHECK_OK(ss_layer_routes(b.c, 2, one, 1));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(b.c, 2, nullptr, &v));
  CHECK_OK(ss_release(b.c, 2, nullptr));
  CHECK_OK(ss_end(b.c));
  CHECK_EQ(ts_usage_tokens(online), uint64_t(120));
  // a table of another shape is refused
  ts_usage* wrong = nullptr;
  const uint32_t shape[] = {16, 8};
  CHECK_OK(ts_usage_new("x", shape, 2, 0, &wrong));
  CHECK(ts_usage_attach(b.c, wrong) != SS_OK);
  ts_usage_free(wrong);
  ts_usage_free(online);
}

TEST(adapt_file_tier_fail_closed) {
  // Unit 3 of layers 1 and 4 has a wrong descriptor hash and lives on disk: every request for it fails (SS_E_HASH),
  // it never lands in a slot (no stale or unchecked bytes served later), and the other units keep streaming exactly.
  Fixture bad("adapt-bad", /*corrupt_hash=*/true);
  Usage usage("synthetic", {0, 16, 8, 1, 16}, 0);
  const int32_t hot[] = {0, 1, 2};
  usage.observe(1, hot, 3);
  usage.observe(4, hot, 3);
  usage.tick();
  std::vector<uint8_t> t = tiers_from_rank(usage.rank(), usage.total_units(), 6, 0);
  CHECK_EQ(int(t[size_t(usage.base(1) + 3)]), int(SS_TIER_DISK));
  Run r;
  if (!open_tiers(r, bad, t, 2, 2)) return;
  const uint32_t rows[5] = {0, 1, 0, 0, 0};
  for (int attempt = 0; attempt < 3; ++attempt) {
    CHECK_OK(ss_begin(r.c, rows, 5));
    const int32_t ids[] = {3, 5};
    CHECK_OK(ss_host_prefetch(r.c, 1, ids, 2));
    CHECK_OK(ss_layer_routes(r.c, 1, ids, 2));
    ss_layer_view v{};
    CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_HASH));
    CHECK(ss_end(r.c) != SS_OK);
    // next forward without unit 3: exact bytes
    CHECK_OK(ss_begin(r.c, rows, 5));
    const int32_t ok_ids[] = {5, 7};
    CHECK_OK(ss_layer_routes(r.c, 1, ok_ids, 2));
    CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
    for (int32_t u : ok_ids) {
      const void* p = unit_ptr(v, u);
      CHECK(p && std::memcmp(p, bad.expected[1][size_t(u)].data(), bad.expected[1][size_t(u)].size()) == 0);
    }
    CHECK_OK(ss_release(r.c, 1, nullptr));
    CHECK_OK(ss_end(r.c));
  }
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
