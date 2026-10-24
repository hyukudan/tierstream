// CPU tests of the host-only pieces: JSON, xxh64, descriptor validation, planner, resident set, lookahead select.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

#include "check.hpp"
#include "desc.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "plan.hpp"
#include "resident_set.hpp"
#include "strix_stream/ss.h"
#include "xxh64.hpp"

using namespace ss;

namespace {

template <class F>
ss_status status_of(F&& f) {
  try {
    f();
    return SS_OK;
  } catch (const Error& e) {
    return e.status();
  }
}

std::string moe_desc(int layers, int experts, uint64_t unit, const std::string& order = "ranked",
                     const std::string& tiers = "pinned|disk", const std::string& extra = "") {
  std::string s = R"({"schema":"strix-stream/desc-v1","engine":"t","order":")" + order + R"(","tiers":")" + tiers +
                  R"(","fixed_device":[{"name":"dense","bytes":1073741824}],"sources":[{"path":"x.bin"}],)" + extra +
                  R"("layers":[)";
  for (int L = 0; L < layers; ++L) {
    if (L) s += ",";
    s += R"({"index":)" + std::to_string(L) + R"(,"kind":"moe","n_units":)" + std::to_string(experts) +
         R"(,"top_k":)" + std::to_string(experts < 2 ? 1 : 2) + R"(,"parts":[{"format":"Q2_K","offset":0,"bytes":)" + std::to_string(unit) +
         R"(,"stride":)" + std::to_string(unit) + "}]}";
  }
  return s + "]}";
}

}  // namespace

TEST(json_roundtrip_and_bounds) {
  Json j = json_parse(R"({"b":[1,2.5,-3,"xé\n"],"a":{"n":18446744073709551615,"t":true,"z":null}})");
  CHECK_EQ(json_u64(*j.find("a")->find("n"), "n"), uint64_t(18446744073709551615ULL));
  CHECK(json_bool(*j.find("a")->find("t"), "t"));
  CHECK_EQ(json_str(j.find("b")->arr[3], "s"), std::string("x\xc3\xa9\n"));
  CHECK_EQ(json_write(j), std::string(R"({"a":{"n":18446744073709551615,"t":true,"z":null},"b":[1,2.5,-3,"x)"
                                      "\xc3\xa9"
                                      R"(\n"]})"));
  CHECK_EQ(json_i64(j.find("b")->arr[2], "i"), int64_t(-3));
  CHECK_EQ(int(status_of([] { json_parse("{\"a\":1,\"a\":2}"); })), int(SS_E_DESC));
  CHECK_EQ(int(status_of([] { json_parse("[1,]"); })), int(SS_E_DESC));
  CHECK_EQ(int(status_of([] { json_parse("{\"a\":1} x"); })), int(SS_E_DESC));
  CHECK_EQ(int(status_of([] { json_u64(json_parse("18446744073709551616"), "n"); })), int(SS_E_DESC));
  CHECK_EQ(int(status_of([] { json_u64(json_parse("-1"), "n"); })), int(SS_E_DESC));
  std::string deep(100, '[');
  CHECK_EQ(int(status_of([&] { json_parse(deep); })), int(SS_E_DESC));
}

TEST(xxh64_vectors) {
  CHECK_EQ(xxh64("", 0), uint64_t(0xEF46DB3751D8E999ULL));
  CHECK_EQ(xxh64("a", 1), uint64_t(0xD24EC4F1A98C6E5BULL));
  CHECK_EQ(xxh64("abc", 3), uint64_t(0x44BC2CF5AD770999ULL));
  const char* s = "Nobody inspects the spammish repetition";
  CHECK_EQ(xxh64(s, std::strlen(s)), uint64_t(0xFBCEA83C8A378BF1ULL));
  // Streaming in uneven pieces equals one shot.
  std::string big(10007, '\0');
  for (size_t i = 0; i < big.size(); ++i) big[i] = char(i * 131 + 7);
  Xxh64 h;
  size_t pos = 0, step = 1;
  while (pos < big.size()) {
    size_t n = std::min(step, big.size() - pos);
    h.update(big.data() + pos, n);
    pos += n;
    step = step * 3 + 1;
  }
  CHECK_EQ(h.digest(), xxh64(big.data(), big.size()));
}

TEST(desc_validation) {
  Desc d = desc_from_json(json_parse(moe_desc(2, 8, 1000)));
  CHECK_EQ(d.total_units, int64_t(16));
  CHECK_EQ(d.unit_base[1], int64_t(8));
  CHECK_EQ(d.unit_stride(0), uint64_t(1024));
  CHECK_EQ(d.max_units_per_layer, 8);
  // Round trip through the canonical writer is stable.
  std::string once = json_write(desc_to_json(d));
  CHECK_EQ(json_write(desc_to_json(desc_from_json(json_parse(once)))), once);

  auto bad = [](const std::string& s) { return int(status_of([&] { desc_from_json(json_parse(s)); })); };
  std::string base = moe_desc(1, 8, 1000);
  CHECK_EQ(bad(R"({"schema":"strix-stream/desc-v2","layers":[]})"), int(SS_E_DESC));
  std::string typo = base;
  typo.replace(typo.find("\"top_k\""), 7, "\"topk\"");
  CHECK_EQ(bad(typo), int(SS_E_DESC));  // unknown key
  std::string order = base;
  order.replace(order.find("\"index\":0"), 9, "\"index\":1");
  CHECK_EQ(bad(order), int(SS_E_DESC));  // layers out of order
  std::string sum = base;
  sum.replace(sum.find("\"top_k\""), 7, "\"unit_bytes\":999,\"top_k\"");
  CHECK_EQ(bad(sum), int(SS_E_DESC));  // unit_bytes != parts
  std::string fmt = base;
  fmt.replace(fmt.find("Q2_K"), 4, "Q9_Z");
  CHECK_EQ(bad(fmt), int(SS_E_DESC));
  std::string xfmt = base;
  xfmt.replace(xfmt.find("Q2_K"), 4, "x-trellis-k4");
  CHECK_EQ(bad(xfmt), int(SS_OK));
  std::string overlap = base;
  overlap.replace(overlap.find("\"stride\":1000"), 13, "\"stride\":999");
  CHECK_EQ(bad(overlap), int(SS_E_DESC));
  std::string note = base;
  note.replace(note.find("\"top_k\""), 7, "\"x-note\":\"free form\",\"top_k\"");
  CHECK_EQ(bad(note), int(SS_OK));
}

TEST(plan_ranked_tiers) {
  // 2 layers x 8 experts x 1 MiB (stride 1 MiB), fixed 1 GiB; margin fixed at 2 GiB.
  Desc d = desc_from_json(json_parse(moe_desc(2, 8, 1 << 20)));
  ss_budget b;
  ss_budget_defaults(&b);
  b.margin_fraction = 0;
  b.ring_slots = 2;
  // Room for everything: no ring, all resident.
  b.device_bytes = 3 * kGiB + 16 * (1 << 20);
  Plan p = plan_make(d, b, {});
  CHECK_EQ(p.n_resident, uint64_t(16));
  CHECK_EQ(p.ring, uint64_t(0));
  // 10 MiB of room: ring 2 MiB, then 8 residents; ties in model order -> layer 0 all, layer 1 none.
  b.device_bytes = 3 * kGiB + 10 * (1 << 20);
  b.host_pinned_bytes = 5 * (1 << 20);
  p = plan_make(d, b, {});
  CHECK_EQ(p.n_resident, uint64_t(8));
  CHECK_EQ(p.ring, uint64_t(2) << 20);
  CHECK_EQ(p.n_pinned, uint64_t(5));
  CHECK_EQ(p.n_disk, uint64_t(3));
  for (int u = 0; u < 8; ++u) CHECK_EQ(int(p.tier[size_t(u)]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(p.tier[8]), int(SS_TIER_PINNED));
  CHECK_EQ(int(p.tier[15]), int(SS_TIER_DISK));
  CHECK(p.peak <= p.cap - p.margin);
  // Benefit ranking: the hottest units are resident, the next hottest pinned.
  std::vector<double> ben(16, 1.0);
  for (int u = 8; u < 16; ++u) ben[size_t(u)] = 10.0 + u;
  p = plan_make(d, b, ben);
  for (int u = 8; u < 16; ++u) CHECK_EQ(int(p.tier[size_t(u)]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(p.tier[0]), int(SS_TIER_PINNED));
  CHECK_EQ(int(p.tier[7]), int(SS_TIER_DISK));
  // Too small for the fixed bytes + margin.
  b.device_bytes = 2 * kGiB;
  CHECK_EQ(int(status_of([&] { plan_make(d, b, {}); })), int(SS_E_CAPACITY));
}

TEST(plan_prefix_and_pinned_only) {
  Desc d = desc_from_json(json_parse(moe_desc(4, 1, 100 << 20, "prefix", "pinned")));
  ss_budget b;
  ss_budget_defaults(&b);
  b.margin_fraction = 0;
  b.ring_slots = 2;
  // Room 300 MiB: prefix k=1 (100) + ring 2 x 100 = 300 fits; k=2 needs 200 + 200.
  b.device_bytes = 3 * kGiB + (300ull << 20);
  b.host_pinned_bytes = 400ull << 20;
  Plan p = plan_make(d, b, {});
  CHECK_EQ(p.n_resident, uint64_t(1));
  CHECK_EQ(int(p.tier[0]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(p.tier[1]), int(SS_TIER_PINNED));
  CHECK_EQ(p.pinned, uint64_t(300) << 20);
  // Pinned-only units over the host budget are refused (vramplan's rule).
  b.host_pinned_bytes = 200ull << 20;
  CHECK_EQ(int(status_of([&] { plan_make(d, b, {}); })), int(SS_E_CAPACITY));
  // Dynamic slots are charged before the units.
  Desc d2 = desc_from_json(json_parse(moe_desc(2, 8, 1 << 20)));
  ss_budget c;
  ss_budget_defaults(&c);
  c.margin_fraction = 0;
  c.ring_slots = 2;
  c.dynamic_slots_per_layer = 2;
  c.device_bytes = 3 * kGiB + 10 * (1 << 20);
  p = plan_make(d2, c, {});
  CHECK_EQ(p.dynamic, uint64_t(4) << 20);
  CHECK_EQ(p.n_resident, uint64_t(4));
}

TEST(resident_set_lru) {
  ResidentSet rs(8, 3, SS_POLICY_LRU, 16);
  CHECK_EQ(rs.place(0, true), 0);  // pinned
  std::vector<char> none;
  Admission a = rs.admit({1, 2}, none, true);
  CHECK_EQ(a.admitted.size(), size_t(2));
  rs.observe({1, 2}, {1, 2});
  a = rs.admit({3}, none, true);  // evicts the LRU unpinned: 1 and 2 tie -> lower id 1
  CHECK_EQ(a.evicted.size(), size_t(1));
  CHECK_EQ(a.evicted[0], 1);
  rs.observe({3}, {3});
  CHECK(rs.slot(0) >= 0 && rs.slot(1) < 0 && rs.slot(2) >= 0 && rs.slot(3) >= 0);
  // A route that needs more slots than are evictable: the rest is transient.
  a = rs.admit({4, 5, 6, 2, 3}, none, true);
  CHECK_EQ(a.hits.size(), size_t(2));
  CHECK_EQ(a.admitted.size(), size_t(0));
  CHECK_EQ(a.transient.size(), size_t(3));
  // admit_misses = false: every miss transient.
  ResidentSet st(8, 3, SS_POLICY_LRU, 16);
  a = st.admit({1}, none, false);
  CHECK_EQ(a.transient.size(), size_t(1));
  CHECK_EQ(st.n_resident(), 0);
  // busy units are never evicted.
  ResidentSet b(4, 1, SS_POLICY_LRU, 16);
  b.admit({0}, none, true);
  std::vector<char> busy(4, 0);
  busy[0] = 1;
  a = b.admit({1}, busy, true);
  CHECK_EQ(a.transient.size(), size_t(1));
  CHECK(b.slot(0) >= 0);
}

TEST(resident_set_scored) {
  // Transition 1 -> 2 seen often; after route {1}, a miss on 2 must outrank a resident 5 that was used once.
  ResidentSet rs(8, 2, SS_POLICY_SCORED, 16);
  std::vector<char> none;
  for (int k = 0; k < 4; ++k) {
    rs.observe({1}, {});
    rs.observe({2}, {});
  }
  rs.place(5, false);
  rs.place(6, false);
  rs.observe({5, 6}, {5, 6});
  rs.observe({1}, {1});
  CHECK(rs.score(2) > rs.score(5));
  Admission a = rs.admit({2, 7}, none, true);
  // Rows: 2 (residents 5 and 6 outside the route) -> the top 2 of {5, 6, 2, 7} by score.
  CHECK(std::find(a.admitted.begin(), a.admitted.end(), 2) != a.admitted.end());
  CHECK_EQ(a.admitted.size() + a.transient.size(), size_t(2));
  // Deterministic: the same history gives the same decision.
  ResidentSet r2(8, 2, SS_POLICY_SCORED, 16);
  for (int k = 0; k < 4; ++k) {
    r2.observe({1}, {});
    r2.observe({2}, {});
  }
  r2.place(5, false);
  r2.place(6, false);
  r2.observe({5, 6}, {5, 6});
  r2.observe({1}, {1});
  Admission b = r2.admit({2, 7}, none, true);
  CHECK(a.admitted == b.admitted && a.transient == b.transient && a.evicted == b.evicted);
}

TEST(lookahead_select_union_order) {
  // 2 rows x 6 experts, top 2. Row 0: 3 (0.9), 1 (0.8). Row 1: 4 (0.95), 3 (0.7).
  float s[12] = {0.1f, 0.8f, 0.2f, 0.9f, 0.3f, 0.0f, 0.1f, 0.2f, 0.3f, 0.7f, 0.95f, 0.0f};
  int32_t out[6];
  int n = 0;
  CHECK_OK(ss_lookahead_select(s, 2, 6, 2, out, 6, &n));
  CHECK_EQ(n, 3);
  // Rank 0: 4 (0.95), 3 (0.9); rank 1: 1.
  CHECK_EQ(out[0], 4);
  CHECK_EQ(out[1], 3);
  CHECK_EQ(out[2], 1);
  CHECK_OK(ss_lookahead_select(s, 2, 6, 2, out, 2, &n));
  CHECK_EQ(n, 2);
  // Ties by lower id.
  float t[4] = {0.5f, 0.5f, 0.5f, 0.5f};
  CHECK_OK(ss_lookahead_select(t, 1, 4, 2, out, 4, &n));
  CHECK_EQ(n, 2);
  CHECK_EQ(out[0], 0);
  CHECK_EQ(out[1], 1);
}

TEST(plan_review_rules) {
  // host_fixed over the host budget is refused even with no pinned unit (vramplan's rule).
  std::string base = moe_desc(2, 8, 1 << 20);
  std::string hf = base;
  hf.replace(hf.find("\"fixed_device\""), 14, "\"host_fixed\":[{\"name\":\"ple\",\"bytes\":1000}],\"fixed_device\"");
  Desc d = desc_from_json(json_parse(hf));
  ss_budget b;
  ss_budget_defaults(&b);
  b.margin_fraction = 0;
  b.device_bytes = 4 * kGiB;
  CHECK_EQ(int(status_of([&] { plan_make(d, b, {}); })), int(SS_E_CAPACITY));
  b.host_pinned_bytes = 1000;
  CHECK_EQ(int(status_of([&] { plan_make(d, b, {}); })), int(SS_OK));
  // Pinned bytes are billed as allocated: 4 KiB strides (a 1000-byte unit takes 4096).
  Desc s = desc_from_json(json_parse(moe_desc(1, 8, 1000)));
  std::vector<uint8_t> t(8, SS_TIER_PINNED);
  ss_budget c;
  ss_budget_defaults(&c);
  c.margin_fraction = 0;
  c.device_bytes = 4 * kGiB;
  c.host_pinned_bytes = 8 * 4096;
  Plan p = plan_from_tiers(s, c, t);
  CHECK_EQ(p.pinned, uint64_t(8 * 4096));
  CHECK_EQ(pinned_layout(s, t).total, p.pinned);
  c.host_pinned_bytes = 8 * 4096 - 1;
  CHECK_EQ(int(status_of([&] { plan_from_tiers(s, c, t); })), int(SS_E_CAPACITY));
  // Dynamic slots are billed exactly: a layer fully resident gets none.
  Desc e = desc_from_json(json_parse(moe_desc(2, 8, 1 << 20)));
  ss_budget dyn;
  ss_budget_defaults(&dyn);
  dyn.margin_fraction = 0;
  dyn.device_bytes = 4 * kGiB;
  dyn.dynamic_slots_per_layer = 3;
  dyn.ring_slots = 1;
  std::vector<uint8_t> t2(16, SS_TIER_RESIDENT);
  for (int u = 8; u < 16; ++u) t2[size_t(u)] = SS_TIER_DISK;
  p = plan_from_tiers(e, dyn, t2);
  CHECK_EQ(p.dynamic, uint64_t(3) << 20);
  // pinned_backing gives residents pinned copies, the lowest benefit per byte first.
  ss_budget bk;
  ss_budget_defaults(&bk);
  bk.margin_fraction = 0;
  bk.ring_slots = 2;
  bk.device_bytes = 3 * kGiB + 10 * (1 << 20);
  bk.host_pinned_bytes = 10 * (1 << 20);
  bk.pinned_backing = 1;
  p = plan_make(e, bk, {});
  CHECK_EQ(p.n_resident, uint64_t(8));
  CHECK_EQ(p.n_pinned, uint64_t(8));
  CHECK_EQ(p.n_backed, uint64_t(2));
  CHECK_EQ(int(p.tier[7]), int(SS_TIER_RESIDENT_BACKED));  // ties: the last in model order goes first
  CHECK_EQ(int(p.tier[0]), int(SS_TIER_RESIDENT));
  // The JSON letters round-trip (B included).
  CHECK(plan_from_tiers(e, bk, tiers_from_json(e, plan_to_json(e, p))).tier == p.tier);
}

TEST(desc_overflow_and_narrowing) {
  auto bad = [](const std::string& s) { return int(status_of([&] { desc_from_json(json_parse(s)); })); };
  std::string huge = moe_desc(1, 8, 1000);
  huge.replace(huge.find("\"stride\":1000"), 13, "\"stride\":4000000000000000000");
  CHECK_EQ(bad(huge), int(SS_E_DESC));  // offset + 7 x stride overflows
  std::string many = moe_desc(1, 8, 1000);
  many.replace(many.find("\"n_units\":8"), 11, "\"n_units\":4294967297");
  CHECK_EQ(bad(many), int(SS_E_DESC));  // would wrap to 1 as an int
  std::string idx = moe_desc(1, 8, 1000);
  idx.replace(idx.find("\"index\":0"), 9, "\"index\":-1");
  CHECK_EQ(bad(idx), int(SS_E_DESC));
  std::string hf = moe_desc(1, 8, 1000);
  hf.replace(hf.find("\"fixed_device\""), 14,
             "\"host_fixed\":[{\"name\":\"a\",\"bytes\":9223372036854775808},"
             "{\"name\":\"b\",\"bytes\":9223372036854775808}],\"fixed_device\"");
  CHECK_EQ(bad(hf), int(SS_E_DESC));  // 2^63 + 2^63 wraps to 0
}

TEST(capi_struct_size_and_budget_match) {
  std::string js = moe_desc(1, 4, 4096);
  ss_desc* d = nullptr;
  CHECK_OK(ss_desc_from_json(js.data(), js.size(), &d));
  ss_budget b;
  ss_budget_defaults(&b);
  b.device_bytes = 4 * kGiB;
  ss_plan* p = nullptr;
  // An older caller's struct (shorter struct_size) is accepted; the missing fields take the defaults.
  ss_budget old = b;
  old.struct_size = uint32_t(offsetof(ss_budget, pinned_backing));
  CHECK_OK(ss_plan_make(d, &old, nullptr, 0, &p));
  ss_plan_free(p);
  p = nullptr;
  // Defaults write only the caller's size (an older caller's smaller struct is never overrun).
  {
    const uint32_t old_size = uint32_t(offsetof(ss_budget, pinned_backing));
    unsigned char buf[sizeof(ss_budget) + 16];
    std::memset(buf, 0xAB, sizeof buf);
    ss_budget_defaults_sized(reinterpret_cast<ss_budget*>(buf), old_size);
    bool untouched = true;
    for (size_t k = old_size; k < sizeof buf; ++k) untouched = untouched && buf[k] == 0xAB;
    CHECK(untouched);
    CHECK_EQ(reinterpret_cast<ss_budget*>(buf)->struct_size, old_size);
  }
  // A newer caller's larger struct: only this build's fields are read.
  {
    unsigned char big[sizeof(ss_budget) + 64];
    std::memset(big, 0, sizeof big);
    std::memcpy(big, &b, sizeof b);
    reinterpret_cast<ss_budget*>(big)->struct_size = uint32_t(sizeof big);
    ss_plan* q = nullptr;
    CHECK_OK(ss_plan_make(d, reinterpret_cast<ss_budget*>(big), nullptr, 0, &q));
    ss_plan_free(q);
  }
  ss_budget zero = b;
  zero.struct_size = 0;
  CHECK_EQ(int(ss_plan_make(d, &zero, nullptr, 0, &p)), int(SS_E_INVALID));
  CHECK_OK(ss_plan_make(d, &b, nullptr, 0, &p));
  ss_budget other = b;
  other.device_bytes = 5 * kGiB;
  ss_ctx* c = nullptr;
  CHECK_EQ(int(ss_open(d, &other, p, nullptr, &c)), int(SS_E_INVALID));  // plan made with another budget
  CHECK(std::strstr(ss_last_error(), "budget") != nullptr);
  ss_plan_free(p);
  ss_desc_free(d);
}

int main() { return sstest::run_all(); }
