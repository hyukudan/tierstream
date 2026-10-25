// Device-gated open on the CPU backend: every v1 scope refusal (design rev 3.2 §1-§2) through ss_open, in the order
// the layout checks them, then the backend refusal for a valid layout; old option structs read as the host contract.
#include <cstring>
#include <string>
#include <vector>

#include "check.hpp"
#include "dg_fixture.hpp"
#include "strix_stream/ss.h"

using sstest::DgModel;

namespace {

struct Case {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  ~Case() {
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
};

// Opens with the model's default gated setup after `tweak`; returns the status and ss_last_error.
template <class F>
std::pair<ss_status, std::string> try_open(const DgModel& m, F tweak, int resident = 16, uint32_t dyn = 4,
                                           uint32_t max_ids = 8) {
  Case k;
  if (ss_desc_from_json(m.json.data(), m.json.size(), &k.d) != SS_OK) return {SS_E_DESC, ss_last_error()};
  sstest::dg_budget(k.b, m, dyn, max_ids);
  sstest::dg_options(k.o, max_ids, SS_BACKEND_CPU);
  std::vector<uint8_t> t = m.tiers(resident);
  tweak(k.b, k.o, t);
  ss_status st = ss_plan_from_tiers(k.d, &k.b, t.data(), t.size(), &k.p);
  if (st != SS_OK) return {st, ss_last_error()};
  st = ss_open(k.d, &k.b, k.p, &k.o, &k.c);
  return {st, st == SS_OK ? std::string() : std::string(ss_last_error())};
}

void expect(const std::pair<ss_status, std::string>& r, ss_status st, const char* needle) {
  if (r.first != st || r.second.find(needle) == std::string::npos)
    std::printf("  got %d '%s', wanted %d with '%s'\n", int(r.first), r.second.c_str(), int(st), needle);
  CHECK_EQ(int(r.first), int(st));
  CHECK(r.second.find(needle) != std::string::npos);
}

using B = ss_budget;
using O = ss_options;
using T = std::vector<uint8_t>;

}  // namespace

TEST(dg_open_valid_layout_needs_cuda) {
  DgModel m("open");
  expect(try_open(m, [](B&, O&, T&) {}), SS_E_UNSUPPORTED, "needs the CUDA backend");
}

TEST(dg_open_refusals) {
  DgModel m("open");
  expect(try_open(m, [](B& b, O&, T&) { b.tier_mode = SS_TIER_MODE_INCLUSIVE; }), SS_E_INVALID, "tier_mode EXCLUSIVE");
  expect(try_open(m, [](B&, O& o, T&) { o.policy = SS_POLICY_SCORED; }), SS_E_UNSUPPORTED, "only policy LRU");
  expect(try_open(m, [](B&, O& o, T&) { o.pin_planned = 1; }), SS_E_INVALID, "pin_planned = 0");
  expect(try_open(m, [](B&, O& o, T&) { o.fill_residents = 0; }), SS_E_INVALID, "fill_residents = 1");
  expect(try_open(m, [](B&, O& o, T&) { o.readahead_units = 2; }), SS_E_INVALID, "readahead_units must be 0");
  expect(try_open(m, [](B&, O& o, T&) { o.spec_persistent = 1; }), SS_E_INVALID, "spec_persistent");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_max_ids = 0; }), SS_E_INVALID, "dg_max_ids must be 1..64");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_max_ids = 65; }), SS_E_INVALID, "dg_max_ids must be 1..64");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_wait_timeout_ms = 0; }), SS_E_INVALID, "dg_wait_timeout_ms");
  // a unit on disk: not covered by slots + blocks
  expect(try_open(m, [](B&, O&, T& t) { t[100] = SS_TIER_DISK; }), SS_E_CAPACITY, "is not resident or pinned");
  // no spare slots
  expect(try_open(m, [](B&, O&, T&) {}, 16, 0), SS_E_INVALID, "dynamic_slots_per_layer > 0");
  // ring too small for the banks + guard
  expect(try_open(m, [](B& b, O&, T&) { b.ring_slots = 16; }), SS_E_CAPACITY, "set ring_slots = 2 x dg_max_ids + 1");
  // fewer granted spare blocks than spare slots (the request is only a request)
  expect(try_open(m, [](B& b, O&, T&) { b.host_spare_units = 3; }), SS_E_CAPACITY, "spare blocks");
}

TEST(dg_open_shape_refusals) {
  {
    // a part that is not a multiple of 256 bytes
    DgModel m("open-odd");
    std::string js = m.json;
    const std::string from = R"("stride":2048,"bytes":2048})";
    const size_t at = js.find(from);
    CHECK(at != std::string::npos);
    js.replace(at, from.size(), R"("stride":2000,"bytes":2000})");
    ss_desc* d = nullptr;
    CHECK_OK(ss_desc_from_json(js.data(), js.size(), &d));
    ss_budget b;
    sstest::dg_budget(b, m, 4, 8);
    ss_plan* p = nullptr;
    std::vector<uint8_t> t = m.tiers(16);
    CHECK_OK(ss_plan_from_tiers(d, &b, t.data(), t.size(), &p));
    ss_options o;
    sstest::dg_options(o, 8, SS_BACKEND_CPU);
    ss_ctx* c = nullptr;
    const ss_status st = ss_open(d, &b, p, &o, &c);
    CHECK_EQ(int(st), int(SS_E_UNSUPPORTED));
    CHECK(std::string(ss_last_error()).find("not a multiple of 256") != std::string::npos);
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
}

// An options struct from before the device-gated fields reads as the host contract.
TEST(dg_open_old_options_are_host) {
  DgModel m("open-old");
  ss_desc* d = nullptr;
  CHECK_OK(ss_desc_from_json(m.json.data(), m.json.size(), &d));
  ss_budget b;
  sstest::dg_budget(b, m, 4, 8);
  b.tier_mode = SS_TIER_MODE_INCLUSIVE;
  b.host_spare_units = 0;
  ss_plan* p = nullptr;
  std::vector<uint8_t> t = m.tiers(16);
  CHECK_OK(ss_plan_from_tiers(d, &b, t.data(), t.size(), &p));
  ss_options o;
  ss_options_defaults(&o);
  o.forward_mode = SS_FORWARD_DEVICE_GATED;  // beyond the old size: never read
  o.struct_size = uint32_t(offsetof(ss_options, forward_mode));
  ss_ctx* c = nullptr;
  CHECK_OK(ss_open(d, &b, p, &o, &c));
  CHECK_EQ(ss_dg_gated_index(c, 0), -1);  // a host-contract context
  int32_t ids[1] = {0};
  CHECK_EQ(int(ss_dg_enqueue_wait(c, 0, ids)), int(SS_E_INVALID));
  CHECK(std::string(ss_last_error()).find("not device-gated") != std::string::npos);
  ss_close(c);
  ss_plan_free(p);
  ss_desc_free(d);
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
