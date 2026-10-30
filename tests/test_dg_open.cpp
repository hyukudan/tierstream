// Device-gated open on the CPU backend: every v1 scope refusal (design rev 3.2 §1-§2) through ss_open, in the order
// the layout checks them, then the backend refusal for a valid layout; old option structs read as the host contract.
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "check.hpp"
#include "dg_fixture.hpp"
#include "dg_score.hpp"
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
                                           uint32_t max_ids = 8, const std::string* json = nullptr) {
  Case k;
  const std::string& js = json ? *json : m.json;
  if (ss_desc_from_json(js.data(), js.size(), &k.d) != SS_OK) return {SS_E_DESC, ss_last_error()};
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
  expect(try_open(m, [](B&, O& o, T&) { o.policy = ss_policy(7); }), SS_E_INVALID, "policy must be LRU or SCORED");
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
  // a global spare count above the non-resident units
  expect(try_open(m, [](B&, O& o, T&) { o.dg_spare_slots = 6 * 48 + 1; }), SS_E_INVALID, "above the non-resident");
}

// Spares as the engine caches have them: one global count of spare slots (not a multiple of the layers), and fewer
// spare blocks than spare slots; both valid layouts (they reach the backend refusal on the CPU).
TEST(dg_open_global_spares) {
  DgModel m("open");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_spare_slots = 7; }), SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_spare_slots = 7; }, 16, 0), SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [](B& b, O&, T&) { b.host_spare_units = 3; }), SS_E_UNSUPPORTED, "needs the CUDA backend");
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
  CHECK_EQ(int(ss_dg_reset_open(c)), int(SS_E_INVALID));
  int32_t ids[1] = {0};
  CHECK_EQ(int(ss_dg_enqueue_wait(c, 0, ids)), int(SS_E_INVALID));
  CHECK(std::string(ss_last_error()).find("not device-gated") != std::string::npos);
  ss_close(c);
  ss_plan_free(p);
  ss_desc_free(d);
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }

// Write-back scheduling options: only 0 / 1 are valid for dg_wb_defer and dg_wb_stream; a valid setting reaches the
// backend refusal on the CPU.
TEST(dg_open_wb_defer_options) {
  DgModel m("open");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_wb_defer = 2; }), SS_E_INVALID, "dg_wb_defer must be 0 or 1");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_wb_stream = 2; }), SS_E_INVALID, "dg_wb_stream must be 0 or 1");
  expect(try_open(m, [](B&, O& o, T&) {
           o.dg_wb_defer = 1;
           o.dg_wb_stream = 1;
           o.dg_wb_layer_bytes = 1;
           o.dg_wb_max_outstanding = 4096;
         }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
  // an older caller's struct ends before the write-back fields: whatever lies past its size is not read (the defaults apply)
  expect(try_open(m, [](B&, O& o, T&) {
           o.struct_size = uint32_t(offsetof(ss_options, dg_wb_defer));
           o.dg_wb_defer = 7;
           o.dg_wb_stream = 9;
         }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
}

// The transport defaults: ss_options_defaults puts the deferred write-backs on the write-back stream
// (dg_wb_stream = 1) with dg_wb_defer = 1; a zeroed struct keeps 0 / 0 (the demand stream, boundary write-backs); 0 is
// accepted with deferral (the OFF switch), and sized defaults of an older struct that ends before dg_wb_stream write
// only its bytes.
TEST(dg_open_wb_stream_default) {
  ss_options o;
  ss_options_defaults(&o);
  CHECK_EQ(o.dg_wb_defer, 1u);
  CHECK_EQ(o.dg_wb_stream, 1u);
  CHECK_EQ(o.dg_wb_layer_bytes, 0u);
  CHECK_EQ(o.dg_wb_max_outstanding, 0u);
  ss_options z;
  std::memset(&z, 0, sizeof z);
  CHECK_EQ(z.dg_wb_stream, 0u);
  std::vector<uint8_t> ob(sizeof(ss_options) + 16, 0x5A);
  const uint32_t old = uint32_t(offsetof(ss_options, dg_wb_stream));
  ss_options_defaults_sized(reinterpret_cast<ss_options*>(ob.data()), old);
  bool canary = true;
  for (size_t i = old; i < ob.size(); ++i) canary = canary && ob[i] == 0x5A;
  CHECK(canary);
  CHECK_EQ(reinterpret_cast<ss_options*>(ob.data())->dg_wb_defer, 1u);
  DgModel m("open-wbs");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_wb_stream = 0; }), SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [](B&, O& o, T&) { o.dg_wb_stream = 1; }), SS_E_UNSUPPORTED, "needs the CUDA backend");
}

// ---- winpf: the tierstream/hint-prior-v1 file (ss.h: Window prefetch), parsed with every refusal ----
#include "dg_ctx.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "sha256.hpp"

namespace {

struct PriorFile {
  std::vector<uint8_t> b;
  void u32(uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
  }
  void u16(uint32_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
  }
  void raw(const std::string& s) { b.insert(b.end(), s.begin(), s.end()); }
  void seal() {  // the checksum: sha256 of every byte before it (raw 32 bytes)
    const std::string hex = ts::sha256_hex(b.data(), b.size());
    for (size_t i = 0; i < 64; i += 2) b.push_back(uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
  }
};

// G gated layers of U units; P layers x k entries for the symbols; mutate(field) breaks one thing.
std::vector<uint8_t> make_prior(const ts::Desc& d, const std::vector<int>& gated, uint32_t P, uint32_t k,
                                const std::vector<int32_t>& syms, const std::string& mutate = "") {
  PriorFile f;
  f.raw(mutate == "magic" ? "TSHPRIO2" : "TSHPRIO1");
  f.u32(mutate == "version" ? 2 : 1);
  f.u32(uint32_t(gated.size()) + (mutate == "n_gated" ? 1 : 0));
  for (size_t g = 0; g < gated.size(); ++g)
    f.u32(uint32_t(d.layers[size_t(gated[g])].n_units) + (mutate == "units" && g == 1 ? 1 : 0));
  std::string id = ts::hint_prior_identity(d, gated);
  if (mutate == "identity") id[5] ^= 1;
  f.raw(id);
  const std::string prov = mutate == "prov" ? std::string(4097, 'x') : std::string("test prior");
  f.u32(uint32_t(prov.size()));
  f.raw(prov);
  f.u32(mutate == "layers" ? 0 : P);
  f.u32(mutate == "k" ? 65 : k);
  f.u32(uint32_t(syms.size()));
  for (size_t i = 0; i < syms.size(); ++i) {
    f.u32(uint32_t(mutate == "dup" && i == 1 ? syms[0] : syms[i]));
    for (uint32_t j = 0; j < P * k; ++j)
      f.u16(mutate == "unit" && i == 0 && j == 0 ? 9999u : j == P * k - 1 ? 0xFFFFu : (uint32_t(syms[i]) * 7 + j) % 64);
    for (uint32_t j = 0; j < P * k; ++j) f.u16(65535u - 1000u * j);
  }
  f.seal();
  if (mutate == "checksum") f.b[40] ^= 1;
  if (mutate == "size") f.b.push_back(0);
  return f.b;
}

}  // namespace

TEST(dg_prior_parse_and_refusals) {
  DgModel m("prior");
  const ts::Desc d = ts::desc_from_json(ts::json_parse(m.json));
  std::vector<int> gated;
  for (int L = 0; L < m.G; ++L) gated.push_back(L);
  const std::vector<int32_t> syms = {5, 17, -3, 128000};
  const std::vector<uint8_t> ok = make_prior(d, gated, 2, 4, syms);
  const ts::DgPrior pr = ts::parse_hint_prior(ok.data(), ok.size(), d, gated);
  CHECK_EQ(pr.layers, 2u);
  CHECK_EQ(pr.k, 4u);
  CHECK_EQ(pr.row.size(), size_t(4));
  CHECK_EQ(pr.row.at(128000), 3u);
  CHECK_EQ(pr.unit[0], uint32_t(5 * 7 % 64));
  CHECK_EQ(pr.unit[7], 0xFFFFFFFFu);  // 0xFFFF = none
  CHECK(pr.p[0] == 1.0f);
  for (const char* bad : {"magic", "version", "n_gated", "units", "identity", "prov", "layers", "k", "checksum",
                          "size", "dup", "unit"}) {
    const std::vector<uint8_t> v = make_prior(d, gated, 2, 4, syms, bad);
    bool refused = false;
    try {
      (void)ts::parse_hint_prior(v.data(), v.size(), d, gated);
    } catch (const ts::Error& e) {
      refused = e.status() == SS_E_INVALID;
    }
    if (!refused) std::printf("  not refused: %s\n", bad);
    CHECK(refused);
  }
  // truncated anywhere: refused, never read past the end
  for (size_t n = 0; n < ok.size(); n += 7) {
    bool refused = false;
    try {
      (void)ts::parse_hint_prior(ok.data(), n, d, gated);
    } catch (const ts::Error& e) {
      refused = e.status() == SS_E_INVALID;
    }
    CHECK(refused);
  }
  // another layout (a part offset moved): another identity
  ts::Desc d2 = d;
  d2.layers[0].parts[0].offset += 256;
  CHECK(ts::hint_prior_identity(d, gated) != ts::hint_prior_identity(d2, gated));
}

// The CPU backend refuses a gated open before reading the prior (the prior is validated in the CUDA open, after the
// layout checks): a bad prior does not change that refusal.
TEST(dg_prior_option_cpu_backend) {
  DgModel m("prior-cpu");
  const std::vector<uint8_t> junk(64, 7);
  expect(try_open(m, [&](B&, O& o, T&) {
           o.dg_prior = junk.data();
           o.dg_prior_len = junk.size();
         }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
}

// winpf ABI on the CPU: the defaults, sized defaults of an older options / params struct write only their bytes, and the
// calls refuse a null context.
TEST(dg_winpf_abi_cpu) {
  ss_dg_winpf_params p;
  ss_dg_winpf_defaults(&p);
  CHECK_EQ(p.struct_size, uint32_t(sizeof p));
  CHECK_EQ(p.enable, 0u);
  CHECK_EQ(p.top_k, 0u);
  CHECK_EQ(p.l1_cap, 1u);
  CHECK_EQ(p.prior_cap, 1u);
  CHECK_EQ(p.revert_unused, 1u);
  CHECK(p.l1_pmin == 0.4f && p.prior_pmin == 0.7f && p.copy_gbps == 30.0f && p.prior_window_us == 1500u);
  std::vector<uint8_t> ob(sizeof(ss_options) + 32, 0xC3);
  const uint32_t old = uint32_t(offsetof(ss_options, dg_prior_len));
  ss_options_defaults_sized(reinterpret_cast<ss_options*>(ob.data()), old);
  bool canary = true;
  for (size_t i = old; i < ob.size(); ++i) canary = canary && ob[i] == 0xC3;
  CHECK(canary);
  CHECK_EQ(reinterpret_cast<ss_options*>(ob.data())->struct_size, old);
  ss_options o;
  ss_options_defaults(&o);
  CHECK(o.dg_prior == nullptr && o.dg_prior_len == 0);
  CHECK_EQ(int(ss_dg_winpf_set(nullptr, &p)), int(SS_E_INVALID));
  int32_t sym = 1;
  CHECK_EQ(int(ss_dg_window_hint(nullptr, &sym, nullptr, 1)), int(SS_E_INVALID));
}

TEST(dg_reset_open_null_refused) {
  CHECK_EQ(int(ss_dg_reset_open(nullptr)), int(SS_E_INVALID));
}

// Scored eviction's open options: weights (each 0..1000, sum 1000; all 0 = the defaults), the profile (needs SCORED,
// both fields, a hashed descriptor, the exact format and identity); valid settings reach the backend refusal. An older
// caller's struct ends before these fields: their bytes are not read.
TEST(dg_open_scored_options) {
  DgModel m("open-scored");
  const std::string hashed = m.json_hashed();
  auto scored = [](O& o) { o.policy = SS_POLICY_SCORED; };
  expect(try_open(m, [](B&, O& o, T&) { o.policy = SS_POLICY_SCORED_AUTO; }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [&](B&, O& o, T&) { scored(o); }), SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [&](B&, O& o, T&) {
           scored(o);
           o.dg_score_w_trans = o.dg_score_w_freq = o.dg_score_w_recency = 0;
         }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, [&](B&, O& o, T&) {
           scored(o);
           o.dg_score_w_recency = 200;
         }),
         SS_E_INVALID, "sum to 1000");
  expect(try_open(m, [&](B&, O& o, T&) {
           scored(o);
           o.dg_score_w_trans = 1001;
           o.dg_score_w_freq = 0;
           o.dg_score_w_recency = 0;
         }),
         SS_E_INVALID, "each be 0..1000");
  // the profile of this model: identity from its descriptor's layers and hashes, zero counts
  std::vector<uint32_t> units(size_t(m.G), uint32_t(m.U));
  std::vector<std::vector<uint64_t>> h(size_t(m.G));
  std::vector<ts::DgScoreIdLayer> idl;
  for (int L = 0; L < m.G; ++L) {
    h[size_t(L)] = m.hashes(L);
    idl.push_back({uint32_t(L), uint32_t(m.U), m.unit(), &h[size_t(L)]});
  }
  const std::string id = ts::dg_score_identity(idl);
  std::vector<std::vector<uint32_t>> cells(size_t(m.G), std::vector<uint32_t>(size_t(m.U) * size_t(m.U), 1));
  const std::string prof = ts::dg_profile_write(units, id, "open test", cells);
  auto with = [&](const std::string& bytes) {
    return [&bytes, scored](B&, O& o, T&) {
      scored(o);
      o.dg_score_profile = bytes.data();
      o.dg_score_profile_len = bytes.size();
    };
  };
  expect(try_open(m, with(prof), 16, 4, 8, &hashed), SS_E_UNSUPPORTED, "needs the CUDA backend");
  expect(try_open(m, with(prof), 16, 4, 8, nullptr), SS_E_INVALID, "per-unit hashes");
  expect(try_open(m, [&](B&, O& o, T&) {
           o.dg_score_profile = prof.data();
           o.dg_score_profile_len = prof.size();
         }, 16, 4, 8, &hashed),
         SS_E_INVALID, "needs policy SCORED");
  expect(try_open(m, [&](B&, O& o, T&) {
           scored(o);
           o.dg_score_profile_len = prof.size();
         }, 16, 4, 8, &hashed),
         SS_E_INVALID, "given together");
  const std::string shorter = prof.substr(0, prof.size() - 4);
  expect(try_open(m, with(shorter), 16, 4, 8, &hashed), SS_E_INVALID, "size");
  std::string flipped = prof;
  flipped[50] ^= 1;  // inside the identity
  expect(try_open(m, with(flipped), 16, 4, 8, &hashed), SS_E_INVALID, "identity mismatch");
  std::vector<uint32_t> other = units;
  other[0] = 63;
  std::vector<std::vector<uint32_t>> c2 = cells;
  c2[0].resize(63 * 63);
  const std::string wrong_units = ts::dg_profile_write(other, id, "", c2);
  expect(try_open(m, with(wrong_units), 16, 4, 8, &hashed), SS_E_INVALID, "units of gated layer 0");
  expect(try_open(m, [&](B&, O& o, T&) {
           o.struct_size = uint32_t(offsetof(ss_options, dg_score_w_trans));
           o.dg_score_w_trans = 5;
           o.dg_score_profile_len = 3;
         }),
         SS_E_UNSUPPORTED, "needs the CUDA backend");
}
