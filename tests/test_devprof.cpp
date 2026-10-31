// Device profiles (ss.h "Device profiles"): the strix/device-profile-v1 text (canonical round trip, strict keys,
// hash), the override merge, the shipped class files against the class table, the planner hooks (caps, the unified
// pool in ss_budget_plan, single box vs pair, the prefill piece) and the C API's detection on a fake Linux tree with
// an override file. The Halo rows must reproduce today's defaults.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "devprof.hpp"
#include "json.hpp"
#include "recipe.hpp"
#include "strix_stream/ss.h"

#if defined(_WIN32)
#include <process.h>
#define TS_GETPID _getpid
#else
#include <unistd.h>
#define TS_GETPID getpid
#endif

namespace fs = std::filesystem;

namespace {

constexpr uint64_t kGiB = uint64_t(1) << 30;
constexpr uint64_t kHaloGtt = 125798105088ull;  // 117.16 GiB (kho=off)
constexpr uint64_t kCarve = 512ull << 20;

// A Strix Halo node as the detector reports it (memory = GTT + carve-out).
std::string halo_text(uint64_t avail = 126500000ull * 1024) {
  return std::string("{\"schema\":\"strix/device-profile-v1\",\"class\":\"strix-halo-395\",\"detected\":true,") +
         "\"generic\":false,\"host\":{\"ram_bytes\":" + std::to_string(std::max<uint64_t>(133120000ull * 1024, avail)) +
         ",\"ram_available_bytes\":" + std::to_string(avail) +
         ",\"unified\":true,\"cpu_cores\":16,\"nvme\":[{\"path\":\"nvme0\",\"model\":\"x\",\"read_gbps\":6.8}]},"
         "\"devices\":[{\"api\":\"hip\",\"id\":\"kfd-node-1\",\"name\":\"AMD Radeon 8060S\",\"arch\":\"gfx1151\","
         "\"compute_units\":40,\"wave\":32,\"lds_bytes\":65536,\"memory_bytes\":" +
         std::to_string(kHaloGtt + kCarve) + ",\"gtt_bytes\":" + std::to_string(kHaloGtt) +
         ",\"visible_bytes\":" + std::to_string(kCarve) +
         ",\"mem_gbps\":244,\"h2d_gbps\":null}],\"links\":[{\"kind\":\"usb4\",\"count\":2,\"gbps_measured\":31.5}],"
         "\"margins\":{\"host_floor_bytes\":" + std::to_string(8 * kGiB) +
         ",\"device_reserve_bytes\":" + std::to_string(2 * kGiB) +
         "},\"limits\":{\"max_piece_rows\":16384,\"uma_ceiling_bytes\":" + std::to_string(120 * kGiB) +
         ",\"persist_blocks\":40}}";
}
std::string rtx_text() {
  return std::string("{\"schema\":\"strix/device-profile-v1\",\"class\":\"rtx-pro-6000-blackwell\",") +
         "\"host\":{\"ram_bytes\":" + std::to_string(128 * kGiB) + ",\"ram_available_bytes\":" +
         std::to_string(100 * kGiB) + "},\"devices\":[{\"api\":\"cuda\",\"name\":\"NVIDIA RTX PRO 6000 Blackwell\","
         "\"arch\":\"sm_120\",\"memory_bytes\":" + std::to_string(97887272960ull) +
         "}],\"margins\":{\"host_floor_bytes\":" + std::to_string(8 * kGiB) +
         ",\"device_reserve_bytes\":" + std::to_string(2 * kGiB) + "}}";
}

ss_devprof* load(const std::string& t) {
  ss_devprof* p = nullptr;
  const ss_status st = ss_devprof_from_json(t.data(), t.size(), &p);
  CHECK_EQ(int(st), int(SS_OK));
  if (st != SS_OK) std::printf("  %s\n", ss_last_error());
  return p;
}
std::string text(const ss_devprof* p) {
  char* s = nullptr;
  size_t n = 0;
  CHECK_EQ(int(ss_devprof_to_json(p, &s, &n)), int(SS_OK));
  std::string r(s ? s : "", n);
  std::free(s);
  return r;
}

struct Tree {
  fs::path root;
  explicit Tree(const char* tag) {
    root = fs::temp_directory_path() / ("ts-devprof-" + std::to_string(TS_GETPID()) + "-" + tag);
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
  }
  ~Tree() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  void put(const std::string& rel, const std::string& t) const {
    const fs::path p = root / rel;
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << t;
  }
  std::string str() const { return root.string(); }
};

}  // namespace

TEST(text_round_trips_canonically_with_a_stable_hash) {
  ss_devprof* p = load(halo_text());
  const std::string a = text(p);
  ss_devprof* q = load(a);
  CHECK(text(q) == a);
  char h1[17], h2[17];
  CHECK_EQ(int(ss_devprof_hash(p, h1)), int(SS_OK));
  CHECK_EQ(int(ss_devprof_hash(q, h2)), int(SS_OK));
  CHECK(std::strcmp(h1, h2) == 0);
  CHECK_EQ(std::strlen(h1), size_t(16));
  CHECK(a.find("\"h2d_gbps\":null") != std::string::npos);  // unknown bandwidth = null
  CHECK(a.find("\"read_gbps\":6.8") != std::string::npos);   // readable, not 6.7999...
  ss_devprof_info i{};
  i.struct_size = sizeof i;
  CHECK_EQ(int(ss_devprof_get(p, &i)), int(SS_OK));
  CHECK(std::string(i.class_name) == "strix-halo-395");
  CHECK(std::string(i.hash) == h1);
  CHECK_EQ(i.unified, 1u);
  CHECK_EQ(i.n_devices, 1u);
  CHECK_EQ(i.n_links, 1u);
  ss_devprof_device d{};
  d.struct_size = sizeof d;
  CHECK_EQ(int(ss_devprof_get_device(p, 0, &d)), int(SS_OK));
  CHECK_EQ(int(d.api), int(SS_BACKEND_HIP));
  CHECK(std::string(d.arch) == "gfx1151");
  CHECK_EQ(d.compute_units, 40u);
  CHECK_EQ(d.memory_bytes, kHaloGtt + kCarve);
  CHECK_EQ(int(ss_devprof_get_device(p, 1, &d)), int(SS_E_INVALID));
  uint64_t v = 0;
  CHECK_EQ(int(ss_devprof_limit(p, "max_piece_rows", &v)), int(SS_OK));
  CHECK_EQ(v, uint64_t(16384));
  CHECK_EQ(int(ss_devprof_limit(p, "no_such_limit", &v)), int(SS_E_INVALID));
  CHECK_EQ(v, uint64_t(16384));
  ss_devprof_free(p);
  ss_devprof_free(q);
}

TEST(strict_reader_refuses_bad_texts) {
  const std::vector<std::string> bad = {
      "{}",
      "{\"schema\":\"strix/device-profile-v2\"}",
      "{\"schema\":\"strix/device-profile-v1\",\"colour\":1}",
      "{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"metal\"}]}",
      "{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"hip\",\"cus\":40}]}",
      "{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"hip\",\"mem_gbps\":-1}]}",
      "{\"schema\":\"strix/device-profile-v1\",\"host\":{\"ram_bytes\":1,\"ram_available_bytes\":2}}",
      "{\"schema\":\"strix/device-profile-v1\",\"links\":[{\"kind\":\"wifi\"}]}",
      "{\"schema\":\"strix/device-profile-v1\",\"limits\":{\"max_piece_rows\":-1}}",
      "not json",
  };
  for (const auto& t : bad) {
    ss_devprof* p = reinterpret_cast<ss_devprof*>(1);
    CHECK_EQ(int(ss_devprof_from_json(t.data(), t.size(), &p)), int(SS_E_RECIPE));
    CHECK(p == nullptr);
  }
}

TEST(override_merges_and_records_paths_atomically) {
  ss_devprof* p = load(halo_text());
  const std::string before = text(p);
  // a refused override changes nothing
  const std::string bad = "{\"devices\":[{\"memory_bytes\":\"lots\"}]}";
  CHECK_EQ(int(ss_devprof_merge_json(p, bad.data(), bad.size(), "bad.json")), int(SS_E_RECIPE));
  CHECK(text(p) == before);
  const std::string lib = "{\"notices\":[\"x\"]}";
  CHECK_EQ(int(ss_devprof_merge_json(p, lib.data(), lib.size(), "x.json")), int(SS_OK));  // provenance: ignored
  CHECK(text(p) == before);
  // a partial one: one device field, a new link, a margin, a limit
  const std::string ov =
      "{\"devices\":[{\"memory_bytes\":" + std::to_string(200 * kGiB) +
      "}],\"links\":[{},{\"kind\":\"ethernet\",\"count\":1}],\"margins\":{\"host_floor_bytes\":" +
      std::to_string(16 * kGiB) + "},\"limits\":{\"max_piece_rows\":32768}}";
  CHECK_EQ(int(ss_devprof_merge_json(p, ov.data(), ov.size(), "host-a.json")), int(SS_OK));
  const std::string t = text(p);
  ts::devprof::Profile q = ts::devprof::from_json(ts::json_parse(t));
  CHECK_EQ(q.devices.size(), size_t(1));
  CHECK_EQ(q.devices[0].memory_bytes, 200 * kGiB);
  CHECK(q.devices[0].arch == "gfx1151");  // untouched fields stay
  CHECK_EQ(q.links.size(), size_t(2));
  CHECK(q.links[1].kind == "ethernet");
  CHECK_EQ(q.host_floor_bytes, 16 * kGiB);
  CHECK_EQ(q.limits["max_piece_rows"], uint64_t(32768));
  CHECK_EQ(q.limits["persist_blocks"], uint64_t(40));
  const std::vector<std::string> want = {"host-a.json:devices[0].memory_bytes", "host-a.json:limits.max_piece_rows",
                                         "host-a.json:links[1].count", "host-a.json:links[1].kind",
                                         "host-a.json:margins.host_floor_bytes"};
  CHECK(q.overrides == want);
  ss_devprof_free(p);
}

TEST(class_files_are_the_class_table) {
  // profiles/<class>.json in the repo, byte for byte (canonical JSON + newline)
  size_t n = 0;
  for (const auto& c : ts::devprof::classes()) {
    const fs::path f = fs::path("profiles") / (std::string(c.name) + ".json");
    std::ifstream in(f, std::ios::binary);
    CHECK(bool(in));
    if (!in) {
      std::printf("  missing %s; expected:\n%s\n", f.string().c_str(), ts::json_write(ts::devprof::class_json(c)).c_str());
      continue;
    }
    std::stringstream s;
    s << in.rdbuf();
    CHECK(s.str() == ts::json_write(ts::devprof::class_json(c)) + "\n");
    ++n;
  }
  size_t files = 0;
  for (const auto& e : fs::directory_iterator("profiles"))
    if (e.path().extension() == ".json") ++files;
  CHECK_EQ(files, n);
}

TEST(halo_class_reproduces_todays_defaults) {
  // the margins are tierstream's current defaults: the recipes' host free-RAM floor and ss_budget's device margin
  ss_budget b;
  ss_budget_defaults(&b);
  ts::Recipe r;
  for (const auto& c : ts::devprof::classes()) {
    CHECK_EQ(c.host_floor_bytes, uint64_t(r.host_free_ram_floor_gib * double(kGiB)));
    if (std::string(c.api) != "cpu") CHECK_EQ(c.device_reserve_bytes, uint64_t(b.margin_gib * double(kGiB)));
  }
  // a detected Halo without margins or limits gets the class's (today's engine values)
  ts::devprof::Profile p;
  ts::devprof::Device d;
  d.api = "hip";
  d.arch = "gfx1151";
  d.memory_bytes = kHaloGtt + kCarve;
  p.devices.push_back(d);
  ts::devprof::apply_class(p);
  CHECK(p.cls == "strix-halo-395");
  CHECK(!p.generic);
  CHECK(p.unified);
  CHECK(p.notices.empty());
  CHECK_EQ(p.host_floor_bytes, 8 * kGiB);
  CHECK_EQ(p.device_reserve_bytes, 2 * kGiB);
  CHECK_EQ(p.limits["max_piece_rows"], uint64_t(16384));
  CHECK_EQ(p.limits["uma_ceiling_bytes"], 120 * kGiB);
  CHECK_EQ(p.limits["persist_blocks"], uint64_t(40));
  CHECK(p.devices[0].mem_gbps == 244.0);
  CHECK(p.evidence.find("devices[0].mem_gbps") != nullptr);
  // a successor arch is not silently a Halo: generic class + notice, no Halo limits
  ts::devprof::Profile g;
  d.arch = "gfx1160";
  d.memory_bytes = 250 * kGiB;
  g.devices.push_back(d);
  ts::devprof::apply_class(g);
  CHECK(g.cls == "generic-hip");
  CHECK(g.generic);
  CHECK(!g.notices.empty());
  CHECK(g.limits.empty());
  CHECK(g.devices[0].mem_gbps == 0);
  // the RTX classes by name / arch / memory, the 4090 not taken for a PRO
  ts::devprof::Profile x, y;
  ts::devprof::Device c;
  c.api = "cuda";
  c.arch = "sm_120";
  c.name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
  c.memory_bytes = 97887272960ull;
  x.devices.push_back(c);
  ts::devprof::apply_class(x);
  CHECK(x.cls == "rtx-pro-6000-blackwell");
  CHECK_EQ(x.limits["long_vram_cap_bytes"], 38 * kGiB);
  c.arch = "sm_89";
  c.name = "NVIDIA GeForce RTX 4090";
  c.memory_bytes = 25757220864ull;
  y.devices.push_back(c);
  ts::devprof::apply_class(y);
  CHECK(y.cls == "rtx-4090");
  CHECK(!y.unified);
}

TEST(budget_caps_from_the_profile) {
  // Halo: one pool = min(available - floor, memory - reserve), unified
  ss_devprof* h = load(halo_text());
  ss_budget_in in;
  ss_budget_in_defaults(&in);
  CHECK_EQ(int(ss_devprof_budget(h, 0, &in)), int(SS_OK));
  const uint64_t avail = 126500000ull * 1024;
  const uint64_t pool = std::min(avail - 8 * kGiB, kHaloGtt + kCarve - 2 * kGiB);
  CHECK_EQ(in.ram_total_bytes, pool);
  CHECK_EQ(in.vram_total_bytes, uint64_t(0));
  CHECK_EQ(in.unified, 1u);
  // explicit caps win
  ss_budget_in_defaults(&in);
  in.ram_total_bytes = 50 * kGiB;
  CHECK_EQ(int(ss_devprof_budget(h, 0, &in)), int(SS_OK));
  CHECK_EQ(in.ram_total_bytes, 50 * kGiB);
  CHECK_EQ(int(ss_devprof_budget(h, 1, &in)), int(SS_E_INVALID));
  // discrete: two caps
  ss_devprof* r = load(rtx_text());
  ss_budget_in_defaults(&in);
  CHECK_EQ(int(ss_devprof_budget(r, 0, &in)), int(SS_OK));
  CHECK_EQ(in.vram_total_bytes, 97887272960ull - 2 * kGiB);
  CHECK_EQ(in.ram_total_bytes, 92 * kGiB);
  CHECK_EQ(in.unified, 0u);
  // margins eating everything: refused, never a 0 (= uncapped) cap
  const std::string tiny = "{\"margins\":{\"device_reserve_bytes\":" + std::to_string(200 * kGiB) + "}}";
  CHECK_EQ(int(ss_devprof_merge_json(r, tiny.data(), tiny.size(), "t")), int(SS_OK));
  ss_budget_in_defaults(&in);
  CHECK_EQ(int(ss_devprof_budget(r, 0, &in)), int(SS_E_CAPACITY));
  // an older caller's struct (before SS_BUDGET_UNIFIED): a unified profile is refused (two independent caps of the
  // pool could each take all of it); a discrete one fills it, writing nothing at or past the prefix
  ss_budget_in s;
  std::memset(static_cast<void*>(&s), 0xA5, sizeof s);
  ss_budget_in_defaults_sized(&s, uint32_t(offsetof(ss_budget_in, unified)));
  CHECK_EQ(int(ss_devprof_budget(h, 0, &s)), int(SS_E_INVALID));
  CHECK_EQ(s.ram_total_bytes, uint64_t(0));
  ss_devprof* r2 = load(rtx_text());
  CHECK_EQ(int(ss_devprof_budget(r2, 0, &s)), int(SS_OK));
  CHECK_EQ(s.vram_total_bytes, 97887272960ull - 2 * kGiB);
  CHECK_EQ(s.struct_size, uint32_t(offsetof(ss_budget_in, unified)));
  const unsigned char* tail = reinterpret_cast<const unsigned char*>(&s) + offsetof(ss_budget_in, unified);
  bool untouched = true;
  for (size_t k = 0; k < sizeof s - offsetof(ss_budget_in, unified); ++k) untouched = untouched && tail[k] == 0xA5;
  CHECK(untouched);
  ss_devprof_free(r2);
  ss_devprof_free(h);
  ss_devprof_free(r);
}

TEST(unified_pool_plans_device_then_host) {
  ss_budget_component dev[] = {{"trunk", 20 * kGiB}, {"kv", 4 * kGiB}};
  ss_budget_component pin[] = {{"staging", 1 * kGiB}};
  ss_budget_in in;
  ss_budget_in_defaults(&in);
  in.device_fixed = dev;
  in.n_device_fixed = 2;
  in.pinned_fixed = pin;
  in.n_pinned_fixed = 1;
  in.keys = 1000;
  in.slot_bytes = in.unit_bytes = 100ull << 20;
  in.arena_spares = 4;
  in.host_spares = 2;
  in.vram_margin_bytes = 2 * kGiB;
  in.ram_margin_bytes = 1 * kGiB;
  // with no cap the flag changes nothing
  ss_budget_out a{}, b{};
  a.struct_size = b.struct_size = sizeof a;
  CHECK_EQ(int(ss_budget_plan(&in, &a)), int(SS_OK));
  in.unified = 1;
  CHECK_EQ(int(ss_budget_plan(&in, &b)), int(SS_OK));
  CHECK(std::memcmp(&a, &b, sizeof a) == 0);
  // a 100 GiB pool: the arena gets pool - host fixed - ram margin - device fixed - vram margin - spares
  in.ram_total_bytes = 100 * kGiB;
  ss_budget_out o{};
  o.struct_size = sizeof o;
  CHECK_EQ(int(ss_budget_plan(&in, &o)), int(SS_OK));
  // (and the host's smallest placement: 2 spare blocks + 8 staging + 1 block)
  const uint64_t arena_room = 100 * kGiB - 1 * kGiB - 1 * kGiB - 11 * in.unit_bytes - 24 * kGiB - 2 * kGiB -
                              4 * in.slot_bytes;
  CHECK_EQ(o.resident, arena_room / in.slot_bytes);
  CHECK_EQ(o.device_capped, 1u);
  // the pinned tier gets what the device's plan leaves: never more than the pool in total
  CHECK(o.device_planned_bytes + 2 * kGiB + o.host_planned_bytes + 1 * kGiB <= 100 * kGiB);
  CHECK_EQ(o.disk + o.pinned + o.resident, in.keys);
  // the same pool given as the VRAM cap (or both, the smaller) plans the same
  ss_budget_in v = in;
  v.ram_total_bytes = 0;
  v.vram_total_bytes = 100 * kGiB;
  ss_budget_out ov{};
  ov.struct_size = sizeof ov;
  CHECK_EQ(int(ss_budget_plan(&v, &ov)), int(SS_OK));
  CHECK(std::memcmp(&o, &ov, sizeof o) == 0);
  // the receipt names the mode; a two-cap plan's text has no such key
  char buf[8192];
  ss_budget_plan_json(&in, &o, buf, sizeof buf);
  CHECK(std::strstr(buf, "\"unified\":true") != nullptr);
  in.unified = 0;
  ss_budget_plan_json(&in, &o, buf, sizeof buf);
  CHECK(std::strstr(buf, "unified") == nullptr);
  // too small: refused with the pool's minimum (device and host parts together)
  in.unified = 1;
  in.ram_total_bytes = 20 * kGiB;
  ss_budget_out r{};
  r.struct_size = sizeof r;
  CHECK_EQ(int(ss_budget_plan(&in, &r)), int(SS_E_CAPACITY));
  CHECK(std::strstr(r.refuse, "unified pool 20.00 GiB") == r.refuse);
  CHECK_EQ(r.min_vram_total_bytes, r.min_ram_total_bytes);
  CHECK(r.min_ram_total_bytes > 26 * kGiB);
  in.ram_total_bytes = r.min_ram_total_bytes;
  CHECK_EQ(int(ss_budget_plan(&in, &r)), int(SS_OK));
}

TEST(single_box_or_pair_is_a_plan_output) {
  ss_devprof* h0 = load(halo_text());
  ss_devprof* h1 = load(halo_text());
  const ss_devprof* boxes[] = {h0, h1};
  ss_devprof_topology_in in;
  ss_devprof_topology_in_defaults(&in);
  in.layers = 43;
  ss_devprof_topology_out o{};
  o.struct_size = sizeof o;
  // DeepSeek's guard: 160 GiB per box single, 86 per box pair (+4): one Halo cannot, two can
  in.fixed_bytes = 164 * kGiB;
  CHECK_EQ(int(ss_devprof_topology(boxes, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.world, 2u);
  CHECK_EQ(o.fits, 1u);
  CHECK_EQ(o.first_layer[0], 0u);
  CHECK_EQ(o.first_layer[1], 22u);  // equal boxes: ceil(43 / 2) on box 0
  CHECK_EQ(o.first_layer[2], 43u);
  // a small model stays on one box
  in.fixed_bytes = 60 * kGiB;
  CHECK_EQ(int(ss_devprof_topology(boxes, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.world, 1u);
  CHECK_EQ(o.first_layer[1], 43u);
  // a pinned recipe keeps its world
  in.pinned_world = 2;
  CHECK_EQ(int(ss_devprof_topology(boxes, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.world, 2u);
  in.pinned_world = 3;
  CHECK_EQ(int(ss_devprof_topology(boxes, 2, &in, &o)), int(SS_E_INVALID));
  in.pinned_world = 0;
  // a 256 GiB successor holds DeepSeek alone
  ss_devprof* g = load(halo_text(250 * kGiB));
  const std::string big = "{\"class\":\"generic-hip\",\"devices\":[{\"memory_bytes\":" + std::to_string(250 * kGiB) +
                          ",\"arch\":\"gfx1160\"}]}";
  CHECK_EQ(int(ss_devprof_merge_json(g, big.data(), big.size(), "g")), int(SS_OK));
  const ss_devprof* gb[] = {g, h1};
  in.fixed_bytes = 164 * kGiB;
  CHECK_EQ(int(ss_devprof_topology(gb, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.world, 1u);
  // unequal boxes split by budget share
  in.fixed_bytes = 300 * kGiB;
  in.pinned_world = 2;
  CHECK_EQ(int(ss_devprof_topology(gb, 2, &in, &o)), int(SS_OK));
  CHECK(o.first_layer[1] > 22u && o.first_layer[1] < 43u);
  in.pinned_world = 0;
  // no measured link: no pair
  const std::string nolink = "{\"links\":[{\"count\":0}]}";
  CHECK_EQ(int(ss_devprof_merge_json(h0, nolink.data(), nolink.size(), "n")), int(SS_OK));
  in.fixed_bytes = 164 * kGiB;
  CHECK_EQ(int(ss_devprof_topology(boxes, 2, &in, &o)), int(SS_E_CAPACITY));
  CHECK_EQ(o.fits, 0u);
  CHECK(std::strstr(o.why, "does not fit") != nullptr);
  CHECK_EQ(int(ss_devprof_topology(boxes, 0, &in, &o)), int(SS_E_INVALID));
  ss_devprof_free(h0);
  ss_devprof_free(h1);
  ss_devprof_free(g);
}

TEST(prefill_piece_from_the_room_left) {
  ss_devprof* h = load(halo_text());
  ss_devprof_piece_in in;
  ss_devprof_piece_in_defaults(&in);
  CHECK_EQ(in.row_align, 1u);
  in.planned_bytes = 80 * kGiB;
  in.bytes_per_row = 1ull << 20;
  in.piece_reserve_bytes = 3 * kGiB;
  in.min_rows = 2048;
  in.max_rows = 16384;  // today's Halo bound binds: unchanged output
  in.row_align = 2048;
  uint32_t rows = 0;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_OK));
  CHECK_EQ(rows, 16384u);
  // less room: aligned down
  in.planned_bytes = 100 * kGiB;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_OK));
  CHECK_EQ(rows % 2048, 0u);
  CHECK(rows < 16384u && rows >= 2048u);
  // nothing left: refused
  in.planned_bytes = 113 * kGiB;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_E_CAPACITY));
  CHECK_EQ(rows, 0u);
  in.bytes_per_row = 0;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_E_INVALID));
  ss_devprof_free(h);
}

TEST(detect_on_a_fake_node_with_an_override_file) {
  Tree t("detect");
  t.put("proc/meminfo", "MemTotal:       133120000 kB\nMemAvailable:   126500000 kB\n");
  t.put("sys/class/kfd/kfd/topology/nodes/0/properties", "cpu_cores_count 32\nsimd_count 0\ngfx_target_version 0\n");
  t.put("sys/class/kfd/kfd/topology/nodes/1/properties",
        "cpu_cores_count 0\nsimd_count 80\nsimd_per_cu 2\nwave_front_size 32\nlds_size_in_kb 64\n"
        "gfx_target_version 110501\ndrm_render_minor 128\n");
  t.put("sys/class/drm/renderD128/device/mem_info_gtt_total", std::to_string(kHaloGtt) + "\n");
  t.put("sys/class/drm/renderD128/device/mem_info_vram_total", std::to_string(kCarve) + "\n");
  t.put("sys/class/drm/renderD128/device/mem_info_vis_vram_total", std::to_string(kCarve) + "\n");
  ss_devprof_detect_opts o;
  ss_devprof_detect_opts_defaults(&o);
  const std::string root = t.str();
  o.root = root.c_str();
  o.flags = SS_DEVPROF_NO_RUNTIME | SS_DEVPROF_NO_ENV | SS_DEVPROF_NO_CACHE;
  ss_devprof* p = nullptr;
  CHECK_EQ(int(ss_devprof_detect(&o, &p)), int(SS_OK));
  ss_devprof_info i{};
  i.struct_size = sizeof i;
  CHECK_EQ(int(ss_devprof_get(p, &i)), int(SS_OK));
  CHECK(std::string(i.class_name) == "strix-halo-395");
  CHECK_EQ(i.detected, 1u);
  CHECK_EQ(i.unified, 1u);
  CHECK_EQ(i.ram_available_bytes, 126500000ull * 1024);
  ss_budget_in in;
  ss_budget_in_defaults(&in);
  CHECK_EQ(int(ss_devprof_budget(p, 0, &in)), int(SS_OK));
  CHECK_EQ(in.ram_total_bytes, std::min<uint64_t>(126500000ull * 1024 - 8 * kGiB, kHaloGtt + kCarve - 2 * kGiB));
  ss_devprof_free(p);
  // --device-profile FILE: partial, merged over the detected one
  t.put("override.json", "{\"margins\":{\"host_floor_bytes\":" + std::to_string(12 * kGiB) + "}}");
  const std::string ov = (t.root / "override.json").string();
  o.override_path = ov.c_str();
  CHECK_EQ(int(ss_devprof_detect(&o, &p)), int(SS_OK));
  i.struct_size = sizeof i;
  CHECK_EQ(int(ss_devprof_get(p, &i)), int(SS_OK));
  CHECK_EQ(i.host_floor_bytes, 12 * kGiB);
  const std::string j = text(p);
  CHECK(j.find("override.json:margins.host_floor_bytes") != std::string::npos);
  // the probe report the recipes match against
  char* rep = nullptr;
  size_t n = 0;
  CHECK_EQ(int(ss_devprof_probe_report(p, "node1", &rep, &n)), int(SS_OK));
  const ts::Json r = ts::json_parse(std::string(rep, n));
  std::free(rep);
  CHECK(r.find("devices") && r.find("devices")->arr.size() == 1);
  const ts::Json& d0 = r.find("devices")->arr[0];
  CHECK(d0.find("arch") && d0.find("arch")->text == "gfx1151");
  CHECK(d0.find("host") && d0.find("host")->text == "node1");
  CHECK(d0.find("kind") && d0.find("kind")->text == "hip");
  ss_devprof_free(p);
  // a missing or bad override is an error, nothing returned
  o.override_path = "no-such-profile.json";
  p = reinterpret_cast<ss_devprof*>(1);
  CHECK_EQ(int(ss_devprof_detect(&o, &p)), int(SS_E_IO));
  CHECK(p == nullptr);
  t.put("bad.json", "{\"devices\":7}");
  const std::string bd = (t.root / "bad.json").string();
  o.override_path = bd.c_str();
  CHECK_EQ(int(ss_devprof_detect(&o, &p)), int(SS_E_RECIPE));
  CHECK(p == nullptr);
}

namespace {
ss_status unified_plan(ss_budget_in in, uint64_t pool, ss_budget_out* o) {
  in.unified = 1;
  in.ram_total_bytes = pool;
  *o = ss_budget_out{};
  o->struct_size = sizeof *o;
  return ss_budget_plan(&in, o);
}
// what a successful unified plan holds: device + margin + host + margin within the pool, every expert placed
void check_within(const ss_budget_in& in, uint64_t pool, const ss_budget_out& o) {
  CHECK(o.device_planned_bytes + in.vram_margin_bytes + o.host_planned_bytes + in.ram_margin_bytes <= pool);
  CHECK_EQ(o.resident + o.pinned + o.disk, in.keys);
}
}  // namespace

TEST(unified_pool_review_counterexamples) {
  ss_budget_in in;
  ss_budget_out o;
  // the derived caps' 1-byte floor never lets a plan past its pool
  ss_budget_in_defaults(&in);
  in.device_context_bytes = 1;
  ss_budget_component one[] = {{"x", 1}};
  in.pinned_fixed = one;
  in.n_pinned_fixed = 1;
  CHECK_EQ(int(unified_plan(in, 1, &o)), int(SS_E_CAPACITY));
  CHECK_EQ(o.min_ram_total_bytes, uint64_t(2));
  CHECK_EQ(int(unified_plan(in, 2, &o)), int(SS_OK));
  check_within(in, 2, o);
  // no experts: no file staging is reserved; the minimum is the fixed parts and attainable
  ss_budget_in_defaults(&in);
  in.unit_bytes = 1;
  in.device_context_bytes = 92;
  CHECK_EQ(int(unified_plan(in, 100, &o)), int(SS_OK));
  CHECK_EQ(int(unified_plan(in, 50, &o)), int(SS_E_CAPACITY));
  CHECK_EQ(o.min_ram_total_bytes, uint64_t(92));
  CHECK_EQ(int(unified_plan(in, 92, &o)), int(SS_OK));
  // no file tier: every expert in host memory fits, the greedy arena must not starve the host
  ss_budget_in_defaults(&in);
  in.file_tier = 0;
  in.keys = 10;
  in.slot_bytes = 2;
  in.unit_bytes = 1;
  CHECK_EQ(int(unified_plan(in, 10, &o)), int(SS_OK));
  check_within(in, 10, o);
  CHECK_EQ(o.resident, uint64_t(0));
  CHECK_EQ(o.pinned, uint64_t(10));
  // everything resident needs no file staging
  ss_budget_in_defaults(&in);
  in.keys = 10;
  in.arena_units = 10;
  in.slot_bytes = in.unit_bytes = 1;
  CHECK_EQ(int(unified_plan(in, 10, &o)), int(SS_OK));
  CHECK_EQ(o.resident, uint64_t(10));
  // a grid: every successful plan stays within its pool, and every refusal's minimum is attainable
  int checked = 0;
  for (int file = 0; file < 2; ++file)
    for (uint64_t keys : {0, 1, 7, 40})
      for (uint64_t slot : {1, 3, 8})
        for (uint64_t unit : {1, 2, 8})
          for (uint64_t spares : {0, 2})
            for (uint64_t pool : {5, 30, 64, 200, 1000}) {
              ss_budget_in_defaults(&in);
              in.file_tier = uint32_t(file);
              in.keys = keys;
              in.slot_bytes = slot;
              in.unit_bytes = unit;
              in.arena_spares = spares;
              in.host_spares = uint32_t(spares);
              in.staging = 2;
              in.device_context_bytes = 3;
              in.vram_margin_bytes = 1;
              in.ram_margin_bytes = 2;
              if (unified_plan(in, pool, &o) == SS_OK) {
                check_within(in, pool, o);
              } else {
                const uint64_t least = o.min_ram_total_bytes;
                CHECK(least > pool);
                CHECK_EQ(int(unified_plan(in, least, &o)), int(SS_OK));
                check_within(in, least, o);
              }
              ++checked;
            }
  CHECK_EQ(checked, 720);
  // huge key counts plan in constant time
  ss_budget_in_defaults(&in);
  in.keys = uint64_t(1) << 40;
  in.slot_bytes = in.unit_bytes = 1;
  CHECK_EQ(int(unified_plan(in, uint64_t(1) << 39, &o)), int(SS_OK));
  check_within(in, uint64_t(1) << 39, o);
  // JSON: "unified" only when a unified cap planned
  ss_budget_in_defaults(&in);
  in.unified = 1;
  o = ss_budget_out{};
  o.struct_size = sizeof o;
  CHECK_EQ(int(ss_budget_plan(&in, &o)), int(SS_OK));
  char buf[4096];
  ss_budget_plan_json(&in, &o, buf, sizeof buf);
  CHECK(std::strstr(buf, "unified") == nullptr);
}

TEST(piece_rows_review_counterexamples) {
  ss_devprof* h = load(halo_text());
  ss_budget_in b;
  ss_budget_in_defaults(&b);
  CHECK_EQ(int(ss_devprof_budget(h, 0, &b)), int(SS_OK));
  ss_devprof_piece_in in;
  ss_devprof_piece_in_defaults(&in);
  // alignment after the max clamp: room for 16 rows, align 8, max 10 -> 8
  in.bytes_per_row = 1;
  in.planned_bytes = b.ram_total_bytes - 16;
  in.row_align = 8;
  in.max_rows = 10;
  uint32_t rows = 0;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_OK));
  CHECK_EQ(rows, 8u);
  // no max and a huge room: the largest uint32, never 0 with success
  in.planned_bytes = 0;
  in.max_rows = 0;
  in.row_align = 1;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_OK));
  CHECK_EQ(rows, UINT32_MAX);
  // bounds with no aligned value
  in.min_rows = 9;
  in.max_rows = 15;
  in.row_align = 16;
  CHECK_EQ(int(ss_devprof_piece_rows(h, &in, &rows)), int(SS_E_INVALID));
  ss_devprof_free(h);
  // a known size the margins exhaust is not "unknown"
  ss_devprof* t = load("{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"cuda\",\"memory_bytes\":" +
                       std::to_string(kGiB) + "}],\"margins\":{\"device_reserve_bytes\":" + std::to_string(2 * kGiB) +
                       "}}");
  ss_devprof_piece_in_defaults(&in);
  in.bytes_per_row = 1;
  in.max_rows = 4096;
  CHECK_EQ(int(ss_devprof_piece_rows(t, &in, &rows)), int(SS_E_CAPACITY));
  CHECK_EQ(rows, 0u);
  ss_devprof_free(t);
  // unknown size: the caller's aligned maximum
  ss_devprof* u = load("{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"cuda\"}]}");
  in.max_rows = 4100;
  in.row_align = 64;
  CHECK_EQ(int(ss_devprof_piece_rows(u, &in, &rows)), int(SS_OK));
  CHECK_EQ(rows, 4096u);
  ss_devprof_free(u);
}

TEST(topology_review_counterexamples) {
  const auto box = [](uint64_t mem, bool measured) {
    return load("{\"schema\":\"strix/device-profile-v1\",\"devices\":[{\"api\":\"cuda\",\"memory_bytes\":" +
                std::to_string(mem) + "}],\"links\":[{\"kind\":\"usb4\",\"count\":2,\"gbps_measured\":" +
                (measured ? std::string("20") : std::string("null")) + "}]}");
  };
  ss_devprof_topology_in in;
  ss_devprof_topology_in_defaults(&in);
  ss_devprof_topology_out o{};
  o.struct_size = sizeof o;
  // byte budgets, not MiB: 700,000 / 300,000 bytes split 10 layers 7 / 3
  ss_devprof* a = box(700000, true);
  ss_devprof* b = box(300000, true);
  const ss_devprof* ab[] = {a, b};
  in.layers = 10;
  in.fixed_bytes = 900000;
  CHECK_EQ(int(ss_devprof_topology(ab, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.world, 2u);
  CHECK_EQ(o.first_layer[1], 7u);
  // budgets near 2^64 split evenly, no product overflow
  ss_devprof* x = box(UINT64_MAX, true);
  ss_devprof* y = box(UINT64_MAX, true);
  const ss_devprof* xy[] = {x, y};
  in.layers = 1u << 19;
  in.pinned_world = 2;
  in.fixed_bytes = 1;
  CHECK_EQ(int(ss_devprof_topology(xy, 2, &in, &o)), int(SS_OK));
  CHECK_EQ(o.first_layer[1], 1u << 18);
  in.pinned_world = 0;
  // a demand past 2^64 is refused, not saturated into a fit; so is a layer count past the split's range
  in.fixed_bytes = UINT64_MAX;
  in.min_expert_bytes = 1;
  CHECK_EQ(int(ss_devprof_topology(xy, 2, &in, &o)), int(SS_E_INVALID));
  in.min_expert_bytes = 0;
  in.fixed_bytes = 1;
  in.layers = 1u << 20;
  CHECK_EQ(int(ss_devprof_topology(xy, 2, &in, &o)), int(SS_E_INVALID));
  // a discovered but unmeasured link does not qualify a pair
  ss_devprof* c = box(300000, false);
  ss_devprof* d = box(300000, false);
  const ss_devprof* cd[] = {c, d};
  in.layers = 10;
  in.fixed_bytes = 500000;
  CHECK_EQ(int(ss_devprof_topology(cd, 2, &in, &o)), int(SS_E_CAPACITY));
  CHECK_EQ(o.world, 1u);
  CHECK(std::strstr(o.why, "no measured link") != nullptr);
  for (ss_devprof* p : {a, b, x, y, c, d}) ss_devprof_free(p);
}

TEST(qualified_class_never_survives_an_identity_change) {
  // a gfx1151 outside the qualified memory range is not a Halo
  ts::devprof::Profile t;
  ts::devprof::Device d;
  d.api = "hip";
  d.arch = "gfx1151";
  d.memory_bytes = 60 * kGiB;
  t.devices.push_back(d);
  ts::devprof::apply_class(t);
  CHECK(t.cls == "generic-hip");
  CHECK(t.limits.empty());
  // sysfs could only read the carve-out: GTT + carve-out still identify the 128 GB Halo, memory completed
  ts::devprof::Profile s;
  d.memory_bytes = kCarve;
  d.gtt_bytes = kHaloGtt;
  d.visible_bytes = kCarve;
  s.devices.push_back(d);
  ts::devprof::apply_class(s);
  CHECK(s.cls == "strix-halo-395");
  CHECK_EQ(s.devices[0].memory_bytes, kHaloGtt + kCarve);
  // a detected Halo, then an override that changes only the arch: the Halo numbers go, nothing class-owned remains
  ss_devprof* p = load(ts::devprof::to_text(s));
  const std::string ov = "{\"devices\":[{\"arch\":\"gfx1160\"}]}";
  CHECK_EQ(int(ss_devprof_merge_json(p, ov.data(), ov.size(), "o")), int(SS_OK));
  ts::devprof::Profile q = ts::devprof::from_json(ts::json_parse(text(p)));
  CHECK(q.cls == "generic-hip");
  CHECK(q.limits.empty());
  CHECK(q.devices[0].mem_gbps == 0);
  CHECK(!q.unified);  // unified came from the class only
  for (const auto& kv : q.evidence.obj) CHECK(kv.second.text.find("strix-halo-395") == std::string::npos);
  CHECK_EQ(q.host_floor_bytes, 8 * kGiB);  // the generic class's margins
  // an override that sets a limit itself keeps it across the change
  ss_devprof* p2 = load(ts::devprof::to_text(s));
  const std::string ov2 = "{\"devices\":[{\"arch\":\"gfx1160\"}],\"limits\":{\"max_piece_rows\":32768}}";
  CHECK_EQ(int(ss_devprof_merge_json(p2, ov2.data(), ov2.size(), "o2")), int(SS_OK));
  uint64_t v = 0;
  CHECK_EQ(int(ss_devprof_limit(p2, "max_piece_rows", &v)), int(SS_OK));
  CHECK_EQ(v, uint64_t(32768));
  CHECK_EQ(int(ss_devprof_limit(p2, "persist_blocks", &v)), int(SS_E_INVALID));
  // forcing an unknown class is refused, nothing changed
  const std::string ov3 = "{\"class\":\"no-such-class\"}";
  const std::string before = text(p2);
  CHECK_EQ(int(ss_devprof_merge_json(p2, ov3.data(), ov3.size(), "o3")), int(SS_E_RECIPE));
  CHECK(text(p2) == before);
  ss_devprof_free(p);
  ss_devprof_free(p2);
}

TEST(saved_full_profile_works_as_an_override) {
  ss_devprof* p = load(halo_text());
  const std::string saved = text(p);
  const std::string seed = "{\"notices\":[],\"sources\":[\"x\"],\"overrides\":[\"y\"]}";
  CHECK_EQ(int(ss_devprof_merge_json(p, seed.data(), seed.size(), "seed")), int(SS_OK));  // metadata ignored
  CHECK(text(p) == saved);
  CHECK_EQ(int(ss_devprof_merge_json(p, saved.data(), saved.size(), "saved.json")), int(SS_OK));
  ts::devprof::Profile q = ts::devprof::from_json(ts::json_parse(text(p)));
  ts::devprof::Profile r = ts::devprof::from_json(ts::json_parse(saved));
  r.overrides = q.overrides;  // the merged paths are the only difference
  CHECK(ts::devprof::to_text(q) == ts::devprof::to_text(r));
  CHECK(!q.overrides.empty());
  ss_devprof_free(p);
}

int main(int argc, char** argv) {
  if (argc > 2 && std::string(argv[1]) == "--write-profiles") {  // regenerates DIR/<class>.json from the table
    for (const auto& c : ts::devprof::classes())
      std::ofstream(fs::path(argv[2]) / (std::string(c.name) + ".json"), std::ios::binary)
          << ts::json_write(ts::devprof::class_json(c)) << '\n';
    return 0;
  }
  return sstest::run_all(argc > 1 ? argv[1] : nullptr);
}
