// CPU tests of placement mode (wave4/placement/contract.md rev 4): the recipe-v1 placement keys (written only when
// set, strict refusals, settable paths, profile guard), optional device binding, the components-v1 reader, the
// planner against the frozen G4 fixtures of golden/place_ref.py (tests/data/placement), G5 determinism.
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "components.hpp"
#include "desc.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "place.hpp"
#include "recipe.hpp"
#include "sha256.hpp"
#include "solve.hpp"
#include "tierstream/ts.h"

using namespace ts;

namespace {

template <class F>
std::string error_of(F&& f) {
  try {
    f();
  } catch (const Error& e) {
    return std::string(e.what());
  } catch (const std::exception& e) {
    return std::string("non-library exception: ") + e.what();
  }
  return "";
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

#define CHECK_REFUSED(expr, part)                                                                          \
  do {                                                                                                     \
    std::string e_ = error_of([&] { expr; });                                                              \
    if (!has(e_, part)) {                                                                                  \
      std::printf("  %s:%d: expected a refusal containing \"%s\", got \"%s\"\n", __FILE__, __LINE__, part, \
                  e_.c_str());                                                                             \
      ++sstest::failures();                                                                                \
    }                                                                                                      \
  } while (0)

Recipe parse_recipe(const std::string& text) { return recipe_from_json(recipe_doc_parse(text)); }

// The contract's §2.5 recipe without its placement keys (a legacy recipe), and pieces that add them.
const char* kHead = R"(schema = "tierstream/recipe-v1"
name = "ds-pro"
[match]
model = "m"
engine = "e"
[[match.devices]]
id = "rtx"
kind = "cuda"
name = "RTX PRO 6000"
memory_min_gib = 94
)";
const char* kAux = R"([[match.devices]]
id = "aux"
kind = "cuda"
memory_min_gib = 8
optional = true
)";
const char* kRoute = R"([[routes]]
id = "decode_rtx"
phase = "decode"
devices = ["rtx"]
execute = "single"
)";
const char* kPlacement = R"([routes.placement.rtx]
device_gib = 90.0
host_pinned_gib = 68.0
max_context_tokens = 131072
max_batch = 1
)";

std::string legacy() { return std::string(kHead) + kRoute + kPlacement; }
const char* kPm = "[routes.assist.aux]\ndevice_gib = 0\n";  // makes the route (and so the recipe) placement mode
std::string with_aux(const std::string& route_extra = "", const std::string& tail = "") {
  return std::string(kHead) + kAux + kRoute + route_extra + kPlacement + tail;
}

std::string canon(const std::string& text) { return json_write(recipe_to_json(parse_recipe(text))); }

const char* kProbe2 = R"({"model":"m","engine":"e","devices":[
  {"id":"GPU-pro","kind":"cuda","name":"NVIDIA RTX PRO 6000 Blackwell","memory_gib":95.59,"host":"pc","ordinal":0,
   "uuid":"GPU-11111111-2222-3333-4444-555555555555","pci":"0000:01:00.0","arch":"sm_120"},
  {"id":"GPU-4090","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":23.99,"host":"pc","ordinal":1,
   "uuid":"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","pci":"0000:06:00.0","arch":"sm_89"}]})";

}  // namespace

// ---- schema: additive keys ----------------------------------------------------------------------------------------
TEST(placement_keys_written_only_when_set) {
  const std::string base = canon(legacy());
  for (const char* k : {"optional", "uuid", "pci", "index", "assist", "components", "max_rows", "max_sequences",
                        "host_pageable_gib", "host_available_floor_gib"})
    CHECK(!has(base, std::string("\"") + k + "\""));
  // Each key, once present, is in the canonical form (and so changes the effective hash).
  std::vector<std::string> variants = {
      with_aux(),
      std::string(kHead) + R"(uuid = "GPU-11111111-2222-3333-4444-555555555555"
)" + kRoute + kPlacement,
      std::string(kHead) + "pci = \"0000:01:00.0\"\n" + kRoute + kPlacement,
      std::string(kHead) + "index = 0\n" + kRoute + kPlacement,
      std::string(kHead) + kRoute + "max_rows = 6\n" + kPlacement + "[routes.components.drafter]\n",
      std::string(kHead) + kRoute + "max_sequences = 1\n" + kPlacement + "[routes.components.drafter]\n",
      legacy() + "host_pageable_gib = 6.0\n",
      legacy() + "host_available_floor_gib = 6.0\n",
      with_aux("", "[routes.assist.aux]\ndevice_gib = 12.0\n"),
      with_aux("", "[routes.assist.aux]\ndevice_gib = 12.0\n[routes.components.drafter]\ndevice = \"aux\"\n"),
      std::string(kHead) + kRoute + kPlacement + "[routes.components.arena]\ngib = 79.0\n",
  };
  std::vector<std::string> seen = {base};
  for (const auto& v : variants) {
    const std::string c = canon(v);
    CHECK(std::find(seen.begin(), seen.end(), c) == seen.end());
    seen.push_back(c);
    // the canonical JSON reads back to itself
    CHECK_EQ(canon(c), c);
  }
  RecipeStack a(legacy(), "a.toml"), b(legacy() + "host_pageable_gib = 6.0\n", "b.toml");
  CHECK(a.effective_sha256() != b.effective_sha256());
  // explicit defaults inside an assist table are written out (the table is what is present)
  const std::string c = canon(with_aux("", "[routes.assist.aux]\n"));
  CHECK(has(c, R"("assist":{"aux":{"device_gib":0,"margin_fraction":0.03,"margin_gib":0.5}})"));
  const std::string d = canon(with_aux("", "[routes.components.drafter]\n"));
  CHECK(has(d, R"("components":{"drafter":{"device":"auto"}})"));
}

TEST(placement_keys_refusals) {
  auto refuse = [](const std::string& text, const char* part) { CHECK_REFUSED(parse_recipe(text), part); };
  refuse(std::string(kHead) + "optional = 1\n" + kRoute + kPlacement, "match.devices.rtx.optional: expected true");
  refuse(std::string(kHead) + "uuid = \"GPU-1\"\n" + kRoute + kPlacement, "PL_E_SCHEMA \"GPU-1\" is not a CUDA UUID");
  refuse(std::string(kHead) + "uuid = \"GPU-11111111-2222-3333-4444-55555555555A\"\n" + kRoute + kPlacement,
         "is not a CUDA UUID");
  refuse(std::string(kHead) + "pci = \"0000:01:00\"\n" + kRoute + kPlacement, "is not a PCI address");
  refuse(std::string(kHead) + "pci = \"0000:0A:00.0\"\n" + kRoute + kPlacement, "is not a PCI address");
  refuse(std::string(kHead) + "index = -1\n" + kRoute + kPlacement, "match.devices.rtx.index: -1 is out of range");
  refuse(std::string(kHead) + "index = 1.5\n" + kRoute + kPlacement, "expected an integer");
  refuse(std::string(kHead) + kAux + "count = 2\n" + kRoute + kPlacement, "only on a cuda or hip set with count 1");
  refuse(std::string(kHead) + "optional = true\n" + kRoute + kPlacement,
         "route \"decode_rtx\" executes on it (only an assist set may be optional)");
  refuse(std::string(kHead) + kRoute + "max_rows = 0\n" + kPlacement, "decode_rtx.max_rows: 0 is out of range");
  // max_rows / max_sequences only on placement-mode routes (a legacy canonical form never changes)
  refuse(std::string(kHead) + kRoute + "max_rows = 6\n" + kPlacement, "decode_rtx.max_rows: PL_E_SCHEMA only on a");
  refuse(std::string(kHead) + kRoute + "max_sequences = 2\n" + kPlacement, "max_sequences: PL_E_SCHEMA only on a");
  refuse(with_aux("", "[routes.components.arena]\ngib = 1\nunits = 4\n"), "PL_E_SCHEMA gib and units are exclusive");
  refuse(with_aux("", "[routes.components.drafter]\nunits = 4\n"), "PL_E_SCHEMA only elastic components");
  refuse(std::string(kHead) + kRoute + "max_sequences = \"1\"\n" + kPlacement + "[routes.components.drafter]\n",
         "expected an integer");
  refuse(legacy() + "host_pageable_gib = -1\n", "host_pageable_gib: -1 is out of range");
  refuse(legacy() + "host_available_floor_gib = \"x\"\n", "expected a number");
  // assists
  refuse(with_aux("", "[routes.assist.nope]\ndevice_gib = 1\n"), "PL_E_UNKNOWN_DEVICE \"nope\"");
  refuse(with_aux("", "[routes.assist.rtx]\ndevice_gib = 1\n"), "PL_E_SCHEMA \"rtx\" executes the route");
  refuse(with_aux("", "[routes.assist.aux]\ndevice_gib = -1\n"), "assist.aux.device_gib: -1 is out of range");
  refuse(with_aux("", "[routes.assist.aux]\nmargin_fraction = 0.9\n"), "margin_fraction: 0.9 is out of range");
  refuse(with_aux("", "[routes.assist.aux]\nbogus = 1\n"), "assist.aux: unknown key \"bogus\"");
  refuse(with_aux("assist = 3\n"), "expected a table keyed by device set");
  // components
  refuse(with_aux("", "[routes.components.kv]\ndevice = \"auto\"\n"), "PL_E_UNKNOWN_COMPONENT \"kv\"");
  refuse(with_aux("", "[routes.components.drafter]\ndevice = \"aux\"\n"), "PL_E_UNKNOWN_DEVICE \"aux\"");
  refuse(with_aux("", "[routes.components.drafter]\ndevice = \"gpu1\"\n"), "PL_E_UNKNOWN_DEVICE \"gpu1\"");
  refuse(with_aux("", "[routes.components.drafter]\ngib = 8\n"), "PL_E_SCHEMA only elastic components");
  refuse(with_aux("", "[routes.components.arena]\ngib = -8\n"), "arena.gib: -8 is out of range");
  refuse(with_aux("", "[routes.components.arena]\nslots = 8\n"), "components.arena: unknown key \"slots\"");
  refuse(with_aux("", "[routes.components.arena]\nlayers = []\n"), "expected at least one layer");
  refuse(with_aux("", "[routes.components.drafter]\ndevice = 3\n"), "drafter.device: expected a string");
  // reserved destination words as set ids, in placement mode only
  const std::string reserved = std::string(kHead) + "[[match.devices]]\nid = \"file\"\nkind = \"cuda\"\n"
                               "memory_min_gib = 8\n" + kRoute + kPlacement;
  CHECK(error_of([&] { parse_recipe(reserved); }).empty());  // legacy: not checked
  refuse(reserved + "[routes.components.arena]\ngib = 1\n", "PL_E_SCHEMA \"file\" is a reserved destination word");
  // one anchor per placement-mode route
  refuse(std::string(kHead) + "[[match.devices]]\nid = \"pair\"\nkind = \"hip\"\ncount = 2\nmemory_min_gib = 8\n" +
             "[[routes]]\nid = \"d\"\nphase = \"decode\"\ndevices = [\"pair\"]\nexecute = \"tp\"\n"
             "[routes.placement.pair]\nmax_context_tokens = 1\nmax_batch = 1\n[routes.components.arena]\ngib = 1\n",
         "PL_E_SCHEMA a placement-mode route");
  // layers is reserved, not a schema error: it parses (the planner refuses it)
  CHECK(error_of([&] { parse_recipe(with_aux("", "[routes.components.dense]\nlayers = [[0, 3]]\n")); }).empty());
}

// The contract's own §2.5 example parses, and its canonical JSON names every placement key.
TEST(placement_contract_example) {
  const std::string t = with_aux("max_rows = 6\n",
                                 "host_pageable_gib = 6.0\nhost_available_floor_gib = 6.0\n"
                                 "[routes.assist.aux]\ndevice_gib = 12.0\n[routes.components.drafter]\ndevice = \"aux\"\n");
  Recipe r = parse_recipe(t);
  const RRoute& rt = r.routes[0];
  CHECK(rt.placement_mode());
  CHECK_EQ(rt.max_rows, int64_t(6));
  CHECK_EQ(rt.assists.size(), size_t(1));
  CHECK_EQ(rt.assist("aux")->device_gib, 12.0);
  CHECK_EQ(rt.component("drafter")->device, std::string("aux"));
  CHECK_EQ(rt.placements[0].get("host_pageable_gib"), 6.0);
  CHECK(!parse_recipe(legacy()).routes[0].placement_mode());
}

// ---- settable paths, profile guard --------------------------------------------------------------------------------
TEST(placement_settable_paths) {
  RecipeStack s(with_aux("", kPm), "r.toml");
  const std::string sha0 = s.effective_sha256();
  s.set("routes.decode_rtx.assist.aux.device_gib", "12", "--set");
  s.set("routes.decode_rtx.components.drafter.device", "\"aux\"", "--set");
  s.set("routes.decode_rtx.components.arena.gib", "70.5", "--set");
  s.set("routes.decode_rtx.max_rows", "6", "--set");
  s.set("routes.decode_rtx.max_sequences", "2", "--set");
  s.set("routes.decode_rtx.placement.rtx.host_pageable_gib", "6", "--set");
  s.set("routes.decode_rtx.placement.rtx.host_available_floor_gib", "4", "--set");
  CHECK(s.effective_sha256() != sha0);
  const RRoute& rt = s.effective().routes[0];
  CHECK_EQ(rt.assist("aux")->device_gib, 12.0);
  CHECK_EQ(rt.component("arena")->gib, 70.5);
  CHECK_EQ(rt.max_sequences, int64_t(2));
  CHECK_REFUSED(s.set("routes.decode_rtx.assist.nope.device_gib", "1", "--set"), "unknown device set \"nope\"");
  CHECK_REFUSED(s.set("routes.decode_rtx.assist.rtx.device_gib", "1", "--set"), "executes the route");
  CHECK_REFUSED(s.set("routes.decode_rtx.assist.aux.bogus", "1", "--set"), "unknown assist field");
  CHECK_REFUSED(s.set("routes.decode_rtx.components.kv.device", "\"auto\"", "--set"), "unknown component \"kv\"");
  CHECK_REFUSED(s.set("routes.decode_rtx.components.arena.layers", "[1]", "--set"), "not a settable component field");
  CHECK_REFUSED(s.set("routes.decode_rtx.components.drafter.gib", "1", "--set"), "PL_E_SCHEMA only elastic");
  CHECK_REFUSED(s.set("match.devices.aux.optional", "false", "--set"), "is not settable");
  // the explain lists the new leaves
  const std::string ex = json_write(s.explain());
  CHECK(has(ex, "\"path\":\"routes.decode_rtx.assist.aux.device_gib\""));
  CHECK(has(ex, "\"path\":\"routes.decode_rtx.components.drafter.device\""));
}

namespace {

const char* kIdentity = R"({"model":"m","engine":"e","workload":"decode",
 "engine_sha256":"e000000000000000000000000000000000000000000000000000000000000000",
 "model_sha256":"a000000000000000000000000000000000000000000000000000000000000000",
 "descriptors":{"rtx":["d1"]},
 "devices":[
  {"id":"GPU-pro","kind":"cuda","name":"NVIDIA RTX PRO 6000 Blackwell","memory_gib":95.59,"host":"pc","ordinal":0,
   "driver":"580.88","disk_read_gbytes_ps":6.8,"h2d_gbytes_ps":25.1,"host_ram_gib":127.7}]})";

std::string profile(const std::string& sha, const std::string& sets, bool aux_bound = false) {
  const std::string aux_dev = aux_bound ? "[\"GPU-4090\"]" : "{ unbound = true }";
  const std::string aux_drv = aux_bound ? "\"580.88\"" : "{ unbound = true }";
  return "schema = \"tierstream/recipe-profile-v1\"\nrecipe = \"ds-pro\"\nrecipe_sha256 = \"" + sha +
         "\"\nworkload = \"decode\"\n"
         "[key]\nengine_sha256 = \"e000000000000000000000000000000000000000000000000000000000000000\"\n"
         "model_sha256 = \"a000000000000000000000000000000000000000000000000000000000000000\"\n"
         "devices = { rtx = [\"GPU-pro\"], aux = " + aux_dev + " }\n"
         "drivers = { rtx = \"580.88\", aux = " + aux_drv + " }\n"
         "descriptors = { rtx = [\"d1\"]" + std::string(aux_bound ? ", aux = []" : "") + " }\n"
         "[probe.rtx]\ndisk_read_gbytes_ps = 6.8\nh2d_gbytes_ps = 25.1\n" +
         (aux_bound ? "[probe.aux]\ndisk_read_gbytes_ps = 6.8\nh2d_gbytes_ps = 25.0\n" : "[probe.aux]\nunbound = true\n") +
         "[set]\n" + sets;
}

}  // namespace

TEST(placement_profile_guard_and_unbound_keys) {
  const Json id = json_parse(kIdentity);
  const std::string text = with_aux("", "[routes.assist.aux]\ndevice_gib = 0\n");
  const std::string ov = "schema = \"tierstream/recipe-overrides-v1\"\nrecipe = \"ds-pro\"\n[set]\n";
  // A profile that tunes the route's layout: no override may change a layout scope or a budget, in either layer
  // order (the stack always applies the profile first).
  for (const char* path : {"\"routes.decode_rtx.assist.aux.device_gib\" = 4", "\"routes.decode_rtx.max_rows\" = 2",
                           "\"routes.decode_rtx.max_sequences\" = 2",
                           "\"routes.decode_rtx.components.drafter.device\" = \"auto\"",
                           "\"routes.decode_rtx.placement.rtx.host_pageable_gib\" = 2",
                           "\"routes.decode_rtx.placement.rtx.host_available_floor_gib\" = 2",
                           "\"routes.decode_rtx.placement.rtx.device_gib\" = 80"}) {
    {
      RecipeStack s(text, "r.toml");
      s.add_profile(profile(s.sha256(), "\"routes.decode_rtx.components.arena.gib\" = 70\n"), "p.toml", id);
      CHECK_REFUSED(s.add_overrides(ov + path + "\n", "o.toml"), "drop the profile (run untuned) or retune");
    }
    {
      RecipeStack s(text, "r.toml");
      s.add_overrides(ov + path + "\n", "o.toml");
      CHECK_REFUSED(s.add_profile(profile(s.sha256(), "\"routes.decode_rtx.max_rows\" = 4\n"), "p.toml", id),
                    "drop the profile (run untuned) or retune");
    }
  }
  {  // a profile tuning the anchor placement guards the new host keys and the route's layout scopes too
    RecipeStack s(text, "r.toml");
    s.add_profile(profile(s.sha256(), "\"routes.decode_rtx.placement.rtx.lookahead_units\" = 4\n"), "p.toml", id);
    CHECK_REFUSED(s.add_overrides(ov + "\"routes.decode_rtx.placement.rtx.host_pageable_gib\" = 2\n", "o.toml"),
                  "retune");
    CHECK_REFUSED(s.add_overrides(ov + "\"routes.decode_rtx.assist.aux.device_gib\" = 2\n", "o.toml"), "retune");
    s.add_overrides(ov + "\"routes.decode_rtx.placement.rtx.lookahead_units\" = 6\n", "o.toml");  // not a budget
  }
  {  // without a profile, the same overrides apply
    RecipeStack s(text, "r.toml");
    s.add_overrides(ov + "\"routes.decode_rtx.assist.aux.device_gib\" = 4\n\"routes.decode_rtx.max_rows\" = 2\n",
                    "o.toml");
    CHECK_EQ(s.effective().routes[0].assist("aux")->device_gib, 4.0);
  }
  // Unbound optional sets in the profile key: {unbound = true}; bound vs unbound never matches.
  {
    RecipeStack s(text, "r.toml");
    s.add_profile(profile(s.sha256(), ""), "p.toml", id);  // the identity has no 4090: aux unbound
  }
  {
    RecipeStack s(text, "r.toml");
    CHECK_REFUSED(s.add_profile(profile(s.sha256(), "", true), "p.toml", id),
                  "the optional set \"aux\" is unbound now; the profile was tuned with it bound");
  }
  {
    Json id2 = id;
    Json dev = json_parse(R"({"id":"GPU-4090","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":23.99,
      "host":"pc","ordinal":1,"driver":"580.88","disk_read_gbytes_ps":6.8,"h2d_gbytes_ps":25.0,"host_ram_gib":127.7})");
    (*const_cast<Json*>(id2.find("devices"))).push(dev);
    (*const_cast<Json*>(id2.find("descriptors")))["aux"] = Json::array();
    RecipeStack s(text, "r.toml");
    CHECK_REFUSED(s.add_profile(profile(s.sha256(), ""), "p.toml", id2),
                  "the optional set \"aux\" is bound now; the profile was tuned without it");
    RecipeStack t(text, "r.toml");
    t.add_profile(profile(t.sha256(), "", true), "p.toml", id2);
  }
}

// ---- optional binding ---------------------------------------------------------------------------------------------
TEST(placement_optional_binding) {
  const Json probe = json_parse(kProbe2);
  {
    Binding b = recipe_bind(parse_recipe(with_aux("", kPm)), probe);
    CHECK_EQ(b.of("rtx")->at(0).id, std::string("GPU-pro"));
    CHECK_EQ(b.of("aux")->at(0).id, std::string("GPU-4090"));
    CHECK_EQ(b.of("aux")->at(0).arch, std::string("sm_89"));
    CHECK_EQ(b.of("aux")->at(0).phys, std::string("0000:06:00.0"));
  }
  {
    // One GPU matching both the anchor and the optional assist: the anchor gets it, the assist is unbound.
    Json one = json_parse(R"({"model":"m","engine":"e","devices":[
      {"id":"GPU-pro","kind":"cuda","name":"NVIDIA RTX PRO 6000 Blackwell","memory_gib":95.59,"host":"pc"}]})");
    Binding b = recipe_bind(parse_recipe(with_aux("", kPm)), one);
    CHECK_EQ(b.of("rtx")->size(), size_t(1));
    CHECK(b.of("aux") && b.of("aux")->empty());
    CHECK_EQ(json_write(binding_json(b)), std::string(R"({"aux":[],"rtx":[{"host":"pc","id":"GPU-pro"}]})"));
  }
  {
    // Reordered optional declarations with overlapping selectors: set-id order decides, in ascending physical
    // order, never the declaration order; the canonical recipes are the same.
    const std::string b1 = "[[match.devices]]\nid = \"b1\"\nkind = \"cuda\"\nmemory_min_gib = 8\noptional = true\n";
    const std::string a1 = "[[match.devices]]\nid = \"a1\"\nkind = \"cuda\"\nmemory_min_gib = 8\noptional = true\n";
    const std::string x = std::string(kHead) + b1 + a1 + kRoute + kPlacement + "[routes.components.drafter]\n";
    const std::string y = std::string(kHead) + a1 + b1 + kRoute + kPlacement + "[routes.components.drafter]\n";
    CHECK_EQ(canon(x), canon(y));
    Json three = json_parse(R"({"model":"m","engine":"e","devices":[
      {"id":"z","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":24,"host":"pc","pci":"0000:07:00.0"},
      {"id":"GPU-pro","kind":"cuda","name":"NVIDIA RTX PRO 6000","memory_gib":95.59,"host":"pc","pci":"0000:01:00.0"},
      {"id":"a","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":24,"host":"pc","pci":"0000:06:00.0"}]})");
    Binding bx = recipe_bind(parse_recipe(x), three), by = recipe_bind(parse_recipe(y), three);
    CHECK_EQ(json_write(binding_json(bx)), json_write(binding_json(by)));
    CHECK_EQ(bx.of("a1")->at(0).id, std::string("a"));  // 0000:06 < 0000:07
    CHECK_EQ(bx.of("b1")->at(0).id, std::string("z"));
    // a remote device never binds an optional set (it is the anchor's host only)
    Json remote = three;
    for (Json& d : const_cast<Json*>(remote.find("devices"))->arr)
      if (d.find("id")->text == "a") d["host"] = Json::string("other");
    Binding br = recipe_bind(parse_recipe(x), remote);
    CHECK_EQ(br.of("a1")->at(0).id, std::string("z"));
    CHECK(br.of("b1")->empty());
  }
  {
    // Selectors: uuid / pci / index (index = position among the host's cuda devices in PCI order).
    auto bound = [&](const std::string& sel) {
      const std::string t = std::string(kHead) + "[[match.devices]]\nid = \"aux\"\nkind = \"cuda\"\nmemory_min_gib = 8\n"
                            "optional = true\n" + sel + kRoute + kPlacement + kPm;
      Binding b = recipe_bind(parse_recipe(t), probe);
      return b.of("aux")->empty() ? std::string("-") : b.of("aux")->at(0).id;
    };
    CHECK_EQ(bound("uuid = \"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\"\n"), std::string("GPU-4090"));
    CHECK_EQ(bound("pci = \"0000:06:00.0\"\n"), std::string("GPU-4090"));
    CHECK_EQ(bound("index = 1\n"), std::string("GPU-4090"));
    CHECK_EQ(bound("index = 0\n"), std::string("-"));  // the PRO is index 0 and the anchor's
    CHECK_EQ(bound("pci = \"0000:09:00.0\"\n"), std::string("-"));
  }
  {
    // Alias rows (one physical device reported twice) are merged; conflicting aliases are refused.
    Json alias = json_parse(kProbe2);
    Json dup = json_parse(R"({"id":"0000:06:00.0","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":23.99,
      "host":"pc","pci":"0000:06:00.0"})");
    const_cast<Json*>(alias.find("devices"))->push(dup);
    Binding b = recipe_bind(parse_recipe(with_aux("", kPm)), alias);
    CHECK_EQ(b.of("aux")->at(0).id, std::string("0000:06:00.0"));  // smallest id of the merged rows
    CHECK_EQ(b.of("aux")->at(0).uuid, std::string("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"));
    Json bad = alias;
    (*const_cast<Json*>(bad.find("devices"))).arr.back()["memory_gib"] = Json::number(int64_t(12));
    CHECK_REFUSED(recipe_bind(parse_recipe(with_aux("", kPm)), bad), "PL_E_DUPLICATE_DEVICE");
    // probe permutation: same binding
    Json perm = alias;
    std::reverse(const_cast<Json*>(perm.find("devices"))->arr.begin(), const_cast<Json*>(perm.find("devices"))->arr.end());
    CHECK_EQ(json_write(binding_json(recipe_bind(parse_recipe(with_aux("", kPm)), perm))), json_write(binding_json(b)));
  }
}

// ---- G4: the frozen fixtures of golden/place_ref.py (tests/data/placement), through the C++ inputs ------------------
namespace {

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

const std::string kData = "tests/data/placement/";
const std::string kBuild(64, '0');  // the G4 fixtures' engine build (rev 16 build_sha256)

Json fixture(const std::string& name) { return json_parse(slurp(kData + name + ".json")); }

int64_t ival(const Json& v) { return json_i64(v, "fixture"); }
double fval(const Json& v) { return json_f64(v, "fixture"); }
std::string dec(double d) {  // shortest decimal text (TOML float)
  char buf[40];
  for (int prec = 15; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*g", prec, d);
    if (std::strtod(buf, nullptr) == d) break;
  }
  std::string s = buf;
  if (s.find_first_of(".e") == std::string::npos) s += ".0";
  return s;
}

// A usage-v1 in the producer's shape (strix-stream-adaptive adapt_usage.cpp): `layers` x `units` (dense layers with
// 0 units first when `dense` > 0), `tokens` tokens, the listed (layer among the expert layers, unit, count, recency)
// entries, every other unit 0; use = rdiv(count x 10^6, tokens).
Json usage_doc(const std::vector<std::array<int64_t, 4>>& hits, int64_t tokens = 1000, const std::string& filter = "",
               int dense = 0, int layers = 40, int units = 384) {
  std::string t = R"({"schema":"tierstream/usage-v1","model":"m","sources":[{"path":"t.txt","sha256":")" +
                  std::string(64, 'a') + R"(","kind":"routes-trace","filter":")" + filter + R"(","tokens":)" +
                  std::to_string(tokens) + R"(}],"decay":{"factor_q32":4283353945,"half_life_tokens":256},"tick":)" +
                  std::to_string(tokens) + R"(,"tokens":)" + std::to_string(tokens) +
                  R"(,"order":"use desc, recency desc, layer asc, unit asc","layers":[)";
  for (int L = 0; L < dense + layers; ++L) {
    const int n = L < dense ? 0 : units;
    std::string c, u, r;
    for (int k = 0; k < n; ++k) {
      int64_t cnt = 0, rec = 0;
      for (const auto& h : hits)
        if (h[0] == L - dense && h[1] == k) cnt = h[2], rec = h[3];
      const std::string sep = k ? "," : "";
      c += sep + std::to_string(cnt);
      u += sep + std::to_string((2 * cnt * 1000000 + tokens) / (2 * tokens));
      r += sep + std::to_string(rec);
    }
    t += std::string(L ? "," : "") + R"({"index":)" + std::to_string(L) + R"(,"n_units":)" + std::to_string(n) +
         R"(,"count":[)" + c + R"(],"use":[)" + u + R"(],"recency":[)" + r + "]}";
  }
  return json_parse(t + "]}");
}

const Json* dev_fx(const Json& fx, const std::string& set) {
  for (const Json& d : fx.find("devices")->arr)
    if (d.find("set")->text == set) return &d;
  return nullptr;
}

// Recipe-v1 TOML of a fixture: anchor "rtx", optional assist "aux" (always listed; budget 0 = disabled), explicit
// destinations and quotas as components.<id> keys, margins written only when they differ from the defaults (so the
// defaults are exercised: anchor 2.0 / 0.03, assist 0.5 / 0.03).
std::string recipe_of(const Json& fx) {
  const Json& rtx = *dev_fx(fx, "rtx");
  const Json& aux = *dev_fx(fx, "aux");
  std::string t = "schema = \"tierstream/recipe-v1\"\nname = \"g4\"\n[match]\nmodel = \"m\"\nengine = \"e\"\n"
                  "[[match.devices]]\nid = \"rtx\"\nkind = \"cuda\"\nmemory_min_gib = " +
                  dec(std::max(95.0, fval(*rtx.find("budget_gib")))) +  // a larger anchor when the budget needs it
                  "\n"
                  "[[match.devices]]\nid = \"aux\"\nkind = \"cuda\"\nmemory_min_gib = 8\noptional = true\n"
                  "[[routes]]\nid = \"decode_rtx\"\nphase = \"decode\"\ndevices = [\"rtx\"]\nexecute = \"single\"\n"
                  "max_rows = 6\n";
  t += "[routes.placement.rtx]\ndevice_gib = " + dec(fval(*rtx.find("budget_gib"))) +
       "\nhost_pinned_gib = " + dec(fval(*fx.find("host_pinned_gib"))) +
       "\nmax_context_tokens = 131072\nmax_batch = 1\n";
  auto margins = [&](const Json& d, double dg, int64_t dp) {
    const double g = fval(*d.find("margin_gib"));
    const int64_t p = ival(*d.find("margin_frac_ppm"));
    if (g == dg && p == dp) return std::string();
    return "margin_gib = " + dec(g) + "\nmargin_fraction = " + dec(double(p) / 1e6) + "\n";
  };
  t += margins(rtx, 2.0, 30000);
  t += "[routes.assist.aux]\ndevice_gib = " + dec(fval(*aux.find("budget_gib"))) + "\n" + margins(aux, 0.5, 30000);
  for (const auto& kv : fx.find("explicit")->obj) t += "[routes.components." + kv.first + "]\ndevice = \"" + kv.second.text + "\"\n";
  if (const Json* a = fx.find("arena_gib")) t += "[routes.components.arena]\ngib = " + dec(fval(*a)) + "\n";
  if (const Json* a = fx.find("arena_units")) t += "[routes.components.arena]\nunits = " + std::to_string(ival(*a)) + "\n";
  if (const Json* f = fx.find("file_units"))  // an exact unit count (rev 6 `units`)
    t += "[routes.components.expert_file]\nunits = " + std::to_string(ival(*f)) + "\n";
  return t;
}

Json probe_of(const Json& fx, bool with_aux = true) {
  // the PRO, or a larger hypothetical anchor when the fixture's budget needs one (G4 N2: 400 GiB)
  const double budget = fval(*dev_fx(fx, "rtx")->find("budget_gib"));
  const std::string mem_gib = dec(std::max(95.59, budget + 1));
  const std::string mem_bytes = std::to_string(std::max<uint64_t>(102641664000, gib_bytes(budget + 1)));
  // the device identity of the fixture (name, driver, platform: the context receipts' key, rev 15)
  auto ident = [&](const char* set) {
    const Json& d = *dev_fx(fx, set);
    return "\"name\":\"" + d.find("name")->text + "\",\"driver\":\"" + d.find("driver")->text +
           "\",\"platform\":\"" + d.find("platform")->text + "\",\"arch\":\"" + d.find("arch")->text + "\"";
  };
  std::string p = R"({"model":"m","engine":"e","devices":[
    {"id":"GPU-pro","kind":"cuda","memory_gib":)" + mem_gib + R"(,"memory_bytes":)" + mem_bytes + R"(,
     "host":"pc","ordinal":0,"uuid":"GPU-11111111-2222-3333-4444-555555555555","pci":"0000:01:00.0",)" +
                  ident("rtx") + ",\"host_ram_gib\":127.7}";
  if (with_aux && dev_fx(fx, "aux")->find("bound")->b)
    p += R"(,{"id":"GPU-4090","kind":"cuda","memory_gib":23.99,"memory_bytes":25757220864,
     "host":"pc","ordinal":1,"uuid":"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee","pci":"0000:05:00.0",)" +
         ident("aux") + ",\"host_ram_gib\":127.7}";
  return json_parse(p + "]}");
}

// components-v1 of a fixture: its components (minus the fixture-only "default" / "anchored" keys, checked against
// contract §1.1), elastic metadata, contexts, capabilities; the drafter carries receipts for both arches.
Json components_of(const Json& fx, const std::string& engine = "e") {
  Json c = Json::object();
  c["schema"] = Json::string("tierstream/components-v1");
  c["engine"] = Json::string(engine);
  c["build_sha256"] = fx.find("build_sha256") ? *fx.find("build_sha256") : Json::string(kBuild);
  c["model_sha256"] = Json::string(std::string(64, 'a'));
  c["desc_sha256"] = Json::string(std::string(64, 'd'));
  Json el = Json::object();
  el["unit_bytes"] = *fx.find("unit_bytes");
  el["units_total"] = *fx.find("total_units");
  el["per_layer_units"] = Json::number(int64_t(384));
  el["spares"] = *fx.find("spares_units");
  el["host_spares"] = *fx.find("host_spare_units");
  el["spare_billing"] = Json::string("global");
  if (const Json* sa = fx.find("slot_allocs")) el["slot_allocs"] = *sa;  // rev 17
  c["elastic"] = el;
  Json cfg = Json::object();
  cfg["max_context_tokens"] = Json::number(int64_t(131072));
  cfg["max_sequences"] = Json::number(int64_t(1));
  cfg["max_rows"] = Json::number(int64_t(6));
  // G4 models the TierSite adapter (dg.* named charges) unless the fixture says otherwise (`cache`, rev 14)
  cfg["cache"] = fx.find("cache") ? *fx.find("cache") : Json::string("tierstream");
  // rev 16: every option that sizes or steers the expert cache (gen_g4.py CACHE_CONFIG)
  cfg["cache_config"] = json_parse(R"({"spares":)" + fx.find("spares_units")->text + R"(,"host_spares":)" +
                                   fx.find("host_spare_units")->text +
                                   R"(,"spare_billing":"global","dg_max_ids":36,"window_slots":73,"policy":"lru",)"
                                   R"("prefetch":"off","wait_timeout_ms":2000})");
  c["config"] = cfg;
  c["capabilities"] = fx.find("capabilities") ? *fx.find("capabilities") : Json::array();
  c["context"] = *fx.find("context");
  if (const Json* g = fx.find("context_growth")) c["context_growth"] = *g;
  // the fixture lists the context receipts' keys [arch, device name, driver, platform, build_sha256] (rev 16: complete
  // keys; a truncated key of a malformed case keeps only the fields it has); each becomes a components-v1 receipt
  Json receipts = Json::array();
  for (const Json& k : fx.find("context_receipts")->arr) {
    Json r = Json::object();
    const char* const fields[] = {"arch", "device_name", "driver", "platform", "build_sha256"};
    for (size_t i = 0; i < 5 && i < k.arr.size(); ++i) r[fields[i]] = k.arr[i];
    const Json* allowance = k.arr.empty() ? nullptr : fx.find("context")->find(k.arr[0].text);
    r["allowance_bytes"] = allowance ? *allowance : Json::number(int64_t(644245094));
    r["max_observed_bytes"] = Json::number(int64_t(590000000));
    // rev 17: the receipt's measured growth and qualified per-allocation overhead G, per arch
    const Json* mg = k.arr.empty() || !fx.find("receipt_max_growth") ? nullptr
                                                                    : fx.find("receipt_max_growth")->find(k.arr[0].text);
    r["max_growth_bytes"] = mg ? *mg : Json::number(int64_t(0));
    const Json* g = k.arr.empty() || !fx.find("alloc_granularity") ? nullptr
                                                                   : fx.find("alloc_granularity")->find(k.arr[0].text);
    r["alloc_granularity_bytes"] = g ? *g : Json::number(int64_t(0));
    r["method"] = Json::string("pdh-process-dedicated");
    r["runs"] = Json::number(int64_t(5));
    r["receipt_sha256"] = Json::string(std::string(64, 'c'));
    receipts.push(r);
  }
  c["context_receipts"] = receipts;
  Json comps = Json::array();
  for (const Json& e : fx.find("components")->arr) {
    Json x = e;
    const std::string id = e.find("id")->text;
    CHECK(component_priority(id) > 0);
    const std::string dflt = component_table()[size_t(component_priority(id) - 1)].dflt;
    CHECK_EQ(e.find("default")->text, dflt == "anchor" ? std::string("anchor") : dflt);
    x.obj.erase("default");
    x.obj.erase("anchored");
    if (const Json* qa = e.find("qualified")) {  // the fixture lists the qualified arches
      Json q = Json::array();
      for (const Json& arch : qa->arr) {
        Json r = Json::object();
        r["arch"] = arch;
        r["build_sha256"] = Json::string(kBuild);
        r["receipt_sha256"] = Json::string(std::string(64, 'e'));
        q.push(r);
      }
      x["qualified"] = q;
    }
    comps.push(x);
  }
  c["components"] = comps;
  return c;
}

struct Run {
  PlaceResult r;
  std::string effective_sha256;
};

Run run_place(const std::string& recipe_text, const Json& probe, const Json& comps_doc) {
  RecipeStack s(recipe_text, "g4.toml");
  Binding b = recipe_bind(s.effective(), probe);
  Components c = components_from_json(comps_doc);
  PlaceInput in;
  in.recipe = &s.effective();
  in.effective_sha256 = s.effective_sha256();
  in.route = "decode_rtx";
  in.binding = &b;
  in.components = &c;
  return {place(in), s.effective_sha256()};
}

const Json* doc_dev(const Json& doc, const std::string& set) {
  for (const Json& d : doc.find("devices")->arr)
    if (d.find("set")->text == set) return &d;
  return nullptr;
}

// A placement line in place_ref's shape: {component, name?, bytes, peak_extra, phases}.
Json ref_line(const Json& l) {
  Json x = Json::object();
  for (const char* k : {"component", "name", "bytes", "peak_extra", "phases", "allocs"})
    if (const Json* v = l.find(k)) x[k] = *v;
  return x;
}

// Lines as a canonical multiset: component priority, then the line's JSON.
std::string line_set(const Json& lines) {
  std::vector<std::string> v;
  char buf[8];
  for (const Json& l : lines.arr) {
    std::snprintf(buf, sizeof buf, "%03d", component_priority(l.find("component")->text));
    v.push_back(buf + json_write(ref_line(l)));
  }
  std::sort(v.begin(), v.end());
  std::string out;
  for (const auto& x : v) out += x + "\n";
  return out;
}

// The planner's numbers vs place_ref.py's (devices + host of the expected or partial object).
void compare_numbers(const std::string& name, const Json& want, const Json& doc) {
  const Json& wd = *want.find("devices");
  CHECK_EQ(doc.find("devices")->arr.size(), wd.obj.size());
  for (const auto& kv : wd.obj) {
    const Json* d = doc_dev(doc, kv.first);
    CHECK(d != nullptr);
    if (!d) continue;
    const Json& w = kv.second;
    for (const char* k : {"budget_bytes", "margin_bytes", "context_bytes", "alloc_overhead_bytes"})
      if (w.find(k) && json_write(*d->find(k)) != json_write(*w.find(k)))
        std::printf("  %s %s.%s: %s vs %s\n", name.c_str(), kv.first.c_str(), k, json_write(*d->find(k)).c_str(),
                    json_write(*w.find(k)).c_str()), ++sstest::failures();
    Json lines = Json::array();
    int64_t active = 0, spares = 0;
    for (const Json& l : d->find("lines")->arr) {
      const std::string c = l.find("component")->text;
      if (c == "arena") active = ival(*l.find("slots"));
      if (c == "spares") spares = ival(*l.find("slots"));
      if (c == "arena" || c == "spares") continue;
      lines.push(ref_line(l));
    }
    CHECK_EQ(line_set(lines), line_set(*w.find("lines")));
    CHECK_EQ(active, ival(*w.find("active_slots")));
    CHECK_EQ(spares, ival(*w.find("spare_slots")));
    CHECK_EQ(json_write(*d->find("phase_peak_bytes")), json_write(*w.find("phase_peak_bytes")));
  }
  // host: the field mapping of contract §4.3 (place_ref name -> placement-v1 field)
  const Json& wh = *want.find("host");
  const Json& h = *doc.find("host");
  const Json* eh = nullptr;
  const Json* ef = nullptr;
  for (const Json& l : h.find("lines")->arr) {
    if (l.find("component")->text == "expert_host") eh = &l;
    if (l.find("component")->text == "expert_file") ef = &l;
  }
  CHECK(eh && ef);
  if (!eh || !ef) return;
  const std::pair<const char*, const Json*> map[] = {
      {"pinned_budget_bytes", h.find("pinned_budget_bytes")}, {"other_pinned_bytes", h.find("other_pinned_bytes")},
      {"pinned_total_bytes", h.find("pinned_total_bytes")},   {"expert_host_blocks", eh->find("blocks")},
      {"expert_host_homes", eh->find("homes")},               {"expert_host_bytes", eh->find("bytes")},
      {"file_units", ef->find("units")},                      {"file_units_spilled", ef->find("spilled")}};
  if (const Json* wl = wh.find("lines")) {  // the other pinned consumers, one line per charge (names included)
    Json mine = Json::array();
    for (const Json& l : h.find("lines")->arr) {
      const std::string c = l.find("component")->text;
      if (l.find("at")->text == "host_pinned" && c != "expert_host") mine.push(l);
    }
    CHECK_EQ(line_set(mine), line_set(*wl));
  }
  for (const auto& m : map) {
    const Json* w = wh.find(m.first);
    if (!w) continue;  // a field the reference does not report
    if (!m.second || json_write(*m.second) != json_write(*w))
      std::printf("  %s host.%s: %s vs %s\n", name.c_str(), m.first, m.second ? json_write(*m.second).c_str() : "-",
                  json_write(*w).c_str()), ++sstest::failures();
  }
}

void check_case(const std::string& name, const PlaceResult& r) {
  const Json want = json_parse(slurp(kData + name + ".expected.json"));
  if (const Json* we = want.find("errors")) {
    CHECK_EQ(r.errors.size(), we->arr.size());
    for (size_t i = 0; i < std::min(r.errors.size(), we->arr.size()); ++i)
      if (r.errors[i].rfind(we->arr[i].text, 0) != 0)
        std::printf("  %s: error \"%s\"\n    expected prefix \"%s\"\n", name.c_str(), r.errors[i].c_str(),
                    we->arr[i].text.c_str()), ++sstest::failures();
    if (const Json* part = want.find("partial")) {
      CHECK(r.computed);
      if (r.computed) compare_numbers(name, *part, r.doc);
    }
  } else {
    for (const auto& e : r.errors) std::printf("  %s: unexpected %s\n", name.c_str(), e.c_str());
    CHECK(r.errors.empty());
    if (r.errors.empty()) compare_numbers(name, want, r.doc);
  }
}

const char* kCases[] = {"A-pro-only-spec-host68", "B-pro-only-spec-host76", "C-pro90-aux12-drafter-aux-host68",
                        "C-tsdefaults", "D-aux8-drafter-aux", "E-aux0-drafter-aux", "F-pro-only-plain-host68",
                        "F-tsdefaults", "G-expert-assist-undeclared", "H-aux-unbound-drafter-auto-host76",
                        "I1-pro-only-spec-host74", "I2-pro-only-spec-host74-file64",
                        "I2b-file64-without-capability", "I3-pro-only-spec-host74-file-auto", "K-dense-on-aux",
                        "L-pro-only-plain-context-growth", "M1-host-cap-tiny-file-auto",
                        "M2-file-quota-above-coverage", "M3-file-quota-above-total",
                        "J3-drafter-aux-no-sm89-receipt", "J4-drafter-aux-no-sm89-context-receipt",
                        "I3b-file-component-without-capability", "N1-file-quota-leaves-no-spare-room",
                        "N2-explicit-arena-leaves-no-spare-room", "N3-zero-device-spares",
                        "N3b-zero-host-spares", "O1-receipt-for-other-driver", "I4-spill-floor",
                        "O2-receipt-for-other-build", "O3-receipt-for-other-device",
                        "O4-receipt-for-other-platform", "O5-truncated-receipt-key", "O5b-empty-receipt-key",
                        "O6-duplicate-receipt-key", "P1-unaligned-host-stride", "P2-engine-cache-full-coverage",
                        "P1b-unit-not-multiple-of-256", "O7-unsorted-receipt-keys",
                        "O8-receipt-growth-above-declared", "P2b-engine-cache-with-dg-charges"};

// The contract's §2.5 TOML, literally (G4 J): it must plan as case C.
const char* kContract25 = R"(schema = "tierstream/recipe-v1"
name = "ds41-rtx-replica-pro-plus-aux"
[match]
model = "DeepSeek-V4.1-Flash"
engine = "strix-cuda-ds41-replica"
[[match.devices]]
id = "rtx"
kind = "cuda"
name = "RTX PRO 6000"
memory_min_gib = 94
host_ram_min_gib = 120
[[match.devices]]
id = "aux"
kind = "cuda"
memory_min_gib = 12
optional = true
[[routes]]
id = "decode_rtx"
phase = "decode"
devices = ["rtx"]
execute = "single"
max_rows = 6
[routes.placement.rtx]
device_gib = 90.0
host_pinned_gib = 68.0
host_pageable_gib = 6.0
host_available_floor_gib = 6.0
max_context_tokens = 131072
max_batch = 1
margin_gib = 0.6                  # explicit (see §2.2 Margins): context/graphs/buffers are explicit lines
margin_fraction = 0.0
[routes.assist.aux]
device_gib = 12.0                 # assist margin defaults 0.5 / 0.03
[routes.components.drafter]
device = "aux"
)";

}  // namespace

TEST(g4_fixtures) {
  for (const char* name : kCases) {
    const Json fx = fixture(name);
    std::string err;
    try {
      Run run = run_place(recipe_of(fx), probe_of(fx), components_of(fx));
      check_case(name, run.r);
    } catch (const std::exception& e) {
      // a malformed context receipt key is refused by the components-v1 reader (PL_E_SCHEMA before any phase):
      // its message ("components: PL_E_SCHEMA ...") must carry the reference text
      const Json want = json_parse(slurp(kData + std::string(name) + ".expected.json"));
      const Json* we = want.find("errors");
      std::string msg = e.what();
      const size_t at = msg.find("PL_E_SCHEMA");
      if (we && we->arr.size() == 1 && at != std::string::npos && msg.substr(at, we->arr[0].text.size()) == we->arr[0].text)
        continue;
      std::printf("  %s: exception %s\n", name, e.what());
      ++sstest::failures();
    }
  }
}

// G4 J / J2 / J3: the frozen literal recipes, probes and components files (golden/g4), through parse -> bind ->
// plan; the components files carry their context receipts (rev 15 list form) and are used as copied.
TEST(g4_literal_recipes) {
  struct J {
    const char *recipe, *probe, *comps, *expected;
  };
  for (const J& c : {J{"J-recipe.toml", "J-probe.json", "J-components.json", "J-literal-recipe"},
                     J{"J2-recipe.toml", "J2-probe.json", "J2-components.json", "J2-literal-single-gpu"},
                     J{"J-recipe.toml", "J-probe.json", "J3-components-no-sm89-receipt.json",
                       "J3-drafter-aux-no-sm89-receipt"},
                     J{"J-recipe.toml", "J-probe.json", "J4-components-no-sm89-context-receipt.json",
                       "J4-drafter-aux-no-sm89-context-receipt"}}) {
    Json comps = json_parse(slurp(kData + c.comps));
    CHECK_EQ(comps.find("config")->find("cache")->text, std::string("tierstream"));  // the TierSite adapter
    Json probe = json_parse(slurp(kData + c.probe));
    // the frozen probes carry no model / engine identity: the recipe's
    probe["model"] = Json::string("DeepSeek-V4.1-Flash");
    probe["engine"] = Json::string("strix-cuda-ds41-replica");
    try {
      Run run = run_place(slurp(kData + c.recipe), probe, comps);
      check_case(c.expected, run.r);
    } catch (const std::exception& e) {
      std::printf("  %s: exception %s\n", c.expected, e.what());
      ++sstest::failures();
    }
  }
}

TEST(g4_extra_cases) {
  const Json c = fixture("C-pro90-aux12-drafter-aux-host68");
  {  // J: the literal §2.5 recipe == C
    Json probe = probe_of(c);
    probe["model"] = Json::string("DeepSeek-V4.1-Flash");
    probe["engine"] = Json::string("strix-cuda-ds41-replica");
    Run run = run_place(kContract25, probe, components_of(c, "strix-cuda-ds41-replica"));
    check_case("C-pro90-aux12-drafter-aux-host68", run.r);
  }
  {  // drafter on an arch without a receipt
    Json comps = components_of(c);
    for (Json& e : const_cast<Json*>(comps.find("components"))->arr)
      if (e.find("id")->text == "drafter") {  // drop the sm_89 receipt
        auto& q = const_cast<Json*>(e.find("qualified"))->arr;
        q.erase(std::remove_if(q.begin(), q.end(), [](const Json& r) { return r.find("arch")->text == "sm_89"; }),
                q.end());
      }
    Run run = run_place(recipe_of(c), probe_of(c), comps);
    CHECK_EQ(run.r.errors.size(), size_t(1));
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_ARCH_UNQUALIFIED drafter on aux (sm_89)"));
  }
  {  // a receipt for another build does not qualify (the context receipts follow the new build: V3 passes)
    Json comps = components_of(c);
    comps["build_sha256"] = Json::string(std::string(64, 'c'));
    for (Json& r : const_cast<Json*>(comps.find("context_receipts"))->arr)
      r["build_sha256"] = Json::string(std::string(64, 'c'));
    Run run = run_place(recipe_of(c), probe_of(c), comps);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_ARCH_UNQUALIFIED"));
    // and context receipts of the old build no longer admit it (V3)
    Json old = components_of(c);
    old["build_sha256"] = Json::string(std::string(64, 'c'));
    Run v3 = run_place(recipe_of(c), probe_of(c), old);
    CHECK(!v3.r.errors.empty() && has(v3.r.errors[0], "PL_E_MISSING_CONTEXT") &&
          has(v3.r.errors[0], "(mismatch: build_sha256)"));
  }
  {  // budget above the bound device
    std::string t = recipe_of(c);
    t.replace(t.find("device_gib = 12.0"), 17, "device_gib = 30.0");
    Run run = run_place(t, probe_of(c), components_of(c));
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_BUDGET_ABOVE_DEVICE aux: budget 32212254720 > device "
                                                         "memory 25757220864"));
    CHECK_EQ(run.r.phase, std::string("V2"));
  }
  {  // one GPU matching both sets: the anchor binds it, the assist is unbound; drafter=auto plans as H
    const Json h = fixture("H-aux-unbound-drafter-auto-host76");
    Json one = probe_of(h, false);
    Run run = run_place(recipe_of(h), one, components_of(h));
    check_case("H-aux-unbound-drafter-auto-host76", run.r);
    // and an explicit reference to it is PL_E_DEVICE_DISABLED
    Run e = run_place(recipe_of(h) + "[routes.components.drafter]\ndevice = \"aux\"\n", one, components_of(h));
    CHECK(!e.r.errors.empty() && has(e.r.errors[0], "PL_E_DEVICE_DISABLED drafter -> aux (the optional set is unbound)"));
  }
  {  // multi-device processes need the engine's multi_device capability
    Json comps = components_of(c);
    comps["capabilities"] = Json::array();
    Run run = run_place(recipe_of(c), probe_of(c), comps);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_CAPABILITY multi_device"));
  }
  {  // V3: stale config, missing context, missing descriptor
    Json comps = components_of(c);
    (*const_cast<Json*>(comps.find("config")))["max_rows"] = Json::number(int64_t(1));
    Run run = run_place(recipe_of(c), probe_of(c), comps);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_STALE config.max_rows: components-v1 1, recipe 6"));
    auto drop_receipts = [](Json& x, const std::string& arch) {
      auto& rs = const_cast<Json*>(x.find("context_receipts"))->arr;
      rs.erase(std::remove_if(rs.begin(), rs.end(), [&](const Json& r) { return r.find("arch")->text == arch; }),
               rs.end());
    };
    Json noctx = components_of(c);
    const_cast<Json*>(noctx.find("context"))->obj.erase("sm_89");
    drop_receipts(noctx, "sm_89");
    Json noreceipt = components_of(c);
    drop_receipts(noreceipt, "sm_89");
    Json nocache = components_of(c);
    const_cast<Json*>(nocache.find("config"))->obj.erase("cache");
    // (a descriptor with dg.* names needs the tierstream cache: drop them to reach the V3 rule)
    for (Json& comp : const_cast<Json*>(nocache.find("components"))->arr)
      if (Json* ch = const_cast<Json*>(comp.find("charges")))
        ch->arr.erase(std::remove_if(ch->arr.begin(), ch->arr.end(), [](const Json& x) { return x.find("name") != nullptr; }),
                      ch->arr.end());
    Run nc = run_place(recipe_of(c), probe_of(c), nocache);
    CHECK(!nc.r.errors.empty() && has(nc.r.errors[0], "PL_E_SCHEMA components-v1 config.cache is required"));
    CHECK_EQ(nc.r.phase, std::string("V3"));
    Run nr = run_place(recipe_of(c), probe_of(c), noreceipt);
    CHECK(!nr.r.errors.empty() && has(nr.r.errors[0], "PL_E_MISSING_CONTEXT aux: no context receipt for sm_89/"));
    CHECK_EQ(nr.r.phase, std::string("V3"));
    // the receipt must match the binding's arch, device name, driver, platform and the build; the message names the
    // first mismatching field (G4 O1: another driver)
    auto field = [&](const char* key, const char* value, const char* part) {
      Json x = components_of(c);
      for (Json& r : const_cast<Json*>(x.find("context_receipts"))->arr)
        if (r.find("arch")->text == "sm_89") r[key] = Json::string(value);
      Run run = run_place(recipe_of(c), probe_of(c), x);
      CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_MISSING_CONTEXT aux: no context receipt for sm_89/") &&
            has(run.r.errors[0], part));
    };
    field("device_name", "NVIDIA GeForce RTX 3090", "(mismatch: device_name)");
    field("driver", "610.00", "(mismatch: driver)");
    field("platform", "linux", "(mismatch: platform)");
    field("build_sha256", std::string(64, 'f').c_str(), "(mismatch: build_sha256)");
    Json nobill = components_of(c);
    const_cast<Json*>(nobill.find("elastic"))->obj.erase("spare_billing");
    CHECK_REFUSED(components_from_json(nobill), "missing \"spare_billing\"");
    Run m = run_place(recipe_of(c), probe_of(c), noctx);
    CHECK(!m.r.errors.empty() && has(m.r.errors[0], "PL_E_MISSING_CONTEXT aux: components-v1 has no context for sm_89"));
    RecipeStack s(recipe_of(c), "g4.toml");
    Binding b = recipe_bind(s.effective(), probe_of(c));
    PlaceInput in;
    in.recipe = &s.effective();
    in.route = "decode_rtx";
    in.binding = &b;
    PlaceResult r = place(in);
    CHECK(!r.errors.empty() && has(r.errors[0], "PL_E_MISSING_DESCRIPTOR"));
  }
  {  // capability refusals: layers, layer_* quotas, an undeclared explicit id, a host component on a device
    const std::string t = recipe_of(c);
    auto first = [&](const std::string& extra) {
      Run run = run_place(t + extra, probe_of(c), components_of(c));
      return run.r.errors.empty() ? std::string() : run.r.errors[0];
    };
    CHECK(has(first("[routes.components.dense]\nlayers = [[0, 3]]\n"), "PL_E_CAPABILITY dense: layers"));
    CHECK(has(first("[routes.components.layer_slots]\ngib = 4\n"), "PL_E_CAPABILITY layer_slots: needs layer-granular"));
    CHECK(has(first("[routes.components.mtp]\ndevice = \"auto\"\n"), "PL_E_CAPABILITY mtp: not declared by the engine"));
    CHECK(has(first("[routes.components.staging]\ndevice = \"aux\"\n"), "PL_E_CAPABILITY staging -> aux"));
    CHECK(has(first("[routes.components.head]\ndevice = \"aux\"\n"), "PL_E_ANCHORED head"));
    CHECK(has(first("[routes.components.arena]\ndevice = \"aux\"\n"), "PL_E_ANCHORED arena"));
    CHECK(has(first("[routes.components.engram]\ndevice = \"host_pinned\"\n"), "PL_E_CAPABILITY engram"));
  }
  {  // explicit arena: requested active units, refused above the room; conflict with dynamic_slots_per_layer
    Run run = run_place(recipe_of(c) + "[routes.components.arena]\ngib = 76.0\n", probe_of(c), components_of(c));
    CHECK(run.r.errors.empty());
    const Json* rtx = run.r.computed ? doc_dev(run.r.doc, "rtx") : nullptr;
    int64_t slots = -1;
    if (rtx)
      for (const Json& l : rtx->find("lines")->arr)
        if (l.find("component")->text == "arena") {
          slots = ival(*l.find("slots"));
          CHECK_EQ(l.find("source")->text, std::string("recipe"));
        }
    CHECK_EQ(slots, int64_t(gib_bytes(76.0) / 9953280));
    Run over = run_place(recipe_of(c) + "[routes.components.arena]\nunits = 15300\n", probe_of(c), components_of(c));
    CHECK(!over.r.errors.empty() && has(over.r.errors[0], "PL_E_BUDGET spares need non-resident units: 60 left for 96"));
    Run big = run_place(recipe_of(c) + "[routes.components.arena]\ngib = 80.0\n", probe_of(c), components_of(c));
    CHECK(!big.r.errors.empty() && has(big.r.errors[0], "PL_E_BUDGET rtx: arena "));
    std::string t = recipe_of(c);
    t.replace(t.find("max_batch = 1\n"), 14, "max_batch = 1\ndynamic_slots_per_layer = 200\n");
    Run cf = run_place(t + "[routes.components.arena]\ngib = 70.0\n", probe_of(c), components_of(c));
    CHECK(!cf.r.errors.empty() && has(cf.r.errors[0], "PL_E_CONFLICT"));
  }
  {  // the explanation and the alternatives list (auto never moves to the assist)
    const Json b = fixture("B-pro-only-spec-host76");
    std::string t = recipe_of(b);
    const std::string off = "[routes.assist.aux]\ndevice_gib = 0.0";
    t.replace(t.find(off), off.size(), "[routes.assist.aux]\ndevice_gib = 12.0");
    Run run = run_place(t, probe_of(b), components_of(b));
    CHECK(run.r.errors.empty());
    CHECK(has(run.r.explain, "drafter -> rtx (default)"));
    CHECK(has(run.r.explain, "drafter -> aux: frees 8321499136 on rtx, needs 10861992776 + context on aux"));
    CHECK(has(run.r.explain, "device aux: not touched"));
    CHECK_EQ(run.r.doc.find("devices")->arr.size(), size_t(1));
  }
}

TEST(frac_ppm_rounding) {
  CHECK_EQ(frac_ppm(0.03), uint64_t(30000));
  CHECK_EQ(frac_ppm(0), uint64_t(0));
  CHECK_EQ(frac_ppm(0.5), uint64_t(500000));
  CHECK_EQ(frac_ppm(0.0000005), uint64_t(1));   // half up
  CHECK_EQ(frac_ppm(0.00000049), uint64_t(0));
  CHECK_EQ(frac_ppm(0.0300005), uint64_t(30001));
  CHECK_EQ(frac_ppm(0.1234564), uint64_t(123456));
  CHECK_EQ(frac_ppm(1e-7), uint64_t(0));
  CHECK_EQ(frac_ppm(0.0300000001), uint64_t(30000));  // G5: the shortest decimal text, not the binary value
}

// ---- components-v1 reader ---------------------------------------------------------------------------------------
TEST(components_reader) {
  const Json c = fixture("C-pro90-aux12-drafter-aux-host68");
  const Json doc = components_of(c);
  Components a = components_from_json(doc);
  CHECK_EQ(a.components.size(), size_t(9));
  CHECK_EQ(a.components[0].id, std::string("dense"));
  CHECK_EQ(a.components.back().id, std::string("staging"));
  // canonical: reading the canonical form gives the same document and hash
  Components back = components_from_json(components_to_json(a));
  CHECK_EQ(back.sha256, a.sha256);
  CHECK_EQ(json_write(components_to_json(back)), json_write(components_to_json(a)));
  // permutations (components, charges, phases, qualified, capabilities) keep components_sha256 (G5)
  Json p = doc;
  auto& arr = const_cast<Json*>(p.find("components"))->arr;
  std::reverse(arr.begin(), arr.end());
  for (Json& e : arr) {
    if (Json* v = const_cast<Json*>(e.find("variants")))
      for (auto& kv : v->obj) std::reverse(kv.second.arr.begin(), kv.second.arr.end());
    if (Json* q = const_cast<Json*>(e.find("qualified"))) std::reverse(q->arr.begin(), q->arr.end());
    for (const char* k : {"charges"})
      if (Json* ch = const_cast<Json*>(e.find(k)))
        for (Json& x : ch->arr) std::reverse(const_cast<Json*>(x.find("phases"))->arr.begin(),
                                             const_cast<Json*>(x.find("phases"))->arr.end());
  }
  Json caps = Json::array();
  caps.push(Json::string("z_cap"));
  caps.push(Json::string("multi_device"));
  p["capabilities"] = caps;
  Json q = doc;
  Json caps2 = Json::array();
  caps2.push(Json::string("multi_device"));
  caps2.push(Json::string("z_cap"));
  q["capabilities"] = caps2;
  CHECK_EQ(components_from_json(p).sha256, components_from_json(q).sha256);
  CHECK(components_from_json(q).sha256 != a.sha256);
  // refusals
  auto refuse = [&](const std::function<void(Json&)>& edit, const char* part) {
    Json x = doc;
    edit(x);
    CHECK_REFUSED(components_from_json(x), part);
  };
  auto comp0 = [](Json& x) -> Json& { return const_cast<Json*>(x.find("components"))->arr[0]; };
  auto charge0 = [&](Json& x) -> Json& { return const_cast<Json*>(comp0(x).find("charges"))->arr[0]; };
  refuse([](Json& x) { x["schema"] = Json::string("tierstream/desc-v1"); }, "PL_E_SCHEMA schema");
  refuse([](Json& x) { x["bogus"] = Json::number(int64_t(1)); }, "unknown key \"bogus\"");
  refuse([](Json& x) { x["build_sha256"] = Json::string("xyz"); }, "build_sha256: expected 64 lowercase hex");
  refuse([&](Json& x) { charge0(x)["bytes"] = Json::number(int64_t(-1)); }, "expected an integer >= 0");
  refuse([&](Json& x) {
    Json n;
    n.kind = Json::Kind::Number;
    n.text = "1.5";
    charge0(x)["bytes"] = n;
  }, "expected an integer >= 0, got 1.5");
  refuse([&](Json& x) {
    Json n;
    n.kind = Json::Kind::Number;
    n.text = "9223372036854775808";
    charge0(x)["bytes"] = n;
  }, "overflows 64 bits");
  refuse([&](Json& x) {
    Json n;
    n.kind = Json::Kind::Number;
    n.text = "9223372036854775807";
    charge0(x)["bytes"] = n;
  }, "byte sum overflows 64 bits");
  refuse([&](Json& x) { charge0(x)["phases"] = Json::array(); }, "non-empty array of phases");
  refuse([&](Json& x) {  // two identical charges: the canonical order would not be total
    Json& ch = *const_cast<Json*>(comp0(x).find("charges"));
    ch.push(ch.arr[0]);
  }, "two identical charges");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("context")))["measured"] = Json::boolean(true); }, "not an arch");
  // charge names (rev 11): unique within a component; dg.* reserved to their component and location; the name is
  // the last sort key, an absent name first
  refuse([&](Json& x) {
    Json& ch = *const_cast<Json*>(comp0(x).find("charges"));
    Json a1 = ch.arr[0], a2 = ch.arr[0];
    a1["name"] = Json::string("w");
    a2["name"] = Json::string("w");
    a2["bytes"] = Json::number(int64_t(7));
    ch.push(a1);
    ch.push(a2);
  }, "two charges named \"w\"");
  refuse([&](Json& x) {
    Json& ch = *const_cast<Json*>(comp0(x).find("charges"));  // dense
    Json a1 = ch.arr[0];
    a1["name"] = Json::string("dg.controls");
    ch.push(a1);
  }, "\"dg.controls\" is reserved");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("config")))["cache"] = Json::string("engine"); },
         "PL_E_SCHEMA window: reserved name dg.slot_table outside the device-gated adapter");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("config")))["cache"] = Json::string("gpu"); },
         "config.cache: \"gpu\" is not tierstream | engine");
  refuse([](Json& x) { const_cast<Json*>(x.find("elastic"))->obj.erase("spare_billing"); },
         "missing \"spare_billing\"");
  {  // the engine cache ignores spare_billing (and carries no dg.* charges)
    Json e = doc;
    (*const_cast<Json*>(e.find("config")))["cache"] = Json::string("engine");
    const_cast<Json*>(e.find("elastic"))->obj.erase("spare_billing");
    for (Json& comp : const_cast<Json*>(e.find("components"))->arr)
      if (Json* ch = const_cast<Json*>(comp.find("charges")))
        ch->arr.erase(std::remove_if(ch->arr.begin(), ch->arr.end(), [](const Json& x) { return x.find("name") != nullptr; }),
                      ch->arr.end());
    CHECK(error_of([&] { components_from_json(e); }).empty());
  }
  {
    Json n1 = doc;
    Json& ch = *const_cast<Json*>(comp0(n1).find("charges"));
    Json a1 = ch.arr[0];
    a1["name"] = Json::string("second");  // equal in the other four fields: the name orders it after the unnamed one
    ch.push(a1);
    const Json canon = components_to_json(components_from_json(n1));
    const Json& cc = *canon.find("components")->arr[0].find("charges");
    CHECK(cc.arr.size() == 2 && !cc.arr[0].find("name") && cc.arr[1].find("name")->text == "second");
    Json n2 = n1;
    auto& c2 = const_cast<Json*>(comp0(n2).find("charges"))->arr;
    std::reverse(c2.begin(), c2.end());
    CHECK_EQ(components_from_json(n2).sha256, components_from_json(n1).sha256);
  }
  // G5: a `qualified` list in another order -> the same components_sha256; a duplicate arch -> PL_E_SCHEMA
  auto drafter = [](Json& x) -> Json& {
    for (Json& e : const_cast<Json*>(x.find("components"))->arr)
      if (e.find("id")->text == "drafter") return e;
    return x;
  };
  {
    Json p1 = doc;
    auto& qs = const_cast<Json*>(drafter(p1).find("qualified"))->arr;
    CHECK_EQ(qs.size(), size_t(2));
    std::reverse(qs.begin(), qs.end());
    CHECK_EQ(components_from_json(p1).sha256, a.sha256);
  }
  refuse([&](Json& x) {
    auto& qd = const_cast<Json*>(drafter(x).find("qualified"))->arr;
    Json dup = qd[0];
    dup["receipt_sha256"] = Json::string(std::string(64, 'f'));  // another receipt, the same arch
    qd.push_back(dup);
  }, "PL_E_SCHEMA components.drafter.qualified: two records for");
  refuse([&](Json& x) { charge0(x)["at"] = Json::string("gpu"); }, "is not device | host_pinned");
  refuse([&](Json& x) { comp0(x)["id"] = Json::string("kv"); }, "\"kv\" is not a component id");
  refuse([&](Json& x) { comp0(x)["id"] = Json::string("head"); }, "components.head: is listed twice");
  refuse([&](Json& x) { comp0(x)["variants"] = Json::object(); }, "exactly one of \"charges\" and \"variants\"");
  refuse([](Json& x) { const_cast<Json*>(x.find("config"))->obj.erase("max_rows"); }, "config: missing \"max_rows\"");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("context")))["gpu"] = Json::number(int64_t(1)); }, "not an arch");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("elastic")))["unit_bytes"] = Json::number(int64_t(0)); },
         "unit_bytes and units_total must be > 0");
  refuse([](Json& x) { (*const_cast<Json*>(x.find("elastic")))["mode"] = Json::string("mixed"); }, "elastic.mode");
  refuse([](Json& x) {
    Json caps3 = Json::array();
    caps3.push(Json::string("a"));
    caps3.push(Json::string("a"));
    x["capabilities"] = caps3;
  }, "\"a\" is listed twice");
}

// ---- G5: determinism ------------------------------------------------------------------------------------------
TEST(g5_determinism) {
  const Json c = fixture("C-pro90-aux12-drafter-aux-host68");
  Run base = run_place(recipe_of(c), probe_of(c), components_of(c));
  CHECK(base.r.errors.empty());
  const std::string sha = base.r.computed ? base.r.doc.find("placement_sha256")->text : "";
  // the placement_sha256 covers the object without itself
  Json body = base.r.doc;
  body.obj.erase("placement_sha256");
  const std::string t = json_write(body);
  CHECK_EQ(sha256_hex(t.data(), t.size()), sha);
  for (int k = 0; k < 100; ++k) {
    // permuted probe rows + an alias row of the 4090, permuted components / charges / receipts
    Json probe = probe_of(c);
    auto& rows = const_cast<Json*>(probe.find("devices"))->arr;
    Json alias = json_parse(R"({"id":"0000:05:00.0","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":23.99,
      "host":"pc","pci":"0000:05:00.0"})");
    rows.insert(rows.begin() + (k % 3 == 0 ? 0 : 1), alias);
    std::rotate(rows.begin(), rows.begin() + (k % int(rows.size())), rows.end());
    Json comps = components_of(c);
    auto& arr = const_cast<Json*>(comps.find("components"))->arr;
    std::rotate(arr.begin(), arr.begin() + (k % int(arr.size())), arr.end());
    for (Json& e : arr)
      if (Json* q = const_cast<Json*>(e.find("qualified")))
        if (k % 2) std::reverse(q->arr.begin(), q->arr.end());
    Run r = run_place(recipe_of(c), probe, comps);
    CHECK(r.r.errors.empty());
    if (r.r.computed && r.r.doc.find("placement_sha256")->text != sha) {
      std::printf("  run %d: placement_sha256 differs\n", k);
      ++sstest::failures();
      break;
    }
  }
  // reordered optional declarations with overlapping selectors and assist tables in another order: the same plan
  const std::string extra = "[[match.devices]]\nid = \"aux2\"\nkind = \"cuda\"\nmemory_min_gib = 8\noptional = true\n";
  std::string r1 = recipe_of(c), r2 = recipe_of(c);
  r1.insert(r1.find("[[routes]]"), extra);
  r2.insert(r2.find("[[match.devices]]\nid = \"aux\""), extra);
  r1 += "[routes.assist.aux2]\ndevice_gib = 0.0\n";
  r2.insert(r2.find("[routes.assist.aux]"), "[routes.assist.aux2]\ndevice_gib = 0.0\n");
  Run a = run_place(r1, probe_of(c), components_of(c)), b = run_place(r2, probe_of(c), components_of(c));
  CHECK(a.r.errors.empty() && b.r.errors.empty());
  CHECK_EQ(a.effective_sha256, b.effective_sha256);
  if (a.r.computed && b.r.computed)
    CHECK_EQ(a.r.doc.find("placement_sha256")->text, b.r.doc.find("placement_sha256")->text);
  // rounding boundaries: gib exactly on a unit edge, the margin fraction at a ppm edge
  const double edge = 8000.0 * 9953280.0 / 1073741824.0;
  Run e = run_place(recipe_of(c) + "[routes.components.arena]\ngib = " + dec(edge) + "\n", probe_of(c), components_of(c));
  int64_t slots = -1;
  if (e.r.computed)
    for (const Json& l : doc_dev(e.r.doc, "rtx")->find("lines")->arr)
      if (l.find("component")->text == "arena") slots = ival(*l.find("slots"));
  CHECK_EQ(slots, int64_t(8000));
}

// ---- C ABI: ts_place / ts_place_from_flags (the CLI inputs of tests/data/placement/cli-*) ------------------------
TEST(capi_place_and_flags) {
  const std::string recipe = slurp(kData + "cli-recipe.toml"), probe = slurp(kData + "cli-probe.json"),
                    comps = slurp(kData + "cli-components.json");
  ss_recipe* r = nullptr;
  CHECK_OK(ss_recipe_from_text(recipe.data(), recipe.size(), "cli-recipe.toml", &r));
  char* out = nullptr;
  char* ex = nullptr;
  size_t n = 0;
  CHECK_OK(ts_place(r, "decode_rtx", probe.data(), probe.size(), comps.data(), comps.size(), nullptr, nullptr, 0, nullptr, 0, &out,
                    &n, &ex, nullptr));
  std::string sha;
  if (out) {
    const Json doc = json_parse(std::string(out, n));
    sha = doc.find("placement_sha256")->text;
    Json want = json_parse(slurp(kData + "C-pro90-aux12-drafter-aux-host68.expected.json"));
    // C's relay is in the CLI components too: the same numbers
    compare_numbers("cli", want, doc);
    CHECK(has(ex, "placement_sha256 " + sha));
    ss_free(out);
    ss_free(ex);
  }
  // a refusal: SS_E_RECIPE, every PL_E line in the message, nothing written
  CHECK_OK(ss_recipe_set(r, "routes.decode_rtx.assist.aux.device_gib", "0", "--set"));
  CHECK_OK(ss_recipe_set(r, "routes.decode_rtx.components.drafter.device", "\"auto\"", "--set"));
  out = nullptr;
  CHECK_EQ(int(ts_place(r, "decode_rtx", probe.data(), probe.size(), comps.data(), comps.size(), nullptr, nullptr, 0,
                        nullptr, 0, &out, nullptr, nullptr, nullptr)),
           int(SS_E_RECIPE));
  CHECK(out == nullptr);
  CHECK(has(ss_last_error(), "PL_E_HOST_PINNED: need 80143418429 > 73014444032"));  // A + the relay
  CHECK(has(ss_last_error(), "717 units short"));
  // no components-v1: PL_E_MISSING_DESCRIPTOR
  CHECK_EQ(int(ts_place(r, "decode_rtx", probe.data(), probe.size(), nullptr, 0, nullptr, nullptr, 0, nullptr, 0, &out, nullptr,
                        nullptr, nullptr)),
           int(SS_E_RECIPE));
  CHECK(has(ss_last_error(), "PL_E_MISSING_DESCRIPTOR"));
  ss_recipe_free(r);

  // the sugar flags plan the same placement as the recipe (sets named gpuN in PCI order)
  const char* argv[] = {"--gpu-budget-gib", "gpu0=90,pci:0000:05:00.0=12", "--device-map", "drafter=name:4090",
                        "--host-pinned-gib", "68", "--margin-gib", "0.6", "--margin-fraction", "0"};
  out = nullptr;
  CHECK_OK(ts_place_from_flags(probe.data(), probe.size(), comps.data(), comps.size(), 10, argv, &out, &n, nullptr,
                               nullptr));
  if (out) {
    const Json doc = json_parse(std::string(out, n));
    Json want = json_parse(slurp(kData + "C-pro90-aux12-drafter-aux-host68.expected.json"));
    Json& wd = (*const_cast<Json*>(want.find("devices")));
    wd["gpu0"] = *wd.find("rtx");
    wd["gpu1"] = *wd.find("aux");
    wd.obj.erase("rtx");
    wd.obj.erase("aux");
    compare_numbers("flags", want, doc);
    CHECK_EQ(doc.find("anchor")->text, std::string("gpu0"));
    ss_free(out);
  }
  const char* bad1[] = {"--anchor", "name:RTX"};  // two matches
  CHECK_EQ(int(ts_place_from_flags(probe.data(), probe.size(), comps.data(), comps.size(), 2, bad1, &out, nullptr,
                                   nullptr, nullptr)),
           int(SS_E_RECIPE));
  CHECK(has(ss_last_error(), "PL_E_UNKNOWN_DEVICE flags: \"name:RTX\" matches 2"));
  const char* bad2[] = {"--device-map", "drafter=gpu7"};
  CHECK_EQ(int(ts_place_from_flags(probe.data(), probe.size(), comps.data(), comps.size(), 2, bad2, &out, nullptr,
                                   nullptr, nullptr)),
           int(SS_E_RECIPE));
  CHECK(has(ss_last_error(), "PL_E_UNKNOWN_DEVICE"));
  const char* bad3[] = {"--arena-gib", "70"};
  CHECK_EQ(int(ts_place_from_flags(probe.data(), probe.size(), comps.data(), comps.size(), 2, bad3, &out, nullptr,
                                   nullptr, nullptr)),
           int(SS_E_RECIPE));
  CHECK(has(ss_last_error(), "unknown flag --arena-gib"));
}

// ---- unit assignment: placement-v1 `units` / `seed` with a desc-v1 or usage-v1 rank ----------------------------
// (G8, the S5 library compile, needs devgate-ts's dg_layout: not on this base, see impl-notes.md)
namespace {

// A DS-like desc-v1: 40 MoE layers x 384 experts of the fixture's unit bytes, 8 GiB fixed (zeroed by the projection).
Desc g8_desc() {
  std::string layers;
  for (int L = 0; L < 40; ++L)
    layers += std::string(L ? "," : "") + R"({"index":)" + std::to_string(L) +
              R"(,"kind":"moe","n_units":384,"top_k":6,"fixed_bytes":1048576,"parts":[{"format":"Q2_K","offset":0,)"
              R"("bytes":9953280,"stride":9953280}]})";
  return desc_from_json(json_parse(
      R"({"schema":"strix-stream/desc-v1","model":"m","engine":"e","order":"ranked","tiers":"pinned|disk",)"
      R"("fixed_device":[{"name":"dense","bytes":8589934592}],"host_fixed":[{"name":"tables","bytes":4096}],)"
      R"("sources":[{"path":"x.bin"}],"state_bytes_per_position":4096,"workspace_bytes_per_row":1048576,)"
      R"("layers":[)" + layers + "]}"));
}

Run run_with_desc(const std::string& recipe_text, const Json& probe, Json comps_doc, const Desc& d) {
  const std::string t = json_write(desc_to_json(d));
  comps_doc["desc_sha256"] = Json::string(sha256_hex(t.data(), t.size()));
  RecipeStack s(recipe_text, "g8.toml");
  Binding b = recipe_bind(s.effective(), probe);
  Components c = components_from_json(comps_doc);
  PlaceInput in;
  in.recipe = &s.effective();
  in.effective_sha256 = s.effective_sha256();
  in.route = "decode_rtx";
  in.binding = &b;
  in.components = &c;
  in.desc = &d;
  return {place(in), s.effective_sha256()};
}

int64_t ranges_total(const Json& units, const char* bucket) {
  int64_t n = 0;
  for (const Json& L : units.arr)
    for (const Json& r : L.find(bucket)->arr) n += ival(r.arr[1]) - ival(r.arr[0]);
  return n;
}

}  // namespace

TEST(units_assignment) {
  const Desc d = g8_desc();
  for (const char* name : {"C-pro90-aux12-drafter-aux-host68", "I3-pro-only-spec-host74-file-auto"}) {
    const Json fx = fixture(name);
    Run run = run_with_desc(recipe_of(fx), probe_of(fx), components_of(fx), d);
    for (const auto& e : run.r.errors) std::printf("  %s: %s\n", name, e.c_str());
    CHECK(run.r.errors.empty());
    if (!run.r.computed) continue;
    const Json& doc = run.r.doc;
    CHECK(doc.find("tiers_plan_id") == nullptr);  // the S5 compile is not on this base
    // the numbers are still place_ref's (the desc only adds the library plan and the benefit rank)
    check_case(name, run.r);
    // the assignment covers every expert once and matches S3's counts
    const Json want = json_parse(slurp(kData + std::string(name) + ".expected.json"));
    const int64_t active = ival(*want.find("devices")->find("rtx")->find("active_slots"));
    CHECK_EQ(ranges_total(*doc.find("units"), "active"), active);
    CHECK_EQ(ranges_total(*doc.find("units"), "file"), ival(*want.find("host")->find("file_units")));
    CHECK_EQ(ranges_total(*doc.find("units"), "host"), ival(*want.find("host")->find("expert_host_homes")));
    CHECK_EQ(int64_t(doc.find("seed")->arr.size()), active);
    CHECK_EQ(doc.find("units")->arr.size(), size_t(40));
  }
  {  // per-layer spare billing is refused in v1 (V4) for the TierSite adapter: placement mode requires global billing
    const Json fx = fixture("C-pro90-aux12-drafter-aux-host68");
    Json comps = components_of(fx);
    (*const_cast<Json*>(comps.find("elastic")))["spare_billing"] = Json::string("per_layer");
    Run run = run_with_desc(recipe_of(fx), probe_of(fx), comps, d);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_CAPABILITY spare_billing: per_layer"));
    CHECK_EQ(run.r.phase, std::string("V4"));
  }
  {  // a descriptor that does not match the elastic metadata is refused, not silently projected
    const Json fx = fixture("C-pro90-aux12-drafter-aux-host68");
    Json comps = components_of(fx);
    (*const_cast<Json*>(comps.find("elastic")))["per_layer_units"] = Json::number(int64_t(256));
    (*const_cast<Json*>(comps.find("elastic")))["units_total"] = Json::number(int64_t(10240));
    Run run = run_with_desc(recipe_of(fx), probe_of(fx), comps, d);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_STALE desc projection"));
  }
  {  // the descriptor's benefit ranks the units without usage-v1; usage-v1 ranks them when given
    const Json fx = fixture("F-pro-only-plain-host68");
    RecipeStack s(recipe_of(fx), "u.toml");
    Binding b = recipe_bind(s.effective(), probe_of(fx));
    Components c = components_from_json(components_of(fx));
    // (layer, unit, count, recency) over 1,000 tokens: use = count x 1,000; three dense layers lead the file
    UsageDoc u = usage_from_json(usage_doc({{39, 383, 900, 0}, {0, 5, 900, 2}, {2, 7, 10, 0}}, 1000, "", 3));
    PlaceInput in;
    in.recipe = &s.effective();
    in.effective_sha256 = s.effective_sha256();
    in.route = "decode_rtx";
    in.binding = &b;
    in.components = &c;
    in.usage = &u;
    PlaceResult r = place(in);
    CHECK(r.errors.empty());
    if (r.computed) {
      const Json& seed = *r.doc.find("seed");
      // selections desc, recency desc, (layer, unit) asc
      CHECK_EQ(json_write(seed.arr[0]), std::string("[0,5]"));
      CHECK_EQ(json_write(seed.arr[1]), std::string("[39,383]"));
      CHECK_EQ(json_write(seed.arr[2]), std::string("[2,7]"));
      CHECK_EQ(json_write(seed.arr[3]), std::string("[0,0]"));
      CHECK_EQ(r.doc.find("usage_sha256")->text, u.sha256);
    }
  }
}

// ---- context-v1: an entry applies only on an exact match of arch, device name, driver, platform, build (§2.7) ------
TEST(context_v1_exact_match) {
  const Json f = fixture("F-pro-only-plain-host68");
  const Json probe = probe_of(f);  // the fixture's binding: the PRO, driver 617.14, windows-wddm
  const std::string pro = dev_fx(f, "rtx")->find("name")->text;
  auto ctx_doc = [&](const std::string& name, const std::string& driver, const std::string& platform, uint64_t bytes,
                     int n = 6) {
    std::string samples;
    for (int k = 0; k < n; ++k)  // oldest first; with 6, the oldest is outside the latest 5
      samples += std::string(k ? "," : "") + R"({"bytes":)" + std::to_string(k == 0 ? 1 : bytes + uint64_t(k)) +
                 R"(,"method":"pdh-process-dedicated"})";
    return context_from_json(json_parse(R"({"schema":"tierstream/context-v1","entries":[{"build_sha256":")" + kBuild +
                                        R"(","arch":"sm_120","device_name":")" + name + R"(","driver":")" + driver +
                                        R"(","platform":")" + platform + R"(","samples":[)" + samples + "]}]}"));
  };
  auto run = [&](const ContextDoc* cd) {
    RecipeStack s(recipe_of(f), "c.toml");
    Binding b = recipe_bind(s.effective(), probe);
    Components c = components_from_json(components_of(f));
    PlaceInput in;
    in.recipe = &s.effective();
    in.effective_sha256 = s.effective_sha256();
    in.route = "decode_rtx";
    in.binding = &b;
    in.components = &c;
    in.context = cd;
    return place(in);
  };
  auto charged = [&](const ContextDoc* cd) {
    PlaceResult r = run(cd);
    return r.computed ? ival(*doc_dev(r.doc, "rtx")->find("context_bytes")) : int64_t(-1);
  };
  const int64_t allowance = 644245094;
  CHECK_EQ(charged(nullptr), allowance);
  const uint64_t above = uint64_t(allowance) + 100000000;
  const ContextDoc exact = ctx_doc(pro, "617.14", "windows-wddm", above);
  CHECK_EQ(charged(&exact), int64_t(above + 3));  // median of the latest 5 (above + 1..5)
  const ContextDoc four = ctx_doc(pro, "617.14", "windows-wddm", above, 5);  // latest 5 = 1, above + 1..4: lower median
  CHECK_EQ(charged(&four), int64_t(above + 2));
  const ContextDoc two = ctx_doc(pro, "617.14", "windows-wddm", above, 3);  // 1, above + 1, above + 2: median above + 1
  CHECK_EQ(charged(&two), int64_t(above + 1));
  const ContextDoc few = ctx_doc(pro, "617.14", "windows-wddm", above, 2);  // fewer than 3: ignored
  CHECK_EQ(charged(&few), allowance);
  for (const ContextDoc& other : {ctx_doc(pro, "610.00", "windows-wddm", above), ctx_doc(pro, "617.14", "linux", above),
                                  ctx_doc("NVIDIA GeForce RTX 4090", "617.14", "windows-wddm", above)})
    CHECK_EQ(charged(&other), allowance);
  CHECK_REFUSED(ctx_doc(pro, "", "windows-wddm", above), "entries[0].driver: expected a non-empty string");  // no wildcard
  CHECK_REFUSED(context_from_json(json_parse(R"({"schema":"tierstream/context-v1","entries":[{"build_sha256":")" + kBuild +
                                             R"(","arch":"sm_120","driver":"617.14","platform":"windows-wddm",)"
                                             R"("samples":[]}]})")),
                "missing \"device_name\"");
  CHECK_REFUSED(context_from_json(json_parse(R"({"schema":"tierstream/context-v1","entries":[{"build_sha256":")" + kBuild +
                                             R"(","arch":"sm_120","device_name":"x","driver":"617.14","platform":"linux",)"
                                             R"("samples":[{"bytes":1,"method":"m","device_used_before_stable":true}]}]})")),
                "unknown key \"device_used_before_stable\"");
  // above 2 x the allowance: PL_E_STALE "context outlier"
  const ContextDoc outlier = ctx_doc(pro, "617.14", "windows-wddm", uint64_t(allowance) * 3);
  PlaceResult r = run(&outlier);
  CHECK(!r.errors.empty() && has(r.errors[0], "PL_E_STALE rtx: context outlier: re-qualify"));
}

// ---- rev 15: no WDDM margin floor (every platform measures per process); probe platform values; Q >= 1 -------------
TEST(platform_and_host_spares) {
  const Json c = fixture("C-pro90-aux12-drafter-aux-host68");
  {  // a 0.25 GiB assist margin on a WDDM binding plans (the rev 12-14 floor is gone)
    std::string t = recipe_of(c);
    const std::string off = "[routes.assist.aux]\ndevice_gib = 12.0\n";
    t.replace(t.find(off), off.size(), off + "margin_gib = 0.25\nmargin_fraction = 0.0\n");
    Run run = run_place(t, probe_of(c), components_of(c));
    for (const auto& e : run.r.errors) std::printf("  unexpected %s\n", e.c_str());
    CHECK(run.r.errors.empty());
    // and a 4 GiB allowance with the recipe's 0.6 GiB anchor margin: no context-proportional band
    Json comps = components_of(c);
    (*const_cast<Json*>(comps.find("context")))["sm_120"] = Json::number(int64_t(4) << 30);
    Run big = run_place(recipe_of(c), probe_of(c), comps);
    CHECK(big.r.errors.empty() || !has(big.r.errors[0], "margin"));
    // the platform still takes the normative values only (placement-mode binding)
    Json bad = probe_of(c);
    for (Json& d : const_cast<Json*>(bad.find("devices"))->arr) d.obj.erase("platform");
    bad["platform"] = Json::string("wddm");
    CHECK_REFUSED(run_place(t, bad, components_of(c)), "is not linux | windows-wddm | windows-tcc");
  }
  {  // the device-gated adapter needs Q >= 1 host spare (S3 step 0, PL_E_SCHEMA; G4 N3b)
    Json comps = components_of(c);
    (*const_cast<Json*>(comps.find("elastic")))["host_spares"] = Json::number(int64_t(0));
    Run run = run_place(recipe_of(c), probe_of(c), comps);
    CHECK(!run.r.errors.empty() && has(run.r.errors[0], "PL_E_SCHEMA device-gated: spares 96 / host spares 0 must be >= 1"));
  }
}

// ---- SV1 + the solver over the planner ---------------------------------------------------------------------------
TEST(sv1_solver_arithmetic) {
  const Json in = json_parse(slurp(kData + "SV1-solver.json"));
  std::string want = slurp(kData + "SV1-solver.expected.json");
  while (!want.empty() && (want.back() == '\n' || want.back() == '\r')) want.pop_back();
  CHECK_EQ(json_write(solve_predict(in)), want);
  // SV1b: an applicability missing a required key (costs without build_sha256) -> every row unpredicted
  const Json inb = json_parse(slurp(kData + "SV1b-incomplete-calibration.json"));
  std::string wantb = slurp(kData + "SV1b-incomplete-calibration.expected.json");
  while (!wantb.empty() && (wantb.back() == '\n' || wantb.back() == '\r')) wantb.pop_back();
  CHECK_EQ(json_write(solve_predict(inb)), wantb);
  // SV1c: the FILE rate per language over the pooled cold set (Spanish 90.909, code 909.091 tok/s); SV1c2: a table
  // change outside the cold set (same rows, another usage_lang_sha256); SV1d: in-sample curves; SV1e: FILE rows need
  // policy lru (rev 17)
  for (const char* name : {"SV1c-language-file-rate", "SV1c2-language-usage-identity", "SV1d-in-sample-curves",
                           "SV1e-file-rows-need-lru"}) {
    const Json inc = json_parse(slurp(kData + name + ".json"));
    std::string wantc = slurp(kData + name + ".expected.json");
    while (!wantc.empty() && (wantc.back() == '\n' || wantc.back() == '\r')) wantc.pop_back();
    const std::string got = json_write(solve_predict(inc));
    if (got != wantc) std::printf("  %s:\n    %s\n    %s\n", name, got.c_str(), wantc.c_str()), ++sstest::failures();
  }
}

// ---- usage-v1 in the producer's shape (strix-stream-adaptive adapt_usage.cpp; tests/data/placement/usage-all.json) --
TEST(usage_v1_producer_form) {
  const std::string text = slurp(kData + "usage-all.json");
  const UsageDoc u = usage_from_json(json_parse(text));
  CHECK_EQ(u.units.size(), size_t(15360));
  CHECK_EQ(u.tokens, int64_t(49430));
  // usage_sha256 = sha256 of the file bytes (the producer writes json_write of the document)
  std::string bytes = text;
  while (!bytes.empty() && (bytes.back() == '\n' || bytes.back() == '\r')) bytes.pop_back();
  CHECK_EQ(u.sha256, sha256_hex(bytes.data(), bytes.size()));
  // the first ranks: use desc, recency desc, layer asc, unit asc (recomputed independently from the file's arrays)
  const Json doc = json_parse(text);
  struct K {
    int64_t use, rec, layer, unit;
  };
  std::vector<K> all;
  for (const Json& l : doc.find("layers")->arr)
    for (size_t k = 0; k < l.find("use")->arr.size(); ++k)
      all.push_back({json_i64(l.find("use")->arr[k], "use"), json_i64(l.find("recency")->arr[k], "rec"),
                     json_i64(*l.find("index"), "index"), int64_t(k)});
  std::stable_sort(all.begin(), all.end(), [](const K& a, const K& b) {
    if (a.use != b.use) return a.use > b.use;
    if (a.rec != b.rec) return a.rec > b.rec;
    return a.layer != b.layer ? a.layer < b.layer : a.unit < b.unit;
  });
  const std::vector<size_t> order = u.rank();
  CHECK_EQ(order.size(), all.size());
  for (size_t i = 0; i < 16 && i < order.size(); ++i) {
    const UsageUnit& x = u.units[order[i]];
    if (x.model_layer != all[i].layer || x.unit != all[i].unit || x.selections != all[i].use)
      std::printf("  rank %zu: (%lld, %lld) vs (%lld, %lld)\n", i, (long long)x.model_layer, (long long)x.unit,
                  (long long)all[i].layer, (long long)all[i].unit), ++sstest::failures();
  }
  // the hottest unit is at least as used as the next, and the coldest has the smallest use
  CHECK(u.units[order[0]].selections >= u.units[order[1]].selections);
  CHECK(u.units[order.back()].selections <= u.units[order[order.size() / 2]].selections);
  // refusals: the old units[] form, an edited use, a layer index out of place
  CHECK_REFUSED(usage_from_json(json_parse(
                    R"({"schema":"tierstream/usage-v1","units":[{"layer":0,"unit":1,"selections":5}]})")),
                "unknown key \"units\"");
  Json edited = doc;
  Json& l0 = const_cast<Json*>(edited.find("layers"))->arr[0];
  const_cast<Json*>(l0.find("use"))->arr[0] = Json::number(int64_t(1));
  CHECK_REFUSED(usage_from_json(edited), "is not rdiv(count x 10^6, tokens)");
  Json moved = doc;
  (*&const_cast<Json*>(moved.find("layers"))->arr[1])["index"] = Json::number(int64_t(7));
  CHECK_REFUSED(usage_from_json(moved), "must equal its position");
  // dense layers (n_units 0) do not count as expert layers; a per-language table names its language
  const UsageDoc d = usage_from_json(usage_doc({{0, 0, 3, 0}}, 1000, "es", 3));
  CHECK_EQ(d.layer_units.size(), size_t(43));
  CHECK_EQ(d.units[0].layer, int64_t(0));
  CHECK_EQ(d.units[0].model_layer, int64_t(3));
  CHECK_EQ(d.filter, std::string("es"));
}

TEST(solver_rows) {
  const Json sv1 = json_parse(slurp(kData + "SV1-solver.json"));
  Json probe = json_parse(slurp(kData + "J2-probe.json"));
  probe["model"] = Json::string("DeepSeek-V4.1-Flash");
  probe["engine"] = Json::string("strix-cuda-ds41-replica");
  // the transfer profile (tierstream/xfer-v1 inputs, rev 17): the PRO's PCI path, LnkCap link, via, NUMA node; the
  // host block; the FILE store; the measured transfer figures
  for (Json& d : const_cast<Json*>(probe.find("devices"))->arr) {
    d["pci_path"] = json_parse(R"(["0000:00:01.1","0000:01:00.0"])");
    d["link"] = json_parse(R"({"gen":5,"width":16})");
    d["via"] = Json::string("direct");
    d["numa_node"] = Json::number(int64_t(0));
  }
  probe["xfer_host"] = json_parse(R"({"cpu_model":"AMD Ryzen 9 9950X","sockets":1,"numa_nodes":1,"pinned_policy":"node"})");
  const std::string store = "nvme:Samsung-990-PRO/gen4x4@cpu;fs=ntfs;io=direct;readers=4";
  probe["file_store"] = Json::string(store);
  const Json figures = json_parse(R"({"h2d_mbps":24000,"d2h_mbps":23000,"h2d_lat_ns":9000,"file_read_mbps":6600})");
  probe["probe_xfer"] = figures;
  RecipeStack s(slurp(kData + "J2-recipe.toml"), "J2-recipe.toml");
  s.set("routes.decode_rtx.placement.rtx.host_pinned_gib", "74", "--set");  // the 74 GiB pinned cap: I1 / I3
  // case B's components-v1 (J2: content, arithmetic, receipts and context receipts of build 0...0)
  const Json comps_b = json_parse(slurp(kData + "J2-components.json"));
  const Json canon = components_to_json(components_from_json(comps_b));
  auto sha = [](const Json& j) {
    const std::string t = json_write(j);
    return sha256_hex(t.data(), t.size());
  };
  // pooled usage: one hot unit; the code table hits the coldest unit of the pooled rank, (39, 383), twice per 1,000
  // tokens (2,000 micro-selections per token: cold = rdiv(2,000, 1,000) = 2 -> R x cold = 12 FILE milli-misses)
  const Json usage = usage_doc({{0, 1, 5, 0}});
  const Json usage_code = usage_doc({{39, 383, 2, 0}}, 1000, "code");
  // the calibration identities of this test's configuration, written by hand from the §3.9 derivations (the SV1 ones
  // name another driver and usage); every required key present
  const std::string build(64, '0'), pro = "sm_120/NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
  Json cost_app = Json::object();
  cost_app["sequences"] = Json::number(int64_t(1));
  cost_app["adapter"] = Json::string("tierstream");
  cost_app["drafter"] = Json::string("anchor");
  cost_app["anchor_device"] = Json::string(pro);
  cost_app["assist_device"] = Json::string("none");
  // xfer = sha256 of the canonical xfer-v1 profile, written here by hand
  cost_app["xfer"] = Json::string(sha(json_parse(
      R"({"schema":"tierstream/xfer-v1","devices":[{"set":"rtx","pci_path":["0000:00:01.1","0000:01:00.0"],)"
      R"("link":{"gen":5,"width":16},"via":"direct","numa_node":0}],"host":{"cpu_model":"AMD Ryzen 9 9950X",)"
      R"("sockets":1,"numa_nodes":1,"pinned_policy":"node"},"drafter":"none"})")));
  {  // cache_sha256 over {elastic, cache_config} without the R-derived fields
    Json cc = Json::object();
    cc["elastic"] = *canon.find("elastic");
    Json cfg = *comps_b.find("config")->find("cache_config");
    for (const char* r : {"dg_max_ids", "banks", "window_slots"}) cfg.obj.erase(r);
    cc["cache_config"] = cfg;
    cost_app["cache_sha256"] = Json::string(sha(cc));
  }
  cost_app["engram_store"] = Json::string("none");
  cost_app["file_store"] = Json::string(store);
  cost_app["build_sha256"] = Json::string(build);
  cost_app["content_sha256"] = Json::string(sha(*canon.find("content")));
  cost_app["arithmetic_sha256"] = Json::string(sha(*canon.find("arithmetic")));
  cost_app["prefetch"] = Json::string("off");
  cost_app["tier_mode"] = Json::string("exclusive");
  cost_app["driver"] = Json::string("617.14");
  cost_app["platform"] = Json::string("windows-wddm");
  Json curve_app = Json::object();
  for (const char* k :
       {"sequences", "adapter", "drafter", "build_sha256", "content_sha256", "prefetch", "tier_mode", "cache_sha256"})
    curve_app[k] = *cost_app.find(k);
  curve_app["policy"] = Json::string("lru");
  curve_app["usage_sha256"] = Json::string(usage_from_json(usage).sha256);
  curve_app["evaluation"] = Json::string("kfold");
  Json in = Json::object();
  in["curves"] = *sv1.find("curves");
  Json costs = *sv1.find("costs");
  costs["applicability"] = cost_app;
  in["curves_applicability"] = curve_app;
  costs["file_staging_bytes"] = Json::number(int64_t(67108864));
  costs["xfer_measured"] = figures;
  costs["provenance"] = json_parse(R"({"verify_us":{"source":"illustrative","estimate":true}})");
  in["costs"] = costs;
  in["usage"] = usage;
  Json langs = Json::object();
  langs["code"] = usage_code;
  in["usage_lang"] = langs;
  Json choices = Json::array();
  {
    Json c = Json::object();
    c["id"] = Json::string("spec-128k");
    c["mode"] = Json::string("spec");
    c["R"] = Json::number(int64_t(6));
    c["ctx"] = Json::number(int64_t(131072));
    c["lang"] = Json::string("code");
    c["components"] = comps_b;
    choices.push(c);
    Json d = c;
    d["id"] = Json::string("spec-128k-other-lang");
    d["lang"] = Json::string("prose");  // no prose/R6 curve: unpredicted, the placement still shown
    choices.push(d);
  }
  in["choices"] = choices;
  const Json out = solve(s, "decode_rtx", probe, in);
  CHECK_EQ(out.find("schema")->text, std::string("tierstream/solver-v1"));
  const Json& r0 = out.find("rows")->arr[0];
  // I1 at host 74 is 64 units short; hypothetically with FILE (its staging from the costs) it is I3: 70 spilled
  CHECK_EQ(json_write(*r0.find("requires")), std::string("[\"file_expert_tier\"]"));
  CHECK_EQ(json_write(*r0.find("placement")->find("file_units")), std::string("76"));
  CHECK_EQ(json_write(*r0.find("placement")->find("devices")->arr[0].find("active_units")), std::string("7415"));
  const Json& p0 = *r0.find("prediction");
  CHECK_EQ(p0.find("tag")->text, std::string("interpolated"));
  // the same row of SV1 (spec-I3: 7418 active, 73 file); the code table's FILE rate over the pooled cold set
  // code/R6 at 7,415 active: rdiv(45,000 x 1,500 - 9,000 x 415, 1,500) = 42,510
  CHECK_EQ(json_write(*p0.find("milli_misses_total")), std::string("42510"));
  CHECK_EQ(json_write(*p0.find("estimates")), std::string("[\"verify_us\"]"));  // provenance says estimate
  CHECK_EQ(json_write(*p0.find("milli_misses_file")), std::string("12"));
  CHECK_EQ(out.find("usage_lang_sha256")->find("code")->text, usage_from_json(usage_code).sha256);
  // the derived calib equals the hand-written identities key by key
  const Json& calib0 = *r0.find("choice")->find("calib");
  for (const Json* app : {&cost_app, &curve_app})
    for (const auto& kv : app->obj) {
      if (kv.first == "evaluation") continue;  // a property of the curves, not a calibration key
      const Json* v = calib0.find(kv.first);
      if (!v || json_write(*v) != json_write(kv.second))
        std::printf("  calib.%s: %s vs %s\n", kv.first.c_str(), v ? json_write(*v).c_str() : "-",
                    json_write(kv.second).c_str()), ++sstest::failures();
    }
  const Json& r1 = out.find("rows")->arr[1];
  CHECK_EQ(r1.find("prediction")->find("tag")->text, std::string("unpredicted"));
  CHECK(r1.find("prediction")->find("milli_tok_s")->is_null());
  CHECK_EQ(json_write(*r1.find("prediction")->find("estimates")), std::string("[]"));
  CHECK(r1.find("placement")->find("placement_sha256") != nullptr);
  // the hash covers the object without itself; the table prints tok/s with three decimals
  Json body = out;
  body.obj.erase("solver_sha256");
  const std::string t = json_write(body);
  CHECK_EQ(sha256_hex(t.data(), t.size()), out.find("solver_sha256")->text);
  const std::string tab = solve_table(out);
  CHECK(has(tab, "requires file_expert_tier"));
  CHECK(has(tab, "unpredicted"));
  auto one = [&](const Json& choice) {
    Json in2 = in;
    Json ch = Json::array();
    ch.push(choice);
    in2["choices"] = ch;
    return in2;
  };
  auto first_tag = [&](const Json& in2) {
    return solve(s, "decode_rtx", probe, in2).find("rows")->arr[0].find("prediction")->find("tag")->text;
  };
  const Json spec = in.find("choices")->arr[0];
  CHECK_EQ(first_tag(one(spec)), std::string("interpolated"));
  {  // rev 17 gates: in-sample curves; probe transfer figures outside +-10 %; FILE rows need lru; another FILE store
    Json in2 = one(spec);
    (*const_cast<Json*>(in2.find("curves_applicability")))["evaluation"] = Json::string("in-sample");
    CHECK_EQ(first_tag(in2), std::string("unpredicted"));
    Json e = spec;
    e["probe_xfer"] = json_parse(R"({"h2d_mbps":21000,"d2h_mbps":23000,"h2d_lat_ns":9000,"file_read_mbps":6600})");
    CHECK_EQ(first_tag(one(e)), std::string("unpredicted"));
    e["probe_xfer"] = json_parse(R"({"h2d_mbps":22000,"d2h_mbps":23000,"h2d_lat_ns":9000,"file_read_mbps":6600})");
    CHECK_EQ(first_tag(one(e)), std::string("interpolated"));
    e["probe_xfer"] = json_parse(R"({"h2d_mbps":24000,"d2h_mbps":23000,"h2d_lat_ns":9000,"file_read_mbps":3500})");
    CHECK_EQ(first_tag(one(e)), std::string("unpredicted"));  // a FILE row: file_read_mbps is checked
    Json in3 = one(spec);
    (*const_cast<Json*>(in3.find("curves_applicability")))["policy"] = Json::string("scored");
    CHECK_EQ(first_tag(in3), std::string("unpredicted"));
    Json bare = probe;
    bare["file_store"] = Json::string("nvme:Samsung-980/gen4x4@cpu;fs=ntfs;io=direct;readers=4");
    CHECK_EQ(solve(s, "decode_rtx", bare, one(spec)).find("rows")->arr[0].find("prediction")->find("tag")->text,
             std::string("unpredicted"));
    // a malformed language table (another shape) -> unpredicted, its sha still recorded
    Json in4 = one(spec);
    Json langs4 = Json::object();
    langs4["code"] = usage_doc({{0, 0, 1, 0}}, 1000, "code", 0, 39);
    in4["usage_lang"] = langs4;
    const Json o4 = solve(s, "decode_rtx", probe, in4);
    CHECK_EQ(o4.find("rows")->arr[0].find("prediction")->find("tag")->text, std::string("unpredicted"));
    CHECK(o4.find("usage_lang_sha256")->find("code") != nullptr);
  }
  {  // FILE units without the row's language table -> unpredicted (rev 16)
    Json in2 = one(spec);
    in2.obj.erase("usage_lang");
    CHECK_EQ(first_tag(in2), std::string("unpredicted"));
  }
  {  // xfer: another link, or a probe without the PCIe fields ("unknown" matches nothing)
    Json in2 = one(spec);
    Json c2 = costs;
    (*const_cast<Json*>(c2.find("applicability")))["xfer"] = Json::string("rtx:gen4x4@chipset;drafter=none");
    in2["costs"] = c2;
    CHECK_EQ(first_tag(in2), std::string("unpredicted"));
    Json bare = probe;
    for (Json& d : const_cast<Json*>(bare.find("devices"))->arr) d.obj.erase("via");
    const Json o = solve(s, "decode_rtx", bare, one(spec));
    CHECK_EQ(o.find("rows")->arr[0].find("choice")->find("calib")->find("xfer")->text, std::string("unknown"));
    CHECK_EQ(o.find("rows")->arr[0].find("prediction")->find("tag")->text, std::string("unpredicted"));
    Json in3 = one(spec);
    Json c3 = costs;
    (*const_cast<Json*>(c3.find("applicability")))["xfer"] = Json::string("unknown");
    in3["costs"] = c3;
    CHECK_EQ(solve(s, "decode_rtx", bare, in3).find("rows")->arr[0].find("prediction")->find("tag")->text,
             std::string("unpredicted"));
  }
  {  // another cache configuration (cache_sha256 over {elastic, cache_config}) -> unpredicted
    Json e = spec;
    Json comps = *e.find("components");
    (*const_cast<Json*>(comps.find("config")->find("cache_config")))["host_spares"] = Json::number(int64_t(1));
    e["components"] = comps;
    CHECK_EQ(first_tag(one(e)), std::string("unpredicted"));
  }
  {  // an incomplete calibration (a required key missing, SV1b) -> unpredicted; another driver / assist -> unpredicted
    for (const char* k : {"build_sha256", "xfer", "cache_sha256", "assist_device", "driver"}) {
      Json in2 = one(spec);
      Json c2 = costs;
      const_cast<Json*>(c2.find("applicability"))->obj.erase(k);
      in2["costs"] = c2;
      CHECK_EQ(first_tag(in2), std::string("unpredicted"));
    }
    for (const char* k : {"policy", "usage_sha256", "cache_sha256"}) {
      Json in2 = one(spec);
      const_cast<Json*>(in2.find("curves_applicability"))->obj.erase(k);
      CHECK_EQ(first_tag(in2), std::string("unpredicted"));
    }
    Json in2 = one(spec);
    Json c2 = costs;
    (*const_cast<Json*>(c2.find("applicability")))["driver"] = Json::string("610.00");
    in2["costs"] = c2;
    CHECK_EQ(first_tag(in2), std::string("unpredicted"));
    Json c3 = costs;
    (*const_cast<Json*>(c3.find("applicability")))["assist_device"] = Json::string("sm_89/NVIDIA GeForce RTX 4090");
    in2["costs"] = c3;
    CHECK_EQ(first_tag(in2), std::string("unpredicted"));
  }
  {  // the calibration identity is derived, never typed
    Json e = spec;
    e["calib"] = json_parse(R"({"adapter":"tierstream"})");
    CHECK_REFUSED(solve(s, "decode_rtx", probe, one(e)), "the calibration identity is derived by the solver");
  }
  {  // a configuration the costs were not measured on (engine cache) is unpredicted, its placement still shown
    Json e = spec;
    Json comps = *e.find("components");
    (*const_cast<Json*>(comps.find("config")))["cache"] = Json::string("engine");
    const_cast<Json*>(comps.find("elastic"))->obj.erase("spare_billing");
    for (Json& comp : const_cast<Json*>(comps.find("components"))->arr)
      if (Json* ch = const_cast<Json*>(comp.find("charges")))
        ch->arr.erase(std::remove_if(ch->arr.begin(), ch->arr.end(), [](const Json& x) { return x.find("name") != nullptr; }),
                      ch->arr.end());
    e["components"] = comps;
    const Json o = solve(s, "decode_rtx", probe, one(e));
    const Json& rr = o.find("rows")->arr[0];
    CHECK_EQ(rr.find("prediction")->find("tag")->text, std::string("unpredicted"));
    CHECK_EQ(rr.find("choice")->find("calib")->find("adapter")->text, std::string("engine"));
    CHECK(rr.find("placement")->find("placement_sha256") != nullptr);
  }
  // without the FILE staging figure the row stays a refusal with its units short
  costs.obj.erase("file_staging_bytes");
  in["costs"] = costs;
  const Json out2 = solve(s, "decode_rtx", probe, in);
  const Json& rf = *out2.find("rows")->arr[0].find("placement")->find("refusal");
  CHECK_EQ(rf.find("code")->text, std::string("PL_E_HOST_PINNED"));
  CHECK_EQ(json_write(*rf.find("units_short")), std::string("69"));
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
