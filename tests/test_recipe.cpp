// CPU tests of recipes (docs/recipes.md): TOML reader, schema validation, profile / override layers and explain,
// hardware binding, admission with fallbacks, and compilation to plan-v1 (parity with ss_plan_make).
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "check.hpp"
#include "desc.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "plan.hpp"
#include "recipe.hpp"
#include "sha256.hpp"
#include "strix_stream/ss.h"
#include "toml.hpp"

using namespace ss;

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

constexpr uint64_t GiB = uint64_t(1) << 30;
constexpr uint64_t MiB = uint64_t(1) << 20;

std::string layers_json(int layers, const std::string& kind, int units, uint64_t unit, const std::string& benefit) {
  std::string s;
  for (int L = 0; L < layers; ++L) {
    if (L) s += ",";
    s += R"({"index":)" + std::to_string(L) + R"(,"kind":")" + kind + R"(","n_units":)" + std::to_string(units) +
         R"(,"top_k":)" + std::to_string(units < 2 ? 1 : 2) + R"(,"parts":[{"format":"Q2_K","offset":0,"bytes":)" +
         std::to_string(unit) + R"(,"stride":)" + std::to_string(unit) + "}]" + benefit + "}";
  }
  return s;
}

std::string desc_text(int layers, const std::string& kind, int units, uint64_t unit, const std::string& order,
                      const std::string& tiers, const std::string& extra = "", const std::string& benefit = "") {
  return R"({"schema":"strix-stream/desc-v1","model":"mm","engine":"t","order":")" + order + R"(","tiers":")" + tiers +
         R"(","fixed_device":[{"name":"dense","bytes":4294967296}],"sources":[{"path":"x.bin"}],)" + extra +
         R"("layers":[)" + layers_json(layers, kind, units, unit, benefit) + "]}";
}

Desc make_desc(const std::string& text) {
  Desc d = desc_from_json(json_parse(text));
  return d;
}

// A MiniMax-like lab: RTX PRO prefill from 4K with resident / pinned / NVMe layer tiers, pair TP decode.
const char* kLab = R"(
schema = "strix-stream/recipe-v1"
name = "mm-lab"
description = "test lab"
pin = ["routes.prefill_long.when"]

[match]
model = "mm"

[[match.devices]]
id = "rtx"
kind = "cuda"
name = "RTX PRO"
memory_min_gib = 90.0
host_ram_min_gib = 100.0
disk_read_gbytes_ps_min = 5.0

[[match.devices]]
id = "pair"
kind = "hip"
count = 2
memory_min_gib = 100.0
internal_link = { kind = "usb4", gbps_min = 30.0 }

[[match.links]]
id = "pc-nodes"
between = ["rtx", "pair"]
kind = "ethernet"
gbps_min = 1.0

[limits]
host_free_ram_floor_gib = 8.0

[[routes]]
id = "prefill_long"
phase = "prefill"
devices = ["rtx"]
execute = "single"
handoff_route = "decode_pair"
links = ["pc-nodes"]
fallback = "prefill_nodes"
when = { prompt_tokens_min = 4096 }
x-cite = "mmrtxpf report: 8K 16.1 s vs 45 s"

[routes.placement.rtx]
device_gib = 90.0
host_pinned_gib = 48.0
workspace_gib = 4.0
max_context_tokens = 32768
max_batch = 1
chunk_tokens = 4096

[[routes.placement.rtx.units]]
id = "layers"
class = "blob"
resident = 32
pinned = 16
rest = "disk"

[[routes]]
id = "prefill_nodes"
phase = "prefill"
devices = ["pair"]
execute = "tp"
handoff_route = "decode_pair"

[routes.placement.pair]
device_gib = 100.0
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 4

[[routes]]
id = "decode_pair"
phase = "decode"
devices = ["pair"]
execute = "tp"

[routes.placement.pair]
device_gib = 100.0
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 8
)";

std::string lab_with(const std::string& from, const std::string& to) {
  std::string s = kLab;
  size_t p = s.find(from);
  if (p == std::string::npos) return "missing anchor: " + from;
  return s.replace(p, from.size(), to);
}

Recipe parse_recipe(const std::string& text) { return recipe_from_json(recipe_doc_parse(text)); }

const char* kProbe =
    R"({"model":"mm","engine":"t","engine_sha256":"e000000000000000000000000000000000000000000000000000000000000000",
 "model_sha256":"a000000000000000000000000000000000000000000000000000000000000000","workload":"prefill-8k",
 "descriptors":{"rtx":["d1"],"pair":["d2","d3"]},
 "devices":[
  {"id":"GPU-4090","kind":"cuda","name":"NVIDIA GeForce RTX 4090","memory_gib":23.6,"host":"pc","host_ram_gib":127.7,"disk_read_gbytes_ps":6.8,"h2d_gbytes_ps":25.0,"ordinal":1,"driver":"580.88"},
  {"id":"GPU-PRO","kind":"cuda","name":"NVIDIA RTX PRO 6000 Blackwell","memory_gib":95.6,"host":"pc","host_ram_gib":127.7,"disk_read_gbytes_ps":6.8,"h2d_gbytes_ps":25.1,"ordinal":0,"driver":"580.88"},
  {"id":"node1","kind":"hip","name":"Strix Halo","memory_gib":117.0,"host":"node1","host_ram_gib":128.0,"disk_read_gbytes_ps":5.9,"driver":"6.4.3"},
  {"id":"node2","kind":"hip","name":"Strix Halo","memory_gib":117.0,"host":"node2","host_ram_gib":128.0,"disk_read_gbytes_ps":5.8,"driver":"6.4.3"}],
 "links":[{"kind":"usb4","hosts":["node1","node2"],"gbps":37.0},
          {"kind":"ethernet","hosts":["pc","node1"],"gbps":2.5},{"kind":"ethernet","hosts":["pc","node2"],"gbps":2.5}]})";

std::string blob_desc() { return desc_text(60, "blob", 1, 2 * GiB, "prefix", "pinned|disk"); }
std::string pair_desc() { return desc_text(4, "moe", 8, 64 * MiB, "ranked", "pinned|disk"); }

std::string desc_sha(const Desc& d) {
  std::string t = json_write(desc_to_json(d));
  return sha256_hex(t.data(), t.size());
}

}  // namespace

// ---- TOML ---------------------------------------------------------------------------------------------------------
TEST(toml_values_and_tables) {
  Json j = toml_parse(R"(
# comment
title = "a \"q\" \u00e9\tx"  # trailing comment
lit = 'C:\path'
n = 1_000
h = 0xff
neg = -7
f = 2.5e3
b = true
arr = [1, [2, 3], "x",]   # mixed, trailing comma
ml = """
line1 \
   line2"""
mll = '''
raw\n'''
dot.ted.key = 1
"quoted key" = 2
inline = { a = 1, b.c = 2 }

[t1]
v = 1
[t1.sub]
w = 2
[[aot]]
k = 1
[aot.sub]
z = 1
[[aot]]
k = 2
)");
  CHECK_EQ(j.find("title")->text, std::string("a \"q\" \xc3\xa9\tx"));
  CHECK_EQ(j.find("lit")->text, std::string("C:\\path"));
  CHECK_EQ(j.find("n")->text, std::string("1000"));
  CHECK_EQ(j.find("h")->text, std::string("255"));
  CHECK_EQ(j.find("neg")->text, std::string("-7"));
  CHECK_EQ(j.find("f")->text, std::string("2.5e3"));
  CHECK(j.find("b")->b);
  CHECK_EQ(j.find("arr")->arr.size(), size_t(3));
  CHECK_EQ(j.find("ml")->text, std::string("line1 line2"));
  CHECK_EQ(j.find("mll")->text, std::string("raw\\n"));
  CHECK_EQ(j.find("dot")->find("ted")->find("key")->text, std::string("1"));
  CHECK_EQ(j.find("quoted key")->text, std::string("2"));
  CHECK_EQ(j.find("inline")->find("b")->find("c")->text, std::string("2"));
  CHECK_EQ(j.find("t1")->find("sub")->find("w")->text, std::string("2"));
  CHECK_EQ(j.find("aot")->arr.size(), size_t(2));
  CHECK_EQ(j.find("aot")->arr[0].find("sub")->find("z")->text, std::string("1"));
  CHECK_EQ(j.find("aot")->arr[1].find("k")->text, std::string("2"));
}

TEST(toml_refusals) {
  CHECK_REFUSED(toml_parse("a = 1\na = 2\n"), "duplicate key");
  CHECK_REFUSED(toml_parse("[t]\n[t]\n"), "defined twice");
  CHECK_REFUSED(toml_parse("d = 1979-05-27\n"), "dates");
  CHECK_REFUSED(toml_parse("t = 07:32:00\n"), "dates");
  CHECK_REFUSED(toml_parse("x = inf\n"), "inf and nan");
  CHECK_REFUSED(toml_parse("x = 012\n"), "leading zero");
  CHECK_REFUSED(toml_parse("x = 1__0\n"), "'_'");
  CHECK_REFUSED(toml_parse("x = \"\\q\"\n"), "unknown escape");
  CHECK_REFUSED(toml_parse("x = {a = 1}\n[x]\n"), "already has a value");
  CHECK_REFUSED(toml_parse("x = {a = 1}\nx.b = 2\n"), "cannot be extended");
  CHECK_REFUSED(toml_parse("a = [1]\n[[a]]\n"), "not an array of tables");
  CHECK_REFUSED(toml_parse("[f]\na.b = 1\n[f.a]\n"), "defined twice");
  CHECK_REFUSED(toml_parse("x = 1 y = 2\n"), "end of the line");
  CHECK_REFUSED(toml_parse("x = 9223372036854775808\n"), "out of range");
  CHECK_REFUSED(toml_parse("x = \"abc\n"), "unterminated");
  CHECK_REFUSED(toml_parse("x = {a = 1,\n b = 2}\n"), "one line");
  CHECK_REFUSED(toml_parse("x-note = 1e999\n"), "out of range");
  CHECK_REFUSED(toml_parse("s = \"\xC3\x28\"\n"), "invalid UTF-8");
  CHECK_REFUSED(toml_parse("s = \"\xED\xA0\x80\"\n"), "invalid UTF-8");
  // Line numbers in messages.
  CHECK_REFUSED(toml_parse("a = 1\n\nb = = 2\n"), "line 3");
}

TEST(sha256_vectors) {
  CHECK_EQ(sha256_hex("", 0), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
  CHECK_EQ(sha256_hex("abc", 3), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
  std::string m(1000, 'a');
  CHECK_EQ(sha256_hex(m.data(), m.size()),
           std::string("41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"));
}

// ---- schema -------------------------------------------------------------------------------------------------------
TEST(recipe_valid_toml_and_json_agree) {
  Recipe r = parse_recipe(kLab);
  CHECK_EQ(r.name, std::string("mm-lab"));
  CHECK_EQ(r.routes.size(), size_t(3));
  const RPlacement* p = r.route("prefill_long")->placement("rtx");
  CHECK(p != nullptr);
  CHECK_EQ(p->get("staging_mib"), 256.0);     // default
  CHECK_EQ(p->get("margin_fraction"), 0.03);  // default
  CHECK_EQ(p->groups[0].resident, int64_t(32));
  CHECK_EQ(p->groups[0].rank == Rank::Model, true);  // default
  // The canonical form re-parses (as JSON) to the same canonical form.
  const std::string canon = json_write(recipe_to_json(r));
  Recipe again = parse_recipe(canon);
  CHECK_EQ(json_write(recipe_to_json(again)), canon);
  // Integers stay integers, floats shortest.
  CHECK(has(canon, "\"max_context_tokens\":32768"));
  CHECK(has(canon, "\"margin_fraction\":0.03"));
}

TEST(recipe_refusals) {
  auto refuse = [](const std::string& from, const std::string& to, const char* part) {
    std::string e = error_of([&] { parse_recipe(lab_with(from, to)); });
    if (!has(e, part)) {
      std::printf("  refusal for [%s -> %s]: expected \"%s\", got \"%s\"\n", from.c_str(), to.c_str(), part, e.c_str());
      ++sstest::failures();
    }
  };
  refuse("chunk_tokens = 4096", "chunk_tokens = 4096\nresidnet = 3",
         "routes.prefill_long.placement.rtx: unknown key \"residnet\"");
  refuse("chunk_tokens = 4096", "chunk_tokens = 4096\nexact = true", "exactness is not a recipe option");
  refuse("rest = \"disk\"", "rest = \"disk\"\nbit_exact = false", "exactness is not a recipe option");
  refuse("max_batch = 1\n", "\n", "missing \"max_batch\"");
  refuse("max_context_tokens = 32768\nmax_batch = 1", "max_batch = 1", "missing \"max_context_tokens\"");
  refuse("device_gib = 90.0", "device_gib = 95.0", "above the set's memory_min_gib");
  refuse("device_gib = 90.0", "device_gib = 0", "out of range (0");
  refuse("chunk_tokens = 4096", "chunk_tokens = 4096.5", "expected an integer");
  refuse("chunk_tokens = 4096", "chunk_tokens = \"4096\"", "expected an integer");
  refuse("resident = 32", "resident = \"all\"", "\"fill\"");
  refuse("rest = \"disk\"", "rest = \"pair\"", "is not one of");
  refuse("execute = \"single\"", "execute = \"tp\"", "\"tp\" runs on one device set with count >= 2");
  refuse("id = \"prefill_nodes\"\nphase = \"prefill\"\ndevices = [\"pair\"]\nexecute = \"tp\"",
         "id = \"prefill_nodes\"\nphase = \"prefill\"\ndevices = [\"pair\"]\nexecute = \"single\"",
         "\"single\" runs on one device set with count 1");
  refuse("fallback = \"prefill_nodes\"", "fallback = \"decode_pair\"", "a route of another phase");
  refuse("fallback = \"prefill_nodes\"", "fallback = \"prefill_long\"", "cannot fall back to itself");
  refuse("handoff_route = \"decode_pair\"\nlinks", "handoff_route = \"prefill_nodes\"\nlinks", "is not a decode route");
  refuse("links = [\"pc-nodes\"]\n", "", "no link connects");
  refuse("when = { prompt_tokens_min = 4096 }", "when = { prompt_tokens_min = 4096, prompt_tokens_max = 100 }",
         "below prompt_tokens_min");
  refuse("id = \"decode_pair\"\nphase = \"decode\"\ndevices = [\"pair\"]\nexecute = \"tp\"",
         "id = \"decode_pair\"\nphase = \"decode\"\ndevices = [\"pair\"]\nexecute = \"tp\"\nwhen = { batch_max = 4 }",
         "must be a catch-all");
  refuse("rest = \"disk\"", "rest = \"peer\"\npeer = \"pair\"", "\"peer\" only in a hetero route");
  refuse("rest = \"disk\"", "rest = \"pinned\"\npinned = \"fill\"", "duplicate key");  // pinned already set
  refuse("pinned = 16\nrest = \"disk\"", "pinned = \"fill\"\nrest = \"pinned\"", "one of them must bound");
  refuse("resident = 32", "resident = 32\nlist = [1, 2]", "only with rank = \"list\"");
  refuse("resident = 32", "resident = 32\nrank = \"list\"", "needs \"list\"");
  refuse("rest = \"disk\"\n", "rest = \"disk\"\n\n[[routes.placement.rtx.units]]\nid = \"two\"\nclass = \"blob\"\n",
         "overlaps group \"layers\"");
  refuse("host_pinned_gib = 48.0", "host_pinned_gib = 0.0", "pinned units need host_pinned_gib > 0");
  refuse("[routes.placement.pair]\ndevice_gib = 100.0\nworkspace_gib = 2.0\nmax_context_tokens = 32768\nmax_batch = 4",
         "", "every set of a route needs a placement");
  refuse("pin = [\"routes.prefill_long.when\"]", "pin = [\"routes.prefill_long.devices\"]", "not settable");
  refuse("pin = [\"routes.prefill_long.when\"]", "pin = [\"routes.nope\"]", "unknown route");
  refuse("internal_link = { kind = \"usb4\", gbps_min = 30.0 }", "internal_link = { kind = \"usb5\" }",
         "is not one of");
  refuse("schema = \"strix-stream/recipe-v1\"", "schema = \"strix-stream/recipe-v2\"",
         "expected \"strix-stream/recipe-v1\"");
  refuse("[limits]\nhost_free_ram_floor_gib = 8.0", "[limits]\nhost_free_ram_floor_gib = -1.0", "out of range");
  // A recipe without a decode route.
  CHECK_REFUSED(parse_recipe(R"(schema = "strix-stream/recipe-v1"
name = "x"
[match]
model = "m"
[[match.devices]]
id = "a"
kind = "cpu"
memory_min_gib = 1.0
[[routes]]
id = "p"
phase = "prefill"
devices = ["a"]
execute = "single"
[routes.placement.a]
max_context_tokens = 10
max_batch = 1
)"),
                "at least one decode route");
}

// ---- layers, precedence, explain ---------------------------------------------------------------------------------
namespace {

std::string profile_text(const std::string& sha, const std::string& set_lines, const std::string& driver = "580.88",
                         const std::string& h2d = "25.1", const std::string& workload = "prefill-8k") {
  return "schema = \"strix-stream/recipe-profile-v1\"\nrecipe = \"mm-lab\"\nrecipe_sha256 = \"" + sha +
         "\"\nworkload = \"" + workload +
         "\"\nsource = \"tune\"\n"
         "[key]\nengine_sha256 = \"e000000000000000000000000000000000000000000000000000000000000000\"\n"
         "model_sha256 = \"a000000000000000000000000000000000000000000000000000000000000000\"\n"
         "devices = { rtx = [\"GPU-PRO\"], pair = [\"node1\", \"node2\"] }\n"
         "drivers = { rtx = \"" +
         driver +
         "\", pair = \"6.4.3\" }\n"
         "descriptors = { rtx = [\"d1\"], pair = [\"d2\", \"d3\"] }\n"
         "[probe.rtx]\ndisk_read_gbytes_ps = 6.8\nh2d_gbytes_ps = " +
         h2d +
         "\n"
         "[probe.pair]\ndisk_read_gbytes_ps = 5.8\n"
         "[set]\n" +
         set_lines;
}

const Json* field_of(const Json& explain, const std::string& path) {
  for (const Json& f : explain.find("fields")->arr)
    if (f.find("path")->text == path) return &f;
  return nullptr;
}

}  // namespace

TEST(layers_precedence_pins_and_explain) {
  RecipeStack s(kLab, "lab.toml");
  const Json probe = json_parse(kProbe);
  // Profile: tunes the resident count (applies) and the prompt threshold (pinned by the recipe: ignored).
  s.add_profile(profile_text(s.sha256(),
                             "\"routes.prefill_long.placement.rtx.units.layers.resident\" = 33\n"
                             "\"routes.prefill_long.when.prompt_tokens_min\" = 3072\n"
                             "\"routes.decode_pair.placement.pair.lookahead_units\" = 4\n"),
                "tune.toml", probe);
  CHECK_EQ(s.effective().route("prefill_long")->placement("rtx")->groups[0].resident, int64_t(33));
  CHECK_EQ(s.effective().route("prefill_long")->when.prompt_tokens_min, int64_t(4096));
  // Override beats the profile (and a pin).
  s.add_overrides(
      "schema = \"strix-stream/recipe-overrides-v1\"\nrecipe = \"mm-lab\"\n[set]\n"
      "\"routes.decode_pair.placement.pair.lookahead_units\" = 6\n"
      "\"routes.prefill_long.when.prompt_tokens_min\" = 2048\n",
      "ops.toml");
  s.set("routes.decode_pair.placement.pair.lookahead_units", "7", "--set");
  CHECK_EQ(s.effective().route("decode_pair")->placement("pair")->get("lookahead_units"), 7.0);
  CHECK_EQ(s.effective().route("prefill_long")->when.prompt_tokens_min, int64_t(2048));

  Json e = s.explain();
  const Json* res = field_of(e, "routes.prefill_long.placement.rtx.units.layers.resident");
  CHECK(res && res->find("from")->text == "profile" && res->find("origin")->text == "tune.toml");
  CHECK(res && res->find("shadowed")->arr[0].find("from")->text == "recipe");
  const Json* pm = field_of(e, "routes.prefill_long.when.prompt_tokens_min");
  CHECK(pm && pm->find("from")->text == "override" && pm->find("origin")->text == "ops.toml");
  bool pinned_note = false;
  for (const Json& x : pm->find("shadowed")->arr)
    pinned_note = pinned_note || has(x.find("reason")->text, "pinned by recipe: routes.prefill_long.when");
  CHECK(pinned_note);
  const Json* la = field_of(e, "routes.decode_pair.placement.pair.lookahead_units");
  CHECK(la && la->find("from")->text == "override" && la->find("origin")->text == "--set");
  CHECK_EQ(la->find("shadowed")->arr.size(), size_t(2));  // profile 4, ops.toml 6
  const Json* st = field_of(e, "routes.prefill_long.placement.rtx.staging_mib");
  CHECK(st && st->find("from")->text == "default");
  const Json* dg = field_of(e, "routes.prefill_long.placement.rtx.device_gib");
  CHECK(dg && dg->find("from")->text == "recipe");
}

TEST(layers_refusals_are_atomic) {
  RecipeStack s(kLab, "lab.toml");
  const std::string before = json_write(s.effective_json());
  const std::string sha_before = s.effective_sha256();
  // Structure is not settable, tables are not leaves, ids must exist, values are checked.
  CHECK_REFUSED(s.set("routes.prefill_long.devices", "[\"pair\"]", "--set"), "not settable");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx", "{}", "--set"), "names a table");
  CHECK_REFUSED(s.set("routes.nope.fallback", "\"x\"", "--set"), "unknown route");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.units.layers.class", "\"moe\"", "--set"), "not settable");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.device_gib", "200.0", "--set"),
                "above the set's memory_min_gib");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.device_gib", "200.0", "--set"),
                "(value set by override --set)");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.chunk_tokens", "nope", "--set"), "not a TOML value");
  CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.exact", "true", "--set"), "unknown placement field");
  CHECK_REFUSED(s.add_overrides("schema = \"strix-stream/recipe-overrides-v1\"\nrecipe = \"other\"\n[set]\n", "o"),
                "is not this recipe");
  CHECK_REFUSED(s.add_overrides("schema = \"strix-stream/recipe-profile-v1\"\nrecipe = \"mm-lab\"\n[set]\n", "o"),
                "schema: expected");
  CHECK_EQ(json_write(s.effective_json()), before);
  CHECK_EQ(s.effective_sha256(), sha_before);
}

TEST(profile_staleness) {
  const Json probe = json_parse(kProbe);
  const std::string set = "\"routes.prefill_long.placement.rtx.units.layers.resident\" = 33\n";
  {
    RecipeStack s(kLab, "lab.toml");
    CHECK_REFUSED(s.add_profile(profile_text(std::string(64, '0'), set), "p", probe), "profile stale: recipe_sha256");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set, "575.00"), "p", probe), "key.drivers.rtx is 575.00");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set, "580.88", "30.0"), "p", probe), "drifted");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set, "580.88", "25.1", "decode-r1"), "p", probe),
                  "tuned for workload");
    Json other = probe;
    other["descriptors"]["rtx"].arr[0] = Json::string("d9");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set), "p", other), "differ from the descriptors in use");
    Json no_engine = probe;
    no_engine.obj.erase("engine_sha256");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set), "p", no_engine), "the identity has no engine_sha256");
    std::string missing_key = profile_text(s.sha256(), set);
    missing_key.replace(missing_key.find("descriptors = {"), 15, "x-descriptors = {");
    CHECK_REFUSED(s.add_profile(missing_key, "p", probe), "missing \"key.descriptors\"");
    // Still untuned: the recipe's value.
    CHECK_EQ(s.effective().route("prefill_long")->placement("rtx")->groups[0].resident, int64_t(32));
    s.add_profile(profile_text(s.sha256(), set), "p", probe);
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set), "p2", probe), "at most one");
    // Budgets of a tuned placement and the hardware match are frozen while the profile applies.
    CHECK_REFUSED(s.set("routes.prefill_long.placement.rtx.device_gib", "80.0", "--set"), "drop the profile");
    CHECK_REFUSED(s.set("match.devices.rtx.memory_min_gib", "80.0", "--set"), "drop the profile");
    s.set("routes.prefill_long.placement.rtx.chunk_tokens", "8192", "--set");  // not a budget: fine
    s.set("routes.decode_pair.placement.pair.device_gib", "90.0", "--set");    // untuned placement: fine
  }
  {
    // The profile added after a budget override of a placement it tunes is refused too.
    RecipeStack s(kLab, "lab.toml");
    s.set("routes.prefill_long.placement.rtx.host_pinned_gib", "40.0", "--set");
    CHECK_REFUSED(s.add_profile(profile_text(s.sha256(), set), "p", probe), "drop the profile");
  }
}

// ---- hardware binding ---------------------------------------------------------------------------------------------
TEST(match_binding_and_refusals) {
  Recipe r = parse_recipe(kLab);
  Binding b = recipe_bind(r, json_parse(kProbe));
  CHECK_EQ(b.of("rtx")->size(), size_t(1));
  CHECK_EQ(b.of("rtx")->at(0).id, std::string("GPU-PRO"));  // the 4090 fails the name and memory
  CHECK_EQ(b.of("pair")->at(0).id, std::string("node1"));
  CHECK_EQ(b.of("pair")->at(1).id, std::string("node2"));
  // Missing hardware: one line per unmet requirement.
  Json p = json_parse(kProbe);
  p["devices"].arr.erase(p["devices"].arr.begin() + 1);  // no PRO
  std::string e = error_of([&] { recipe_bind(r, p); });
  CHECK(has(e, "match.devices.rtx: needs 1 cuda device(s) named \"RTX PRO\" with >= 90 GiB"));
  CHECK(has(e, "NVIDIA GeForce RTX 4090 (GPU-4090): name"));
  // Internal link down in the probe: no assignment satisfies it.
  Json q = json_parse(kProbe);
  q["links"].arr.erase(q["links"].arr.begin());
  CHECK_REFUSED(recipe_bind(r, q), "match.devices.pair.internal_link");
  // A PCIe bandwidth minimum the probed card does not reach.
  Recipe h = parse_recipe(
      lab_with("disk_read_gbytes_ps_min = 5.0", "disk_read_gbytes_ps_min = 5.0\nh2d_gbytes_ps_min = 30.0"));
  CHECK_REFUSED(recipe_bind(h, json_parse(kProbe)), "NVIDIA RTX PRO 6000 Blackwell (GPU-PRO): H2D 25.1 GB/s");
  CHECK_REFUSED(parse_recipe(lab_with("count = 2\n", "count = 2\nh2d_gbytes_ps_min = 1.0\n")), "only for a cuda set");
  // Wrong model.
  Json m = json_parse(kProbe);
  m["model"] = Json::string("other");
  CHECK_REFUSED(recipe_bind(r, m), "match.model: recipe is for \"mm\"");
}

TEST(match_search_is_topology_aware) {
  // Two eligible cards for "a"; only the second one's host links to "b". First-fit would pick the first and fail.
  Recipe r = parse_recipe(R"(schema = "strix-stream/recipe-v1"
name = "topo"
[match]
model = "m"
[[match.devices]]
id = "a"
kind = "cuda"
memory_min_gib = 10.0
[[match.devices]]
id = "b"
kind = "hip"
memory_min_gib = 10.0
[[match.links]]
id = "ab"
between = ["a", "b"]
kind = "ethernet"
[[routes]]
id = "d"
phase = "decode"
devices = ["a"]
execute = "single"
[routes.placement.a]
max_context_tokens = 10
max_batch = 1
)");
  Json probe = json_parse(R"({"model":"m","devices":[
    {"id":"c1","kind":"cuda","name":"x","memory_gib":24,"host":"h1"},
    {"id":"c2","kind":"cuda","name":"x","memory_gib":24,"host":"h2"},
    {"id":"s1","kind":"hip","name":"y","memory_gib":100,"host":"h3"}],
    "links":[{"kind":"ethernet","hosts":["h3","h2"],"gbps":1}]})");
  Binding b = recipe_bind(r, probe);
  CHECK_EQ(b.of("a")->at(0).id, std::string("c2"));
  // Same report, same binding.
  CHECK_EQ(json_write(binding_json(recipe_bind(r, probe))), json_write(binding_json(b)));
}

// ---- compilation ------------------------------------------------------------------------------------------------
TEST(plan_blob_tiers_32_16_12_and_plan_json_roundtrip) {
  RecipeStack s(kLab, "lab.toml");
  Desc d = make_desc(blob_desc());
  RecipePlan p = recipe_plan(s.effective(), s.effective_sha256(), "prefill_long", "rtx", 0, "GPU-PRO", d, {}, {});
  CHECK_EQ(p.plan.n_resident, uint64_t(32));
  CHECK_EQ(p.plan.n_pinned, uint64_t(16));
  CHECK_EQ(p.plan.n_disk, uint64_t(12));
  // Model rank: layers 0-31 resident, 32-47 pinned, 48-59 disk.
  CHECK_EQ(int(p.plan.tier[0]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(p.plan.tier[31]), int(SS_TIER_RESIDENT));
  CHECK_EQ(int(p.plan.tier[32]), int(SS_TIER_PINNED));
  CHECK_EQ(int(p.plan.tier[48]), int(SS_TIER_DISK));
  CHECK_EQ(p.budget.device_bytes, 90 * GiB);
  CHECK_EQ(p.budget.host_pinned_bytes, 48 * GiB);
  CHECK_EQ(p.budget.workspace_bytes, 4 * GiB + 16 * 60);  // + the device unit tables of a discrete card
  CHECK_EQ(int(p.options.backend), int(SS_BACKEND_CUDA));
  CHECK_EQ(p.options.staging_bytes, 256 * MiB);
  // The plan-v1 in the output is what ss_plan_from_json reads back to the same plan.
  Plan back = plan_from_tiers(d, p.budget, tiers_from_json(d, *p.doc.find("plan")));
  CHECK(back.tier == p.plan.tier);
  CHECK_EQ(back.peak, p.plan.peak);
  CHECK_EQ(p.doc.find("engine")->find("chunk_tokens")->text, std::string("4096"));
  CHECK_EQ(p.doc.find("engine")->find("forward_rows")->text, std::string("4096"));
  CHECK_EQ(p.doc.find("engine")->find("idle_preload_layers")->text, std::string("0"));
  CHECK_EQ(json_u64(*p.doc.find("resources")->find("disk_bytes"), "d"), 12 * 2 * GiB);
  // The exclusive tier mode is part of the schema; this build compiles inclusive tiers only and says so.
  RecipeStack x(kLab, "lab.toml");
  x.set("routes.prefill_long.placement.rtx.tier_mode", "\"exclusive\"", "--set");
  CHECK_REFUSED(recipe_plan(x.effective(), x.effective_sha256(), "prefill_long", "rtx", 0, "GPU-PRO", d, {}, {}),
                "inclusive tiers only");
  const Json* res = p.doc.find("resources");
  CHECK(json_u64(*res->find("host_bytes"), "h") > 16 * 2 * GiB);  // pinned tier + staging + tables + metadata
  CHECK(json_u64(*res->find("host_pinned_bytes"), "h") == p.plan.pinned);
  CHECK_EQ(p.doc.find("plan_id")->text.size(), size_t(64));
  // Too many integer units for the group: refused, never shrunk.
  RecipeStack t(kLab, "lab.toml");
  t.set("routes.prefill_long.placement.rtx.units.layers.pinned", "30", "--set");
  CHECK_REFUSED(recipe_plan(t.effective(), t.effective_sha256(), "prefill_long", "rtx", 0, "GPU-PRO", d, {}, {}),
                "resident + pinned = 62 but the group has 60 units");
  // A count that does not fit the device: the planner's refusal.
  RecipeStack u(kLab, "lab.toml");
  u.set("routes.prefill_long.placement.rtx.units.layers.resident", "44", "--set");
  CHECK_REFUSED(recipe_plan(u.effective(), u.effective_sha256(), "prefill_long", "rtx", 0, "GPU-PRO", d, {}, {}),
                "above the cap");
  // A list rank on a prefix descriptor breaks the prefix: refused.
  RecipeStack v(kLab, "lab.toml");
  // rank and list depend on each other: one layer sets both (each --set alone would be refused).
  CHECK_REFUSED(v.set("routes.prefill_long.placement.rtx.units.layers.rank", "\"list\"", "--set"), "needs \"list\"");
  v.add_overrides(
      "schema = \"strix-stream/recipe-overrides-v1\"\nrecipe = \"mm-lab\"\n[set]\n"
      "\"routes.prefill_long.placement.rtx.units.layers.rank\" = \"list\"\n"
      "\"routes.prefill_long.placement.rtx.units.layers.list\" = [59, 58]\n",
      "o.toml");
  CHECK_REFUSED(recipe_plan(v.effective(), v.effective_sha256(), "prefill_long", "rtx", 0, "GPU-PRO", d, {}, {}),
                "order is prefix");
}

namespace {

// One-route recipe on a cpu set with one unit group, for planner parity.
std::string fill_recipe(const std::string& group, double device_gib, double host_gib, bool backing = false) {
  return R"(schema = "strix-stream/recipe-v1"
name = "fill"
[match]
model = "mm"
[[match.devices]]
id = "c"
kind = "cuda"
memory_min_gib = 200.0
[[routes]]
id = "d"
phase = "decode"
devices = ["c"]
execute = "single"
[routes.placement.c]
device_gib = )" +
         std::to_string(device_gib) + "\nhost_pinned_gib = " + std::to_string(host_gib) +
         "\nworkspace_gib = 1.0\nmax_context_tokens = 100\nmax_batch = 1\nprefetch_layers = 0\npinned_backing = " +
         (backing ? "true" : "false") + "\n[[routes.placement.c.units]]\nid = \"g\"\n" + group;
}

}  // namespace

TEST(plan_fill_matches_ss_plan_make) {
  // Ranked descriptor, measured frequency, resident fill + pinned fill + disk = ss_plan_make with that benefit.
  Desc d = make_desc(desc_text(6, "moe", 16, 256 * MiB, "ranked", "pinned|disk"));
  std::vector<double> benefit;
  for (int64_t g = 0; g < d.total_units; ++g) benefit.push_back(double((g * 7919) % 101));
  for (double dev : {20.0, 30.0, 40.0})
    for (double host : {0.5, 3.0, 40.0})
      for (bool backing : {false, true}) {
        Recipe r = parse_recipe(fill_recipe(
            "class = \"moe\"\nrank = \"frequency\"\nresident = \"fill\"\npinned = \"fill\"\nrest = \"disk\"\n", dev,
            host, backing));
        RecipePlan p = recipe_plan(r, "x", "d", "c", 0, "c0", d, benefit, {});
        Plan ref = plan_make(d, p.budget, benefit);
        if (!(ref.tier == p.plan.tier)) {
          std::printf("  parity: device %.1f host %.1f backing %d differs\n", dev, host, int(backing));
          ++sstest::failures();
        }
      }
  // Prefix descriptor (layer blobs), uniform benefit: the longest prefix with the exact remaining rings.
  Desc b = make_desc(desc_text(20, "blob", 1, 1 * GiB, "prefix", "pinned|disk"));
  for (double dev : {10.0, 14.0, 26.0, 40.0}) {
    Recipe r = parse_recipe(
        fill_recipe("class = \"blob\"\nresident = \"fill\"\npinned = \"fill\"\nrest = \"disk\"\n", dev, 6.0));
    RecipePlan p = recipe_plan(r, "x", "d", "c", 0, "c0", b, {}, {});
    Plan ref = plan_make(b, p.budget, {});
    CHECK(ref.tier == p.plan.tier);
  }
  // Explicit residents kept in a prefix search: 3 integer + fill.
  Recipe r =
      parse_recipe(fill_recipe("class = \"blob\"\nlayers = [[0, 2]]\nresident = 3\nrest = \"disk\"\n"
                               "[[routes.placement.c.units]]\nid = \"h\"\nclass = \"blob\"\nlayers = [[3, 19]]\n"
                               "resident = \"fill\"\nrest = \"disk\"\n",
                               14.0, 0.0));
  RecipePlan p = recipe_plan(r, "x", "d", "c", 0, "c0", b, {}, {});
  CHECK(p.plan.n_resident > 3);
  for (size_t g = 1; g < p.plan.tier.size(); ++g)
    CHECK(!(p.plan.tier[g] == SS_TIER_RESIDENT && p.plan.tier[g - 1] != SS_TIER_RESIDENT));
}

TEST(plan_refusals) {
  Desc d = make_desc(desc_text(4, "moe", 8, 64 * MiB, "ranked", "disk"));
  // Pinned units on a disk-only descriptor.
  Recipe r = parse_recipe(fill_recipe("class = \"moe\"\nresident = 2\npinned = 2\nrest = \"disk\"\n", 20, 4));
  CHECK_REFUSED(recipe_plan(r, "x", "d", "c", 0, "c0", d, {}, {}), "tiers allow pinned");
  // Frequency rank without a measured benefit.
  Recipe f = parse_recipe(fill_recipe("class = \"moe\"\nrank = \"frequency\"\nresident = 2\nrest = \"disk\"\n", 20, 0));
  CHECK_REFUSED(recipe_plan(f, "x", "d", "c", 0, "c0", d, {}, {}), "needs a measured benefit");
  // A group naming a layer the descriptor does not have, or of another kind.
  Recipe l = parse_recipe(fill_recipe("class = \"moe\"\nlayers = [7]\nrest = \"disk\"\n", 20, 0));
  CHECK_REFUSED(recipe_plan(l, "x", "d", "c", 0, "c0", d, {}, {}), "past the descriptor's 4 layers");
  Recipe k = parse_recipe(fill_recipe("class = \"blob\"\nrest = \"disk\"\n", 20, 0));
  CHECK_REFUSED(recipe_plan(k, "x", "d", "c", 0, "c0", d, {}, {}), "is in no unit group");
  Recipe e = parse_recipe(fill_recipe("class = \"moe\"\nrank = \"list\"\nlist = [9]\nrest = \"disk\"\n", 20, 0));
  CHECK_REFUSED(recipe_plan(e, "x", "d", "c", 0, "c0", d, {}, {}), "expert 9 is past layer 0's 8 units");
  // Ring too small for the prefetch depth.
  RecipeStack gs(fill_recipe("class = \"moe\"\nresident = 2\nrest = \"disk\"\n", 20, 0), "f");
  gs.set("routes.d.placement.c.ring_slots", "6", "--set");
  gs.set("routes.d.placement.c.prefetch_layers", "1", "--set");
  CHECK_REFUSED(recipe_plan(gs.effective(), "x", "d", "c", 0, "c0", d, {}, {}), "below (prefetch_layers + 1) x 6");
  // Derived workspace needs the descriptor's sizes.
  std::string text = fill_recipe("class = \"moe\"\nresident = 2\nrest = \"disk\"\n", 20, 0);
  text.replace(text.find("workspace_gib = 1.0\n"), 20, "");
  Recipe nw = parse_recipe(text);
  CHECK_REFUSED(recipe_plan(nw, "x", "d", "c", 0, "c0", d, {}, {}), "set workspace_gib");
  Desc dw = make_desc(desc_text(4, "moe", 8, 64 * MiB, "ranked", "disk",
                                R"("workspace_bytes_per_row":1000,"state_bytes_per_position":100,)"));
  RecipePlan pw = recipe_plan(nw, "x", "d", "c", 0, "c0", dw, {}, {});
  // decode: rows = max_batch; + the device unit tables (2 x 8 bytes x 32 units) of a discrete card
  CHECK_EQ(pw.budget.workspace_bytes, uint64_t(1000 * 1 + 100 * 100 * 1 + 16 * 32));
  // The whole device requirement stays within the cap.
  CHECK(json_u64(*pw.doc.find("resources")->find("device_bytes"), "d") <= pw.budget.device_bytes);
  // Staging must hold 4 x the source alignment.
  RecipeStack sa(fill_recipe("class = \"moe\"\nresident = 2\nrest = \"disk\"\n", 20, 0), "f");
  Desc big_align = make_desc(R"({"schema":"strix-stream/desc-v1","model":"mm","engine":"t","tiers":"disk",)"
                             R"("sources":[{"path":"x.bin","align":1048576}],"layers":[)" +
                             layers_json(2, "moe", 8, 64 * MiB, "") + "]}");
  sa.set("routes.d.placement.c.staging_mib", "2", "--set");
  CHECK_REFUSED(recipe_plan(sa.effective(), "x", "d", "c", 0, "c0", big_align, {}, {}),
                "4 x the largest source alignment");
}

namespace {

const char* kHetero = R"(schema = "strix-stream/recipe-v1"
name = "het"
[match]
model = "mm"
[[match.devices]]
id = "rtx"
kind = "cuda"
memory_min_gib = 90.0
[[match.devices]]
id = "pair"
kind = "hip"
count = 2
memory_min_gib = 100.0
internal_link = { kind = "usb4" }
[[match.links]]
id = "pc-nodes"
between = ["rtx", "pair"]
[[routes]]
id = "dec"
phase = "decode"
devices = ["rtx", "pair"]
execute = "hetero"
links = ["pc-nodes"]
[routes.placement.rtx]
device_gib = 40.0
workspace_gib = 1.0
max_context_tokens = 1000
max_batch = 1
[[routes.placement.rtx.units]]
id = "hot"
class = "moe"
rank = "frequency"
resident = 3
rest = "peer"
peer = "pair"
[routes.placement.pair]
device_gib = 100.0
workspace_gib = 1.0
max_context_tokens = 1000
max_batch = 1
[[routes.placement.pair.units]]
id = "cold"
class = "moe"
rest = "disk"
)";

}  // namespace

TEST(plan_hetero_ownership) {
  Recipe r = parse_recipe(kHetero);
  Desc dr = make_desc(desc_text(4, "moe", 8, 64 * MiB, "ranked", "pinned|disk"));
  Desc dp = make_desc(desc_text(4, "moe", 8, 32 * MiB, "ranked", "pinned|disk"));
  std::vector<double> ben;
  for (int64_t g = 0; g < dr.total_units; ++g) ben.push_back(double(g % 8));  // experts 7, 6, 5 are hot
  std::vector<RefInput> refs = {{&dr, ben}, {&dp, {}}};
  RecipePlan a = recipe_plan(r, "x", "dec", "rtx", 0, "GPU-PRO", dr, ben, refs);
  RecipePlan b0 = recipe_plan(r, "x", "dec", "pair", 0, "node1", dp, {}, refs);
  RecipePlan b1 = recipe_plan(r, "x", "dec", "pair", 1, "node2", dp, {}, refs);
  CHECK_EQ(a.doc.find("ownership_sha256")->text, b0.doc.find("ownership_sha256")->text);
  CHECK_EQ(b0.doc.find("ownership_sha256")->text, b1.doc.find("ownership_sha256")->text);
  for (int64_t g = 0; g < dr.total_units; ++g) {
    const bool hot = g % 8 >= 5;
    const bool rtx_owns = a.plan.tier[size_t(g)] != SS_TIER_PAIR;
    const bool pair_owns = b0.plan.tier[size_t(g)] != SS_TIER_PAIR;
    CHECK(rtx_owns != pair_owns);  // exactly one owner
    CHECK_EQ(rtx_owns, hot);
  }
  CHECK_EQ(a.plan.n_pair, uint64_t(4 * 5));
  CHECK_EQ(a.plan.ring, uint64_t(0));  // pair units take no ring
  // A resident fill next to peer units reserves no ring for them: 6 of 8 units fit ~1.6 GiB of room.
  std::string fill = kHetero;
  const std::string hot = "rank = \"frequency\"\nresident = 3", cap = "device_gib = 40.0";
  fill.replace(fill.find(hot), hot.size(), "resident = \"fill\"");
  fill.replace(fill.find(cap), cap.size(), "device_gib = 8.6");
  Recipe rf = parse_recipe(fill);
  Desc big = make_desc(desc_text(1, "moe", 8, 256 * MiB, "ranked", "pinned|disk"));
  Desc bigp = make_desc(desc_text(1, "moe", 8, 256 * MiB, "ranked", "pinned|disk"));
  std::vector<RefInput> rr = {{&big, {}}, {&bigp, {}}};
  RecipePlan af = recipe_plan(rf, "x", "dec", "rtx", 0, "GPU-PRO", big, {}, rr);
  CHECK_EQ(af.plan.n_resident, uint64_t(6));
  CHECK_EQ(af.plan.n_pair, uint64_t(2));
  // Frequency ranking needs a measurement for every ranked unit.
  Desc part2 = make_desc(
      R"({"schema":"strix-stream/desc-v1","model":"mm","engine":"t","sources":[{"path":"x.bin"}],"layers":[)" +
      layers_json(1, "moe", 8, 64 * MiB, R"(,"benefit":[1,2,3,4,5,6,7,8])") + "," +
      R"({"index":1,"kind":"moe","n_units":8,"top_k":2,"parts":[{"format":"Q2_K","offset":0,"bytes":67108864,"stride":67108864}]}]})");
  std::vector<RefInput> pr = {{&part2, {}}, {&part2, {}}};
  CHECK_REFUSED(recipe_plan(r, "x", "dec", "rtx", 0, "GPU-PRO", part2, {}, pr), "needs a measured benefit");

  // Without the reference inputs a hetero route cannot be compiled.
  CHECK_REFUSED(recipe_plan(r, "x", "dec", "rtx", 0, "GPU-PRO", dr, ben, {}), "reference descriptor of every set");
  // Different layer structures cannot share an ownership map.
  Desc odd = make_desc(desc_text(4, "moe", 6, 32 * MiB, "ranked", "pinned|disk"));
  std::vector<RefInput> bad = {{&dr, ben}, {&odd, {}}};
  CHECK_REFUSED(recipe_plan(r, "x", "dec", "rtx", 0, "GPU-PRO", dr, ben, bad), "same layers, kinds and units");
}

// ---- admission ----------------------------------------------------------------------------------------------------
namespace {

struct Lab {
  RecipeStack s{kLab, "lab.toml"};
  Desc rtx = make_desc(blob_desc());
  Desc pair = make_desc(pair_desc());
  Json state;

  Lab() {
    state = json_parse(R"({"phase":"prefill","prompt_tokens":8192,"context_tokens":12288,"batch":1,
      "binding":{"rtx":[{"id":"GPU-PRO","host":"pc"}],"pair":[{"id":"node1","host":"node1"},{"id":"node2","host":"node2"}]},
      "devices":{"GPU-PRO":{"available":true,"disk_available":true},"node1":{"available":true},"node2":{"available":true}},
      "links":{"pc-nodes":{"up":true},"pair.internal":{"up":true}},
      "hosts":{"pc":{"free_ram_gib":100.0},"node1":{"free_ram_gib":120.0},"node2":{"free_ram_gib":120.0}},
      "running":[],"requirements":[]})");
    Json descs = Json::object();
    Json a = Json::array();
    a.push(Json::string(desc_sha(rtx)));
    descs["rtx"] = a;
    Json b = Json::array();
    b.push(Json::string(desc_sha(pair)));
    b.push(Json::string(desc_sha(pair)));
    descs["pair"] = b;
    state["descriptors"] = descs;
    compile_all();
  }
  void require_plan(const std::string& route, const std::string& set, int member, const std::string& id,
                    const Desc& d) {
    RecipePlan p = recipe_plan(s.effective(), s.effective_sha256(), route, set, member, id, d, {}, {});
    Json q = Json::object();
    for (const char* k : {"route", "set", "member", "id", "effective_sha256", "descriptor_sha256", "plan_id"})
      q[k] = *p.doc.find(k);
    for (const char* k : {"host_bytes", "commit_bytes", "disk_bytes"}) q[k] = *p.doc.find("resources")->find(k);
    state["requirements"].push(q);
  }
  void compile_all() {
    state["requirements"] = Json::array();
    require_plan("prefill_long", "rtx", 0, "GPU-PRO", rtx);
    for (const char* r : {"prefill_nodes", "decode_pair"}) {
      require_plan(r, "pair", 0, "node1", pair);
      require_plan(r, "pair", 1, "node2", pair);
    }
  }
  std::string route() { return recipe_select(s.effective(), s.effective_sha256(), state).find("route")->text; }
};

}  // namespace

TEST(select_routes_and_fallbacks) {
  Lab lab;
  CHECK_EQ(lab.route(), std::string("prefill_long"));
  Json sel = recipe_select(lab.s.effective(), lab.s.effective_sha256(), lab.state);
  CHECK_EQ(sel.find("handoff_route")->text, std::string("decode_pair"));
  CHECK(json_u64(*sel.find("charge")->find("pc"), "c") > 32 * GiB);
  // Short prompt: the RTX route does not match; the nodes take it.
  lab.state["prompt_tokens"] = Json::number(uint64_t(1024));
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["prompt_tokens"] = Json::number(uint64_t(8192));
  // PC-BIG busy -> fallback to the nodes.
  lab.state["devices"]["GPU-PRO"]["available"] = Json::boolean(false);
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  Json t = recipe_select(lab.s.effective(), lab.s.effective_sha256(), lab.state);
  CHECK(has(json_write(*t.find("trace")), "GPU-PRO of \\\"rtx\\\" is not available"));
  lab.state["devices"]["GPU-PRO"]["available"] = Json::boolean(true);
  // Link down -> nodes only.
  lab.state["links"]["pc-nodes"]["up"] = Json::boolean(false);
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["links"]["pc-nodes"]["up"] = Json::boolean(true);
  // Low RAM on the PC (the pinned tier would cross the floor) -> nodes.
  lab.state["hosts"]["pc"]["free_ram_gib"] = Json::number(30.0);
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  // ...unless the RTX service is already running with these exact plans (its memory is counted as used).
  Json run = Json::object();
  run["route"] = Json::string("prefill_long");
  run["effective_sha256"] = Json::string(lab.s.effective_sha256());
  Json ids = Json::array();
  ids.push(*lab.state["requirements"].arr[0].find("plan_id"));
  run["plan_ids"] = ids;
  lab.state["running"].push(run);
  CHECK_EQ(lab.route(), std::string("prefill_long"));
  // A running entry for other plans gets no credit.
  // A credited route still needs its hosts' reports and floors.
  lab.state["hosts"]["pc"]["free_ram_gib"] = Json::number(7.0);
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  Json saved_pc = lab.state["hosts"]["pc"];
  lab.state["hosts"].obj.erase("pc");
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["hosts"]["pc"] = saved_pc;
  lab.state["hosts"]["pc"]["free_ram_gib"] = Json::number(30.0);
  lab.state["running"].arr[0]["plan_ids"].arr[0] = Json::string("stale");
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["running"] = Json::array();
  lab.state["hosts"]["pc"]["free_ram_gib"] = Json::number(100.0);
  // Requirements compiled for another descriptor do not count: the route is not admitted.
  lab.state["descriptors"]["rtx"].arr[0] = Json::string("other");
  t = recipe_select(lab.s.effective(), lab.s.effective_sha256(), lab.state);
  CHECK_EQ(t.find("route")->text, std::string("prefill_nodes"));
  CHECK(has(json_write(*t.find("trace")), "not compiled for this recipe"));
  lab.state["descriptors"]["rtx"].arr[0] = Json::string(desc_sha(lab.rtx));
  // Context above the RTX placement's bound: the RTX route is not admitted; the nodes' bound is the same, so none is.
  lab.state["context_tokens"] = Json::number(uint64_t(40000));
  std::string e = error_of([&] { lab.route(); });
  CHECK(has(e, "no route of \"mm-lab\" can admit this prefill request"));
  CHECK(has(e, "above max_context_tokens 32768"));
  lab.state["context_tokens"] = Json::number(uint64_t(12288));
  // Batch above the RTX max_batch 1: the nodes (max_batch 4) take it.
  lab.state["batch"] = Json::number(uint64_t(3));
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["batch"] = Json::number(uint64_t(1));
  // Internal link of the pair down: neither prefill route can hand off to the pair decode.
  lab.state["links"]["pair.internal"]["up"] = Json::boolean(false);
  CHECK_REFUSED(lab.route(), "link pair.internal is down");
  lab.state["links"]["pair.internal"]["up"] = Json::boolean(true);
  // The RTX plan streams 12 layers from its disk: a busy disk (or no report) makes it inadmissible.
  lab.state["devices"]["GPU-PRO"]["disk_available"] = Json::boolean(false);
  t = recipe_select(lab.s.effective(), lab.s.effective_sha256(), lab.state);
  CHECK_EQ(t.find("route")->text, std::string("prefill_nodes"));
  CHECK(has(json_write(*t.find("trace")), "is not available and the plan streams from it"));
  lab.state["devices"]["GPU-PRO"]["disk_available"] = Json::boolean(true);
  // A missing host RAM report makes a route inadmissible.
  Json pc = lab.state["hosts"]["pc"];
  lab.state["hosts"].obj.erase("pc");
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
  lab.state["hosts"]["pc"] = pc;
  CHECK_EQ(lab.route(), std::string("prefill_long"));
}

TEST(select_disabled_routes_and_commit_floor) {
  Lab lab;
  // A disabled route admits nothing; the walk goes on (and a disabled fallback is skipped).
  lab.s.set("routes.prefill_long.enabled", "false", "--set");
  lab.compile_all();
  Json t = recipe_select(lab.s.effective(), lab.s.effective_sha256(), lab.state);
  CHECK_EQ(t.find("route")->text, std::string("prefill_nodes"));
  CHECK(has(json_write(*t.find("trace")), "\"verdict\":\"disabled\""));
  lab.s.set("routes.prefill_long.enabled", "true", "--set");
  lab.s.set("routes.prefill_nodes.enabled", "false", "--set");
  lab.compile_all();
  lab.state["devices"]["GPU-PRO"]["available"] = Json::boolean(false);
  CHECK_REFUSED(lab.route(), "prefill_nodes: disabled");
  lab.state["devices"]["GPU-PRO"]["available"] = Json::boolean(true);
  lab.s.set("routes.prefill_nodes.enabled", "true", "--set");
  // Commit floor: checked only when set; then every charged host must report its free commit.
  lab.s.set("limits.host_free_commit_floor_gib", "14.0", "--set");
  lab.compile_all();
  std::string e = error_of([&] { lab.route(); });
  CHECK(has(e, "no free commit reported for host node1"));
  for (const char* h : {"pc", "node1", "node2"}) lab.state["hosts"][h]["free_commit_gib"] = Json::number(200.0);
  CHECK_EQ(lab.route(), std::string("prefill_long"));
  // The RTX's device bytes count toward the PC's commit (WDDM): ~33 GiB host + ~81 GiB device + the 14 GiB floor
  // do not fit in 120 GiB.
  lab.state["hosts"]["pc"]["free_commit_gib"] = Json::number(120.0);
  CHECK_EQ(lab.route(), std::string("prefill_nodes"));
}

TEST(select_session_and_stale_recipe) {
  Lab lab;
  // Decode of a running session stays on its route; when it cannot continue it is refused, not moved.
  lab.state["phase"] = Json::string("decode");
  lab.state["session_route"] = Json::string("decode_pair");
  CHECK_EQ(lab.route(), std::string("decode_pair"));
  lab.state["devices"]["node2"]["available"] = Json::boolean(false);
  CHECK_REFUSED(lab.route(), "a running request is never moved");
  lab.state["devices"]["node2"]["available"] = Json::boolean(true);
  // Requirements compiled for an earlier effective recipe stop counting after an override.
  lab.s.set("routes.decode_pair.placement.pair.lookahead_units", "3", "--set");
  CHECK_REFUSED(lab.route(), "not compiled for this recipe");
  lab.compile_all();
  CHECK_EQ(lab.route(), std::string("decode_pair"));
}

// ---- C ABI --------------------------------------------------------------------------------------------------------
TEST(capi_recipe_roundtrip) {
  ss_recipe* r = nullptr;
  CHECK_EQ(int(ss_recipe_from_text("schema = 1", 10, "bad.toml", &r)), int(SS_E_RECIPE));
  CHECK(has(ss_last_error(), "bad.toml: schema: expected a string"));
  CHECK(r == nullptr);
  CHECK_OK(ss_recipe_from_text(kLab, std::strlen(kLab), "lab.toml", &r));
  CHECK_OK(ss_recipe_set(r, "routes.prefill_long.placement.rtx.chunk_tokens", "8192", nullptr));
  char* out = nullptr;
  size_t len = 0;
  CHECK_OK(ss_recipe_to_json(r, &out, &len));
  CHECK(out && has(out, "\"chunk_tokens\":8192") && std::strlen(out) == len);
  ss_free(out);
  out = nullptr;
  const char* no_devices = "{\"model\":\"x\",\"devices\":[]}";
  CHECK_EQ(int(ss_recipe_match(r, no_devices, std::strlen(no_devices), &out, nullptr)), int(SS_E_RECIPE));
  CHECK(out == nullptr);
  CHECK_OK(ss_recipe_match(r, kProbe, std::strlen(kProbe), &out, nullptr));
  CHECK(out && has(out, "\"GPU-PRO\""));
  ss_free(out);
  // Compile through the ABI and feed the plan-v1 to ss_plan_from_json: same tiers.
  const std::string dt = blob_desc();
  ss_desc* d = nullptr;
  CHECK_OK(ss_desc_from_json(dt.data(), dt.size(), &d));
  ss_budget b;
  ss_budget_defaults(&b);
  ss_options o;
  ss_options_defaults(&o);
  o.device = 3;
  ss_plan* plan = nullptr;
  out = nullptr;
  CHECK_OK(ss_recipe_plan(r, "prefill_long", "rtx", 0, "GPU-PRO", d, nullptr, 0, nullptr, nullptr, nullptr, 0, &b, &o,
                          &plan, &out, nullptr));
  CHECK_EQ(o.device, 3);
  CHECK_EQ(int(o.backend), int(SS_BACKEND_CUDA));
  CHECK_EQ(b.device_bytes, 90 * GiB);
  Json doc = json_parse(out);
  const std::string pj = json_write(*doc.find("plan"));
  ss_plan* back = nullptr;
  CHECK_OK(ss_plan_from_json(d, &b, pj.data(), pj.size(), &back));
  for (int L = 0; L < 60; ++L) CHECK_EQ(int(ss_plan_unit_tier(back, L, 0)), int(ss_plan_unit_tier(plan, L, 0)));
  ss_plan_totals t1, t2;
  ss_plan_get_totals(plan, &t1);
  ss_plan_get_totals(back, &t2);
  CHECK_EQ(t1.peak_device_bytes, t2.peak_device_bytes);
  CHECK_EQ(t1.n_resident, uint64_t(32));
  ss_plan_free(back);
  ss_plan_free(plan);
  ss_free(out);
  // An old caller's smaller ss_budget gets only its bytes written.
  struct {
    ss_budget b;
    char guard[8];
  } small;
  std::memset(&small, 0x5a, sizeof small);
  ss_budget_defaults_sized(&small.b, uint32_t(offsetof(ss_budget, dynamic_slots_per_layer)));
  ss_options o2;
  ss_options_defaults(&o2);
  CHECK_OK(ss_recipe_plan(r, "prefill_long", "rtx", 0, "GPU-PRO", d, nullptr, 0, nullptr, nullptr, nullptr, 0, &small.b,
                          &o2, nullptr, nullptr, nullptr));
  CHECK_EQ(small.b.device_bytes, 90 * GiB);
  CHECK_EQ(int(small.guard[0]), 0x5a);
  CHECK_EQ(small.b.struct_size, uint32_t(offsetof(ss_budget, dynamic_slots_per_layer)));  // size kept
  CHECK_EQ(int(reinterpret_cast<unsigned char*>(&small.b.dynamic_slots_per_layer)[0]), 0x5a);
  // Explain through the ABI.
  CHECK_OK(ss_recipe_explain(r, &out, nullptr));
  CHECK(has(out, "\"origin\":\"--set\""));
  ss_free(out);
  ss_desc_free(d);
  ss_recipe_free(r);
}

int main() { return sstest::run_all(); }
