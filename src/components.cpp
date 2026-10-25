#include "components.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>

#include "errors.hpp"
#include "recipe.hpp"
#include "schema_id.hpp"
#include "sha256.hpp"

namespace ts {
namespace {

constexpr const char* kSchema = "tierstream/components-v1";
constexpr uint64_t kMaxBytes = uint64_t(INT64_MAX);  // every byte count and sum stays a signed 64-bit integer

const char* kPhaseWords[kPlacePhases] = {"load", "prefill", "decode", "spec"};

[[noreturn]] void bad(const std::string& path, const std::string& msg) {
  fail(SS_E_RECIPE, "components: PL_E_SCHEMA " + path + ": " + msg);
}

void keys(const Json& o, const std::string& path, std::initializer_list<const char*> allowed) {
  if (!o.is_object()) bad(path, "expected an object");
  for (const auto& kv : o.obj) {
    bool ok = false;
    for (const char* k : allowed) ok = ok || kv.first == k;
    if (!ok) bad(path, "unknown key \"" + kv.first + "\"");
  }
}

const Json& need(const Json& o, const char* key, const std::string& path) {
  const Json* v = o.find(key);
  if (!v) bad(path, std::string("missing \"") + key + "\"");
  return *v;
}

std::string str(const Json& v, const std::string& path) {
  if (!v.is_string() || v.text.empty()) bad(path, "expected a non-empty string");
  return v.text;
}

std::string hex64(const Json& v, const std::string& path) {
  std::string s = str(v, path);
  bool ok = s.size() == 64;
  for (char c : s) ok = ok && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
  if (!ok) bad(path, "expected 64 lowercase hex digits");
  return s;
}

// Non-negative integer literal <= INT64_MAX (negative, fractional, exponent or overflowing text is refused).
uint64_t u64(const Json& v, const std::string& path) {
  if (!v.is_number()) bad(path, "expected an integer");
  const std::string& t = v.text;
  if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos)
    bad(path, "expected an integer >= 0, got " + t);
  if (t.size() > 19) bad(path, t + " overflows 64 bits");
  const unsigned long long x = std::strtoull(t.c_str(), nullptr, 10);
  if (x > kMaxBytes) bad(path, t + " overflows 64 bits");
  return x;
}

bool boolean(const Json& v, const std::string& path) {
  if (v.kind != Json::Kind::Bool) bad(path, "expected true or false");
  return v.b;
}

// sm_XY[Z][a] (CUDA) or gfxNNN[N] (HIP).
bool arch_ok(const std::string& a) {
  auto digits = [&](size_t from, size_t lo, size_t hi) {
    size_t n = 0;
    while (from + n < a.size() && a[from + n] >= '0' && a[from + n] <= '9') ++n;
    return n >= lo && n <= hi ? from + n : std::string::npos;
  };
  if (a.compare(0, 3, "sm_") == 0) {
    size_t e = digits(3, 2, 3);
    return e != std::string::npos && (e == a.size() || (e + 1 == a.size() && a[e] >= 'a' && a[e] <= 'z'));
  }
  if (a.compare(0, 3, "gfx") == 0 && a.size() >= 6 && a.size() <= 8) {
    for (size_t i = 3; i < a.size(); ++i)
      if (!((a[i] >= '0' && a[i] <= '9') || (a[i] >= 'a' && a[i] <= 'f'))) return false;
    return true;
  }
  return false;
}

uint64_t add_ck(uint64_t a, uint64_t b, const std::string& path) {
  if (a > kMaxBytes - b) bad(path, "byte sum overflows 64 bits");
  return a + b;
}

std::vector<Charge> charges_of(const Json& v, const std::string& path, uint64_t& total) {
  if (!v.is_array()) bad(path, "expected an array of charges");
  std::vector<Charge> out;
  for (size_t i = 0; i < v.arr.size(); ++i) {
    const std::string w = path + "[" + std::to_string(i) + "]";
    const Json& e = v.arr[i];
    keys(e, w, {"at", "bytes", "phases", "peak_extra", "name", "allocs", "kind"});
    Charge c;
    if (const Json* x = e.find("name")) c.name = str(*x, w + ".name");
    const std::string at = str(need(e, "at", w), w + ".at");
    if (at == "device")
      c.at = ChargeAt::Device;
    else if (at == "host_pinned")
      c.at = ChargeAt::HostPinned;
    else if (at == "host_pageable")
      c.at = ChargeAt::HostPageable;
    else if (at == "file")
      c.at = ChargeAt::File;
    else
      bad(w + ".at", "\"" + at + "\" is not device | host_pinned | host_pageable | file");
    c.bytes = u64(need(e, "bytes", w), w + ".bytes");
    if (const Json* x = e.find("peak_extra")) c.peak_extra = u64(*x, w + ".peak_extra");
    if (const Json* x = e.find("kind")) {
      const std::string k = str(*x, w + ".kind");
      if (k != "payload" && k != "driver") bad(w + ".kind", "\"" + k + "\" is not payload | driver");
      c.driver = k == "driver";
    }
    c.allocs = c.driver ? 0 : 1;  // a driver charge is no allocation of the engine's (rev 18)
    if (const Json* x = e.find("allocs")) c.allocs = u64(*x, w + ".allocs");
    if (c.allocs > 1000000) bad(w + ".allocs", "more than 10^6 allocations");
    const Json& ph = need(e, "phases", w);
    if (!ph.is_array() || ph.arr.empty()) bad(w + ".phases", "expected a non-empty array of phases");
    for (const Json& p : ph.arr) {
      const std::string s = str(p, w + ".phases");
      int k = -1;
      for (int q = 0; q < kPlacePhases; ++q)
        if (s == kPhaseWords[q]) k = q;
      if (k < 0) bad(w + ".phases", "\"" + s + "\" is not load | prefill | decode | spec");
      if (c.phases & (1u << k)) bad(w + ".phases", "\"" + s + "\" is listed twice");
      c.phases |= 1u << k;
    }
    total = add_ck(total, add_ck(c.bytes, c.peak_extra, w), w);
    out.push_back(c);
  }
  return out;
}

// Canonical charge order (§2.4): (at word, phases in lifetime order, bytes, peak_extra).
std::vector<int> phase_list(unsigned m) {
  std::vector<int> v;
  for (int p = 0; p < kPlacePhases; ++p)
    if (m & (1u << p)) v.push_back(p);
  return v;
}

bool charge_less(const Charge& a, const Charge& b) {
  const std::string x = charge_at_word(a.at), y = charge_at_word(b.at);
  if (x != y) return x < y;
  const std::vector<int> pa = phase_list(a.phases), pb = phase_list(b.phases);
  if (pa != pb) return pa < pb;
  if (a.bytes != b.bytes) return a.bytes < b.bytes;
  if (a.peak_extra != b.peak_extra) return a.peak_extra < b.peak_extra;
  if (a.name != b.name) return a.name < b.name;  // an absent name ("") sorts first
  if (a.allocs != b.allocs) return a.allocs < b.allocs;
  return a.driver < b.driver;
}

// The device-gated adapter's reserved charge names (contract §2.4): component and location of each.
bool dg_name_ok(const std::string& name, const std::string& component, ChargeAt at) {
  if (name == "dg.slot_table" || name == "dg.controls") return component == "window" && at == ChargeAt::Device;
  if (name == "dg.mapped" || name == "dg.bounce" || name == "dg.staging" || name == "dg.table_pin")
    return component == "staging" && at == ChargeAt::HostPinned;
  return false;
}

// Canonical order; two charges of one component equal in all five fields, or with one name, are refused (the order
// would not be total / a name would not identify its line); "dg." names are reserved.
void sort_distinct(std::vector<Charge>& cs, const std::string& path, const std::string& component, bool gated) {
  // V1 (rev 17): the dg. prefix outside the device-gated adapter, in document order (the reference's message)
  if (!gated)
    for (const Charge& c : cs)
      if (c.name.compare(0, 3, "dg.") == 0) bad(component, "reserved name " + c.name + " outside the device-gated adapter");
  std::sort(cs.begin(), cs.end(), charge_less);
  std::set<std::string> names;
  for (size_t i = 0; i < cs.size(); ++i) {
    if (i && !charge_less(cs[i - 1], cs[i])) bad(path, "two identical charges (merge them into one)");
    const std::string& n = cs[i].name;
    if (n.empty()) continue;
    if (!names.insert(n).second) bad(path, "two charges named \"" + n + "\"");
    if (n.compare(0, 3, "dg.") == 0 && !dg_name_ok(n, component, cs[i].at))
      bad(path, "\"" + n + "\" is reserved: dg.slot_table / dg.controls are device charges of window, dg.mapped / "
                "dg.bounce / dg.staging / dg.table_pin host_pinned charges of staging");
  }
}

Json charges_json(const std::vector<Charge>& cs) {
  Json a = Json::array();
  for (const Charge& c : cs) {
    Json e = Json::object();
    if (!c.name.empty()) e["name"] = Json::string(c.name);
    e["at"] = Json::string(charge_at_word(c.at));
    e["bytes"] = Json::number(c.bytes);
    e["peak_extra"] = Json::number(c.peak_extra);
    if (c.allocs != (c.driver ? 0u : 1u)) e["allocs"] = Json::number(c.allocs);  // when not the kind's default
    if (c.driver) e["kind"] = Json::string("driver");          // "payload" is the default, omitted
    Json ph = Json::array();
    for (int p : phase_list(c.phases)) ph.push(Json::string(kPhaseWords[p]));
    e["phases"] = ph;
    a.push(e);
  }
  return a;
}

// Scalars of the free-form option tables (config extras, arithmetic options): kept as written (canonical JSON).
void scalars_only(const Json& o, const std::string& path) {
  if (!o.is_object()) bad(path, "expected an object");
  for (const auto& kv : o.obj)
    if (kv.second.is_object() || kv.second.is_array() || kv.second.is_null())
      bad(path + "." + kv.first, "expected a number, string or boolean");
}

Json content_of(const Json& v, const std::string& w) {
  keys(v, w, {"draft_sha256", "overlays", "vocabulary_sha256"});
  Json out = Json::object();
  if (const Json* x = v.find("draft_sha256")) out["draft_sha256"] = Json::string(hex64(*x, w + ".draft_sha256"));
  if (const Json* x = v.find("vocabulary_sha256"))
    out["vocabulary_sha256"] = Json::string(hex64(*x, w + ".vocabulary_sha256"));
  if (const Json* x = v.find("overlays")) {
    if (!x->is_array()) bad(w + ".overlays", "expected an array");
    std::vector<std::pair<std::string, std::string>> ov;
    for (size_t i = 0; i < x->arr.size(); ++i) {
      const std::string ow = w + ".overlays[" + std::to_string(i) + "]";
      keys(x->arr[i], ow, {"path", "sha256"});
      ov.push_back({str(need(x->arr[i], "path", ow), ow + ".path"), hex64(need(x->arr[i], "sha256", ow), ow + ".sha256")});
    }
    std::sort(ov.begin(), ov.end());
    for (size_t i = 1; i < ov.size(); ++i)
      if (ov[i].first == ov[i - 1].first) bad(w + ".overlays", "\"" + ov[i].first + "\" is listed twice");
    Json a = Json::array();
    for (const auto& o : ov) {
      Json e = Json::object();
      e["path"] = Json::string(o.first);
      e["sha256"] = Json::string(o.second);
      a.push(e);
    }
    out["overlays"] = a;
  }
  return out;
}

Json arithmetic_of(const Json& v, const std::string& w) {
  keys(v, w, {"options", "dispatch"});
  Json out = Json::object();
  if (const Json* x = v.find("options")) {
    scalars_only(*x, w + ".options");
    out["options"] = *x;
  }
  if (const Json* x = v.find("dispatch")) {
    if (!x->is_object()) bad(w + ".dispatch", "expected an object keyed by arch");
    Json d = Json::object();
    for (const auto& kv : x->obj) {
      const std::string dw = w + ".dispatch." + kv.first;
      if (!arch_ok(kv.first)) bad(dw, "not an arch (sm_XY | gfxNNNN)");
      if (!kv.second.is_array()) bad(dw, "expected an array of variant names");
      std::vector<std::string> names;
      for (const Json& n : kv.second.arr) names.push_back(str(n, dw));
      std::sort(names.begin(), names.end());  // canonical: lexicographic
      Json a = Json::array();
      for (const auto& n : names) a.push(Json::string(n));
      d[kv.first] = a;
    }
    out["dispatch"] = d;
  }
  return out;
}

}  // namespace

const char* place_phase_word(int p) { return p >= 0 && p < kPlacePhases ? kPhaseWords[p] : "?"; }

const char* charge_at_word(ChargeAt a) {
  switch (a) {
    case ChargeAt::Device:
      return "device";
    case ChargeAt::HostPinned:
      return "host_pinned";
    case ChargeAt::HostPageable:
      return "host_pageable";
    case ChargeAt::File:
      return "file";
  }
  return "?";
}

const std::vector<Charge>* ComponentDecl::for_role(bool assist_role) const {
  if (!has_variants) return &charges;
  if (assist_role) return has_assist ? &assist : nullptr;
  return has_anchor ? &anchor : nullptr;
}

const ComponentDecl* Components::find(const std::string& cid) const {
  for (const auto& c : components)
    if (c.id == cid) return &c;
  return nullptr;
}

bool Components::has(const std::string& capability) const {
  return std::binary_search(capabilities.begin(), capabilities.end(), capability);
}

Components components_from_json(const Json& doc) {
  keys(doc, "components-v1",
       {"schema", "engine", "build_sha256", "model_sha256", "desc_sha256", "content", "arithmetic", "elastic", "config",
        "capabilities", "context", "context_growth", "context_receipts", "receipts_sha256", "components",
        "envelope", "stack"});
  const std::string schema = str(need(doc, "schema", "components-v1"), "schema");
  if (schema != kSchema && schema != "strix-stream/components-v1")
    bad("schema", std::string("expected \"") + kSchema + "\"");
  Components c;
  c.engine = str(need(doc, "engine", "components-v1"), "engine");
  c.build_sha256 = hex64(need(doc, "build_sha256", "components-v1"), "build_sha256");
  c.model_sha256 = hex64(need(doc, "model_sha256", "components-v1"), "model_sha256");
  c.desc_sha256 = hex64(need(doc, "desc_sha256", "components-v1"), "desc_sha256");
  if (const Json* v = doc.find("receipts_sha256")) c.receipts_sha256 = hex64(*v, "receipts_sha256");
  if (const Json* v = doc.find("content")) c.content = content_of(*v, "content");
  if (const Json* v = doc.find("arithmetic")) c.arithmetic = arithmetic_of(*v, "arithmetic");

  if (const Json* e = doc.find("elastic")) {
    keys(*e, "elastic",
         {"unit_bytes", "units_total", "per_layer_units", "per_layer_max_active", "spares", "host_spares", "mode",
          "spare_billing", "slot_allocs", "host_chunk_bytes"});
    if (const Json* x = e->find("host_chunk_bytes")) {
      c.host_chunk_bytes = u64(*x, "elastic.host_chunk_bytes");
      if (!c.host_chunk_bytes) bad("elastic.host_chunk_bytes", "must be > 0");
    }
    if (const Json* x = e->find("slot_allocs")) c.slot_allocs = u64(*x, "elastic.slot_allocs");
    c.has_elastic = true;
    c.elastic_doc = *e;
    c.unit_bytes = u64(need(*e, "unit_bytes", "elastic"), "elastic.unit_bytes");
    c.units_total = u64(need(*e, "units_total", "elastic"), "elastic.units_total");
    if (const Json* x = e->find("per_layer_units")) c.per_layer_units = u64(*x, "elastic.per_layer_units");
    if (const Json* x = e->find("spares")) c.spares = u64(*x, "elastic.spares");
    if (const Json* x = e->find("host_spares")) c.host_spares = u64(*x, "elastic.host_spares");
    if (const Json* x = e->find("per_layer_max_active")) {
      c.per_layer_max_active = u64(*x, "elastic.per_layer_max_active");
      if (!c.per_layer_units) bad("elastic.per_layer_max_active", "needs per_layer_units");
    }
    if (const Json* x = e->find("spare_billing")) {  // required for the TierSite adapter (checked with config.cache)
      c.has_spare_billing = true;
      const std::string m = str(*x, "elastic.spare_billing");
      if (m != "global" && m != "per_layer") bad("elastic.spare_billing", "\"" + m + "\" is not global | per_layer");
      c.spare_billing_per_layer = m == "per_layer";
      if (c.spare_billing_per_layer && !c.per_layer_units) bad("elastic.spare_billing", "per_layer needs per_layer_units");
    }
    if (const Json* x = e->find("mode")) {
      const std::string m = str(*x, "elastic.mode");
      if (m != "exclusive" && m != "inclusive") bad("elastic.mode", "\"" + m + "\" is not exclusive | inclusive");
      c.exclusive = m == "exclusive";
    }
    if (c.unit_bytes == 0 || c.units_total == 0) bad("elastic", "unit_bytes and units_total must be > 0");
    if (c.units_total > (uint64_t(1) << 32)) bad("elastic.units_total", "is above 2^32");
    if (c.units_total > kMaxBytes / c.unit_bytes ||
        c.units_total + c.spares + c.host_spares > kMaxBytes / c.unit_bytes)
      bad("elastic", "units x unit_bytes overflows 64 bits");
    if (c.per_layer_units && c.units_total % c.per_layer_units)
      bad("elastic.per_layer_units", "does not divide units_total");
  }

  const Json& cfg = need(doc, "config", "components-v1");
  {  // config.cache_config (rev 16): every option that sizes or steers the expert cache, a table of scalars
    // the other config values: scalars or tables of scalars (e.g. config.window, rev 18)
    Json rest = cfg;
    if (rest.is_object()) rest.obj.erase("cache_config");
    if (rest.is_object()) rest.obj.erase("shapes");
    if (rest.is_object())
      for (auto it = rest.obj.begin(); it != rest.obj.end();)
        if (it->second.is_object()) {
          scalars_only(it->second, "config." + it->first);
          it = rest.obj.erase(it);
        } else {
          ++it;
        }
    scalars_only(rest, "config");
    if (const Json* sh = cfg.find("shapes")) {  // config.shapes (rev 18): the finite shape set, tables of scalars
      if (!sh->is_array()) bad("config.shapes", "expected an array of shape tables");
      for (size_t i = 0; i < sh->arr.size(); ++i) scalars_only(sh->arr[i], "config.shapes[" + std::to_string(i) + "]");
    }
    if (const Json* cc = cfg.find("cache_config")) {
      scalars_only(*cc, "config.cache_config");
      c.cache_config = *cc;
    }
  }
  c.config = cfg;
  if (const Json* st = doc.find("stack")) {  // rev 18: the build's stack facts, kept as written
    keys(*st, "stack", {"stack_bytes", "dynamic_stack"});
    if (const Json* x = st->find("stack_bytes")) u64(*x, "stack.stack_bytes");
    if (const Json* x = st->find("dynamic_stack")) boolean(*x, "stack.dynamic_stack");
    c.stack = *st;
  }
  if (const Json* ev = doc.find("envelope")) {  // rev 18: the configuration envelope (kept as written, canonical JSON)
    keys(*ev, "envelope", {"shapes", "limits", "pools", "allocator", "dispatch", "cache_config"});
    c.envelope = *ev;
    const std::string t = json_write(*ev);
    c.envelope_sha256 = sha256_hex(t.data(), t.size());
  }
  auto cfg_int = [&](const char* k, bool required) -> int64_t {
    const Json* v = cfg.find(k);
    if (!v) {
      if (required) bad("config", std::string("missing \"") + k + "\"");
      return -1;
    }
    return int64_t(u64(*v, std::string("config.") + k));
  };
  c.max_context_tokens = cfg_int("max_context_tokens", true);
  c.max_sequences = cfg_int("max_sequences", true);
  c.max_rows = cfg_int("max_rows", true);
  c.chunk_rows = cfg_int("chunk_rows", false);
  if (const Json* v = cfg.find("cache")) {
    c.cache = str(*v, "config.cache");
    if (c.cache != "tierstream" && c.cache != "engine") bad("config.cache", "\"" + c.cache + "\" is not tierstream | engine");
  }
  if (c.has_elastic && c.device_gated() && !c.has_spare_billing)
    bad("elastic", "missing \"spare_billing\" (global | per_layer): required for the TierSite adapter (config.cache = "
                   "\"tierstream\"); the engine cache ignores it");

  if (const Json* v = doc.find("capabilities")) {
    if (!v->is_array()) bad("capabilities", "expected an array of strings");
    for (const Json& x : v->arr) c.capabilities.push_back(str(x, "capabilities"));
    std::sort(c.capabilities.begin(), c.capabilities.end());
    for (size_t i = 1; i < c.capabilities.size(); ++i)
      if (c.capabilities[i] == c.capabilities[i - 1])
        bad("capabilities", "\"" + c.capabilities[i] + "\" is listed twice");
  }

  const Json& ctx = need(doc, "context", "components-v1");
  if (!ctx.is_object()) bad("context", "expected an object keyed by arch");
  for (const auto& kv : ctx.obj) {  // allowances only: a measured value belongs to context-v1
    if (!arch_ok(kv.first)) bad("context." + kv.first, "not an arch (sm_XY | gfxNNNN)");
    c.context[kv.first] = u64(kv.second, "context." + kv.first);
  }
  if (const Json* rc = doc.find("context_receipts")) {
    if (!rc->is_array()) bad("context_receipts", "expected an array of receipts");
    for (size_t i = 0; i < rc->arr.size(); ++i) {
      const std::string w = "context_receipts[" + std::to_string(i) + "]";
      const Json& e = rc->arr[i];
      keys(e, w,
           {"arch", "device_name", "driver", "platform", "build_sha256", "allowance_bytes", "max_observed_bytes",
            "max_growth_bytes", "alloc_granularity_bytes", "envelope_sha256", "method", "runs", "receipt_sha256",
            "host_alloc_granularity_bytes", "graph_bytes"});
      {  // the key is complete (rev 16): five non-empty strings, else PL_E_SCHEMA naming the key as written
        std::string seen;
        bool complete = true;
        for (const char* k : {"arch", "device_name", "driver", "platform", "build_sha256"}) {
          const Json* v = e.find(k);
          if (v) seen += (seen.empty() ? "" : ", ") + json_write(*v);
          complete = complete && v && v->is_string() && !v->text.empty();
        }
        if (!complete)
          bad("context receipt key [" + seen + "]", "needs [arch, device_name, driver, platform, build_sha256]");
      }
      Components::ContextReceipt r;
      r.arch = str(need(e, "arch", w), w + ".arch");
      if (!arch_ok(r.arch)) bad(w + ".arch", "not an arch (sm_XY | gfxNNNN)");
      if (!c.context.count(r.arch)) bad(w + ".arch", "no context allowance for " + r.arch);
      r.device_name = str(need(e, "device_name", w), w + ".device_name");
      r.driver = str(need(e, "driver", w), w + ".driver");
      r.platform = str(need(e, "platform", w), w + ".platform");
      if (r.platform != "linux" && r.platform != "windows-wddm" && r.platform != "windows-tcc")
        bad(w + ".platform", "\"" + r.platform + "\" is not linux | windows-wddm | windows-tcc");
      r.build_sha256 = hex64(need(e, "build_sha256", w), w + ".build_sha256");
      r.allowance_bytes = u64(need(e, "allowance_bytes", w), w + ".allowance_bytes");
      r.max_observed_bytes = u64(need(e, "max_observed_bytes", w), w + ".max_observed_bytes");
      if (r.max_observed_bytes > r.allowance_bytes) bad(w, "max_observed_bytes is above allowance_bytes");
      r.max_growth_bytes = u64(need(e, "max_growth_bytes", w), w + ".max_growth_bytes");
      r.alloc_granularity_bytes = u64(need(e, "alloc_granularity_bytes", w), w + ".alloc_granularity_bytes");
      if (r.alloc_granularity_bytes > (uint64_t(1) << 32))
        bad(w + ".alloc_granularity_bytes", "above 4 GiB per allocation");
      r.envelope_sha256 = hex64(need(e, "envelope_sha256", w), w + ".envelope_sha256");
      if (const Json* x = e.find("host_alloc_granularity_bytes")) {
        r.host_alloc_granularity_bytes = u64(*x, w + ".host_alloc_granularity_bytes");
        if (r.host_alloc_granularity_bytes > (uint64_t(1) << 32))
          bad(w + ".host_alloc_granularity_bytes", "above 4 GiB per allocation");
      }
      if (const Json* x = e.find("graph_bytes")) r.graph_bytes = int64_t(u64(*x, w + ".graph_bytes"));
      r.method = str(need(e, "method", w), w + ".method");
      r.runs = u64(need(e, "runs", w), w + ".runs");
      if (r.runs < 3) bad(w + ".runs", "a context receipt needs >= 3 quiescent runs");
      r.receipt_sha256 = hex64(need(e, "receipt_sha256", w), w + ".receipt_sha256");
      c.context_receipts.push_back(r);
    }
    auto key = [](const Components::ContextReceipt& r) {
      return r.arch + "|" + r.device_name + "|" + r.driver + "|" + r.platform + "|" + r.build_sha256;
    };
    std::sort(c.context_receipts.begin(), c.context_receipts.end(),
              [&](const Components::ContextReceipt& a, const Components::ContextReceipt& b) { return key(a) < key(b); });
    for (size_t i = 1; i < c.context_receipts.size(); ++i)
      if (key(c.context_receipts[i]) == key(c.context_receipts[i - 1]))
        bad("duplicate context receipt key", key(c.context_receipts[i]));
  }
  if (const Json* g = doc.find("context_growth")) {
    if (!g->is_object()) bad("context_growth", "expected an object keyed by arch");
    for (const auto& kv : g->obj) {
      if (!arch_ok(kv.first)) bad("context_growth." + kv.first, "not an arch (sm_XY | gfxNNNN)");
      c.context_growth[kv.first] = u64(kv.second, "context_growth." + kv.first);
    }
  }

  const Json& comps = need(doc, "components", "components-v1");
  if (!comps.is_array()) bad("components", "expected an array");
  uint64_t total = 0;
  for (const auto& kv : c.context) total = add_ck(total, kv.second, "context");
  for (const auto& kv : c.context_growth) total = add_ck(total, kv.second, "context_growth");
  for (size_t i = 0; i < comps.arr.size(); ++i) {
    std::string w = "components[" + std::to_string(i) + "]";
    const Json& e = comps.arr[i];
    keys(e, w, {"id", "charges", "variants", "qualified", "requires"});
    ComponentDecl d;
    d.id = str(need(e, "id", w), w + ".id");
    d.priority = component_priority(d.id);
    if (!d.priority || d.id == "context")
      bad(w + ".id", "\"" + d.id + "\" is not a component id (context is the \"context\" table)");
    w = "components." + d.id;
    if (c.find(d.id)) bad(w, "is listed twice");
    const Json* ch = e.find("charges");
    const Json* va = e.find("variants");
    if (!!ch == !!va) bad(w, "needs exactly one of \"charges\" and \"variants\"");
    if (ch) {
      d.charges = charges_of(*ch, w + ".charges", total);
      sort_distinct(d.charges, w + ".charges", d.id, c.device_gated());
    } else {
      d.has_variants = true;
      keys(*va, w + ".variants", {"anchor", "assist"});
      if (va->obj.empty()) bad(w + ".variants", "expected \"anchor\" and / or \"assist\"");
      if (const Json* x = va->find("anchor")) {
        d.has_anchor = true;
        d.anchor = charges_of(*x, w + ".variants.anchor", total);
        sort_distinct(d.anchor, w + ".variants.anchor", d.id, c.device_gated());
      }
      if (const Json* x = va->find("assist")) {
        d.has_assist = true;
        d.assist = charges_of(*x, w + ".variants.assist", total);
        sort_distinct(d.assist, w + ".variants.assist", d.id, c.device_gated());
      }
    }
    if (const Json* q = e.find("qualified")) {
      if (!q->is_array()) bad(w + ".qualified", "expected an array of receipts");
      for (size_t k = 0; k < q->arr.size(); ++k) {
        const std::string qw = w + ".qualified[" + std::to_string(k) + "]";
        const Json& r = q->arr[k];
        keys(r, qw, {"arch", "build_sha256", "receipt_sha256", "content_sha256", "arithmetic_sha256", "variants"});
        Receipt rc;
        rc.arch = str(need(r, "arch", qw), qw + ".arch");
        if (!arch_ok(rc.arch)) bad(qw + ".arch", "not an arch (sm_XY | gfxNNNN)");
        rc.build_sha256 = hex64(need(r, "build_sha256", qw), qw + ".build_sha256");
        rc.receipt_sha256 = hex64(need(r, "receipt_sha256", qw), qw + ".receipt_sha256");
        if (const Json* x = r.find("content_sha256")) rc.content_sha256 = hex64(*x, qw + ".content_sha256");
        if (const Json* x = r.find("arithmetic_sha256")) rc.arithmetic_sha256 = hex64(*x, qw + ".arithmetic_sha256");
        if (const Json* x = r.find("variants")) {
          if (!x->is_array()) bad(qw + ".variants", "expected an array");
          for (const Json& n : x->arr) {
            const std::string s = str(n, qw + ".variants");
            if (s != "graph" && s != "eager") bad(qw + ".variants", "\"" + s + "\" is not graph | eager");
            rc.variants.push_back(s);
          }
          std::sort(rc.variants.begin(), rc.variants.end());
          rc.variants.erase(std::unique(rc.variants.begin(), rc.variants.end()), rc.variants.end());
        }
        d.qualified.push_back(rc);
      }
      std::sort(d.qualified.begin(), d.qualified.end(), [](const Receipt& a, const Receipt& b) {
        return a.arch != b.arch ? a.arch < b.arch : a.build_sha256 < b.build_sha256;
      });
      for (size_t k = 1; k < d.qualified.size(); ++k)  // one record per arch (contract rev 9)
        if (d.qualified[k].arch == d.qualified[k - 1].arch)
          bad(w + ".qualified", "two records for " + d.qualified[k].arch);
    }
    if (const Json* r = e.find("requires")) {
      keys(*r, w + ".requires", {"relay"});
      if (const Json* x = r->find("relay")) d.requires_relay = boolean(*x, w + ".requires.relay");
    }
    c.components.push_back(d);
  }
  std::sort(c.components.begin(), c.components.end(),
            [](const ComponentDecl& a, const ComponentDecl& b) { return a.priority < b.priority; });
  const std::string t = json_write(components_to_json(c));
  c.sha256 = sha256_hex(t.data(), t.size());
  return c;
}

int64_t context_aggregate(const ContextEntry& e) {
  std::vector<uint64_t> v;
  for (size_t i = e.samples.size(); i-- > 0 && v.size() < 5;) v.push_back(e.samples[i].bytes);
  if (v.size() < 3) return -1;
  std::sort(v.begin(), v.end());
  return int64_t(v[(v.size() - 1) / 2]);
}

const ContextEntry* ContextDoc::find(const std::string& build, const std::string& arch, const std::string& device_name,
                                     const std::string& driver, const std::string& platform) const {
  for (const auto& e : entries)  // contract §2.7: an entry applies only on an exact match of all five key fields
    if (e.build_sha256 == build && e.arch == arch && e.device_name == device_name && e.driver == driver &&
        e.platform == platform)
      return &e;
  return nullptr;
}

ContextDoc context_from_json(const Json& doc) {
  auto cbad = [](const std::string& path, const std::string& msg) {
    fail(SS_E_RECIPE, "context: PL_E_SCHEMA " + path + ": " + msg);
  };
  if (!doc.is_object()) cbad("context-v1", "expected an object");
  for (const auto& kv : doc.obj)
    if (kv.first != "schema" && kv.first != "entries") cbad("context-v1", "unknown key \"" + kv.first + "\"");
  const Json* s = doc.find("schema");
  if (!s || !s->is_string() || s->text != "tierstream/context-v1") cbad("schema", "expected \"tierstream/context-v1\"");
  const Json* es = doc.find("entries");
  if (!es || !es->is_array()) cbad("entries", "expected an array");
  ContextDoc c;
  for (size_t i = 0; i < es->arr.size(); ++i) {
    const std::string w = "entries[" + std::to_string(i) + "]";
    const Json& e = es->arr[i];
    keys(e, w, {"arch", "device_name", "driver", "platform", "build_sha256", "samples"});
    ContextEntry x;
    x.build_sha256 = hex64(need(e, "build_sha256", w), w + ".build_sha256");
    x.arch = str(need(e, "arch", w), w + ".arch");
    if (!arch_ok(x.arch)) cbad(w + ".arch", "not an arch (sm_XY | gfxNNNN)");
    x.device_name = str(need(e, "device_name", w), w + ".device_name");  // the five key fields: required, exact
    x.driver = str(need(e, "driver", w), w + ".driver");
    x.platform = str(need(e, "platform", w), w + ".platform");
    const Json& ss = need(e, "samples", w);
    if (!ss.is_array()) cbad(w + ".samples", "expected an array (oldest first)");
    for (size_t k = 0; k < ss.arr.size(); ++k) {
      const std::string sw = w + ".samples[" + std::to_string(k) + "]";
      keys(ss.arr[k], sw, {"bytes", "method"});  // attributable per-process methods only (§2.4)
      ContextSample smp;
      smp.bytes = u64(need(ss.arr[k], "bytes", sw), sw + ".bytes");
      smp.method = str(need(ss.arr[k], "method", sw), sw + ".method");
      x.samples.push_back(smp);
    }
    c.entries.push_back(x);
  }
  auto key = [](const ContextEntry& e) {
    return e.arch + "|" + e.device_name + "|" + e.driver + "|" + e.platform + "|" + e.build_sha256;
  };
  std::sort(c.entries.begin(), c.entries.end(), [&](const ContextEntry& a, const ContextEntry& b) { return key(a) < key(b); });
  Json arr = Json::array();
  for (size_t i = 0; i < c.entries.size(); ++i) {
    const ContextEntry& e = c.entries[i];
    if (i && key(e) == key(c.entries[i - 1])) cbad("entries", "two entries for " + key(e));
    Json x = Json::object();
    x["build_sha256"] = Json::string(e.build_sha256);
    x["arch"] = Json::string(e.arch);
    x["device_name"] = Json::string(e.device_name);
    x["driver"] = Json::string(e.driver);
    x["platform"] = Json::string(e.platform);
    Json sa = Json::array();
    for (const auto& smp : e.samples) {
      Json y = Json::object();
      y["bytes"] = Json::number(smp.bytes);
      y["method"] = Json::string(smp.method);
      sa.push(y);
    }
    x["samples"] = sa;
    arr.push(x);
  }
  Json canon = Json::object();
  canon["schema"] = Json::string("tierstream/context-v1");
  canon["entries"] = arr;
  const std::string t = json_write(canon);
  c.sha256 = sha256_hex(t.data(), t.size());
  return c;
}

UsageDoc usage_from_json(const Json& doc) {
  auto ubad = [](const std::string& path, const std::string& msg) {
    fail(SS_E_RECIPE, "usage: PL_E_SCHEMA " + path + ": " + msg);
  };
  // the producer's shape (strix-stream-adaptive adapt_usage.cpp Usage::to_json)
  auto num = [&](const Json* v, const std::string& path) -> int64_t {
    if (!v) ubad(path, "missing");
    if (!v->is_number() || v->text.empty() || v->text.find_first_not_of("0123456789") != std::string::npos ||
        v->text.size() > 19)
      ubad(path, "expected an integer >= 0");
    const unsigned long long x = std::strtoull(v->text.c_str(), nullptr, 10);
    if (x > uint64_t(INT64_MAX)) ubad(path, "overflows 63 bits");
    return int64_t(x);
  };
  if (!doc.is_object()) ubad("usage-v1", "expected an object");
  for (const auto& kv : doc.obj) {
    static const char* const allowed[] = {"schema", "model", "sources", "decay", "tick", "tokens", "order", "layers"};
    bool ok = false;
    for (const char* k : allowed) ok = ok || kv.first == k;
    if (!ok) ubad("usage-v1", "unknown key \"" + kv.first + "\"");
  }
  const Json* s = doc.find("schema");
  if (!s || !s->is_string() || s->text != "tierstream/usage-v1") ubad("schema", "expected \"tierstream/usage-v1\"");
  UsageDoc u;
  if (const Json* m = doc.find("model")) {
    if (!m->is_string()) ubad("model", "expected a string");
    u.model = m->text;
  }
  if (const Json* o = doc.find("order"))
    if (!o->is_string() || o->text != "use desc, recency desc, layer asc, unit asc")
      ubad("order", "expected \"use desc, recency desc, layer asc, unit asc\"");
  if (const Json* d = doc.find("decay")) {
    if (!d->is_object()) ubad("decay", "expected {factor_q32, half_life_tokens}");
    for (const auto& kv : d->obj)
      if (kv.first != "factor_q32" && kv.first != "half_life_tokens") ubad("decay", "unknown key \"" + kv.first + "\"");
    num(d->find("factor_q32"), "decay.factor_q32");
    num(d->find("half_life_tokens"), "decay.half_life_tokens");
  }
  u.tokens = num(doc.find("tokens"), "tokens");
  num(doc.find("tick"), "tick");
  if (const Json* v = doc.find("sources")) {
    if (!v->is_array()) ubad("sources", "expected an array of {path, sha256, kind, filter, tokens}");
    for (size_t i = 0; i < v->arr.size(); ++i) {
      const std::string w = "sources[" + std::to_string(i) + "]";
      const Json& e = v->arr[i];
      keys(e, w, {"path", "sha256", "kind", "filter", "tokens"});
      str(need(e, "path", w), w + ".path");
      u.sources.push_back(hex64(need(e, "sha256", w), w + ".sha256"));
      str(need(e, "kind", w), w + ".kind");
      std::string filter;
      if (const Json* f = e.find("filter")) {
        if (!f->is_string()) ubad(w + ".filter", "expected a string");
        filter = f->text;
      }
      if (const Json* t = e.find("tokens")) num(t, w + ".tokens");
      if (i == 0) u.filter = filter;  // a per-language table names its language in every source's filter
      else if (u.filter != filter) u.filter.clear();
    }
  }
  const Json* ls = doc.find("layers");
  if (!ls || !ls->is_array() || ls->arr.empty()) ubad("layers", "expected a non-empty array");
  int64_t moe = 0;
  for (size_t L = 0; L < ls->arr.size(); ++L) {
    const std::string w = "layers[" + std::to_string(L) + "]";
    const Json& l = ls->arr[L];
    keys(l, w, {"index", "n_units", "count", "use", "recency"});
    if (num(l.find("index"), w + ".index") != int64_t(L)) ubad(w + ".index", "must equal its position");
    const int64_t n = num(l.find("n_units"), w + ".n_units");  // 0 = a dense layer
    if (n >= (int64_t(1) << 20)) ubad(w + ".n_units", "out of range");
    auto arr = [&](const char* k) -> const Json& {
      const Json* a = l.find(k);
      if (!a || !a->is_array() || int64_t(a->arr.size()) != n) ubad(w + "." + k, "expected one integer per unit");
      return *a;
    };
    const Json& cnt = arr("count");
    const Json& use = arr("use");
    const Json& rec = arr("recency");
    u.layer_units.push_back(n);
    for (int64_t k = 0; k < n; ++k) {
      const std::string kw = "[" + std::to_string(k) + "]";
      UsageUnit x;
      x.layer = moe;
      x.model_layer = int64_t(L);
      x.unit = k;
      x.count = num(&cnt.arr[size_t(k)], w + ".count" + kw);
      x.selections = num(&use.arr[size_t(k)], w + ".use" + kw);
      x.recency = num(&rec.arr[size_t(k)], w + ".recency" + kw);
      // use is derived: rdiv(count x 10^6, tokens) (0 without tokens)
      const int64_t want =
          u.tokens ? int64_t((2 * uint64_t(x.count) * 1000000u + uint64_t(u.tokens)) / (2 * uint64_t(u.tokens))) : 0;
      if (x.count > (int64_t(1) << 40) || x.selections != want)
        ubad(w + ".use" + kw, "is not rdiv(count x 10^6, tokens)");
      u.units.push_back(x);
    }
    if (n) ++moe;
  }
  // usage_sha256: sha256 of the document as the producer writes it (json_write: equal to the file bytes it writes)
  const std::string t = json_write(doc);
  u.sha256 = sha256_hex(t.data(), t.size());
  return u;
}

std::vector<size_t> UsageDoc::rank() const {
  std::vector<size_t> order(units.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {  // units are layer-major, unit ascending
    if (units[a].selections != units[b].selections) return units[a].selections > units[b].selections;
    return units[a].recency > units[b].recency;
  });
  return order;
}

Json components_to_json(const Components& c) {
  Json j = Json::object();
  j["schema"] = Json::string(kSchema);
  j["engine"] = Json::string(c.engine);
  j["build_sha256"] = Json::string(c.build_sha256);
  j["model_sha256"] = Json::string(c.model_sha256);
  j["desc_sha256"] = Json::string(c.desc_sha256);
  if (!c.receipts_sha256.empty()) j["receipts_sha256"] = Json::string(c.receipts_sha256);
  if (!c.content.is_null()) j["content"] = c.content;
  if (!c.arithmetic.is_null()) j["arithmetic"] = c.arithmetic;
  if (c.has_elastic) {
    Json e = Json::object();
    e["unit_bytes"] = Json::number(c.unit_bytes);
    e["units_total"] = Json::number(c.units_total);
    e["per_layer_units"] = Json::number(c.per_layer_units);
    e["spares"] = Json::number(c.spares);
    e["host_spares"] = Json::number(c.host_spares);
    if (c.per_layer_max_active) e["per_layer_max_active"] = Json::number(c.per_layer_max_active);
    if (c.slot_allocs != 1) e["slot_allocs"] = Json::number(c.slot_allocs);
    if (c.host_chunk_bytes != (uint64_t(1) << 30)) e["host_chunk_bytes"] = Json::number(c.host_chunk_bytes);
    e["mode"] = Json::string(c.exclusive ? "exclusive" : "inclusive");
    if (c.has_spare_billing) e["spare_billing"] = Json::string(c.spare_billing_per_layer ? "per_layer" : "global");
    j["elastic"] = e;
  }
  j["config"] = c.config;
  if (!c.envelope.is_null()) j["envelope"] = c.envelope;
  if (!c.stack.is_null()) j["stack"] = c.stack;
  Json caps = Json::array();
  for (const auto& s : c.capabilities) caps.push(Json::string(s));
  j["capabilities"] = caps;
  Json ctx = Json::object();
  for (const auto& kv : c.context) ctx[kv.first] = Json::number(kv.second);
  j["context"] = ctx;
  Json growth = Json::object();
  for (const auto& kv : c.context_growth) growth[kv.first] = Json::number(kv.second);
  j["context_growth"] = growth;
  Json receipts = Json::array();
  for (const auto& r : c.context_receipts) {
    Json e = Json::object();
    e["arch"] = Json::string(r.arch);
    e["device_name"] = Json::string(r.device_name);
    e["driver"] = Json::string(r.driver);
    e["platform"] = Json::string(r.platform);
    e["build_sha256"] = Json::string(r.build_sha256);
    e["allowance_bytes"] = Json::number(r.allowance_bytes);
    e["max_observed_bytes"] = Json::number(r.max_observed_bytes);
    e["max_growth_bytes"] = Json::number(r.max_growth_bytes);
    e["alloc_granularity_bytes"] = Json::number(r.alloc_granularity_bytes);
    e["envelope_sha256"] = Json::string(r.envelope_sha256);
    if (r.host_alloc_granularity_bytes != 4096)
      e["host_alloc_granularity_bytes"] = Json::number(r.host_alloc_granularity_bytes);
    if (r.graph_bytes >= 0) e["graph_bytes"] = Json::number(r.graph_bytes);
    e["method"] = Json::string(r.method);
    e["runs"] = Json::number(r.runs);
    e["receipt_sha256"] = Json::string(r.receipt_sha256);
    receipts.push(e);
  }
  j["context_receipts"] = receipts;
  Json comps = Json::array();
  for (const auto& d : c.components) {
    Json e = Json::object();
    e["id"] = Json::string(d.id);
    if (d.has_variants) {
      Json v = Json::object();
      if (d.has_anchor) v["anchor"] = charges_json(d.anchor);
      if (d.has_assist) v["assist"] = charges_json(d.assist);
      e["variants"] = v;
    } else {
      e["charges"] = charges_json(d.charges);
    }
    if (!d.qualified.empty()) {
      Json q = Json::array();
      for (const auto& r : d.qualified) {
        Json x = Json::object();
        x["arch"] = Json::string(r.arch);
        x["build_sha256"] = Json::string(r.build_sha256);
        x["receipt_sha256"] = Json::string(r.receipt_sha256);
        if (!r.content_sha256.empty()) x["content_sha256"] = Json::string(r.content_sha256);
        if (!r.arithmetic_sha256.empty()) x["arithmetic_sha256"] = Json::string(r.arithmetic_sha256);
        if (!r.variants.empty()) {
          Json vv = Json::array();
          for (const auto& s : r.variants) vv.push(Json::string(s));
          x["variants"] = vv;
        }
        q.push(x);
      }
      e["qualified"] = q;
    }
    if (d.requires_relay) {
      Json r = Json::object();
      r["relay"] = Json::boolean(true);
      e["requires"] = r;
    }
    comps.push(e);
  }
  j["components"] = comps;
  return j;
}

}  // namespace ts
