#include "recipe.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>

#include "errors.hpp"
#include "schema_id.hpp"
#include "toml.hpp"

namespace ts {
namespace {

constexpr const char* kSchema = "strix-stream/recipe-v1";
constexpr double kMaxGib = 1048576.0;  // 2^20 GiB: gib x 2^30 stays exact in a double
constexpr int64_t kMaxTokens = int64_t(1) << 40;
constexpr int kMaxLayer = 4095;  // desc.cpp kMaxLayers - 1
constexpr int kMaxUnit = 65535;  // desc.cpp kMaxUnits - 1

class Reader {
 public:
  explicit Reader(NoteFn note) : note_(note) {}

  [[noreturn]] void bad(const std::string& path, const std::string& msg) const {
    std::string m = path + ": " + msg;
    if (note_.fn) m += note_.fn(note_.user, path);
    fail(SS_E_RECIPE, m);
  }

  void keys(const Json& o, const std::string& path, std::initializer_list<const char*> allowed) const {
    if (!o.is_object()) bad(path, "expected a table");
    for (const auto& kv : o.obj) {
      if (kv.first.rfind("x-", 0) == 0) continue;
      bool ok = false;
      for (const char* k : allowed) ok = ok || kv.first == k;
      if (!ok) bad(path, "unknown key \"" + kv.first + "\"");
    }
  }

  const Json& need(const Json& o, const char* key, const std::string& path) const {
    const Json* v = o.find(key);
    if (!v) bad(path, std::string("missing \"") + key + "\"");
    return *v;
  }

  std::string str(const Json& v, const std::string& path) const {
    if (!v.is_string()) bad(path, "expected a string");
    return v.text;
  }

  std::string ident(const Json& v, const std::string& path, size_t max, bool upper_dot) const {
    std::string s = str(v, path);
    bool ok = !s.empty() && s.size() <= max;
    for (char c : s)
      ok = ok && ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-' ||
                  (upper_dot && ((c >= 'A' && c <= 'Z') || c == '.')));
    if (!ok)
      bad(path, "\"" + s + "\" must be 1-" + std::to_string(max) + " characters of " +
                    (upper_dot ? "[A-Za-z0-9._-]" : "[a-z0-9_-]"));
    return s;
  }

  std::string hex64(const Json& v, const std::string& path) const {
    std::string s = str(v, path);
    bool ok = s.size() == 64;
    for (char c : s) ok = ok && ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
    if (!ok) bad(path, "expected 64 lowercase hex digits");
    return s;
  }

  std::string choice(const Json& v, const std::string& path, std::initializer_list<const char*> options) const {
    std::string s = str(v, path);
    std::string all;
    for (const char* o : options) {
      if (s == o) return s;
      all += std::string(all.empty() ? "" : " | ") + "\"" + o + "\"";
    }
    bad(path, "\"" + s + "\" is not one of " + all);
  }

  double num(const Json& v, const std::string& path, double lo, double hi, bool lo_open = false) const {
    if (!v.is_number()) bad(path, "expected a number");
    char* end = nullptr;
    double d = std::strtod(v.text.c_str(), &end);
    if (!end || *end || !std::isfinite(d)) bad(path, "bad number " + v.text);
    if (d < lo || d > hi || (lo_open && d == lo))
      bad(path, v.text + " is out of range " + (lo_open ? "(" : "[") + fmt(lo) + ", " + fmt(hi) + "]");
    return d;
  }

  int64_t integer(const Json& v, const std::string& path, int64_t lo, int64_t hi) const {
    if (!v.is_number()) bad(path, "expected an integer");
    const std::string& t = v.text;
    size_t k = t.size() && t[0] == '-' ? 1 : 0;
    if (k == t.size() || t.find_first_not_of("0123456789", k) != std::string::npos)
      bad(path, "expected an integer, got " + t);
    if (t.size() - k > 18) bad(path, t + " is out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    int64_t x = std::strtoll(t.c_str(), nullptr, 10);
    if (x < lo || x > hi) bad(path, t + " is out of range [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return x;
  }

  bool boolean(const Json& v, const std::string& path) const {
    if (v.kind != Json::Kind::Bool) bad(path, "expected true or false");
    return v.b;
  }

  const Json& array(const Json& v, const std::string& path) const {
    if (!v.is_array()) bad(path, "expected an array");
    return v;
  }

  static std::string fmt(double d) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%g", d);
    return buf;
  }

 private:
  NoteFn note_;
};

// Exactness is never a recipe option: refuse the keys someone would use to ask for it, wherever they are.
void refuse_exactness(const Json& v, const std::string& path) {
  if (v.is_object()) {
    for (const auto& kv : v.obj) {
      const std::string p = path.empty() ? kv.first : path + "." + kv.first;
      if (kv.first == "exact" || kv.first == "exactness" || kv.first == "bit_exact")
        fail(SS_E_RECIPE, p + ": exactness is not a recipe option (every placement is bit-identical)");
      if (kv.first.rfind("x-", 0) != 0) refuse_exactness(kv.second, p);
    }
  } else if (v.is_array()) {
    for (size_t i = 0; i < v.arr.size(); ++i) refuse_exactness(v.arr[i], path + "[" + std::to_string(i) + "]");
  }
}

// Array elements are named by their id in paths; before the id is read, by index.
std::string elem(const Reader& rd, const Json& e, const std::string& base, size_t i, std::string& id_out,
                 bool upper_dot = false) {
  std::string w = base + "[" + std::to_string(i) + "]";
  if (!e.is_object()) rd.bad(w, "expected a table");
  id_out = rd.ident(rd.need(e, "id", w), w + ".id", 32, upper_dot);
  return base + "." + id_out;
}

const char* kind_name(ss_backend_kind k) { return k == SS_BACKEND_CUDA ? "cuda" : k == SS_BACKEND_HIP ? "hip" : "cpu"; }
const char* phase_name(Phase p) { return p == Phase::Prefill ? "prefill" : "decode"; }
const char* execute_name(Execute e) {
  switch (e) {
    case Execute::Single:
      return "single";
    case Execute::Tp:
      return "tp";
    case Execute::Replicas:
      return "replicas";
    case Execute::Hetero:
      return "hetero";
  }
  return "?";
}
const char* rank_name(Rank r) { return r == Rank::Model ? "model" : r == Rank::List ? "list" : "frequency"; }
const char* rest_name(Rest r) { return r == Rest::Disk ? "disk" : r == Rest::Pinned ? "pinned" : "peer"; }

std::vector<std::pair<int, int>> layer_ranges(const Reader& rd, const Json& v, const std::string& path) {
  std::vector<std::pair<int, int>> out;
  rd.array(v, path);
  if (v.arr.empty()) rd.bad(path, "expected at least one layer (omit the key for every layer of the class)");
  for (size_t i = 0; i < v.arr.size(); ++i) {
    const Json& e = v.arr[i];
    std::string w = path + "[" + std::to_string(i) + "]";
    if (e.is_array()) {
      if (e.arr.size() != 2) rd.bad(w, "a range is [first, last]");
      int a = int(rd.integer(e.arr[0], w + "[0]", 0, kMaxLayer));
      int b = int(rd.integer(e.arr[1], w + "[1]", 0, kMaxLayer));
      if (b < a) rd.bad(w, "range [" + std::to_string(a) + ", " + std::to_string(b) + "] is empty");
      out.push_back({a, b});
    } else {
      int a = int(rd.integer(e, w, 0, kMaxLayer));
      out.push_back({a, a});
    }
  }
  std::vector<std::pair<int, int>> s = out;
  std::sort(s.begin(), s.end());
  for (size_t i = 1; i < s.size(); ++i)
    if (s[i].first <= s[i - 1].second) rd.bad(path, "layer " + std::to_string(s[i].first) + " is listed twice");
  return out;
}

bool ranges_overlap(const std::vector<std::pair<int, int>>& x, const std::vector<std::pair<int, int>>& y) {
  if (x.empty() || y.empty()) return true;  // "every layer of the class"
  for (const auto& a : x)
    for (const auto& b : y)
      if (a.first <= b.second && b.first <= a.second) return true;
  return false;
}

Json ranges_json(const std::vector<std::pair<int, int>>& r) {
  Json a = Json::array();
  for (const auto& x : r) {
    if (x.first == x.second) {
      a.push(Json::number(int64_t(x.first)));
    } else {
      Json p = Json::array();
      p.push(Json::number(int64_t(x.first)));
      p.push(Json::number(int64_t(x.second)));
      a.push(p);
    }
  }
  return a;
}

// Shortest text that reads back as the same double (Json::number(double) always writes 17 digits).
Json fnum(double d) {
  char buf[40];
  for (int prec = 15; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*g", prec, d);
    if (std::strtod(buf, nullptr) == d) break;
  }
  Json j;
  j.kind = Json::Kind::Number;
  j.text = buf;
  return j;
}

Json count_json(int64_t n) { return n == kFill ? Json::string("fill") : Json::number(n); }

int64_t count_of(const Reader& rd, const Json& v, const std::string& path) {
  if (v.is_string()) {
    if (v.text != "fill") rd.bad(path, "expected an integer >= 0 or \"fill\"");
    return kFill;
  }
  return rd.integer(v, path, 0, kMaxUnit + 1);
}

bool hex_run(const std::string& s, size_t at, size_t n) {
  if (at + n > s.size()) return false;
  for (size_t i = at; i < at + n; ++i)
    if (!((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'a' && s[i] <= 'f'))) return false;
  return true;
}

// "GPU-" + 8-4-4-4-12 lowercase hex: cudaDeviceProp.uuid as nvidia-smi prints it.
bool cuda_uuid_ok(const std::string& s) {
  if (s.size() != 40 || s.compare(0, 4, "GPU-") != 0) return false;
  size_t at = 4;
  for (size_t n : {8, 4, 4, 4, 12}) {
    if (!hex_run(s, at, n)) return false;
    at += n;
    if (at < s.size() && s[at++] != '-') return false;
  }
  return true;
}

// HIP has no normative UUID text yet (a hip uuid selector is refused by the planner): a token of [A-Za-z0-9-].
bool hip_uuid_ok(const std::string& s) {
  if (s.empty() || s.size() > 64) return false;
  for (char c : s)
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-')) return false;
  return true;
}

// dddd:bb:dd.f, lowercase hex, function 0-7.
bool pci_ok(const std::string& s) {
  return s.size() == 12 && hex_run(s, 0, 4) && s[4] == ':' && hex_run(s, 5, 2) && s[7] == ':' && hex_run(s, 8, 2) &&
         s[10] == '.' && s[11] >= '0' && s[11] <= '7';
}

RGroup group_from(const Reader& rd, const Json& e, const std::string& w, const std::string& id) {
  rd.keys(e, w, {"id", "class", "layers", "rank", "list", "resident", "pinned", "rest", "peer"});
  RGroup g;
  g.id = id;
  g.cls = rd.choice(rd.need(e, "class", w), w + ".class", {"moe", "blob"}) == "moe" ? LayerKind::Moe : LayerKind::Blob;
  if (const Json* v = e.find("layers")) g.layers = layer_ranges(rd, *v, w + ".layers");
  if (const Json* v = e.find("rank")) {
    std::string r = rd.choice(*v, w + ".rank", {"model", "list", "frequency"});
    g.rank = r == "model" ? Rank::Model : r == "list" ? Rank::List : Rank::Frequency;
  }
  if (const Json* v = e.find("list")) {
    if (g.rank != Rank::List) rd.bad(w + ".list", "only with rank = \"list\"");
    rd.array(*v, w + ".list");
    if (v->arr.empty()) rd.bad(w + ".list", "expected at least one unit");
    std::set<int> seen;
    for (size_t i = 0; i < v->arr.size(); ++i) {
      int u = int(rd.integer(v->arr[i], w + ".list[" + std::to_string(i) + "]", 0, kMaxUnit));
      if (!seen.insert(u).second) rd.bad(w + ".list", std::to_string(u) + " is listed twice");
      g.list.push_back(u);
    }
  } else if (g.rank == Rank::List) {
    rd.bad(w, "rank = \"list\" needs \"list\"");
  }
  if (const Json* v = e.find("resident")) g.resident = count_of(rd, *v, w + ".resident");
  if (const Json* v = e.find("pinned")) g.pinned = count_of(rd, *v, w + ".pinned");
  if (const Json* v = e.find("rest")) {
    std::string r = rd.choice(*v, w + ".rest", {"disk", "pinned", "peer"});
    g.rest = r == "disk" ? Rest::Disk : r == "pinned" ? Rest::Pinned : Rest::Peer;
  }
  if (const Json* v = e.find("peer")) {
    if (g.rest != Rest::Peer) rd.bad(w + ".peer", "only with rest = \"peer\"");
    g.peer = rd.ident(*v, w + ".peer", 32, false);
  } else if (g.rest == Rest::Peer) {
    rd.bad(w, "rest = \"peer\" needs \"peer\" (the set that computes the units)");
  }
  if (g.rest == Rest::Pinned && g.pinned == kFill)
    rd.bad(w, "rest = \"pinned\" with pinned = \"fill\": one of them must bound the pinned tier");
  if (g.cls == LayerKind::Moe && g.resident != kFill && g.pinned != kFill && g.resident + g.pinned > kMaxUnit + 1)
    rd.bad(w, "resident + pinned is above the most units a layer can have");
  return g;
}

// Placement-mode keys of a route (docs/recipes.md §5.4): max_rows, max_sequences, assist.<set>, components.<id>.
// Schema-time checks only; whatever needs the binding or the engine's components-v1 belongs to the planner.
void route_placement_keys(const Reader& rd, const Recipe& r, const Json& e, const std::string& w, RRoute& rt) {
  if (const Json* v = e.find("max_rows")) rt.max_rows = rd.integer(*v, w + ".max_rows", 1, 65536);
  if (const Json* v = e.find("max_sequences")) rt.max_sequences = rd.integer(*v, w + ".max_sequences", 1, 65536);
  if (const Json* as = e.find("assist")) {
    const std::string aw = w + ".assist";
    if (!as->is_object()) rd.bad(aw, "expected a table keyed by device set");
    for (const auto& kv : as->obj) {  // std::map: set id order
      if (kv.first.rfind("x-", 0) == 0) continue;
      const std::string sw = aw + "." + kv.first;
      const RDevice* dev = r.device(kv.first);
      if (!dev) rd.bad(sw, "PL_E_UNKNOWN_DEVICE \"" + kv.first + "\" is not a device set of match.devices");
      if (std::find(rt.devices.begin(), rt.devices.end(), kv.first) != rt.devices.end())
        rd.bad(sw, "PL_E_SCHEMA \"" + kv.first + "\" executes the route (an assist is not listed in its devices)");
      if (dev->count != 1 || dev->kind == SS_BACKEND_CPU)
        rd.bad(sw, "PL_E_SCHEMA an assist is one GPU: a cuda or hip set with count 1");
      rd.keys(kv.second, sw, {"device_gib", "margin_gib", "margin_fraction"});
      RAssist a;
      a.set = kv.first;
      if (const Json* x = kv.second.find("device_gib")) a.device_gib = rd.num(*x, sw + ".device_gib", 0, kMaxGib);
      if (const Json* x = kv.second.find("margin_gib")) a.margin_gib = rd.num(*x, sw + ".margin_gib", 0, 64);
      if (const Json* x = kv.second.find("margin_fraction"))
        a.margin_fraction = rd.num(*x, sw + ".margin_fraction", 0, 0.5);
      rt.assists.push_back(a);
    }
  }
  if (const Json* cs = e.find("components")) {
    const std::string cw = w + ".components";
    if (!cs->is_object()) rd.bad(cw, "expected a table keyed by component id");
    for (const auto& kv : cs->obj) {
      if (kv.first.rfind("x-", 0) == 0) continue;
      const std::string iw = cw + "." + kv.first;
      const int prio = component_priority(kv.first);
      if (!prio) rd.bad(iw, "PL_E_UNKNOWN_COMPONENT \"" + kv.first + "\" is not a component id");
      rd.keys(kv.second, iw, {"device", "gib", "units", "layers"});
      RComponent c;
      c.id = kv.first;
      if (const Json* x = kv.second.find("device")) {
        c.device = rd.str(*x, iw + ".device");
        if (!reserved_destination(c.device) && c.device != rt.devices[0] && !rt.assist(c.device))
          rd.bad(iw + ".device", "PL_E_UNKNOWN_DEVICE \"" + c.device +
                                     "\" is not the route's anchor, one of its assists, host_pinned, host_pageable, "
                                     "file or auto");
      }
      if (const Json* x = kv.second.find("gib")) {
        if (!component_table()[size_t(prio - 1)].elastic)
          rd.bad(iw + ".gib", "PL_E_SCHEMA only elastic components (arena, expert_*, layer_*) take a size");
        c.has_gib = true;
        c.gib = rd.num(*x, iw + ".gib", 0, kMaxGib);
      }
      if (const Json* x = kv.second.find("units")) {
        if (!component_table()[size_t(prio - 1)].elastic)
          rd.bad(iw + ".units", "PL_E_SCHEMA only elastic components (arena, expert_*, layer_*) take a size");
        if (c.has_gib) rd.bad(iw, "PL_E_SCHEMA gib and units are exclusive (units: an exact count)");
        c.units = rd.integer(*x, iw + ".units", 0, int64_t(1) << 32);
      }
      if (const Json* x = kv.second.find("layers")) {  // reserved: the planner refuses it (PL_E_CAPABILITY)
        c.has_layers = true;
        c.layers = layer_ranges(rd, *x, iw + ".layers");
      }
      rt.components.push_back(c);
    }
    std::sort(rt.components.begin(), rt.components.end(), [](const RComponent& a, const RComponent& b) {
      return component_priority(a.id) < component_priority(b.id);
    });
  }
  // max_rows / max_sequences exist only in placement mode, so a legacy recipe's canonical form never changes
  if (!rt.placement_mode() && (rt.max_rows > 0 || rt.max_sequences > 0))
    rd.bad(w + (rt.max_rows > 0 ? ".max_rows" : ".max_sequences"),
           "PL_E_SCHEMA only on a placement-mode route (with an assist or components table)");
  if (rt.placement_mode() && (rt.execute != Execute::Single))
    rd.bad(w + ".execute", "PL_E_SCHEMA a placement-mode route (assist / components) has one anchor: \"single\"");
}

}  // namespace

const std::vector<PlacementField>& placement_fields() {
  static const std::vector<PlacementField> f = {
      {"device_gib", FieldType::Gib, 0, kMaxGib, 0},  // default: the set's memory_min_gib
      {"host_pinned_gib", FieldType::Gib, 0, kMaxGib, 0},
      {"host_pageable_gib", FieldType::Gib, 0, kMaxGib, 0},         // placement: 0 = not checked
      {"host_available_floor_gib", FieldType::Gib, 0, kMaxGib, 0},  // placement: 0 = not checked
      {"workspace_gib",FieldType::Gib, 0, kMaxGib, 0},                  // optional: derived when absent
      {"max_context_tokens", FieldType::Int, 1, double(kMaxTokens), 0},  // required
      {"max_batch", FieldType::Int, 1, 65536, 0},                        // required
      {"margin_gib", FieldType::Float, 0, 64, 2.0},
      {"margin_fraction", FieldType::Float, 0, 0.5, 0.03},
      {"dynamic_slots_per_layer", FieldType::Int, 0, 65536, 0},
      {"ring_slots", FieldType::Int, 0, 65536, 0},
      {"pinned_backing", FieldType::Bool, 0, 1, 0},
      {"host_spare_units", FieldType::Int, 0, 65536, 0},
      {"policy", FieldType::Policy, 0, 1, double(SS_POLICY_LRU)},
      {"tier_mode", FieldType::TierMode, 0, 1, kTierInclusive},
      {"lookahead_units", FieldType::Int, 0, 64, 2},
      {"reader_threads", FieldType::Int, 1, 64, 2},
      {"weight_dirs", FieldType::WeightDirs, 0, 0, 0},
      {"drive_readers", FieldType::Int, 1, 32, 4},
      {"weight_verify", FieldType::WeightVerify, 0, 1, SS_MFILE_VERIFY_CACHED},
      {"staging_buffers", FieldType::Int, 2, 64, 4},
      {"staging_mib", FieldType::Int, 1, 16384, 256},
      {"copy_piece_mib", FieldType::Int, 1, 4096, 64},
      {"score_window", FieldType::Int, 1, 4096, 16},
      {"pin_planned", FieldType::Bool, 0, 1, 1},
      {"fill_residents", FieldType::Bool, 0, 1, 1},
      {"direct_io", FieldType::Bool, 0, 1, 1},
      {"verify_hash", FieldType::Bool, 0, 1, 0},
      {"group_units", FieldType::Int, 0, 65536, 0},
      {"spec_persistent", FieldType::Bool, 0, 1, 0},
      {"readahead_units", FieldType::Int, 0, 64, 0},
      {"prefetch_layers", FieldType::Int, 0, 8, 1},
      {"host_prefetch_layers", FieldType::Int, 0, 64, 0},
      {"chunk_tokens", FieldType::Int, 0, 1048576, 0},
      {"routed_only_max_tokens", FieldType::Int, 0, 1048576, 0},
      {"idle_preload_layers", FieldType::Int, 0, 64, 0},
      {"max_copies_per_token", FieldType::Int, 0, 65536, 0},
  };
  return f;
}

const std::vector<ComponentInfo>& component_table() {
  static const std::vector<ComponentInfo> t = {
      {"context", "anchor", false},     {"dense", "anchor", false},          {"head", "anchor", false},
      {"mtp", "anchor", false},         {"buffers", "anchor", false},        {"state", "anchor", false},
      {"window", "anchor", false},      {"graphs", "anchor", false},         {"drafter", "anchor", false},
      {"relay", "host_pinned", false},  {"ple", "host_pinned", false},       {"spares", "anchor", false},
      {"arena", "anchor", true},        {"layer_slots", "anchor", true},     {"expert_host", "host_pinned", true},
      {"layer_host", "host_pinned", true}, {"expert_assist", "", true},      {"expert_compute", "", true},
      {"expert_file", "file", true},    {"engram", "file", false},           {"staging", "host_pinned", false},
      {"pageable", "host_pageable", false},
  };
  return t;
}

int component_priority(const std::string& id) {
  const auto& t = component_table();
  for (size_t i = 0; i < t.size(); ++i)
    if (id == t[i].id) return int(i) + 1;
  return 0;
}

bool reserved_destination(const std::string& w) {
  return w == "auto" || w == "host_pinned" || w == "host_pageable" || w == "file";
}

std::string normalize_uuid(const std::string& s) {
  std::string t = s;
  for (char& c : t) c = char(std::tolower(static_cast<unsigned char>(c)));
  if (t.compare(0, 4, "gpu-") == 0) t = "GPU-" + t.substr(4);
  return t;
}

std::string normalize_pci(const std::string& s) {
  std::string t = s;
  for (char& c : t) c = char(std::tolower(static_cast<unsigned char>(c)));
  // nvidia-smi prints an 8-digit domain (00000000:01:00.0)
  if (t.size() == 16 && t.compare(0, 4, "0000") == 0 && t[8] == ':') t = t.substr(4);
  return pci_ok(t) ? t : std::string();
}

const RAssist* RRoute::assist(const std::string& set) const {
  for (const auto& a : assists)
    if (a.set == set) return &a;
  return nullptr;
}

const RComponent* RRoute::component(const std::string& cid) const {
  for (const auto& c : components)
    if (c.id == cid) return &c;
  return nullptr;
}

double RPlacement::get(const char* key) const {
  const auto& f = placement_fields();
  for (size_t i = 0; i < f.size(); ++i)
    if (std::string(f[i].key) == key) return v[i];
  fail(SS_E_RECIPE, std::string("internal: no placement field ") + key);
}

const RPlacement* RRoute::placement(const std::string& set) const {
  for (const auto& p : placements)
    if (p.set == set) return &p;
  return nullptr;
}

const RDevice* Recipe::device(const std::string& id) const {
  for (const auto& d : devices)
    if (d.id == id) return &d;
  return nullptr;
}

const RLink* Recipe::link(const std::string& id) const {
  for (const auto& l : links)
    if (l.id == id) return &l;
  return nullptr;
}

const RRoute* Recipe::route(const std::string& id) const {
  for (const auto& r : routes)
    if (r.id == id) return &r;
  return nullptr;
}

uint64_t gib_bytes(double gib) {
  require(std::isfinite(gib) && gib >= 0 && gib <= kMaxGib, SS_E_RECIPE, "GiB value out of range");
  return uint64_t(std::floor(gib * 1073741824.0 + 0.5));
}

Json recipe_doc_parse(const std::string& text) {
  size_t k = text.compare(0, 3, "\xEF\xBB\xBF") == 0 ? 3 : 0;
  while (k < text.size() && (text[k] == ' ' || text[k] == '\t' || text[k] == '\r' || text[k] == '\n')) ++k;
  if (k < text.size() && text[k] == '{') {
    try {
      return json_parse(text.substr(k));
    } catch (const Error& e) {
      fail(SS_E_RECIPE, e.what());
    }
  }
  return toml_parse(text);
}

Recipe recipe_from_json(const Json& doc, NoteFn note) {
  Reader rd(note);
  refuse_exactness(doc, "");
  rd.keys(doc, "recipe", {"schema", "name", "description", "generated_by", "pin", "match", "limits", "routes"});
  if (!schema_accepts(rd.str(rd.need(doc, "schema", "recipe"), "schema"), kSchema))
    rd.bad("schema", std::string("expected \"") + kSchema + "\"");
  Recipe r;
  r.name = rd.ident(rd.need(doc, "name", "recipe"), "name", 64, true);
  if (const Json* v = doc.find("description")) r.description = rd.str(*v, "description");
  if (const Json* v = doc.find("generated_by")) r.generated_by = rd.choice(*v, "generated_by", {"human", "autotuner"});

  // match
  const Json& m = rd.need(doc, "match", "recipe");
  rd.keys(m, "match", {"model", "model_sha256", "engine", "devices", "links"});
  r.model = rd.str(rd.need(m, "model", "match"), "match.model");
  if (r.model.empty()) rd.bad("match.model", "must not be empty");
  if (const Json* v = m.find("model_sha256")) r.model_sha256 = rd.hex64(*v, "match.model_sha256");
  if (const Json* v = m.find("engine")) r.engine = rd.str(*v, "match.engine");
  const Json& devs = rd.array(rd.need(m, "devices", "match"), "match.devices");
  if (devs.arr.empty()) rd.bad("match.devices", "expected at least one device set");
  for (size_t i = 0; i < devs.arr.size(); ++i) {
    RDevice d;
    std::string w = elem(rd, devs.arr[i], "match.devices", i, d.id);
    const Json& e = devs.arr[i];
    if (r.device(d.id)) rd.bad(w, "duplicate id");
    rd.keys(e, w,
            {"id", "kind", "count", "name", "memory_min_gib", "host_ram_min_gib", "disk_read_gbytes_ps_min",
             "h2d_gbytes_ps_min", "internal_link", "optional", "uuid", "pci", "index"});
    std::string k = rd.choice(rd.need(e, "kind", w), w + ".kind", {"cuda", "hip", "cpu"});
    d.kind = k == "cuda" ? SS_BACKEND_CUDA : k == "hip" ? SS_BACKEND_HIP : SS_BACKEND_CPU;
    if (const Json* v = e.find("count")) d.count = int(rd.integer(*v, w + ".count", 1, 64));
    if (const Json* v = e.find("name")) d.name = rd.str(*v, w + ".name");
    d.memory_min_gib = rd.num(rd.need(e, "memory_min_gib", w), w + ".memory_min_gib", 0, kMaxGib, true);
    if (const Json* v = e.find("host_ram_min_gib"))
      d.host_ram_min_gib = rd.num(*v, w + ".host_ram_min_gib", 0, kMaxGib);
    if (const Json* v = e.find("disk_read_gbytes_ps_min"))
      d.disk_read_gbytes_ps_min = rd.num(*v, w + ".disk_read_gbytes_ps_min", 0, 1e6);
    if (const Json* v = e.find("h2d_gbytes_ps_min")) {
      d.h2d_gbytes_ps_min = rd.num(*v, w + ".h2d_gbytes_ps_min", 0, 1e6);
      if (d.kind != SS_BACKEND_CUDA && d.h2d_gbytes_ps_min > 0) rd.bad(w + ".h2d_gbytes_ps_min", "only for a cuda set");
    }
    if (const Json* v = e.find("internal_link")) {
      std::string lw = w + ".internal_link";
      rd.keys(*v, lw, {"kind", "gbps_min"});
      d.has_internal_link = true;
      if (const Json* x = v->find("kind"))
        d.internal_kind = rd.choice(*x, lw + ".kind", {"usb4", "ethernet", "pcie", "other"});
      if (const Json* x = v->find("gbps_min")) d.internal_gbps_min = rd.num(*x, lw + ".gbps_min", 0, 1e6);
      if (d.count < 2) rd.bad(lw, "only for a set with count >= 2");
    }
    // Placement selectors: one physical device each, so only on count = 1 sets of a GPU kind.
    if (const Json* v = e.find("optional")) d.optional = rd.boolean(*v, w + ".optional");
    if (const Json* v = e.find("uuid")) {
      d.uuid = rd.str(*v, w + ".uuid");
      if (d.kind == SS_BACKEND_CUDA && !cuda_uuid_ok(d.uuid))
        rd.bad(w + ".uuid", "PL_E_SCHEMA \"" + d.uuid +
                                "\" is not a CUDA UUID (GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx, lowercase hex)");
      if (d.kind == SS_BACKEND_HIP && !hip_uuid_ok(d.uuid))
        rd.bad(w + ".uuid", "PL_E_SCHEMA \"" + d.uuid + "\" is not a device UUID text");
    }
    if (const Json* v = e.find("pci")) {
      d.pci = rd.str(*v, w + ".pci");
      if (!pci_ok(d.pci)) rd.bad(w + ".pci", "PL_E_SCHEMA \"" + d.pci + "\" is not a PCI address dddd:bb:dd.f (lowercase)");
    }
    if (const Json* v = e.find("index")) d.index = rd.integer(*v, w + ".index", 0, 1023);
    if ((d.optional || !d.uuid.empty() || !d.pci.empty() || d.index >= 0) && (d.count != 1 || d.kind == SS_BACKEND_CPU))
      rd.bad(w, "PL_E_SCHEMA optional / uuid / pci / index select one GPU: only on a cuda or hip set with count 1");
    r.devices.push_back(d);
  }
  if (const Json* links = m.find("links")) {
    rd.array(*links, "match.links");
    for (size_t i = 0; i < links->arr.size(); ++i) {
      RLink l;
      std::string w = elem(rd, links->arr[i], "match.links", i, l.id);
      const Json& e = links->arr[i];
      if (r.link(l.id)) rd.bad(w, "duplicate id");
      rd.keys(e, w, {"id", "between", "kind", "gbps_min"});
      const Json& b = rd.array(rd.need(e, "between", w), w + ".between");
      if (b.arr.size() != 2) rd.bad(w + ".between", "expected two device set ids");
      l.a = rd.str(b.arr[0], w + ".between[0]");
      l.b = rd.str(b.arr[1], w + ".between[1]");
      if (!r.device(l.a)) rd.bad(w + ".between", "unknown device set \"" + l.a + "\"");
      if (!r.device(l.b)) rd.bad(w + ".between", "unknown device set \"" + l.b + "\"");
      if (l.a == l.b) rd.bad(w + ".between", "a link joins two different sets");
      if (const Json* v = e.find("kind")) l.kind = rd.choice(*v, w + ".kind", {"usb4", "ethernet", "pcie", "other"});
      if (const Json* v = e.find("gbps_min")) l.gbps_min = rd.num(*v, w + ".gbps_min", 0, 1e6);
      r.links.push_back(l);
    }
  }

  // limits
  if (const Json* lim = doc.find("limits")) {
    rd.keys(*lim, "limits", {"host_free_ram_floor_gib", "host_free_commit_floor_gib"});
    if (const Json* v = lim->find("host_free_ram_floor_gib"))
      r.host_free_ram_floor_gib = rd.num(*v, "limits.host_free_ram_floor_gib", 0, kMaxGib);
    if (const Json* v = lim->find("host_free_commit_floor_gib"))
      r.host_free_commit_floor_gib = rd.num(*v, "limits.host_free_commit_floor_gib", 0, kMaxGib);
  }

  // routes
  const Json& routes = rd.array(rd.need(doc, "routes", "recipe"), "routes");
  const auto& fields = placement_fields();
  for (size_t i = 0; i < routes.arr.size(); ++i) {
    RRoute rt;
    std::string w = elem(rd, routes.arr[i], "routes", i, rt.id);
    const Json& e = routes.arr[i];
    if (r.route(rt.id)) rd.bad(w, "duplicate id");
    rd.keys(
        e, w,
        {"id", "phase", "devices", "execute", "handoff_route", "links", "when", "fallback", "enabled", "placement",
         "max_rows", "max_sequences", "assist", "components"});
    rt.phase = rd.choice(rd.need(e, "phase", w), w + ".phase", {"prefill", "decode"}) == "prefill" ? Phase::Prefill
                                                                                                   : Phase::Decode;
    const Json& ds = rd.array(rd.need(e, "devices", w), w + ".devices");
    if (ds.arr.empty()) rd.bad(w + ".devices", "expected at least one device set");
    for (size_t k = 0; k < ds.arr.size(); ++k) {
      std::string s = rd.str(ds.arr[k], w + ".devices[" + std::to_string(k) + "]");
      if (!r.device(s)) rd.bad(w + ".devices", "unknown device set \"" + s + "\"");
      if (std::find(rt.devices.begin(), rt.devices.end(), s) != rt.devices.end())
        rd.bad(w + ".devices", "\"" + s + "\" is listed twice");
      rt.devices.push_back(s);
    }
    std::string ex = rd.choice(rd.need(e, "execute", w), w + ".execute", {"single", "tp", "replicas", "hetero"});
    rt.execute = ex == "single"     ? Execute::Single
                 : ex == "tp"       ? Execute::Tp
                 : ex == "replicas" ? Execute::Replicas
                                    : Execute::Hetero;
    const RDevice* first = r.device(rt.devices[0]);
    switch (rt.execute) {
      case Execute::Single:
        if (rt.devices.size() != 1 || first->count != 1)
          rd.bad(w + ".execute", "\"single\" runs on one device set with count 1");
        break;
      case Execute::Tp:
      case Execute::Replicas:
        if (rt.devices.size() != 1 || first->count < 2)
          rd.bad(w + ".execute", "\"" + ex + "\" runs on one device set with count >= 2");
        break;
      case Execute::Hetero:
        if (rt.devices.size() < 2) rd.bad(w + ".execute", "\"hetero\" needs at least two device sets");
        break;
    }
    if (const Json* v = e.find("handoff_route")) {
      if (rt.phase != Phase::Prefill) rd.bad(w + ".handoff_route", "only prefill routes hand off");
      rt.handoff_route = rd.ident(*v, w + ".handoff_route", 32, false);
    }
    if (const Json* v = e.find("links")) {
      rd.array(*v, w + ".links");
      for (size_t k = 0; k < v->arr.size(); ++k) {
        std::string s = rd.str(v->arr[k], w + ".links[" + std::to_string(k) + "]");
        if (!r.link(s)) rd.bad(w + ".links", "unknown link \"" + s + "\"");
        if (std::find(rt.links.begin(), rt.links.end(), s) != rt.links.end())
          rd.bad(w + ".links", "\"" + s + "\" is listed twice");
        rt.links.push_back(s);
      }
    }
    if (const Json* v = e.find("when")) {
      std::string ww = w + ".when";
      rd.keys(*v, ww, {"prompt_tokens_min", "prompt_tokens_max", "batch_min", "batch_max"});
      if (const Json* x = v->find("prompt_tokens_min"))
        rt.when.prompt_tokens_min = rd.integer(*x, ww + ".prompt_tokens_min", 0, kMaxTokens);
      if (const Json* x = v->find("prompt_tokens_max"))
        rt.when.prompt_tokens_max = rd.integer(*x, ww + ".prompt_tokens_max", 0, kMaxTokens);
      if (const Json* x = v->find("batch_min")) rt.when.batch_min = rd.integer(*x, ww + ".batch_min", 1, 1 << 20);
      if (const Json* x = v->find("batch_max")) rt.when.batch_max = rd.integer(*x, ww + ".batch_max", 1, 1 << 20);
      if (rt.when.prompt_tokens_max != kNoLimit && rt.when.prompt_tokens_max < rt.when.prompt_tokens_min)
        rd.bad(ww + ".prompt_tokens_max", "is below prompt_tokens_min");
      if (rt.when.batch_max != kNoLimit && rt.when.batch_max < rt.when.batch_min)
        rd.bad(ww + ".batch_max", "is below batch_min");
    }
    if (const Json* v = e.find("fallback")) rt.fallback = rd.ident(*v, w + ".fallback", 32, false);
    if (const Json* v = e.find("enabled")) rt.enabled = rd.boolean(*v, w + ".enabled");
    if (const Json* pl = e.find("placement")) {
      if (!pl->is_object()) rd.bad(w + ".placement", "expected a table keyed by device set");
      for (const auto& kv : pl->obj) {
        if (kv.first.rfind("x-", 0) == 0) continue;
        std::string pw = w + ".placement." + kv.first;
        if (std::find(rt.devices.begin(), rt.devices.end(), kv.first) == rt.devices.end())
          rd.bad(pw, "\"" + kv.first + "\" is not one of the route's devices");
        const RDevice* dev = r.device(kv.first);
        const Json& pe = kv.second;
        if (!pe.is_object()) rd.bad(pw, "expected a table");
        RPlacement p;
        p.set = kv.first;
        p.v.resize(fields.size());
        for (const auto& f : pe.obj) {
          if (f.first == "units" || f.first.rfind("x-", 0) == 0) continue;
          bool known = false;
          for (const auto& pf : fields) known = known || f.first == pf.key;
          if (!known) rd.bad(pw, "unknown key \"" + f.first + "\"");
        }
        for (size_t k = 0; k < fields.size(); ++k) {
          const PlacementField& f = fields[k];
          const std::string fw = pw + "." + f.key;
          const Json* v = pe.find(f.key);
          double dflt = std::string(f.key) == "device_gib" ? dev->memory_min_gib : f.dflt;
          if (!v) {
            if (std::string(f.key) == "max_context_tokens" || std::string(f.key) == "max_batch")
              rd.bad(pw, std::string("missing \"") + f.key + "\" (context and batch bounds are never assumed)");
            p.v[k] = dflt;
            continue;
          }
          switch (f.type) {
            case FieldType::Gib:
              p.v[k] = rd.num(*v, fw, f.lo, f.hi, std::string(f.key) == "device_gib");
              break;
            case FieldType::Float:
              p.v[k] = rd.num(*v, fw, f.lo, f.hi);
              break;
            case FieldType::Int:
              p.v[k] = double(rd.integer(*v, fw, int64_t(f.lo), int64_t(f.hi)));
              break;
            case FieldType::Bool:
              p.v[k] = rd.boolean(*v, fw) ? 1 : 0;
              break;
            case FieldType::Policy:
              p.v[k] = rd.choice(*v, fw, {"lru", "scored"}) == "lru" ? double(SS_POLICY_LRU) : double(SS_POLICY_SCORED);
              break;
            case FieldType::TierMode:
              p.v[k] = rd.choice(*v, fw, {"inclusive", "exclusive"}) == "inclusive" ? kTierInclusive : kTierExclusive;
              break;
            case FieldType::WeightDirs:
              p.weight_dirs = rd.str(*v, fw);
              if (p.weight_dirs.find('\0') != std::string::npos) rd.bad(fw, "embedded NUL in directory list");
              break;
            case FieldType::WeightVerify:
              p.v[k] = rd.choice(*v, fw, {"cached", "always"}) == "cached" ? SS_MFILE_VERIFY_CACHED : SS_MFILE_VERIFY_ALWAYS;
              break;
          }
          if (std::string(f.key) == "workspace_gib") p.has_workspace = true;
        }
        if (p.get("device_gib") > dev->memory_min_gib)
          rd.bad(pw + ".device_gib", "is above the set's memory_min_gib " + Reader::fmt(dev->memory_min_gib));
        if (dev->host_ram_min_gib > 0 && p.get("host_pinned_gib") > dev->host_ram_min_gib)
          rd.bad(pw + ".host_pinned_gib", "is above the set's host_ram_min_gib " + Reader::fmt(dev->host_ram_min_gib));
        if (p.has_workspace && p.get("workspace_gib") >= p.get("device_gib"))
          rd.bad(pw + ".workspace_gib", "must be below device_gib");
        if (const Json* units = pe.find("units")) {
          rd.array(*units, pw + ".units");
          for (size_t k = 0; k < units->arr.size(); ++k) {
            std::string gid;
            std::string gw = elem(rd, units->arr[k], pw + ".units", k, gid);
            for (const auto& g : p.groups)
              if (g.id == gid) rd.bad(gw, "duplicate id");
            RGroup g = group_from(rd, units->arr[k], gw, gid);
            for (const auto& o : p.groups)
              if (o.cls == g.cls && ranges_overlap(o.layers, g.layers))
                rd.bad(gw + ".layers", "overlaps group \"" + o.id + "\" of the same class");
            if (g.pinned != 0 && p.get("host_pinned_gib") <= 0)
              rd.bad(gw + ".pinned", "pinned units need host_pinned_gib > 0 in the placement");
            if (g.rest == Rest::Pinned && p.get("host_pinned_gib") <= 0)
              rd.bad(gw + ".rest", "pinned units need host_pinned_gib > 0 in the placement");
            if (g.rest == Rest::Peer) {
              if (rt.execute != Execute::Hetero) rd.bad(gw + ".rest", "\"peer\" only in a hetero route");
              auto me = std::find(rt.devices.begin(), rt.devices.end(), p.set);
              auto it = std::find(rt.devices.begin(), rt.devices.end(), g.peer);
              if (it == rt.devices.end() || it <= me)
                rd.bad(gw + ".peer",
                       "\"" + g.peer + "\" must be a device set after \"" + p.set + "\" in the route's devices");
            }
            p.groups.push_back(g);
          }
        }
        rt.placements.push_back(p);
      }
    }
    for (const auto& s : rt.devices) {
      const RPlacement* p = rt.placement(s);
      if (!p) rd.bad(w + ".placement." + s, "every set of a route needs a placement (budget and context bound)");
      if (rt.execute == Execute::Hetero && p->groups.empty())
        rd.bad(w + ".placement." + s, "a hetero route needs unit groups for every set");
    }
    route_placement_keys(rd, r, e, w, rt);
    r.routes.push_back(rt);
  }

  // Cross-route rules.
  bool any_decode = false;
  for (const Phase ph : {Phase::Prefill, Phase::Decode}) {
    const RRoute* last = nullptr;
    for (const auto& rt : r.routes)
      if (rt.phase == ph) last = &rt;
    if (ph == Phase::Decode) any_decode = last != nullptr;
    if (last && !last->when.always())
      rd.bad("routes." + last->id + ".when",
             std::string("the last ") + phase_name(ph) + " route must be a catch-all (no \"when\")");
  }
  if (!any_decode) rd.bad("routes", "the recipe needs at least one decode route");
  for (const auto& rt : r.routes) {
    const std::string w = "routes." + rt.id;
    std::vector<std::string> sets = rt.devices;
    if (!rt.handoff_route.empty()) {
      const RRoute* h = r.route(rt.handoff_route);
      if (!h) rd.bad(w + ".handoff_route", "unknown route \"" + rt.handoff_route + "\"");
      if (h->phase != Phase::Decode) rd.bad(w + ".handoff_route", "\"" + h->id + "\" is not a decode route");
      for (const auto& s : h->devices)
        if (std::find(sets.begin(), sets.end(), s) == sets.end()) sets.push_back(s);
    }
    // The route's links must touch only its sets and connect them all.
    std::map<std::string, std::string> parent;
    for (const auto& s : sets) parent[s] = s;
    auto root = [&](std::string s) {
      while (parent[s] != s) s = parent[s];
      return s;
    };
    for (const auto& lid : rt.links) {
      const RLink* l = r.link(lid);
      if (!parent.count(l->a) || !parent.count(l->b))
        rd.bad(w + ".links", "link \"" + lid + "\" joins a set the route does not use");
      parent[root(l->a)] = root(l->b);
    }
    for (const auto& s : sets)
      if (root(s) != root(sets[0]))
        rd.bad(w + ".links", "no link connects \"" + s + "\" to \"" + sets[0] + "\" (list the links the route needs)");
    if (!rt.fallback.empty()) {
      const RRoute* f = r.route(rt.fallback);
      if (!f) rd.bad(w + ".fallback", "unknown route \"" + rt.fallback + "\"");
      if (f->phase != rt.phase) rd.bad(w + ".fallback", "\"" + f->id + "\" is a route of another phase");
      if (f == &rt) rd.bad(w + ".fallback", "a route cannot fall back to itself");
      std::set<std::string> seen{rt.id};
      for (const RRoute* c = f; c; c = c->fallback.empty() ? nullptr : r.route(c->fallback))
        if (!seen.insert(c->id).second) rd.bad(w + ".fallback", "the fallback chain cycles at \"" + c->id + "\"");
    }
  }

  // Placement: reserved destination words are no set ids in a placement-mode recipe; executing sets are never
  // optional; optional sets follow the others in set-id order (their binding order), so the canonical form does not
  // depend on how they were declared.
  bool any_placement = false;
  for (const auto& rt : r.routes) any_placement = any_placement || rt.placement_mode();
  for (const auto& d : r.devices) {
    if (any_placement && reserved_destination(d.id))
      rd.bad("match.devices." + d.id, "PL_E_SCHEMA \"" + d.id + "\" is a reserved destination word (placement mode)");
    if (d.optional)
      for (const auto& rt : r.routes)
        if (std::find(rt.devices.begin(), rt.devices.end(), d.id) != rt.devices.end())
          rd.bad("match.devices." + d.id + ".optional",
                 "PL_E_SCHEMA route \"" + rt.id + "\" executes on it (only an assist set may be optional)");
  }
  std::stable_sort(r.devices.begin(), r.devices.end(), [](const RDevice& a, const RDevice& b) {
    if (a.optional != b.optional) return !a.optional;
    return a.optional && a.id < b.id;
  });

  if (const Json* v = doc.find("pin")) {
    rd.array(*v, "pin");
    for (size_t i = 0; i < v->arr.size(); ++i) {
      std::string p = rd.str(v->arr[i], "pin[" + std::to_string(i) + "]");
      if (std::find(r.pin.begin(), r.pin.end(), p) != r.pin.end()) rd.bad("pin", "\"" + p + "\" is listed twice");
      try {
        check_path(r, p, true);
      } catch (const Error& e) {
        rd.bad("pin[" + std::to_string(i) + "]", e.what());
      }
      r.pin.push_back(p);
    }
  }
  return r;
}

Json recipe_to_json(const Recipe& r) {
  Json j = Json::object();
  j["schema"] = Json::string(kSchema);
  j["name"] = Json::string(r.name);
  j["description"] = Json::string(r.description);
  j["generated_by"] = Json::string(r.generated_by);
  Json pin = Json::array();
  for (const auto& p : r.pin) pin.push(Json::string(p));
  j["pin"] = pin;
  Json m = Json::object();
  m["model"] = Json::string(r.model);
  if (!r.model_sha256.empty()) m["model_sha256"] = Json::string(r.model_sha256);
  if (!r.engine.empty()) m["engine"] = Json::string(r.engine);
  Json devs = Json::array();
  for (const auto& d : r.devices) {
    Json e = Json::object();
    e["id"] = Json::string(d.id);
    e["kind"] = Json::string(kind_name(d.kind));
    e["count"] = Json::number(int64_t(d.count));
    if (!d.name.empty()) e["name"] = Json::string(d.name);
    e["memory_min_gib"] = fnum(d.memory_min_gib);
    e["host_ram_min_gib"] = fnum(d.host_ram_min_gib);
    e["disk_read_gbytes_ps_min"] = fnum(d.disk_read_gbytes_ps_min);
    if (d.kind == SS_BACKEND_CUDA) e["h2d_gbytes_ps_min"] = fnum(d.h2d_gbytes_ps_min);
    if (d.has_internal_link) {
      Json il = Json::object();
      il["kind"] = Json::string(d.internal_kind);
      il["gbps_min"] = fnum(d.internal_gbps_min);
      e["internal_link"] = il;
    }
    // Placement selectors (after recipe-v1 shipped): written only when set.
    if (d.optional) e["optional"] = Json::boolean(true);
    if (!d.uuid.empty()) e["uuid"] = Json::string(d.uuid);
    if (!d.pci.empty()) e["pci"] = Json::string(d.pci);
    if (d.index >= 0) e["index"] = Json::number(d.index);
    devs.push(e);
  }
  m["devices"] = devs;
  Json links = Json::array();
  for (const auto& l : r.links) {
    Json e = Json::object();
    e["id"] = Json::string(l.id);
    Json b = Json::array();
    b.push(Json::string(l.a));
    b.push(Json::string(l.b));
    e["between"] = b;
    e["kind"] = Json::string(l.kind);
    e["gbps_min"] = fnum(l.gbps_min);
    links.push(e);
  }
  m["links"] = links;
  j["match"] = m;
  Json lim = Json::object();
  lim["host_free_ram_floor_gib"] = fnum(r.host_free_ram_floor_gib);
  lim["host_free_commit_floor_gib"] = fnum(r.host_free_commit_floor_gib);
  j["limits"] = lim;
  const auto& fields = placement_fields();
  Json routes = Json::array();
  for (const auto& rt : r.routes) {
    Json e = Json::object();
    e["id"] = Json::string(rt.id);
    e["phase"] = Json::string(phase_name(rt.phase));
    Json ds = Json::array();
    for (const auto& s : rt.devices) ds.push(Json::string(s));
    e["devices"] = ds;
    e["execute"] = Json::string(execute_name(rt.execute));
    if (!rt.handoff_route.empty()) e["handoff_route"] = Json::string(rt.handoff_route);
    Json ls = Json::array();
    for (const auto& s : rt.links) ls.push(Json::string(s));
    e["links"] = ls;
    Json w = Json::object();
    w["prompt_tokens_min"] = Json::number(rt.when.prompt_tokens_min);
    if (rt.when.prompt_tokens_max != kNoLimit) w["prompt_tokens_max"] = Json::number(rt.when.prompt_tokens_max);
    w["batch_min"] = Json::number(rt.when.batch_min);
    if (rt.when.batch_max != kNoLimit) w["batch_max"] = Json::number(rt.when.batch_max);
    e["when"] = w;
    if (!rt.fallback.empty()) e["fallback"] = Json::string(rt.fallback);
    e["enabled"] = Json::boolean(rt.enabled);
    Json pl = Json::object();
    for (const auto& p : rt.placements) {
      Json pe = Json::object();
      for (size_t k = 0; k < fields.size(); ++k) {
        const PlacementField& f = fields[k];
        if (std::string(f.key) == "workspace_gib" && !p.has_workspace) continue;
        // Added after recipe-v1 shipped: written only when set, so existing recipes keep their canonical bytes (and
        // the effective hash their profiles and compiled plans are bound to).
        if ((std::string(f.key) == "host_spare_units" || std::string(f.key) == "readahead_units" ||
             std::string(f.key) == "host_prefetch_layers" || std::string(f.key) == "host_pageable_gib" ||
             std::string(f.key) == "host_available_floor_gib") &&
            p.v[k] == 0)
          continue;
        if ((std::string(f.key) == "weight_dirs" && p.weight_dirs.empty()) ||
            (std::string(f.key) == "drive_readers" && p.v[k] == 4) ||
            (std::string(f.key) == "weight_verify" && p.v[k] == SS_MFILE_VERIFY_CACHED))
          continue;
        switch (f.type) {
          case FieldType::Gib:
          case FieldType::Float:
            pe[f.key] = fnum(p.v[k]);
            break;
          case FieldType::Int:
            pe[f.key] = Json::number(int64_t(p.v[k]));
            break;
          case FieldType::Bool:
            pe[f.key] = Json::boolean(p.v[k] != 0);
            break;
          case FieldType::Policy:
            pe[f.key] = Json::string(p.v[k] == double(SS_POLICY_LRU) ? "lru" : "scored");
            break;
          case FieldType::TierMode:
            pe[f.key] = Json::string(p.v[k] == kTierInclusive ? "inclusive" : "exclusive");
            break;
          case FieldType::WeightDirs:
            pe[f.key] = Json::string(p.weight_dirs);
            break;
          case FieldType::WeightVerify:
            pe[f.key] = Json::string(p.v[k] == SS_MFILE_VERIFY_CACHED ? "cached" : "always");
            break;
        }
      }
      Json units = Json::array();
      for (const auto& g : p.groups) {
        Json ge = Json::object();
        ge["id"] = Json::string(g.id);
        ge["class"] = Json::string(g.cls == LayerKind::Moe ? "moe" : "blob");
        if (!g.layers.empty()) ge["layers"] = ranges_json(g.layers);
        ge["rank"] = Json::string(rank_name(g.rank));
        if (!g.list.empty()) {
          Json l = Json::array();
          for (int u : g.list) l.push(Json::number(int64_t(u)));
          ge["list"] = l;
        }
        ge["resident"] = count_json(g.resident);
        ge["pinned"] = count_json(g.pinned);
        ge["rest"] = Json::string(rest_name(g.rest));
        if (!g.peer.empty()) ge["peer"] = Json::string(g.peer);
        units.push(ge);
      }
      pe["units"] = units;
      pl[p.set] = pe;
    }
    e["placement"] = pl;
    // Placement mode (after recipe-v1 shipped): written only when set.
    if (rt.max_rows > 0) e["max_rows"] = Json::number(rt.max_rows);
    if (rt.max_sequences > 0) e["max_sequences"] = Json::number(rt.max_sequences);
    if (!rt.assists.empty()) {
      Json as = Json::object();
      for (const auto& a : rt.assists) {
        Json ae = Json::object();
        ae["device_gib"] = fnum(a.device_gib);
        ae["margin_gib"] = fnum(a.margin_gib);
        ae["margin_fraction"] = fnum(a.margin_fraction);
        as[a.set] = ae;
      }
      e["assist"] = as;
    }
    if (!rt.components.empty()) {
      Json cs = Json::object();
      for (const auto& c : rt.components) {
        Json ce = Json::object();
        ce["device"] = Json::string(c.device);
        if (c.has_gib) ce["gib"] = fnum(c.gib);
        if (c.units >= 0) ce["units"] = Json::number(c.units);
        if (c.has_layers) ce["layers"] = ranges_json(c.layers);
        cs[c.id] = ce;
      }
      e["components"] = cs;
    }
    routes.push(e);
  }
  j["routes"] = routes;
  return j;
}

}  // namespace ts
