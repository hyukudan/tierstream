// Settable paths, profile / override layers and explain (docs/recipes.md §6).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "errors.hpp"
#include "recipe.hpp"
#include "sha256.hpp"
#include "toml.hpp"

namespace ss {
namespace {

constexpr const char* kProfileSchema = "strix-stream/recipe-profile-v1";
constexpr const char* kOverridesSchema = "strix-stream/recipe-overrides-v1";

std::vector<std::string> split(const std::string& path) {
  std::vector<std::string> out;
  size_t b = 0;
  while (true) {
    size_t e = path.find('.', b);
    out.push_back(path.substr(b, e == std::string::npos ? std::string::npos : e - b));
    if (e == std::string::npos) break;
    b = e + 1;
  }
  return out;
}

bool one_of(const std::string& s, std::initializer_list<const char*> options) {
  for (const char* o : options)
    if (s == o) return true;
  return false;
}

bool placement_field(const std::string& s) {
  for (const auto& f : placement_fields())
    if (s == f.key) return true;
  return false;
}

// The node at `path` in a document tree: array elements by their "id". With create, missing tables on the way are
// made (the caller has checked the path, so only placement / when / limits tables can be missing).
Json* walk(Json& doc, const std::vector<std::string>& segs, bool create) {
  Json* node = &doc;
  for (const std::string& s : segs) {
    if (node->is_array()) {
      Json* next = nullptr;
      for (Json& e : node->arr) {
        const Json* id = e.find("id");
        if (id && id->is_string() && id->text == s) next = &e;
      }
      if (!next) return nullptr;
      node = next;
    } else if (node->is_object()) {
      auto it = node->obj.find(s);
      if (it == node->obj.end()) {
        if (!create) return nullptr;
        node = &((*node)[s] = Json::object());
      } else {
        node = &it->second;
      }
    } else {
      return nullptr;
    }
  }
  return node;
}

// Leaves of a canonical recipe: tables recurse, arrays of tables recurse by id, other arrays are leaves.
void leaves(const Json& v, const std::string& path, std::vector<std::pair<std::string, const Json*>>& out) {
  auto join = [&](const std::string& k) { return path.empty() ? k : path + "." + k; };
  if (v.is_object()) {
    for (const auto& kv : v.obj) leaves(kv.second, join(kv.first), out);
  } else if (v.is_array() && !v.arr.empty() && v.arr[0].is_object() && v.arr[0].find("id")) {
    for (const Json& e : v.arr) {
      std::string base = join(e.find("id")->text);
      for (const auto& kv : e.obj)
        if (kv.first != "id") leaves(kv.second, base + "." + kv.first, out);
    }
  } else {
    out.push_back({path, &v});
  }
}

const char* kind_word(RecipeLayer::Kind k) { return k == RecipeLayer::Profile ? "profile" : "override"; }

std::string note_for(const void* user, const std::string& path) {
  // The newest layer that wrote this path (or a field under / above it) is named in the error.
  const auto* layers = static_cast<const std::vector<RecipeLayer>*>(user);
  for (auto it = layers->rbegin(); it != layers->rend(); ++it)
    for (const auto& e : it->set)
      if (e.first == path || e.first.rfind(path + ".", 0) == 0 || path.rfind(e.first + ".", 0) == 0)
        return std::string(" (value set by ") + kind_word(it->kind) + " " + it->origin + ")";
  return "";
}

std::string with_origin(const std::string& origin, const Error& e) { return origin + ": " + e.what(); }

std::string canonical_sha(const Recipe& r) {
  const std::string t = json_write(recipe_to_json(r));
  return sha256_hex(t.data(), t.size());
}

}  // namespace

void check_path(const Recipe& r, const std::string& path, bool allow_prefix) {
  const std::vector<std::string> s = split(path);
  for (const auto& x : s) require(!x.empty(), SS_E_RECIPE, "path \"" + path + "\": empty segment");
  // need(n): the path continues past n segments; at the end, a prefix is fine only when allowed.
  auto end_here = [&](size_t n, bool leaf) {
    if (s.size() == n) {
      require(leaf || allow_prefix, SS_E_RECIPE, "path \"" + path + "\" names a table, not a settable field");
      return true;
    }
    require(!leaf, SS_E_RECIPE, "path \"" + path + "\": \"" + s[n - 1] + "\" is a field, nothing is under it");
    return false;
  };
  auto bad = [&](const std::string& why) { fail(SS_E_RECIPE, "path \"" + path + "\": " + why); };
  const std::string& top = s[0];
  if (top == "limits") {
    if (end_here(1, false)) return;
    if (s[1] != "host_free_ram_floor_gib" && s[1] != "host_free_commit_floor_gib") bad("not a settable limits field");
    end_here(2, true);
    return;
  }
  if (top == "match") {
    if (end_here(1, false)) return;
    if (s[1] != "devices" && s[1] != "links") bad("only match.devices.<set>.* and match.links.<link>.* are settable");
    if (end_here(2, false)) return;
    if (s[1] == "devices") {
      if (!r.device(s[2])) bad("unknown device set \"" + s[2] + "\"");
      if (end_here(3, false)) return;
      if (!one_of(s[3], {"name", "memory_min_gib", "host_ram_min_gib", "disk_read_gbytes_ps_min", "h2d_gbytes_ps_min"}))
        bad("\"" + s[3] + "\" is not settable (structure belongs to the recipe)");
    } else {
      if (!r.link(s[2])) bad("unknown link \"" + s[2] + "\"");
      if (end_here(3, false)) return;
      if (s[3] != "gbps_min") bad("\"" + s[3] + "\" is not settable (structure belongs to the recipe)");
    }
    end_here(4, true);
    return;
  }
  if (top == "routes") {
    if (end_here(1, false)) return;
    const RRoute* rt = r.route(s[1]);
    if (!rt) bad("unknown route \"" + s[1] + "\"");
    if (end_here(2, false)) return;
    if (s[2] == "fallback" || s[2] == "enabled") {
      end_here(3, true);
      return;
    }
    if (s[2] == "when") {
      if (end_here(3, false)) return;
      if (!one_of(s[3], {"prompt_tokens_min", "prompt_tokens_max", "batch_min", "batch_max"}))
        bad("unknown when field \"" + s[3] + "\"");
      end_here(4, true);
      return;
    }
    if (s[2] != "placement") bad("\"" + s[2] + "\" is not settable (structure belongs to the recipe)");
    if (end_here(3, false)) return;
    if (std::find(rt->devices.begin(), rt->devices.end(), s[3]) == rt->devices.end())
      bad("\"" + s[3] + "\" is not one of the route's devices");
    if (end_here(4, false)) return;
    if (placement_field(s[4])) {
      end_here(5, true);
      return;
    }
    if (s[4] != "units") bad("unknown placement field \"" + s[4] + "\"");
    if (end_here(5, false)) return;
    const RPlacement* p = rt->placement(s[3]);
    bool found = false;
    for (const auto& g : p ? p->groups : std::vector<RGroup>{}) found = found || g.id == s[5];
    if (!found) bad("unknown unit group \"" + s[5] + "\" in the recipe's placement");
    if (end_here(6, false)) return;
    if (!one_of(s[6], {"layers", "rank", "list", "resident", "pinned"}))
      bad("\"" + s[6] + "\" is not settable (structure belongs to the recipe)");
    end_here(7, true);
    return;
  }
  bad("\"" + top + "\" is not settable (structure belongs to the recipe)");
}

void set_path(Json& doc, const std::string& path, const Json& value) {
  std::vector<std::string> s = split(path);
  const std::string leaf = s.back();
  s.pop_back();
  Json* parent = walk(doc, s, true);
  require(parent && parent->is_object(), SS_E_RECIPE, "path \"" + path + "\": not found in the recipe");
  (*parent)[leaf] = value;
}

bool pin_covers(const std::string& pin, const std::string& path) {
  return path == pin || path.rfind(pin + ".", 0) == 0;
}

RecipeStack::RecipeStack(const std::string& text, const std::string& origin) : origin_(origin) {
  sha_ = sha256_hex(text.data(), text.size());
  try {
    doc_ = recipe_doc_parse(text);
    base_ = recipe_from_json(doc_);
  } catch (const Error& e) {
    fail(e.status(), with_origin(origin, e));
  }
  eff_ = base_;
  eff_sha_ = canonical_sha(eff_);
}

RecipeLayer RecipeStack::read_layer(const std::string& text, const std::string& origin, RecipeLayer::Kind kind,
                                    const Json* identity) const {
  RecipeLayer L;
  L.kind = kind;
  L.origin = origin;
  const bool profile = kind == RecipeLayer::Profile;
  auto bad = [&](const std::string& msg) { fail(SS_E_RECIPE, origin + ": " + msg); };
  auto stale = [&](const std::string& msg) { bad("profile stale: " + msg); };
  Json doc;
  try {
    doc = recipe_doc_parse(text);
  } catch (const Error& e) {
    fail(e.status(), with_origin(origin, e));
  }
  if (!doc.is_object()) bad("expected a table");
  for (const auto& kv : doc.obj) {
    if (kv.first.rfind("x-", 0) == 0) continue;
    bool ok = profile ? one_of(kv.first, {"schema", "recipe", "recipe_sha256", "source", "workload", "key", "probe",
                                          "max_drift", "set"})
                      : one_of(kv.first, {"schema", "recipe", "recipe_sha256", "source", "set"});
    if (!ok) bad("unknown key \"" + kv.first + "\"");
  }
  auto str_in = [&](const Json& o, const std::string& key, const std::string& where, bool required) -> std::string {
    const Json* v = o.find(key);
    if (!v) {
      if (required) bad("missing \"" + where + "\"");
      return "";
    }
    if (!v->is_string()) bad(where + ": expected a string");
    return v->text;
  };
  auto table = [&](const Json& o, const std::string& key, const std::string& where) -> const Json& {
    const Json* v = o.find(key);
    if (!v) bad("missing \"" + where + "\"");
    if (!v->is_object()) bad(where + ": expected a table");
    return *v;
  };
  const char* want = profile ? kProfileSchema : kOverridesSchema;
  if (str_in(doc, "schema", "schema", true) != want)
    bad(std::string("schema: expected \"") + want + "\" for " + (profile ? "a profile" : "overrides"));
  const std::string rname = str_in(doc, "recipe", "recipe", true);
  if (rname != eff_.name) bad("recipe: \"" + rname + "\" is not this recipe (\"" + eff_.name + "\")");
  const std::string sha = str_in(doc, "recipe_sha256", "recipe_sha256", profile);
  if (!sha.empty() && sha != sha_) {
    const std::string why =
        "recipe_sha256 " + sha + " is not the loaded recipe's " + sha_ + " (made for another version of the recipe)";
    if (profile) stale(why);
    bad(why);
  }
  L.source = str_in(doc, "source", "source", false);

  if (profile) {
    // Every compatibility key is required and checked against the current identity (docs/recipes.md §6.1).
    require(identity != nullptr && identity->is_object(), SS_E_RECIPE,
            origin + ": a profile is applied against the current identity (probe report + workload + descriptors)");
    const std::string workload = str_in(doc, "workload", "workload", true);
    if (workload.empty()) bad("workload: must not be empty");
    const Json* iw = identity->find("workload");
    if (!iw || !iw->is_string()) stale("the identity has no workload class");
    if (iw->text != workload) stale("tuned for workload \"" + workload + "\", running \"" + iw->text + "\"");
    double max_drift = 0.10;
    if (const Json* v = doc.find("max_drift")) {
      char* end = nullptr;
      max_drift = v->is_number() ? std::strtod(v->text.c_str(), &end) : -1;
      if (!v->is_number() || !end || *end || !(max_drift >= 0 && max_drift <= 1)) bad("max_drift: expected 0..1");
    }
    Binding b;
    try {
      b = recipe_bind(eff_, *identity);
    } catch (const Error& e) {
      stale(std::string("the hardware check fails: ") + e.what());
    }
    const Json& key = table(doc, "key", "key");
    for (const auto& kv : key.obj)
      if (kv.first.rfind("x-", 0) != 0 &&
          !one_of(kv.first, {"engine_sha256", "model_sha256", "devices", "drivers", "descriptors"}))
        bad("key: unknown key \"" + kv.first + "\"");
    for (const char* k : {"engine_sha256", "model_sha256"}) {
      const std::string v = str_in(key, k, std::string("key.") + k, true);
      const Json* p = identity->find(k);
      if (!p || !p->is_string()) stale(std::string("the identity has no ") + k);
      if (p->text != v) stale(std::string("key.") + k + " is " + v + ", probed " + p->text);
    }
    auto ids = [&](const Json& a, const std::string& where) {
      if (!a.is_array()) bad(where + ": expected an array of strings");
      std::string out;
      for (const Json& x : a.arr) {
        if (!x.is_string()) bad(where + ": expected an array of strings");
        out += (out.empty() ? "" : ",") + x.text;
      }
      return out;
    };
    const Json& kdev = table(key, "devices", "key.devices");
    const Json& kdrv = table(key, "drivers", "key.drivers");
    const Json& kdesc = table(key, "descriptors", "key.descriptors");
    const Json* idesc = identity->find("descriptors");
    const Json& pr = table(doc, "probe", "probe");
    for (const Json* t : {&kdev, &kdrv, &kdesc, &pr})
      for (const auto& kv : t->obj)
        if (kv.first.rfind("x-", 0) != 0 && !eff_.device(kv.first)) bad("unknown device set \"" + kv.first + "\"");
    for (const auto& set : b.sets) {
      const RDevice* dev = eff_.device(set.first);
      const std::vector<BoundMember>& mem = set.second;
      std::string got;
      for (const auto& m : mem) got += (got.empty() ? "" : ",") + m.id;
      const Json* d = kdev.find(set.first);
      if (!d) bad("missing \"key.devices." + set.first + "\"");
      const std::string want_ids = ids(*d, "key.devices." + set.first);
      if (want_ids != got) stale("key.devices." + set.first + " is [" + want_ids + "], bound [" + got + "]");
      if (dev->kind != SS_BACKEND_CPU) {
        const std::string drv = str_in(kdrv, set.first, "key.drivers." + set.first, true);
        for (const auto& m : mem)
          if (m.driver != drv)
            stale("key.drivers." + set.first + " is " + drv + ", probed " +
                  (m.driver.empty() ? std::string("nothing") : m.driver) + " on " + m.id);
      }
      const Json* ds = kdesc.find(set.first);
      if (!ds) bad("missing \"key.descriptors." + set.first + "\"");
      const std::string want_desc = ids(*ds, "key.descriptors." + set.first);
      const Json* cur = idesc && idesc->is_object() ? idesc->find(set.first) : nullptr;
      if (!cur) stale("the identity has no descriptors for \"" + set.first + "\"");
      const std::string now_desc = ids(*cur, "identity descriptors." + set.first);
      if (want_desc != now_desc) stale("key.descriptors." + set.first + " differ from the descriptors in use");
      const Json* ps = pr.find(set.first);
      if (!ps || !ps->is_object()) bad("missing table \"probe." + set.first + "\"");
      for (const auto& kv : ps->obj)
        if (kv.first.rfind("x-", 0) != 0 && kv.first != "disk_read_gbytes_ps" && kv.first != "h2d_gbytes_ps")
          bad("probe." + set.first + ": unknown key \"" + kv.first + "\"");
      std::vector<const char*> needed = {"disk_read_gbytes_ps"};
      if (dev->kind == SS_BACKEND_CUDA) needed.push_back("h2d_gbytes_ps");
      for (const char* f : needed) {
        const std::string w = "probe." + set.first + "." + f;
        const Json* v = ps->find(f);
        if (!v) bad("missing \"" + w + "\"");
        char* end = nullptr;
        double tuned = v->is_number() ? std::strtod(v->text.c_str(), &end) : -1;
        if (!v->is_number() || !end || *end || !std::isfinite(tuned) || tuned <= 0) bad(w + ": expected a number > 0");
        double now = -1;
        bool all = true;
        for (const auto& m : mem) {
          double x = std::string(f) == "disk_read_gbytes_ps" ? m.disk_read_gbytes_ps : m.h2d_gbytes_ps;
          all = all && x >= 0;
          now = now < 0 ? x : std::min(now, x);
        }
        if (!all) stale(w + " is set but the probe has no " + f + " for every member");
        if (std::fabs(now - tuned) > max_drift * tuned) {
          char buf[160];
          std::snprintf(buf, sizeof buf, " drifted: tuned %.4g, probed %.4g (more than %.0f %%)", tuned, now,
                        100 * max_drift);
          stale(w + buf);
        }
      }
    }
  }
  const Json& set = table(doc, "set", "set");
  for (const auto& kv : set.obj) {
    try {
      check_path(base_, kv.first, false);
    } catch (const Error& e) {
      fail(e.status(), with_origin(origin, e));
    }
    L.set.push_back({kv.first, kv.second});
  }
  return L;
}

void RecipeStack::rebuild(const std::vector<RecipeLayer>& layers) {
  // Precedence: recipe < profile (except pinned paths) < overrides in order. Applied to a copy; committed only when
  // the result validates, so a refused layer leaves the effective recipe as it was.
  std::vector<const RecipeLayer*> order;
  for (const auto& L : layers)
    if (L.kind == RecipeLayer::Profile) order.push_back(&L);
  for (const auto& L : layers)
    if (L.kind == RecipeLayer::Override) order.push_back(&L);
  // A profile is tuned for its budgets: no override may change a budget of a placement it tunes, or the hardware match.
  std::vector<std::string> tuned;
  for (const RecipeLayer* L : order)
    if (L->kind == RecipeLayer::Profile)
      for (const auto& e : L->set) {
        const std::vector<std::string> s = split(e.first);
        if (s.size() >= 4 && s[0] == "routes" && s[2] == "placement")
          tuned.push_back(s[0] + "." + s[1] + ".placement." + s[3]);
      }
  bool has_profile = order.size() && order[0]->kind == RecipeLayer::Profile;
  for (const RecipeLayer* L : order) {
    if (L->kind != RecipeLayer::Override || !has_profile) continue;
    for (const auto& e : L->set) {
      const std::vector<std::string> s = split(e.first);
      bool budget = s.size() == 5 &&
                    one_of(s[4], {"device_gib", "host_pinned_gib", "workspace_gib", "max_context_tokens", "max_batch",
                                  "margin_gib", "margin_fraction", "dynamic_slots_per_layer", "ring_slots",
                                  "pinned_backing", "host_spare_units", "staging_buffers", "staging_mib",
                                  "tier_mode", "readahead_units"});
      bool hit = s[0] == "match";
      for (const auto& t : tuned) hit = hit || (budget && pin_covers(t, e.first));
      require(!hit, SS_E_RECIPE,
              L->origin + ": " + e.first + " changes what the profile " + order[0]->origin +
                  " was tuned with; drop the profile (run untuned) or retune");
    }
  }
  Json doc = doc_;
  for (const RecipeLayer* L : order)
    for (const auto& e : L->set) {
      bool pinned = false;
      for (const auto& p : base_.pin) pinned = pinned || pin_covers(p, e.first);
      if (L->kind == RecipeLayer::Profile && pinned) continue;
      set_path(doc, e.first, e.second);
    }
  Recipe eff;
  try {
    eff = recipe_from_json(doc, NoteFn{note_for, &layers});
  } catch (const Error& e) {
    fail(e.status(), with_origin(origin_ + " + layers", e));
  }
  eff_ = std::move(eff);
  eff_sha_ = canonical_sha(eff_);
  layers_ = layers;
}

void RecipeStack::add_profile(const std::string& text, const std::string& origin, const Json& probe) {
  for (const auto& L : layers_)
    require(L.kind != RecipeLayer::Profile, SS_E_RECIPE,
            origin + ": a profile is already applied (" + L.origin + "); at most one");
  std::vector<RecipeLayer> next = layers_;
  next.push_back(read_layer(text, origin, RecipeLayer::Profile, &probe));
  rebuild(next);
}

void RecipeStack::add_overrides(const std::string& text, const std::string& origin) {
  std::vector<RecipeLayer> next = layers_;
  next.push_back(read_layer(text, origin, RecipeLayer::Override, nullptr));
  rebuild(next);
}

void RecipeStack::set(const std::string& path, const std::string& value_text, const std::string& origin) {
  check_path(base_, path, false);
  Json v;
  try {
    Json t = toml_parse("v = " + value_text + "\n");
    v = *t.find("v");
  } catch (const Error& e) {
    fail(SS_E_RECIPE, origin + ": " + path + " = " + value_text + ": not a TOML value (" + e.what() + ")");
  }
  RecipeLayer L;
  L.kind = RecipeLayer::Override;
  L.origin = origin;
  L.set.push_back({path, v});
  std::vector<RecipeLayer> next = layers_;
  next.push_back(L);
  rebuild(next);
}

Json RecipeStack::explain() const {
  Json eff = recipe_to_json(eff_);
  std::vector<std::pair<std::string, const Json*>> fields;
  leaves(eff, "", fields);
  Json out = Json::object();
  out["schema"] = Json::string("strix-stream/recipe-explain-v1");
  out["recipe"] = Json::string(eff_.name);
  out["recipe_sha256"] = Json::string(sha_);
  Json ls = Json::array();
  Json o0 = Json::object();
  o0["kind"] = Json::string("recipe");
  o0["origin"] = Json::string(origin_);
  ls.push(o0);
  for (const auto& L : layers_) {
    Json o = Json::object();
    o["kind"] = Json::string(kind_word(L.kind));
    o["origin"] = Json::string(L.origin);
    o["source"] = Json::string(L.source);
    ls.push(o);
  }
  out["layers"] = ls;
  Json arr = Json::array();
  for (const auto& f : fields) {
    // Every writer of this path, lowest precedence first: recipe, profile, overrides in order.
    struct W {
      std::string from, origin, why;
      const Json* value;
    };
    std::vector<W> writers;
    if (const Json* raw = walk(const_cast<Json&>(doc_), split(f.first), false))
      writers.push_back({"recipe", origin_, "", raw});
    for (int pass = 0; pass < 2; ++pass)
      for (const auto& L : layers_) {
        if ((pass == 0) != (L.kind == RecipeLayer::Profile)) continue;
        for (const auto& e : L.set) {
          if (e.first != f.first) continue;
          std::string why;
          if (L.kind == RecipeLayer::Profile)
            for (const auto& p : base_.pin)
              if (pin_covers(p, e.first)) why = "pinned by recipe: " + p;
          writers.push_back({kind_word(L.kind), L.origin, why, &e.second});
        }
      }
    int win = -1;
    for (int i = int(writers.size()) - 1; i >= 0 && win < 0; --i)
      if (writers[size_t(i)].why.empty()) win = i;
    Json e = Json::object();
    e["path"] = Json::string(f.first);
    e["value"] = *f.second;
    e["from"] = Json::string(win < 0 ? "default" : writers[size_t(win)].from);
    if (win >= 0) e["origin"] = Json::string(writers[size_t(win)].origin);
    Json sh = Json::array();
    for (int i = 0; i < int(writers.size()); ++i) {
      if (i == win) continue;
      Json s = Json::object();
      s["from"] = Json::string(writers[size_t(i)].from);
      s["origin"] = Json::string(writers[size_t(i)].origin);
      s["value"] = *writers[size_t(i)].value;
      s["reason"] = Json::string(writers[size_t(i)].why.empty() ? "overridden" : writers[size_t(i)].why);
      sh.push(s);
    }
    if (!sh.arr.empty()) e["shadowed"] = sh;
    arr.push(e);
  }
  out["fields"] = arr;
  return out;
}

}  // namespace ss
