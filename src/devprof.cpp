// Device profiles: strix/device-profile-v1 JSON (canonical writer, strict reader), the override merge and the
// detection order (host -> runtimes -> class -> override).
#include "devprof.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <set>
#include <sstream>

#include "errors.hpp"
#include "xxh64.hpp"

namespace ts::devprof {

namespace {

[[noreturn]] void bad(const std::string& m) { fail(SS_E_RECIPE, "device profile: " + m); }

// Bandwidths: 0 = unknown = null; else at most 4 significant decimals so the text stays readable and canonical.
Json gbps(double v) {
  if (v == 0) return Json{};
  char b[40];
  std::snprintf(b, sizeof b, "%.4f", v);
  std::string t = b;
  while (t.back() == '0') t.pop_back();
  if (t.back() == '.') t.pop_back();
  Json j;
  j.kind = Json::Kind::Number;
  j.text = t;
  return j;
}
Json strings(const std::vector<std::string>& v) {
  Json a = Json::array();
  for (const auto& s : v) a.push(Json::string(s));
  return a;
}

Json device_json(const Device& d) {
  Json j = Json::object();
  j["api"] = Json::string(d.api);
  j["id"] = Json::string(d.id);
  j["name"] = Json::string(d.name);
  j["arch"] = Json::string(d.arch);
  j["compute_units"] = Json::number(uint64_t(d.compute_units));
  j["wave"] = Json::number(uint64_t(d.wave));
  j["lds_bytes"] = Json::number(d.lds_bytes);
  j["memory_bytes"] = Json::number(d.memory_bytes);
  j["gtt_bytes"] = Json::number(d.gtt_bytes);
  j["visible_bytes"] = Json::number(d.visible_bytes);
  j["mem_gbps"] = gbps(d.mem_gbps);
  j["h2d_gbps"] = gbps(d.h2d_gbps);
  j["cc_major"] = Json::number(int64_t(d.cc_major));
  j["cc_minor"] = Json::number(int64_t(d.cc_minor));
  j["pcie_gen"] = Json::number(int64_t(d.pcie_gen));
  j["pcie_width"] = Json::number(int64_t(d.pcie_width));
  j["driver"] = Json::string(d.driver);
  return j;
}
Json link_json(const Link& l) {
  Json j = Json::object();
  j["kind"] = Json::string(l.kind);
  j["count"] = Json::number(uint64_t(l.count));
  j["gbps_measured"] = gbps(l.gbps_measured);
  return j;
}
Json nvme_json(const Nvme& n) {
  Json j = Json::object();
  j["path"] = Json::string(n.path);
  j["model"] = Json::string(n.model);
  j["read_gbps"] = gbps(n.read_gbps);
  return j;
}

// Strict object reading: every key must be known; each getter marks its key.
struct Obj {
  const Json& j;
  std::string where;
  std::set<std::string> seen;
  Obj(const Json& v, std::string w) : j(v), where(std::move(w)) {
    if (!j.is_object()) bad(where + ": expected an object");
  }
  const Json* get(const char* k) {
    seen.insert(k);
    return j.find(k);
  }
  void done() const {
    for (const auto& kv : j.obj)
      if (!seen.count(kv.first)) bad(where + "." + kv.first + ": unknown key");
  }
  std::string path(const char* k) const { return where + "." + k; }
  void u64(const char* k, uint64_t& v) {
    if (const Json* x = get(k)) v = json_u64(*x, path(k));
  }
  void u32(const char* k, uint32_t& v) {
    if (const Json* x = get(k)) {
      const uint64_t t = json_u64(*x, path(k));
      if (t > 0xFFFFFFFFu) bad(path(k) + ": out of range");
      v = uint32_t(t);
    }
  }
  void i32(const char* k, int32_t& v) {
    if (const Json* x = get(k)) {
      const int64_t t = json_i64(*x, path(k));
      if (t < -2147483647 - 1 || t > 2147483647) bad(path(k) + ": out of range");
      v = int32_t(t);
    }
  }
  void f64(const char* k, double& v) {
    if (const Json* x = get(k)) {
      if (x->is_null()) {
        v = 0;
        return;
      }
      v = json_f64(*x, path(k));
      if (!(v >= 0)) bad(path(k) + ": expected a bandwidth >= 0 or null");
    }
  }
  void str(const char* k, std::string& v) {
    if (const Json* x = get(k)) v = json_str(*x, path(k));
  }
  void boolean(const char* k, bool& v) {
    if (const Json* x = get(k)) v = json_bool(*x, path(k));
  }
  void strs(const char* k, std::vector<std::string>& v) {
    if (const Json* x = get(k)) {
      if (!x->is_array()) bad(path(k) + ": expected an array of strings");
      v.clear();
      for (size_t i = 0; i < x->arr.size(); ++i) v.push_back(json_str(x->arr[i], path(k) + "[" + std::to_string(i) + "]"));
    }
  }
};

Device device_from(const Json& j, const std::string& where) {
  Device d;
  Obj o(j, where);
  o.str("api", d.api);
  if (d.api != "hip" && d.api != "cuda" && d.api != "cpu") bad(where + ".api: expected \"hip\", \"cuda\" or \"cpu\"");
  o.str("id", d.id);
  o.str("name", d.name);
  o.str("arch", d.arch);
  o.u32("compute_units", d.compute_units);
  o.u32("wave", d.wave);
  o.u64("lds_bytes", d.lds_bytes);
  o.u64("memory_bytes", d.memory_bytes);
  o.u64("gtt_bytes", d.gtt_bytes);
  o.u64("visible_bytes", d.visible_bytes);
  o.f64("mem_gbps", d.mem_gbps);
  o.f64("h2d_gbps", d.h2d_gbps);
  o.i32("cc_major", d.cc_major);
  o.i32("cc_minor", d.cc_minor);
  o.i32("pcie_gen", d.pcie_gen);
  o.i32("pcie_width", d.pcie_width);
  o.str("driver", d.driver);
  o.done();
  return d;
}
Link link_from(const Json& j, const std::string& where) {
  Link l;
  Obj o(j, where);
  o.str("kind", l.kind);
  if (l.kind != "usb4" && l.kind != "ethernet" && l.kind != "pcie" && l.kind != "other")
    bad(where + ".kind: expected \"usb4\", \"ethernet\", \"pcie\" or \"other\"");
  o.u32("count", l.count);
  o.f64("gbps_measured", l.gbps_measured);
  o.done();
  return l;
}
Nvme nvme_from(const Json& j, const std::string& where) {
  Nvme n;
  Obj o(j, where);
  o.str("path", n.path);
  o.str("model", n.model);
  o.f64("read_gbps", n.read_gbps);
  o.done();
  return n;
}
template <class T>
void list(Obj& o, const char* k, std::vector<T>& v, T (*from)(const Json&, const std::string&)) {
  if (const Json* x = o.get(k)) {
    if (!x->is_array()) bad(o.path(k) + ": expected an array");
    v.clear();
    for (size_t i = 0; i < x->arr.size(); ++i) v.push_back(from(x->arr[i], o.path(k) + "[" + std::to_string(i) + "]"));
  }
}

// Deep merge of a partial text over the profile's JSON: objects by key, arrays by index (an element past the end
// starts from the default element), scalars replaced. Every replaced leaf is recorded as its path.
void merge_into(Json& base, const Json& over, const std::string& path, const Json* elem_default,
                std::vector<std::string>& leaves) {
  if (over.is_object() && (base.is_object() || base.is_null())) {
    if (base.is_null()) base = Json::object();
    for (const auto& kv : over.obj) {
      const std::string p = path.empty() ? kv.first : path + "." + kv.first;
      // provenance stays the target's: a saved full profile can be passed back as an override
      if (path.empty() && (kv.first == "overrides" || kv.first == "notices" || kv.first == "sources")) continue;
      if (path.empty() && kv.first == "schema") {
        if (!kv.second.is_string() || kv.second.text != SS_DEVPROF_SCHEMA) bad("schema: expected " SS_DEVPROF_SCHEMA);
        continue;
      }
      const Json* d = nullptr;
      static const Json dev = device_json(Device{}), lnk = link_json(Link{}), nv = nvme_json(Nvme{});
      if (p == "devices") d = &dev;
      if (p == "links") d = &lnk;
      if (p == "host.nvme") d = &nv;
      Json& slot = base.obj[kv.first];
      if (path.empty() && (kv.first == "evidence" || kv.first == "limits")) {  // flat maps, merged by key
        if (!kv.second.is_object()) bad(kv.first + ": expected an object");
        if (!slot.is_object()) slot = Json::object();
        for (const auto& e : kv.second.obj) {
          slot.obj[e.first] = e.second;
          leaves.push_back(kv.first + "." + e.first);
        }
        continue;
      }
      merge_into(slot, kv.second, p, d, leaves);
    }
    return;
  }
  if (over.is_array() && elem_default) {
    if (!base.is_array()) base = Json::array();
    for (size_t i = 0; i < over.arr.size(); ++i) {
      if (i >= base.arr.size()) base.push(*elem_default);
      merge_into(base.arr[i], over.arr[i], path + "[" + std::to_string(i) + "]", nullptr, leaves);
    }
    return;
  }
  if (over.is_object() || over.is_array()) bad(path + ": cannot merge " + (over.is_object() ? "an object" : "an array") +
                                               " over a value");
  base = over;
  leaves.push_back(path);
}

std::string env(const char* k) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string();
}

}  // namespace

Json to_json(const Profile& p) {
  Json j = Json::object();
  j["schema"] = Json::string(SS_DEVPROF_SCHEMA);
  j["class"] = Json::string(p.cls);
  j["detected"] = Json::boolean(p.detected);
  j["generic"] = Json::boolean(p.generic);
  Json h = Json::object();
  h["ram_bytes"] = Json::number(p.ram_bytes);
  h["ram_available_bytes"] = Json::number(p.ram_available_bytes);
  h["unified"] = Json::boolean(p.unified);
  h["cpu_cores"] = Json::number(uint64_t(p.cpu_cores));
  Json nv = Json::array();
  for (const auto& n : p.nvme) nv.push(nvme_json(n));
  h["nvme"] = nv;
  j["host"] = h;
  Json ds = Json::array();
  for (const auto& d : p.devices) ds.push(device_json(d));
  j["devices"] = ds;
  Json ls = Json::array();
  for (const auto& l : p.links) ls.push(link_json(l));
  j["links"] = ls;
  Json m = Json::object();
  m["host_floor_bytes"] = Json::number(p.host_floor_bytes);
  m["device_reserve_bytes"] = Json::number(p.device_reserve_bytes);
  j["margins"] = m;
  Json lim = Json::object();
  for (const auto& kv : p.limits) lim[kv.first] = Json::number(kv.second);
  j["limits"] = lim;
  j["evidence"] = p.evidence.is_object() ? p.evidence : Json::object();
  j["overrides"] = strings(p.overrides);
  j["notices"] = strings(p.notices);
  j["sources"] = strings(p.sources);
  return j;
}

std::string to_text(const Profile& p) { return json_write(to_json(p)); }

std::string hash(const Profile& p) {
  const std::string t = to_text(p);
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(xxh64(t.data(), t.size())));
  return b;
}

namespace {
Profile read_profile(const Json& j) {
  Profile p;
  Obj o(j, "profile");
  const Json* s = o.get("schema");
  if (!s || !s->is_string() || s->text != SS_DEVPROF_SCHEMA) bad("schema: expected " SS_DEVPROF_SCHEMA);
  o.str("class", p.cls);
  o.boolean("detected", p.detected);
  o.boolean("generic", p.generic);
  if (const Json* h = o.get("host")) {
    Obj ho(*h, "host");
    ho.u64("ram_bytes", p.ram_bytes);
    ho.u64("ram_available_bytes", p.ram_available_bytes);
    ho.boolean("unified", p.unified);
    ho.u32("cpu_cores", p.cpu_cores);
    list(ho, "nvme", p.nvme, nvme_from);
    ho.done();
  }
  list(o, "devices", p.devices, device_from);
  list(o, "links", p.links, link_from);
  if (const Json* m = o.get("margins")) {
    Obj mo(*m, "margins");
    mo.u64("host_floor_bytes", p.host_floor_bytes);
    mo.u64("device_reserve_bytes", p.device_reserve_bytes);
    mo.done();
  }
  if (const Json* l = o.get("limits")) {
    if (!l->is_object()) bad("limits: expected an object");
    for (const auto& kv : l->obj) p.limits[kv.first] = json_u64(kv.second, "limits." + kv.first);
  }
  if (const Json* e = o.get("evidence")) {
    if (!e->is_object()) bad("evidence: expected an object");
    for (const auto& kv : e->obj) json_str(kv.second, "evidence." + kv.first);
    p.evidence = *e;
  }
  o.strs("overrides", p.overrides);
  o.strs("notices", p.notices);
  o.strs("sources", p.sources);
  o.done();
  if (p.ram_available_bytes > p.ram_bytes && p.ram_bytes) bad("host.ram_available_bytes is above host.ram_bytes");
  return p;
}
}  // namespace

Profile from_json(const Json& j) {
  try {
    return read_profile(j);
  } catch (const Error& e) {  // the typed getters' SS_E_DESC: every bad profile text is SS_E_RECIPE
    if (e.status() == SS_E_RECIPE) throw;
    bad(e.what());
  }
}

void merge(Profile& p, const Json& partial, const std::string& origin) {
  if (!partial.is_object()) bad(origin + ": expected an object");
  Json base = to_json(p);
  std::vector<std::string> leaves;
  merge_into(base, partial, "", nullptr, leaves);
  Profile q = from_json(base);  // validates the result; p is untouched on a throw
  // A changed identity (the class, or device 0's api / arch / name / memory) takes back everything the old class
  // filled - except what this override sets itself - and picks the class again: qualified numbers never survive a
  // change of machine.
  bool forced = false, ident = false;
  for (const auto& l : leaves) {
    forced = forced || l == "class";
    for (const char* f : {"api", "arch", "name", "memory_bytes", "gtt_bytes", "visible_bytes"})
      ident = ident || l == std::string("devices[0].") + f;
  }
  if (forced || ident) {
    unclass(q, leaves);
    apply_class(q, forced ? q.cls : std::string());
  }
  for (const auto& l : leaves) q.overrides.push_back(origin + ":" + l);
  p = std::move(q);
}

// Detection never fails: a source that throws becomes a notice.
template <class F>
void source(Profile& p, const char* what, F&& f) {
  try {
    f();
  } catch (const std::exception& e) {
    p.notices.push_back(std::string(what) + ": " + e.what());
  }
}

Profile detect(const DetectOptions& o, const HipApi* hip, const CudaApi* cuda) {
  Profile p;
  p.detected = true;
#if defined(_WIN32)
  if (o.root.empty() || o.root == "/") source(p, "windows", [&] { detect_windows_host(p); });
  else source(p, "linux", [&] { detect_linux(p, o.root); });  // a fake tree (tests)
#else
  source(p, "linux", [&] { detect_linux(p, o.root.empty() ? std::string("/") : o.root); });
#endif
  const std::string cache = (o.flags & SS_DEVPROF_NO_CACHE) ? std::string()  // empty = no cache
                            : o.cache_dir.empty()            ? default_gpu_cache_dir()
                                                             : o.cache_dir;
  source(p, "hip", [&] {
    if (hip) {
      detect_hip(p, *hip);
    } else if (!(o.flags & SS_DEVPROF_NO_RUNTIME)) {
      const HipApi h = hip_runtime();
      if (h.present()) detect_hip(p, h);
    }
  });
  source(p, "cuda", [&] {
    if (cuda) {
      detect_cuda(p, *cuda, cache);
    } else if (!(o.flags & SS_DEVPROF_NO_RUNTIME)) {
      const CudaApi c = cuda_runtime();
      if (c.present()) detect_cuda(p, c, cache);
    }
  });
  apply_class(p);
  std::string ov = o.override_path;
  if (ov.empty() && !(o.flags & SS_DEVPROF_NO_ENV)) ov = env("STRIX_DEVICE_PROFILE");
  if (!ov.empty()) {
    std::ifstream f(ov, std::ios::binary);
    require(bool(f), SS_E_IO, "device profile: cannot open the override " + ov);
    std::stringstream t;
    t << f.rdbuf();
    Json j;
    try {
      j = json_parse(t.str());
    } catch (const Error& e) {
      bad(ov + ": " + e.what());
    }
    merge(p, j, ov);
  }
  return p;
}

}  // namespace ts::devprof
