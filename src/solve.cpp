#include "solve.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "components.hpp"
#include "errors.hpp"
#include "place.hpp"
#include "sha256.hpp"

namespace ts {
namespace {

constexpr int64_t kBound = 10000000;  // milli-misses and microseconds are <= 10^7 (§3.9): every product fits 64 bits

[[noreturn]] void bad(const std::string& where, const std::string& msg) {
  fail(SS_E_RECIPE, "solver: PL_E_SCHEMA " + where + ": " + msg);
}

bool int_text(const Json& v) {
  return v.is_number() && !v.text.empty() && v.text.size() <= 18 &&
         v.text.find_first_not_of("0123456789") == std::string::npos;
}

int64_t nonneg(const Json& v, const std::string& where, int64_t hi) {
  if (!int_text(v)) bad(where, "expected an integer >= 0");
  const int64_t x = json_i64(v, where);
  if (x > hi) bad(where, v.text + " is above " + std::to_string(hi));
  return x;
}

std::string sha_of(const Json& j) {
  const std::string t = json_write(j);
  return sha256_hex(t.data(), t.size());
}

// rdiv(n, d) = (2n + d) div (2d): round half up on non-negative integers.
int64_t rdiv(int64_t n, int64_t d) { return (2 * n + d) / (2 * d); }

using Knots = std::vector<std::pair<int64_t, int64_t>>;

// On a knot -> it ("knot"); between two -> one rounded linear interpolation ("interpolated"); above the last ->
// the last ("clamped_high"); below the first -> unpredicted (clamping there would overstate tok/s).
bool interp(const Knots& k, int64_t a, int64_t& value, const char*& tag) {
  if (a < k.front().first) return false;
  if (a >= k.back().first) {
    value = k.back().second;
    tag = a == k.back().first ? "knot" : "clamped_high";
    return true;
  }
  for (size_t i = 0; i + 1 < k.size(); ++i) {
    const auto& p = k[i];
    const auto& q = k[i + 1];
    if (a == p.first) {
      value = p.second;
      tag = "knot";
      return true;
    }
    if (p.first < a && a < q.first) {
      value = rdiv(p.second * (q.first - p.first) + (q.second - p.second) * (a - p.first), q.first - p.first);
      tag = "interpolated";
      return true;
    }
  }
  return false;
}

std::map<std::string, Knots> curves_of(const Json& in) {
  std::map<std::string, Knots> out;
  const Json* c = in.find("curves");
  if (!c) return out;
  if (const Json* inner = c->find("curves")) c = inner;  // tierstream/miss-curve-v1 wrapper
  if (!c->is_object()) bad("curves", "expected an object keyed by \"<lang>/R<r>\"");
  for (const auto& kv : c->obj) {
    const std::string w = "curves." + kv.first;
    if (!kv.second.is_array() || kv.second.arr.empty()) bad(w, "expected knots [[active_units, milli_misses]]");
    Knots k;
    for (const Json& x : kv.second.arr) {
      if (!x.is_array() || x.arr.size() < 2) bad(w, "a knot is [active_units, milli_misses]");
      k.push_back({nonneg(x.arr[0], w, int64_t(1) << 32), nonneg(x.arr[1], w, kBound)});
    }
    for (size_t i = 1; i < k.size(); ++i)
      if (k[i].first <= k[i - 1].first) bad(w, "knots must ascend in active_units");
    out[kv.first] = k;
  }
  return out;
}

// A cost in [0, 10^7], or -1 when absent / out of range (the row is unpredicted).
int64_t cost(const Json* costs, const std::string& key, const std::string& a = "", const std::string& b = "") {
  const Json* v = costs ? costs->find(key) : nullptr;
  if (v && !a.empty()) v = v->is_object() ? v->find(a) : nullptr;
  if (v && !b.empty()) v = v->is_object() ? v->find(b) : nullptr;
  if (!v || !int_text(*v)) return -1;
  const int64_t x = json_i64(*v, key);
  return x <= kBound ? x : -1;
}

struct Prediction {
  bool ok = false;
  const char* tag = "unpredicted";
  int64_t total = 0, pinned = 0, file = 0, window = 0, tok = 0;
  std::vector<std::string> used;  // the cost fields the prediction used (for `estimates`)
};

// Calibration identity (§3.9, rev 17; reference golden/solver_ref.py): costs (`costs.applicability`) and curves
// (`curves_applicability`) predict only the configuration they were measured on. Each required key must be present,
// not "unknown", and equal to the choice's `calib`; `drafter` is compared for spec rows only (costs and curves),
// `file_store` for rows with FILE units only (costs). Keys outside the required sets are not compared. The curves must
// be out of sample (`evaluation` holdout | kfold); the probe's transfer figures must lie within +-10 % of the
// calibration run's (`costs.xfer_measured`; `file_read_mbps` for FILE rows only); FILE rows need policy lru; v1
// predicts sequences = 1 only.
const char* const kCostKeys[] = {"sequences",      "adapter",           "anchor_device", "assist_device",
                                 "xfer",           "cache_sha256",      "build_sha256",  "content_sha256",
                                 "arithmetic_sha256", "prefetch",       "tier_mode",     "driver",
                                 "platform",       "engram_store"};
const char* const kCurveKeys[] = {"sequences",    "adapter",      "policy",         "usage_sha256", "cache_sha256",
                                  "build_sha256", "content_sha256", "prefetch", "tier_mode"};

bool applies(const Json* app, const Json& calib, std::vector<const char*> keys, bool spec, bool file) {
  if (!app || !app->is_object() || app->obj.empty()) return false;
  if (spec) keys.push_back("drafter");
  if (file) keys.push_back("file_store");
  for (const char* k : keys) {
    const Json* v = app->find(k);
    if (!v || (v->is_string() && v->text == "unknown")) return false;  // a key whose value is unknown never matches
    const Json* c = calib.is_object() ? calib.find(k) : nullptr;
    if (!c || json_write(*c) != json_write(*v)) return false;
  }
  return true;
}

// Every probe figure within +-10 % of the calibration's (10 x |p - c| <= c, integers, c > 0), grouped (rev 18): the
// transfer figures of every device set the calibration names, and the store's read rate for FILE rows.
// rev 19: the groups are the choice's exec_sets (sorted); a set missing on either side fails.
bool within(const Json* probe, const Json* calib, std::vector<std::string> exec_sets, bool file) {
  if (!calib || !calib->is_object()) return false;
  std::sort(exec_sets.begin(), exec_sets.end());
  std::vector<std::pair<std::string, std::vector<const char*>>> groups;
  for (const std::string& s : exec_sets) groups.push_back({s, {"h2d_mbps", "d2h_mbps", "h2d_lat_ns"}});
  if (groups.empty()) return false;
  if (file) groups.push_back({"store", {"file_read_mbps"}});
  for (const auto& g : groups) {
    const Json* pg = probe && probe->is_object() ? probe->find(g.first) : nullptr;
    const Json* cg = calib->find(g.first);
    for (const char* n : g.second) {
      const Json* pv = pg && pg->is_object() ? pg->find(n) : nullptr;
      const Json* cv = cg && cg->is_object() ? cg->find(n) : nullptr;
      if (!pv || !cv || !int_text(*pv) || !int_text(*cv)) return false;
      const int64_t pp = json_i64(*pv, n), cc = json_i64(*cv, n);
      if (cc <= 0 || 10 * (pp > cc ? pp - cc : cc - pp) > cc) return false;
    }
  }
  return true;
}

std::vector<std::string> strings_of(const Json* v) {
  std::vector<std::string> out;
  if (v && v->is_array())
    for (const Json& x : v->arr)
      if (x.is_string()) out.push_back(x.text);
  return out;
}

// FILE rows (rev 18): holdout curves; the language table's sources are evaluation traces of the curve and disjoint from
// the pooled ranking table's sources (the cold set's FILE rate is measured out of sample).
bool file_sources_ok(const Json* capp, const std::vector<std::string>& lang_src, const std::vector<std::string>& pooled) {
  const Json* ev = capp ? capp->find("evaluation") : nullptr;
  if (!ev || !ev->is_string() || ev->text != "holdout" || lang_src.empty()) return false;
  const std::vector<std::string> eval = strings_of(capp->find("evaluation_sources"));
  for (const std::string& s : lang_src) {
    if (std::find(eval.begin(), eval.end(), s) == eval.end()) return false;
    if (std::find(pooled.begin(), pooled.end(), s) != pooled.end()) return false;
  }
  return true;
}

const Json* curves_applicability(const Json& in) {
  if (const Json* a = in.find("curves_applicability")) return a;
  const Json* c = in.find("curves");  // a tierstream/miss-curve-v1 document carries its own
  return c ? c->find("applicability") : nullptr;
}

// The calibration gate of one row (everything but the curve lookup and the arithmetic).
bool calibration_applies(const Json& in, const Json& calib, int64_t sequences, bool spec, bool file,
                         const Json* probe_xfer, const std::vector<std::string>& exec_sets,
                         const std::vector<std::string>& lang_src, const std::vector<std::string>& pooled_src) {
  const Json* costs = in.find("costs");
  const Json* capp = curves_applicability(in);
  if (sequences != 1) return false;
  if (!applies(costs ? costs->find("applicability") : nullptr, calib,
               std::vector<const char*>(std::begin(kCostKeys), std::end(kCostKeys)), spec, file))
    return false;
  if (!applies(capp, calib, std::vector<const char*>(std::begin(kCurveKeys), std::end(kCurveKeys)), spec, false))
    return false;
  const Json* ev = capp ? capp->find("evaluation") : nullptr;  // in-sample curves understate misses
  if (!ev || !ev->is_string() || (ev->text != "holdout" && ev->text != "kfold")) return false;
  if (ev->text == "holdout") {
    const Json* sources = capp->find("evaluation_sources");
    if (!sources || !sources->is_array() || sources->arr.empty()) return false;
    for (const Json& s : sources->arr) {
      if (!s.is_string() || s.text.size() != 64 ||
          s.text.find_first_not_of("0123456789abcdef") != std::string::npos)
        return false;
      if (std::find(pooled_src.begin(), pooled_src.end(), s.text) != pooled_src.end()) return false;
    }
  }
  if (!within(probe_xfer, costs ? costs->find("xfer_measured") : nullptr, exec_sets, file)) return false;
  if (file && !file_sources_ok(capp, lang_src, pooled_src)) return false;
  if (file) {  // the curves are measured at file_units = 0; only LRU keeps the total invariant to FILE homes
    const Json* pol = calib.is_object() ? calib.find("policy") : nullptr;
    if (!pol || !pol->is_string() || pol->text != "lru") return false;
  }
  return true;
}

// cold_sum: the sum of the file_units coldest units' micro-selections per token; -1 = not available.
Prediction predict(const std::map<std::string, Knots>& curves, const Json* costs, const std::string& mode, int64_t r,
                   int64_t ctx, const std::string& lang, int64_t active, int64_t cold_sum, bool file) {
  Prediction p;
  const std::string key = lang + "/R" + std::to_string(r), c = std::to_string(ctx), rs = std::to_string(r);
  auto it = curves.find(key);
  if (it == curves.end() || cold_sum < 0) return p;
  int64_t total = 0;
  const char* tag = "unpredicted";
  if (!interp(it->second, active, total, tag)) return p;
  // the FILE per-miss cost is needed only by FILE rows
  const int64_t mp = cost(costs, "miss_pinned_us"), mf = file ? cost(costs, "miss_file_us") : 0;
  const int64_t bd = cost(costs, "boundary_us", c), en = cost(costs, "engram_us", c);
  int64_t base = 0, accepted = 1000;
  p.used = {"boundary_us", "engram_us", "miss_pinned_us"};
  if (file) p.used.push_back("miss_file_us");
  if (mode == "plain") {
    base = cost(costs, "allhit_us", c);
    p.used.push_back("allhit_us");
  } else {
    p.used.insert(p.used.end(), {"draft_us", "verify_us", "accepted_milli"});
    const int64_t dr = cost(costs, "draft_us", rs), vr = cost(costs, "verify_us", c, rs);
    base = dr < 0 || vr < 0 ? -1 : dr + vr;
    accepted = cost(costs, "accepted_milli", key);
  }
  // every consumed scalar is checked on its own (cost(): an integer in [0, 10^7]); sums are not bounded (rev 18)
  if (mp < 0 || mf < 0 || bd < 0 || en < 0 || base < 0 || accepted < 0) return p;
  const int64_t cold = rdiv(cold_sum, 1000);
  p.file = std::min(r * cold, total);
  p.pinned = total - p.file;
  p.window = base + rdiv(p.pinned * mp, 1000) + rdiv(p.file * mf, 1000) + bd + en;
  if (p.window == 0) return p;
  p.total = total;
  p.tok = rdiv(1000000 * accepted, p.window);
  p.tag = tag;
  p.ok = true;
  return p;
}

// estimates: the cost fields a prediction used whose costs.provenance[field].estimate is true (sorted)
Json prediction_json(const Prediction& p, const Json* costs = nullptr) {
  Json e = Json::object();
  e["tag"] = Json::string(p.tag);
  std::vector<std::string> est;
  const Json* prov = p.ok && costs ? costs->find("provenance") : nullptr;
  if (prov && prov->is_object())
    for (const std::string& f : p.used)
      if (const Json* pf = prov->find(f))
        if (const Json* x = pf->is_object() ? pf->find("estimate") : nullptr)
          if (x->kind == Json::Kind::Bool && x->b) est.push_back(f);
  std::sort(est.begin(), est.end());
  Json ea = Json::array();
  for (const auto& f : est) ea.push(Json::string(f));
  e["estimates"] = ea;
  auto n = [&](int64_t v) { return p.ok ? Json::number(v) : Json(); };
  e["milli_misses_total"] = n(p.total);
  e["milli_misses_pinned"] = n(p.pinned);
  e["milli_misses_file"] = n(p.file);
  e["window_us"] = n(p.window);
  e["milli_tok_s"] = n(p.tok);
  return e;
}

struct Choice {
  std::string id, mode, lang;
  int64_t r = 1, ctx = 0, seq = 0;
};

Choice choice_of(const Json& c, size_t i) {
  const std::string w = "choices[" + std::to_string(i) + "]";
  if (!c.is_object()) bad(w, "expected an object");
  Choice x;
  auto s = [&](const char* k) {
    const Json* v = c.find(k);
    if (!v || !v->is_string() || v->text.empty()) bad(w, std::string("missing \"") + k + "\"");
    return v->text;
  };
  x.id = s("id");
  x.mode = s("mode");
  if (x.mode != "plain" && x.mode != "spec") bad(w + ".mode", "plain | spec");
  x.lang = s("lang");
  const Json* r = c.find("R");
  x.r = r ? nonneg(*r, w + ".R", 64) : 1;
  if (x.r < 1 || (x.mode == "plain" && x.r != 1)) bad(w + ".R", "plain is R = 1; spec R >= 1");
  const Json* ctx = c.find("ctx");
  if (!ctx) bad(w, "missing \"ctx\"");
  x.ctx = nonneg(*ctx, w + ".ctx", int64_t(1) << 40);
  if (const Json* q = c.find("sequences")) x.seq = nonneg(*q, w + ".sequences", 65536);
  return x;
}

// prefetch: components-v1 config's prefetch options (keys starting "prefetch") as a canonical string, "off" when
// none is set or every one is 0 / false.
std::string prefetch_of(const Components& c) {
  std::string out;
  bool on = false;
  if (c.config.is_object())
    for (const auto& kv : c.config.obj) {  // std::map: sorted keys
      if (kv.first.rfind("prefetch", 0) != 0) continue;
      const std::string v = json_write(kv.second);
      on = on || !(v == "0" || v == "false" || v == "\"off\"" || v == "\"\"");
      out += (out.empty() ? "" : ",") + kv.first + "=" + v;
    }
  return on ? out : "off";
}

// The probe rows of a placed device (its PCI address; alias rows included).
std::vector<const Json*> probe_rows(const Json& probe, const Json& dev) {
  std::vector<const Json*> out;
  const Json* pci = dev.find("pci");
  if (const Json* devs = probe.find("devices"))
    for (const Json& r : devs->arr) {
      const Json* rp = r.find("pci");
      if (pci && rp && rp->is_string() && normalize_pci(rp->text) == pci->text) out.push_back(&r);
    }
  return out;
}

// xfer (rev 17 §3.9) = sha256 of the canonical transfer profile tierstream/xfer-v1 built from the probe: per executing
// device in placement order {set, pci_path: [root port BDF, bridge BDFs..., device BDF], link: {gen, width} (the
// narrowest LnkCap over that path), via: chipset | direct, numa_node} from its probe rows (matched by PCI address,
// alias rows included), the host block (probe `xfer_host`: {cpu_model, sockets, numa_nodes, pinned_policy: node |
// interleave}) and the drafter link none | host-staged | peer (host-staged when the probe does not report peer
// access). Any missing field gives "unknown", which never matches.
std::string xfer_of(const Json& doc, const Json& probe, const std::string& drafter_role) {
  bool peer = false;
  if (const Json* p = probe.find("peer")) peer = p->kind == Json::Kind::Bool && p->b;
  Json devs = Json::array();
  for (const Json& d : doc.find("devices")->arr) {
    const Json *path = nullptr, *link = nullptr, *via = nullptr, *numa = nullptr;
    for (const Json* r : probe_rows(probe, d)) {
      if (const Json* v = r->find("pci_path"))
        if (v->is_array() && !v->arr.empty() &&
            std::all_of(v->arr.begin(), v->arr.end(), [](const Json& b) { return b.is_string() && !b.text.empty(); }))
          path = v;
      if (const Json* v = r->find("link"))
        if (v->is_object() && v->find("gen") && int_text(*v->find("gen")) && v->find("width") &&
            int_text(*v->find("width")))
          link = v;
      if (const Json* v = r->find("via"))
        if (v->is_string() && (v->text == "chipset" || v->text == "direct")) via = v;
      if (const Json* v = r->find("numa_node"))
        if (int_text(*v)) numa = v;
      if (d.find("role")->text != "anchor")
        if (const Json* p = r->find("peer")) peer = p->kind == Json::Kind::Bool && p->b;
    }
    if (!path || !link || !via || !numa) return "unknown";
    Json e = Json::object();
    e["set"] = *d.find("set");
    Json pp = Json::array();
    for (const Json& b : path->arr) pp.push(Json::string(normalize_pci(b.text).empty() ? b.text : normalize_pci(b.text)));
    e["pci_path"] = pp;
    e["link"] = Json::string("gen" + link->find("gen")->text + "x" + link->find("width")->text);
    e["via"] = *via;
    e["numa_node"] = *numa;
    devs.push(e);
  }
  const Json* host = probe.find("xfer_host");
  if (!host || !host->is_object()) return "unknown";
  Json h = Json::object();
  for (const char* k : {"cpu_model", "sockets", "numa_nodes", "pinned_policy"}) {
    const Json* v = host->find(k);
    if (!v || (v->is_string() ? v->text.empty() : !int_text(*v))) return "unknown";
    h[k] = *v;
  }
  if (h.find("pinned_policy")->text != "node" && h.find("pinned_policy")->text != "interleave") return "unknown";
  // the canonical profile as golden/gen_sv1.py writes it: {devices: [{set, pci_path, link: "gen<G>x<W>", via,
  // numa_node}], host, drafter_link}
  Json prof = Json::object();
  prof["devices"] = devs;
  prof["host"] = h;
  prof["drafter_link"] = Json::string(drafter_role != "assist" ? "none" : peer ? "peer" : "host-staged");
  return sha_of(prof);
}

// A store identity from the probe (`file_store` / `engram_store`: "<bus>:<model>/gen<G>x<W>@<root>;fs=..;io=..;
// readers=<n>"): "none" when the placement does not use it, "unknown" when used and not reported.
std::string store_of(const Json& probe, const char* key, bool used) {
  if (!used) return "none";
  const Json* v = probe.find(key);
  return v && v->is_string() && !v->text.empty() ? v->text : "unknown";
}

// The calibration identity of a placed choice (§3.9): anchor_device / assist_device "<arch>/<binding name>" ("none"
// without an assist), xfer (sha256 of the xfer-v1 profile), cache_sha256 (sha256 of the canonical {elastic,
// cache_config} without the R-derived dg_max_ids / banks / window_slots), content / arithmetic sha256 of the canonical
// components-v1 objects, prefetch, tier_mode, driver / platform of the binding, usage_sha256, build_sha256, sequences,
// adapter (config.cache), drafter, policy, file_store (FILE units) / engram_store (an engram component placed).
Json derive_calib(const RecipeStack& s, const std::string& route, const std::string& anchor, const Json& doc,
                  const Components& comps, const UsageDoc* usage, const Json& probe, int64_t file_units) {
  Json calib = Json::object();
  calib["sequences"] = Json::number(comps.max_sequences);
  calib["adapter"] = Json::string(comps.cache);
  calib["build_sha256"] = Json::string(comps.build_sha256);
  if (!comps.content.is_null()) calib["content_sha256"] = Json::string(sha_of(comps.content));
  if (!comps.arithmetic.is_null()) calib["arithmetic_sha256"] = Json::string(sha_of(comps.arithmetic));
  if (!comps.cache_config.is_null()) calib["cache_sha256"] = Json::string(cache_sha256_of(comps));
  if (usage) calib["usage_sha256"] = Json::string(usage->sha256);
  calib["prefetch"] = Json::string(prefetch_of(comps));
  calib["tier_mode"] = Json::string(comps.exclusive ? "exclusive" : "inclusive");
  calib["drafter"] = Json::string("none");
  calib["assist_device"] = Json::string("none");
  bool engram = false;
  for (const Json& d : doc.find("devices")->arr) {
    for (const Json& l : d.find("lines")->arr) {
      if (l.find("component")->text == "drafter") calib["drafter"] = *d.find("role");
      engram = engram || l.find("component")->text == "engram";
    }
    const std::string dev = d.find("arch")->text + "/" + d.find("name")->text;
    if (d.find("role")->text != "anchor") {
      calib["assist_device"] = Json::string(dev);
      continue;
    }
    calib["anchor_device"] = Json::string(dev);
    if (const Json* v = d.find("driver")) calib["driver"] = *v;
    if (const Json* v = d.find("platform")) calib["platform"] = *v;
  }
  if (const Json* h = doc.find("host"))
    for (const Json& l : h->find("lines")->arr) engram = engram || l.find("component")->text == "engram";
  calib["xfer"] = Json::string(xfer_of(doc, probe, calib.find("drafter")->text));
  calib["file_store"] = Json::string(store_of(probe, "file_store", file_units > 0));
  calib["engram_store"] = Json::string(store_of(probe, "engram_store", engram));
  const RPlacement* ap = s.effective().route(route)->placement(anchor);
  calib["policy"] = Json::string(ap->get("policy") == double(SS_POLICY_LRU) ? "lru" : "scored");
  return calib;
}

}  // namespace

// cache_sha256 (rev 18 §3.9, golden/calib_ref.py): sha256 of the canonical {elastic (as written), cache_config}; the
// R-derived fields live in config.window and never enter it.
std::string cache_sha256_of(const Components& c) {
  Json cc = Json::object();
  cc["elastic"] = c.elastic_doc;
  cc["cache_config"] = c.cache_config;
  return sha_of(cc);
}

Json solve_predict(const Json& in) {
  const auto curves = curves_of(in);
  const Json* costs = in.find("costs");
  std::vector<int64_t> usage;  // coldest first
  if (const Json* u = in.find("usage")) {
    if (!u->is_array()) bad("usage", "expected the coldest-first micro-selections per token");
    for (const Json& x : u->arr) usage.push_back(nonneg(x, "usage", int64_t(1) << 50));
  }
  // usage_lang: per language, the micro-selections per token of the same units in the same (pooled cold) order; a
  // malformed table (length, values) makes its FILE rows unpredicted; usage_lang_sha256 = sha256 of each canonical
  // array as given
  std::map<std::string, std::vector<int64_t>> usage_lang;
  Json lang_sha = Json::object();
  if (const Json* ul = in.find("usage_lang")) {
    if (!ul->is_object()) bad("usage_lang", "expected {lang: [micro-selections per unit, pooled cold order]}");
    for (const auto& kv : ul->obj) {
      lang_sha[kv.first] = Json::string(sha_of(kv.second));
      bool ok = kv.second.is_array() && kv.second.arr.size() == usage.size();
      std::vector<int64_t> v;
      if (ok)
        for (const Json& x : kv.second.arr) {
          if (!int_text(x)) {
            ok = false;
            break;
          }
          v.push_back(json_i64(x, "usage_lang"));
        }
      if (ok) usage_lang[kv.first] = v;
    }
  }
  const Json* cs = in.find("choices");
  if (!cs || !cs->is_array()) bad("choices", "expected an array");
  Json rows = Json::array();
  for (size_t i = 0; i < cs->arr.size(); ++i) {
    const Json& c = cs->arr[i];
    const Choice x = choice_of(c, i);
    const Json* a = c.find("active_units");
    const Json* f = c.find("file_units");
    if (!a || !f) bad("choices[" + std::to_string(i) + "]", "needs active_units and file_units");
    const int64_t file = nonneg(*f, "file_units", int64_t(1) << 32);
    const Json* seq = c.find("sequences");
    const Json* calib = c.find("calib");
    const bool spec = x.mode == "spec", has_file = file > 0;
    const int64_t sequences = seq && int_text(*seq) ? json_i64(*seq, "sequences") : -1;
    Json row;
    const Json* uls = in.find("usage_lang_sources");
    const std::vector<std::string> lang_src = strings_of(uls ? uls->find(x.lang) : nullptr);
    if (!calib || !calibration_applies(in, *calib, sequences, spec, has_file, c.find("probe_xfer"),
                                       strings_of(c.find("exec_sets")), lang_src,
                                       strings_of(in.find("usage_sources")))) {
      row = prediction_json(Prediction{});
    } else {
      // FILE traffic of the row's language over the file_units coldest units of the pooled order (solver_ref:
      // usage_lang[lang][:file_units]); a FILE row without a (well-formed) table for its language is unpredicted
      int64_t cold = 0;
      if (has_file) {
        auto it = usage_lang.find(x.lang);
        if (it == usage_lang.end())
          cold = -1;
        else
          for (int64_t k = 0; k < file && k < int64_t(it->second.size()); ++k) cold += it->second[size_t(k)];
      }
      row = prediction_json(predict(curves, costs, x.mode, x.r, x.ctx, x.lang,
                                    nonneg(*a, "active_units", int64_t(1) << 32), cold, has_file),
                            costs);
    }
    row["id"] = Json::string(x.id);
    rows.push(row);
  }
  Json out = Json::object();
  out["usage_lang_sha256"] = lang_sha;
  out["rows"] = rows;
  return out;
}

Json solve(const RecipeStack& stack, const std::string& route, const Json& probe, const Json& in) {
  const auto curves = curves_of(in);
  const Json* costs = in.find("costs");
  if (!costs || !costs->is_object()) bad("costs", "expected tierstream/solver-costs-v1");
  // usage-v1 (pooled: ranks the units, chooses the cold set) and usage_lang {lang: usage-v1} (rev 16: each
  // language's own table gives its FILE rate over that cold set); each embedded or named by a path
  auto load_usage = [](const Json& v) {
    if (!v.is_string()) return usage_from_json(v);
    std::ifstream f(v.text, std::ios::binary);
    require(bool(f), SS_E_IO, "solver: cannot open usage " + v.text);
    std::stringstream ss;
    ss << f.rdbuf();
    return usage_from_json(json_parse(ss.str()));
  };
  UsageDoc usage;
  const bool have_usage = in.find("usage") != nullptr;
  if (have_usage) usage = load_usage(*in.find("usage"));
  std::map<std::string, UsageDoc> usage_lang;  // the well-formed tables (same layers and units as the pooled one)
  Json lang_sha = Json::object();               // usage_lang_sha256: every table given
  if (const Json* ul = in.find("usage_lang")) {
    if (!ul->is_object()) bad("usage_lang", "expected {lang: usage-v1}");
    for (const auto& kv : ul->obj) {
      UsageDoc d = load_usage(kv.second);
      lang_sha[kv.first] = Json::string(d.sha256);
      if (have_usage && d.layer_units == usage.layer_units) usage_lang[kv.first] = d;  // else: its FILE rows unpredicted
    }
  }
  const RRoute* rt = stack.effective().route(route);
  require(rt != nullptr, SS_E_RECIPE, "solver: unknown route \"" + route + "\"");
  const std::string anchor = rt->devices[0];
  const Json* cs = in.find("choices");
  if (!cs || !cs->is_array() || cs->arr.empty()) bad("choices", "expected a non-empty array");
  Json rows = Json::array();
  for (size_t i = 0; i < cs->arr.size(); ++i) {
    const Json& c = cs->arr[i];
    const Choice x = choice_of(c, i);
    const Json* cdoc = c.find("components");
    if (!cdoc) bad("choices[" + std::to_string(i) + "]", "missing \"components\" (components-v1 for this choice)");
    if (c.find("calib"))
      bad("choices[" + std::to_string(i) + "].calib", "the calibration identity is derived by the solver, never typed");
    RecipeStack s = stack;  // the choice's configuration as explicit settings of the route
    s.set("routes." + route + ".placement." + anchor + ".max_context_tokens", std::to_string(x.ctx), "solver");
    s.set("routes." + route + ".max_rows", std::to_string(x.r), "solver");
    if (x.seq) s.set("routes." + route + ".max_sequences", std::to_string(x.seq), "solver");
    Json named;
    if (cdoc->is_string()) {  // §2.7: `components` names the components-v1 document of that configuration
      std::ifstream f(cdoc->text, std::ios::binary);
      require(bool(f), SS_E_IO, "solver: cannot open components " + cdoc->text);
      std::stringstream ss;
      ss << f.rdbuf();
      named = json_parse(ss.str());
      cdoc = &named;
    }
    Components comps = components_from_json(*cdoc);
    Binding b = recipe_bind(s.effective(), probe);
    PlaceInput pin;
    pin.recipe = &s.effective();
    pin.effective_sha256 = s.effective_sha256();
    pin.route = route;
    pin.binding = &b;
    pin.components = &comps;
    if (have_usage) pin.usage = &usage;
    PlaceResult pr = place(pin);
    Json requires = Json::array();
    Components hypo;
    if (!pr.errors.empty() && pr.computed && !comps.has("file_expert_tier")) {
      // FILE not declared yet (§3.9 Availability): plan the row hypothetically with the capability and the FILE
      // owner's declared staging; the row is tagged and never writable as a recipe.
      bool pinned_only = true;
      for (const auto& e : pr.errors) pinned_only = pinned_only && e.rfind("PL_E_HOST_PINNED", 0) == 0;
      int64_t staging = -1;  // bytes, not microseconds: no 10^7 bound
      if (const Json* fs = costs->find("file_staging_bytes"))
        if (int_text(*fs)) staging = json_i64(*fs, "file_staging_bytes");
      if (pinned_only && staging >= 0) {
        Json h = *cdoc;
        Json caps = h.find("capabilities") ? *h.find("capabilities") : Json::array();
        caps.push(Json::string("file_expert_tier"));
        h["capabilities"] = caps;
        bool has_file = false;
        for (const Json& e : h.find("components")->arr) has_file = has_file || e.find("id")->text == "expert_file";
        if (!has_file && staging > 0) {
          Json ch = Json::object();
          ch["at"] = Json::string("host_pinned");
          ch["bytes"] = Json::number(staging);
          Json ph = Json::array();
          for (int p = 0; p < kPlacePhases; ++p) ph.push(Json::string(place_phase_word(p)));
          ch["phases"] = ph;
          Json f = Json::object();
          f["id"] = Json::string("expert_file");
          Json chs = Json::array();
          chs.push(ch);
          f["charges"] = chs;
          (*const_cast<Json*>(h.find("components"))).push(f);
        }
        hypo = components_from_json(h);
        pin.components = &hypo;
        PlaceResult g = place(pin);
        if (g.errors.empty()) {
          pr = g;
          requires.push(Json::string("file_expert_tier"));
        }
      }
    }
    Json row = Json::object();
    row["id"] = Json::string(x.id);
    Json ch = Json::object();
    ch["mode"] = Json::string(x.mode);
    ch["R"] = Json::number(x.r);
    ch["ctx"] = Json::number(x.ctx);
    ch["batch"] = Json::number(comps.max_sequences);
    ch["lang"] = Json::string(x.lang);
    row["choice"] = ch;
    row["components_sha256"] = Json::string(comps.sha256);
    row["requires"] = requires;
    Json pl = Json::object();
    if (!pr.errors.empty()) {
      Json rf = Json::object();
      const std::string& e0 = pr.errors[0];
      rf["code"] = Json::string(e0.substr(0, e0.find_first_of(" :")));
      const size_t at = e0.find(" units short");
      if (at != std::string::npos) {
        size_t b0 = e0.rfind(' ', at - 1);
        rf["units_short"] = Json::number(int64_t(std::atoll(e0.c_str() + b0 + 1)));
      }
      Json es = Json::array();
      for (const auto& e : pr.errors) es.push(Json::string(e));
      rf["errors"] = es;
      pl["refusal"] = rf;
      row["placement"] = pl;
      row["prediction"] = prediction_json(Prediction{});
      rows.push(row);
      continue;
    }
    const Json& doc = pr.doc;
    int64_t active = 0;
    Json devs = Json::array();
    for (const Json& d : doc.find("devices")->arr) {
      Json e = Json::object();
      e["set"] = *d.find("set");
      int64_t arena = 0, units = 0, peak = 0;
      for (const Json& l : d.find("lines")->arr)
        if (l.find("component")->text == "arena") {
          arena = json_i64(*l.find("bytes"), "arena");
          units = json_i64(*l.find("slots"), "arena");
        }
      for (const auto& kv : d.find("phase_peak_bytes")->obj) peak = std::max(peak, json_i64(kv.second, "peak"));
      if (d.find("role")->text == "anchor") active = units;
      e["active_units"] = Json::number(units);
      e["arena_bytes"] = Json::number(arena);
      e["peak_bytes"] = Json::number(peak);
      devs.push(e);
    }
    int64_t file = 0;
    for (const Json& l : doc.find("host")->find("lines")->arr)
      if (l.find("component")->text == "expert_file") file = json_i64(*l.find("units"), "file");
    pl["placement_sha256"] = *doc.find("placement_sha256");
    pl["devices"] = devs;
    pl["pinned_total_bytes"] = *doc.find("host")->find("pinned_total_bytes");
    pl["file_units"] = Json::number(file);
    row["placement"] = pl;
    // FILE rate (rev 16): the row's language's selections over the file_units coldest units of the pooled rank (the
    // units S3 sent to FILE); no FILE units needs no table; FILE units without the pooled usage or the language's
    // table are unpredicted
    int64_t cold = file == 0 ? 0 : -1;
    auto lang = usage_lang.find(x.lang);
    if (file > 0 && have_usage && lang != usage_lang.end()) {
      const std::vector<size_t> order = usage.rank();  // hottest first
      cold = 0;
      for (int64_t k = 0; k < file && k < int64_t(order.size()); ++k)
        cold = std::min(cold + lang->second.units[order[order.size() - 1 - size_t(k)]].selections, kBound * 1000);
    }
    // the choice's calibration identity, derived (never typed) from the route, the placement, the binding,
    // components-v1 and usage-v1 (contract §3.9)
    Json calib = derive_calib(s, route, anchor, doc, comps, have_usage ? &usage : nullptr, probe, file);
    (*const_cast<Json*>(row.find("choice")))["calib"] = calib;
    // the probe's transfer figures: the choice's own probe_xfer, else the probe report's
    const Json* pxf = c.find("probe_xfer") ? c.find("probe_xfer") : probe.find("probe_xfer");
    const auto ul = usage_lang.find(x.lang);
    // exec_sets (rev 19): the anchor, plus the assist when the drafter executes there
    std::vector<std::string> exec_sets = {anchor};
    for (const Json& d : doc.find("devices")->arr)
      if (d.find("role")->text != "anchor")
        for (const Json& l : d.find("lines")->arr)
          if (l.find("component")->text == "drafter") exec_sets.push_back(d.find("set")->text);
    std::sort(exec_sets.begin(), exec_sets.end());
    (*const_cast<Json*>(row.find("choice")))["exec_sets"] = [&] {
      Json a = Json::array();
      for (const auto& e : exec_sets) a.push(Json::string(e));
      return a;
    }();
    const bool ok = calibration_applies(in, calib, comps.max_sequences, x.mode == "spec", file > 0, pxf, exec_sets,
                                        ul == usage_lang.end() ? std::vector<std::string>() : ul->second.sources,
                                        have_usage ? usage.sources : std::vector<std::string>());
    row["prediction"] = ok ? prediction_json(predict(curves, costs, x.mode, x.r, x.ctx, x.lang, active, cold, file > 0),
                                             costs)
                           : prediction_json(Prediction{});
    rows.push(row);
  }
  Json out = Json::object();
  out["schema"] = Json::string("tierstream/solver-v1");
  out["recipe"] = Json::string(stack.effective().name);
  out["effective_sha256"] = Json::string(stack.effective_sha256());
  out["route"] = Json::string(route);
  if (have_usage) out["usage_sha256"] = Json::string(usage.sha256);
  out["usage_lang_sha256"] = lang_sha;
  if (const Json* cv = in.find("curves")) out["curves_sha256"] = Json::string(sha_of(*cv));
  out["costs_sha256"] = Json::string(sha_of(*costs));
  out["rows"] = rows;
  out["solver_sha256"] = Json::string(sha_of(out));
  return out;
}

std::string solve_table(const Json& sv) {
  std::string t;
  char buf[512];
  std::snprintf(buf, sizeof buf, "%-22s %-5s %3s %8s %6s %-6s %8s %9s %9s %6s %10s  %s\n", "choice", "mode", "R", "ctx",
                "batch", "lang", "active", "pinned", "file", "tag", "tok/s", "requires / refusal");
  t += buf;
  for (const Json& r : sv.find("rows")->arr) {
    const Json& c = *r.find("choice");
    const Json& p = *r.find("placement");
    const Json& pr = *r.find("prediction");
    std::string active = "-", pinned = "-", file = "-", tok = "-", note;
    if (const Json* rf = p.find("refusal")) {
      note = rf->find("code")->text;
      if (const Json* u = rf->find("units_short")) note += " (" + u->text + " units short)";
    } else {
      active = p.find("devices")->arr[0].find("active_units")->text;
      std::snprintf(buf, sizeof buf, "%.2fG", double(json_i64(*p.find("pinned_total_bytes"), "p")) / 1073741824.0);
      pinned = buf;
      file = p.find("file_units")->text;
      for (const Json& q : r.find("requires")->arr) note += (note.empty() ? "requires " : ", ") + q.text;
    }
    if (pr.find("milli_tok_s") && pr.find("milli_tok_s")->is_number()) {
      const int64_t m = json_i64(*pr.find("milli_tok_s"), "tok");
      std::snprintf(buf, sizeof buf, "%lld.%03lld", (long long)(m / 1000), (long long)(m % 1000));
      tok = buf;
    }
    std::snprintf(buf, sizeof buf, "%-22s %-5s %3s %8s %6s %-6s %8s %9s %9s %6s %10s  %s\n", r.find("id")->text.c_str(),
                  c.find("mode")->text.c_str(), c.find("R")->text.c_str(), c.find("ctx")->text.c_str(),
                  c.find("batch")->text.c_str(), c.find("lang")->text.c_str(), active.c_str(), pinned.c_str(),
                  file.c_str(), pr.find("tag")->text.c_str(), tok.c_str(), note.c_str());
    t += buf;
  }
  return t;
}

}  // namespace ts
