#include "adapt_solve.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>

#include "adapt_sim.hpp"
#include "errors.hpp"
#include "strix_stream/ss.h"

namespace ts {
namespace adapt {

namespace {

const Json& need(const Json& o, const std::string& k, const std::string& where) {
  const Json* v = o.find(k);
  require(v != nullptr, SS_E_INVALID, where + "." + k + " missing");
  return *v;
}

uint64_t u64(const Json& o, const std::string& k, const std::string& where) {
  return json_u64(need(o, k, where), where + "." + k);
}

uint64_t u64_or(const Json& o, const std::string& k, uint64_t dflt, const std::string& where) {
  const Json* v = o.find(k);
  return v ? json_u64(*v, where + "." + k) : dflt;
}

std::string str_or(const Json& o, const std::string& k, const std::string& dflt) {
  const Json* v = o.find(k);
  return v && v->is_string() ? v->text : dflt;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  require(bool(f), SS_E_IO, "cannot open " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string resolve(const std::string& base, const std::string& p) {
  std::filesystem::path q(p);
  if (q.is_absolute() || base.empty()) return p;
  return (std::filesystem::path(base) / q).string();
}

// A document given inline (object) or as a path (string).
Json doc(const Json& v, const std::string& base, const std::string& where) {
  if (v.is_object()) return v;
  require(v.is_string(), SS_E_INVALID, where + ": expected an object or a path");
  return json_parse(read_file(resolve(base, v.text)));
}

struct Interp {
  bool ok;
  uint64_t value;
  const char* tag;
};

// Contract rev 5 section 3.9 (golden/solver_ref.py): knot | interpolated | clamped_high (above the last knot: the last
// knot's value, pessimistic) | unpredicted (below the first knot: clamping would understate misses).
Interp interp(const Json& knots, uint64_t a, const std::string& where) {
  require(knots.is_array() && !knots.arr.empty(), SS_E_INVALID, where + ": no knots");
  std::vector<std::pair<uint64_t, uint64_t>> k;
  for (const Json& p : knots.arr) {
    require(p.is_array() && p.arr.size() >= 2, SS_E_INVALID, where + ": a knot is [active_units, milli_misses]");
    k.emplace_back(json_u64(p.arr[0], where), json_u64(p.arr[1], where));
  }
  for (size_t i = 1; i < k.size(); ++i)
    require(k[i].first > k[i - 1].first, SS_E_INVALID, where + ": knots must be sorted by active_units");
  if (a < k.front().first) return {false, 0, "unpredicted"};
  if (a >= k.back().first) return {true, k.back().second, a == k.back().first ? "knot" : "clamped_high"};
  for (size_t i = 0; i + 1 < k.size(); ++i) {
    const uint64_t a0 = k[i].first, m0 = k[i].second, a1 = k[i + 1].first, m1 = k[i + 1].second;
    if (a == a0) return {true, m0, "knot"};
    if (a0 < a && a < a1) {
      // m0 (a1 - a) + m1 (a - a0) = solver_ref's m0 (a1 - a0) + (m1 - m0)(a - a0), without a negative term
      const uint64_t num = m0 * (a1 - a) + m1 * (a - a0);
      return {true, rdiv(num, a1 - a0), "interpolated"};
    }
  }
  fail(SS_E_INVALID, where + ": interpolation fell through");
}

constexpr uint64_t kBound = 10000000;  // milli-misses and microseconds (contract: every product fits in 64 bits)

// tierstream/solver-costs-v1: {allhit_us: {ctx}, verify_us: {ctx: {R}}, draft_us: {R}, accepted_milli: {lang/R},
// miss_pinned_us, miss_file_us, boundary_us: {ctx}, engram_us: {ctx}}. An absent key makes the row unpredicted.
struct Costs {
  const Json* j;
  bool get(std::initializer_list<std::string> path, uint64_t& v, std::string& missing) const {
    const Json* o = j;
    std::string name;
    for (const std::string& k : path) {
      name += (name.empty() ? "" : ".") + k;
      o = o && o->is_object() ? o->find(k) : nullptr;
    }
    if (!o || !o->is_number() || o->text.find_first_not_of("0123456789") != std::string::npos) {
      if (missing.empty()) missing = name;
      return false;
    }
    v = json_u64(*o, "costs." + name);
    return true;
  }
};

struct Window {
  bool ok = false;
  std::string missing;
  uint64_t window_us = 0, accepted_m = 0, milli_tok_s = 0;
};

// The section 3.9 window time for a mode, given the milli-misses served from each home.
Window window_time(const Costs& c, const std::string& mode, uint32_t r, uint64_t ctx, const std::string& key,
                   uint64_t total_m, uint64_t pinned_m, uint64_t file_m) {
  Window w;
  const std::string cs = std::to_string(ctx), rs = std::to_string(r);
  uint64_t mp = 0, mf = 0, bd = 0, en = 0, base = 0, acc = 1000;
  bool ok = c.get({"miss_pinned_us"}, mp, w.missing);
  ok = c.get({"miss_file_us"}, mf, w.missing) && ok;
  ok = c.get({"boundary_us", cs}, bd, w.missing) && ok;
  ok = c.get({"engram_us", cs}, en, w.missing) && ok;
  if (mode == "plain") {
    ok = c.get({"allhit_us", cs}, base, w.missing) && ok;
  } else {
    uint64_t dr = 0, vr = 0;
    ok = c.get({"draft_us", rs}, dr, w.missing) && ok;
    ok = c.get({"verify_us", cs, rs}, vr, w.missing) && ok;
    ok = c.get({"accepted_milli", key}, acc, w.missing) && ok;
    base = dr + vr;
  }
  if (!ok) return w;
  const uint64_t fixed = bd + en;
  for (uint64_t v : {total_m, mp, mf, base, fixed, acc})
    if (v > kBound) {
      w.missing = "a value outside [0, 10^7]";
      return w;
    }
  w.window_us = base + rdiv(pinned_m * mp, 1000) + rdiv(file_m * mf, 1000) + fixed;
  if (w.window_us == 0) {
    w.missing = "window_us = 0";
    return w;
  }
  w.accepted_m = acc;
  w.milli_tok_s = rdiv(uint64_t(1000000) * w.accepted_m, w.window_us);
  w.ok = true;
  return w;
}

uint64_t line_sum(const Json* lines, const std::string& where, Json* out) {
  uint64_t s = 0;
  if (!lines) return 0;
  require(lines->is_array(), SS_E_INVALID, where + ": expected an array of {id, bytes}");
  for (size_t i = 0; i < lines->arr.size(); ++i) {
    const Json& l = lines->arr[i];
    const std::string w = where + "[" + std::to_string(i) + "]";
    const uint64_t b = u64(l, "bytes", w);
    s += b;
    if (out) {
      Json o = Json::object();
      o["id"] = Json::string(str_or(l, "id", "?"));
      o["bytes"] = Json::number(b);
      o["source"] = Json::string(str_or(l, "source", ""));
      out->push(o);
    }
  }
  return s;
}

}  // namespace

namespace {

// Calibration identity keys (contract 3.9, golden/solver_ref.py 5722f6e2): an applicability lacking any of them is
// incomplete, and every key it has must equal the choice's `calib` value.
const char* const kCostKeys[] = {"sequences", "adapter", "drafter", "anchor_device", "assist_device", "link",
                                 "build_sha256", "content_sha256", "arithmetic_sha256", "prefetch", "tier_mode",
                                 "driver", "platform"};
const char* const kCurveKeys[] = {"sequences", "adapter", "drafter", "policy", "usage_sha256", "build_sha256",
                                  "content_sha256", "prefetch", "tier_mode"};

template <size_t N>
bool applies(const Json* app, const char* const (&req)[N], const Json* calib) {
  if (!app || !app->is_object() || app->obj.empty()) return false;
  for (const char* k : req)
    if (!app->find(k)) return false;
  for (const auto& kv : app->obj) {
    const Json* c = calib && calib->is_object() ? calib->find(kv.first) : nullptr;
    if (!c || json_write(*c) != json_write(kv.second)) return false;
  }
  return true;
}

Json unpredicted_row(const Json& id) {
  Json o = Json::object();
  o["id"] = id;
  o["tag"] = Json::string("unpredicted");
  for (const char* k : {"milli_misses_total", "milli_misses_pinned", "milli_misses_file", "window_us", "milli_tok_s"})
    o[k] = Json();
  return o;
}

}  // namespace

Json solve_ref(const Json& fx) {
  const Json& curves = need(fx, "curves", "fixture");
  const Json& usage = need(fx, "usage", "fixture");
  require(usage.is_array(), SS_E_INVALID, "fixture.usage: coldest-first micro-selections");
  const Json& costs = need(fx, "costs", "fixture");
  const Costs c{&costs};
  Json rows = Json::array();
  for (const Json& ch : need(fx, "choices", "fixture").arr) {
    const Json& id = need(ch, "id", "choice");
    const uint32_t r = uint32_t(u64(ch, "R", "choice"));
    const std::string mode = str_or(ch, "mode", "plain");
    require(mode == "spec" || r == 1, SS_E_INVALID, "fixture: plain needs R = 1");
    const std::string key = str_or(ch, "lang", "") + "/R" + std::to_string(r);
    // v1 predicts single-sequence choices only; costs and curves apply only to their calibration identity
    const Json* seq = ch.find("sequences");
    const Json* calib = ch.find("calib");
    if (!seq || json_write(*seq) != "1" || !applies(costs.find("applicability"), kCostKeys, calib) ||
        !applies(fx.find("curves_applicability"), kCurveKeys, calib) || !curves.find(key)) {
      rows.push(unpredicted_row(id));
      continue;
    }
    const Interp t = interp(*curves.find(key), u64(ch, "active_units", "choice"), "curves." + key);
    Window w;
    uint64_t file_m = 0, pinned_m = 0;
    if (t.ok) {
      const uint64_t fu = std::min<uint64_t>(u64(ch, "file_units", "choice"), usage.arr.size());  // = a Python slice
      uint64_t sum = 0;
      for (uint64_t i = 0; i < fu; ++i) sum += json_u64(usage.arr[size_t(i)], "usage");
      file_m = std::min<uint64_t>(r * rdiv(sum, 1000), t.value);
      pinned_m = t.value - file_m;
      w = window_time(c, mode, r, u64(ch, "ctx", "choice"), key, t.value, pinned_m, file_m);
    }
    if (!w.ok) {
      rows.push(unpredicted_row(id));
      continue;
    }
    Json o = Json::object();
    o["id"] = id;
    o["tag"] = Json::string(t.tag);
    o["milli_misses_total"] = Json::number(t.value);
    o["milli_misses_pinned"] = Json::number(pinned_m);
    o["milli_misses_file"] = Json::number(file_m);
    o["window_us"] = Json::number(w.window_us);
    o["milli_tok_s"] = Json::number(w.milli_tok_s);
    rows.push(o);
  }
  Json out = Json::object();
  out["rows"] = rows;
  return out;
}

Json solve(const Json& in, const std::string& base) {
  require(str_or(in, "schema", "") == "tierstream/adapt-solve-v1", SS_E_INVALID,
          "solve: schema must be tierstream/adapt-solve-v1");
  const Json& model = need(in, "model", "solve");
  const uint64_t unit = u64(model, "unit_bytes", "model");
  const uint64_t units = u64(model, "units", "model");
  const uint64_t hstride = u64_or(model, "host_stride", (unit + 4095) / 4096 * 4096, "model");
  require(unit > 0 && units > 0, SS_E_INVALID, "model: unit_bytes and units > 0");
  const Json& dev = need(in, "device", "solve");
  const uint64_t budget = u64(dev, "budget_bytes", "device");
  const uint64_t margin = u64_or(dev, "margin_bytes", 0, "device");
  const Json& host = need(in, "host", "solve");
  const uint64_t pinned = u64(host, "pinned_bytes", "host");
  const uint64_t aux = u64_or(host, "aux_pinned_bytes", 0, "host");
  const uint64_t spare_blocks = u64_or(host, "spare_blocks", 0, "host");
  const Json& modes = need(in, "modes", "solve");
  const Json& state = need(in, "state", "solve");

  // Usage ranking (also the cold rate of the §3.9 prediction).
  Usage usage = Usage::from_json(doc(need(in, "usage", "solve"), base, "solve.usage"));
  require(uint64_t(usage.total_units()) == units, SS_E_INVALID, "solve: the usage table does not cover model.units");
  const std::vector<int64_t> order = usage.rank();
  std::vector<uint64_t> cold_prefix(size_t(units) + 1, 0);  // sum of the k coldest micro-selections
  for (uint64_t k = 0; k < units; ++k) cold_prefix[size_t(k + 1)] = cold_prefix[size_t(k)] + usage.use(order[size_t(units - 1 - k)]);

  std::unique_ptr<Json> curves;
  if (const Json* cv = in.find("curves")) curves = std::make_unique<Json>(doc(*cv, base, "solve.curves"));
  std::unique_ptr<Json> costs_doc;
  if (const Json* cs = in.find("costs")) costs_doc = std::make_unique<Json>(doc(*cs, base, "solve.costs"));
  std::unique_ptr<Trace> trace;
  const Json* sim = in.find("sim");
  if (sim) trace = std::make_unique<Trace>(trace_load(resolve(base, json_str(need(*sim, "trace", "sim"), "sim.trace")),
                                                      int(units / uint64_t(usage.n_layers()))));
  const Json* ref = in.find("reference");
  const uint64_t floor_us = ref ? u64_or(*ref, "floor_us", 0, "reference") : 0;
  const uint64_t w3_milli = ref ? u64_or(*ref, "w3_milli_tok_s", 0, "reference") : 0;

  Json out = Json::object();
  // A what-if document, deliberately NOT tierstream/solver-v1: the operator solver (ts-recipe solve, placement
  // contract 3.9) belongs to the placement planner; solve_ref above is its arithmetic for the goldens.
  out["schema"] = Json::string("tierstream/adapt-whatif-v1");
  out["usage_sha256"] = Json::string(usage.sha256());
  if (curves) out["curves_source"] = Json::string(str_or(*curves, "schema", "?"));
  if (trace) out["trace_sha256"] = Json::string(trace->sha256);
  Json common = Json::array();
  const uint64_t common_bytes = line_sum(dev.find("lines"), "device.lines", &common);
  out["device_lines"] = common;
  Json rows = Json::array();
  const Json& choices = need(in, "choices", "solve");
  require(choices.is_array(), SS_E_INVALID, "solve.choices: expected an array");
  for (size_t ci = 0; ci < choices.arr.size(); ++ci) {
    const Json& ch = choices.arr[ci];
    const std::string where = "choices[" + std::to_string(ci) + "]";
    const std::string mode = json_str(need(ch, "mode", where), where + ".mode");
    const uint64_t ctx = u64(ch, "ctx", where);
    const uint64_t batch = u64_or(ch, "batch", 1, where);
    const Json& m = need(modes, mode, "modes");
    const uint32_t r = uint32_t(u64_or(m, "R", 1, "modes." + mode));
    // The cost formula of the mode: "plain" or "spec" (a mode may be named otherwise, e.g. "plain_today").
    const std::string kind = str_or(m, "kind", mode.rfind("spec", 0) == 0 ? "spec" : "plain");
    require(kind == "plain" || kind == "spec", SS_E_INVALID, "modes." + mode + ".kind: plain | spec");
    require(kind == "spec" || r == 1, SS_E_INVALID, "modes." + mode + ": a plain mode has R = 1");
    Json row = Json::object();
    row["id"] = Json::string(str_or(ch, "id", where));
    row["mode"] = Json::string(mode);
    row["ctx"] = Json::number(ctx);
    row["batch"] = Json::number(batch);
    row["R"] = Json::number(uint64_t(r));
    row["kind"] = Json::string(kind);
    Json mlines = Json::array();
    const uint64_t mode_bytes = line_sum(m.find("lines"), "modes." + mode + ".lines", &mlines);
    row["mode_lines"] = mlines;
    // State per sequence at this context: an exact table entry, else a declared linear model (tagged).
    const Json& st = need(state, mode, "state");
    uint64_t per_seq = 0;
    std::string state_tag;
    const Json* tab = st.find("per_sequence");
    if (tab && tab->find(std::to_string(ctx))) {
      per_seq = json_u64(*tab->find(std::to_string(ctx)), "state." + mode + ".per_sequence");
      state_tag = "table";
    } else if (const Json* lin = st.find("linear")) {
      per_seq = u64(*lin, "base", "state.linear") + u64(*lin, "per_token", "state.linear") * ctx;
      state_tag = "linear-model";
    } else {
      fail(SS_E_INVALID, "state." + mode + ": no entry for context " + std::to_string(ctx) + " and no linear model");
    }
    const uint64_t state_bytes = per_seq * batch;
    row["state_bytes"] = Json::number(state_bytes);
    row["state_tag"] = Json::string(state_tag);
    const uint64_t charged = margin + common_bytes + mode_bytes + state_bytes;
    row["device_charged_bytes"] = Json::number(charged);
    if (charged > budget) {
      row["status"] = Json::string("PL_E_BUDGET");
      row["overflow_bytes"] = Json::number(charged - budget);
      rows.push(row);
      continue;
    }
    uint64_t active = std::min<uint64_t>((budget - charged) / unit, units);
    if (const Json* cap = ch.find("arena_units")) active = std::min<uint64_t>(active, json_u64(*cap, where + ".arena_units"));
    uint64_t host_fixed = aux + spare_blocks * hstride;
    uint64_t host_units = 0;
    if (pinned > host_fixed) host_units = std::min<uint64_t>((pinned - host_fixed) / hstride, units - active);
    uint64_t file_units = units - active - host_units;
    // A FILE tier needs its pinned staging (solver-costs-v1 file_staging_bytes): charged when there is one.
    const uint64_t staging = costs_doc ? u64_or(*costs_doc, "file_staging_bytes", 0, "costs") : 0;
    if (file_units > 0 && staging > 0) {
      host_fixed += staging;
      host_units = pinned > host_fixed ? std::min<uint64_t>((pinned - host_fixed) / hstride, units - active) : 0;
      file_units = units - active - host_units;
    }
    row["status"] = Json::string("ok");
    row["active_units"] = Json::number(active);
    row["arena_bytes"] = Json::number(active * unit);
    row["host_units"] = Json::number(host_units);
    row["host_pinned_bytes"] = Json::number(host_units * hstride + host_fixed);
    row["file_units"] = Json::number(file_units);
    row["file_bytes"] = Json::number(file_units * unit);
    row["device_free_bytes"] = Json::number(budget - charged - active * unit);

    Json langs = Json::object();
    const Json* lang_list = ch.find("langs");
    std::vector<std::string> ls;
    if (lang_list && lang_list->is_array())
      for (const Json& l : lang_list->arr) ls.push_back(json_str(l, where + ".langs"));
    else
      ls.push_back(str_or(ch, "lang", "all"));
    for (const std::string& lang : ls) {
      const std::string key = lang + "/R" + std::to_string(r);
      Json lr = Json::object();
      // (a) the §3.9 arithmetic: interpolated curve + cold-rate bound for the file share.
      const Json* cvs = curves ? curves->find("curves") : nullptr;
      const Json* cv = cvs ? cvs->find(key) : nullptr;
      if (cv && costs_doc) {
        const Json& knots = cv->is_array() ? *cv : need(*cv, "knots", "curves." + key);
        const Interp t = interp(knots, active, "curves." + key);
        const uint64_t cold = rdiv(cold_prefix[size_t(file_units)], 1000);
        const uint64_t file_m = t.ok ? std::min<uint64_t>(uint64_t(r) * cold, t.value) : 0;
        const uint64_t pinned_m = t.ok ? t.value - file_m : 0;
        Window w;
        if (t.ok) w = window_time(Costs{costs_doc.get()}, kind, r, ctx, key, t.value, pinned_m, file_m);
        else w.missing = "active units below the curve's first knot";
        if (batch != 1) w = Window{false, "sequences != 1 (v1 predicts single-sequence rows only)"};
        Json a = Json::object();
        a["tag"] = Json::string(w.ok ? t.tag : "unpredicted");
        a["milli_misses_total"] = t.ok ? Json::number(t.value) : Json();
        a["milli_misses_pinned"] = t.ok ? Json::number(pinned_m) : Json();
        a["milli_misses_file_bound"] = t.ok ? Json::number(file_m) : Json();
        if (w.ok) {
          a["window_us"] = Json::number(w.window_us);
          a["milli_tok_s"] = Json::number(w.milli_tok_s);
          if (floor_us) a["floor_pct_milli"] = Json::number(rdiv(100 * floor_us * w.accepted_m, w.window_us));
          if (w3_milli) a["vs_w3_pct_milli"] = Json::number(rdiv(w.milli_tok_s * 100000, w3_milli));
        } else {
          a["unpredicted"] = Json::string("missing cost " + w.missing);
        }
        lr["contract"] = a;
      } else {
        lr["contract"] = Json::string(curves ? "no curve " + key : "no curves given");
      }
      // (b) simulated split for exactly this (active, file) pair on the evaluation trace.
      if (trace) {
        const Json* lf = sim->find("langs");
        const std::string lfilter = lf && lf->find(lang) ? json_str(*lf->find(lang), "sim.langs") : "";
        std::string filter = str_or(*sim, "eval_filter", "");
        if (!lfilter.empty()) filter += (filter.empty() ? "" : "&") + lfilter;
        SimConfig sc;
        sc.active = int64_t(active);
        sc.file_units = int64_t(file_units);
        sc.spares = int64_t(u64_or(*sim, "spares", 96, "sim"));
        sc.rows = r;
        sc.wb_windows = uint32_t(u64_or(*sim, "wb_windows", 1, "sim"));
        sc.evict = str_or(*sim, "evict", "lru") == "file-last" ? Evict::FileLast : Evict::Lru;
        const SimResult sr = simulate(*trace, filter, order, sc);
        Json b = Json::object();
        b["filter"] = Json::string(filter);
        b["windows"] = Json::number(sr.windows);
        b["milli_misses_pinned"] = Json::number(sr.milli_per_window(sr.miss_pinned));
        b["milli_misses_file"] = Json::number(sr.milli_per_window(sr.miss_file));
        b["milli_file_uses"] = Json::number(sr.milli_per_window(sr.file_uses));
        if (costs_doc) {
          const uint64_t pm = sr.milli_per_window(sr.miss_pinned), fm = sr.milli_per_window(sr.miss_file);
          Window w = window_time(Costs{costs_doc.get()}, kind, r, ctx, key, pm + fm, pm, fm);
          if (batch != 1) w = Window{false, "sequences != 1 (v1 predicts single-sequence rows only)"};
          if (w.ok) {
            b["window_us"] = Json::number(w.window_us);
            b["milli_tok_s"] = Json::number(w.milli_tok_s);
            if (floor_us) b["floor_pct_milli"] = Json::number(rdiv(100 * floor_us * w.accepted_m, w.window_us));
            if (w3_milli) b["vs_w3_pct_milli"] = Json::number(rdiv(w.milli_tok_s * 100000, w3_milli));
          } else {
            b["unpredicted"] = Json::string("missing cost " + w.missing);
          }
        }
        lr["simulated"] = b;
      }
      langs[lang] = lr;
    }
    row["predicted"] = langs;
    rows.push(row);
  }
  out["rows"] = rows;
  return out;
}

}  // namespace adapt
}  // namespace ts
