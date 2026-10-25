// Placement planner S0-S5 (wave4/placement contract rev 4 §3.2), integer bytes only. The arithmetic is the one of
// golden/place_ref.py: margins = max(gib_bytes(margin_gib), budget x frac_ppm / 10^6), context (+ growth) charged
// once per touched device, phase peaks over allocation lifetimes, device spares on the anchor, arena active =
// min(room / unit, units left), exclusive host homes = total - active - assist - file + host spares, other pinned
// consumers reserved first, S3e spill of the coldest units to the file tier when the engine declares it.
#include "place.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

#include "errors.hpp"
#include "sha256.hpp"

namespace ts {
namespace {

constexpr const char* kSchema = "tierstream/placement-v1";
constexpr unsigned kAllPhases = (1u << kPlacePhases) - 1;
constexpr int64_t kMax = INT64_MAX;

// Checked signed 64-bit arithmetic: an overflow is a refusal (PL_E_SCHEMA), never a wrap.
int64_t add(int64_t a, int64_t b) {
  require(!((b > 0 && a > kMax - b) || (b < 0 && a < -kMax - b)), SS_E_RECIPE,
          "PL_E_SCHEMA placement arithmetic overflows 64 bits");
  return a + b;
}
int64_t mul(int64_t a, int64_t b) {
  require(a >= 0 && b >= 0 && (a == 0 || b <= kMax / a), SS_E_RECIPE,
          "PL_E_SCHEMA placement arithmetic overflows 64 bits");
  return a * b;
}
int64_t i64(uint64_t v) {
  require(v <= uint64_t(kMax), SS_E_RECIPE, "PL_E_SCHEMA placement arithmetic overflows 64 bits");
  return int64_t(v);
}
int64_t ceil_div(int64_t n, int64_t d) { return n <= 0 ? 0 : (n + d - 1) / d; }

std::string num(int64_t v) { return std::to_string(v); }

std::string gib_text(int64_t bytes) {
  char buf[48];
  std::snprintf(buf, sizeof buf, "%.2f GiB", double(bytes) / 1073741824.0);
  return buf;
}

// Shortest decimal text that reads back as d (the canonical recipe writes numbers so).
std::string shortest(double d) {
  char buf[40];
  for (int prec = 15; prec <= 17; ++prec) {
    std::snprintf(buf, sizeof buf, "%.*g", prec, d);
    if (std::strtod(buf, nullptr) == d) break;
  }
  return buf;
}

Json phases_json(unsigned m) {
  Json a = Json::array();
  for (int p = 0; p < kPlacePhases; ++p)
    if (m & (1u << p)) a.push(Json::string(place_phase_word(p)));
  return a;
}

std::string phases_text(unsigned m) {
  std::string s;
  for (int p = 0; p < kPlacePhases; ++p)
    if (m & (1u << p)) s += std::string(s.empty() ? "" : ",") + place_phase_word(p);
  return s;
}

struct Line {
  std::string component, name, source;  // name: the charge's (optional)
  ChargeAt at = ChargeAt::Device;
  int64_t bytes = 0, peak_extra = 0, slots = -1;  // slots: elastic lines only
  int64_t allocs = 0;  // separate allocations (rev 17; the charge's, 0 for the elastic lines: slot_allocs bills them)
  bool driver = false;  // kind "driver" (rev 18): printed on the line, the same arithmetic
  unsigned phases = kAllPhases;
  int64_t live() const { return bytes + peak_extra; }
};

struct Dev {
  std::string set;
  bool anchor = false, bound = false, enabled = false;
  const RDevice* rd = nullptr;
  const BoundMember* bm = nullptr;
  double budget_gib = 0, margin_gib = 0, margin_fraction = 0;
  bool margin_default = true;
  int64_t memory = 0, budget = 0, margin = 0, context = 0, growth = 0, allowance = 0, measured = -1;
  int64_t granularity = 0, overhead = 0;  // G of the context receipt; alloc_overhead_bytes (max over phases)
  const Components::ContextReceipt* receipt = nullptr;  // the matched context receipt (V3)
  bool touched = false;
  std::vector<Line> lines;
  int64_t peak[kPlacePhases] = {0, 0, 0, 0};
  int64_t max_peak() const { return *std::max_element(peak, peak + kPlacePhases); }
  int64_t room() const { return budget - margin - max_peak(); }
  void add_line(const Line& l) {
    lines.push_back(l);
    for (int p = 0; p < kPlacePhases; ++p)
      if (l.phases & (1u << p)) peak[p] = add(peak[p], l.live());
  }
};

// Ids that run only on the anchor in v1 (contract §3.7 V4): placed off it -> PL_E_ANCHORED unless the engine
// declares the capability that moves them.
bool anchored_id(const std::string& id) {
  for (const char* a : {"context", "dense", "head", "mtp", "buffers", "state", "window", "graphs", "spares", "arena",
                        "layer_slots"})
    if (id == a) return true;
  return false;
}
const char* unanchor_capability(const std::string& id) {
  return id == "head" ? "assist_head" : id == "dense" ? "layers" : nullptr;
}
// Destinations that exist only through a capability (contract §0 / §8).
const char* capability_of(const std::string& id) {
  return id == "expert_assist"    ? "staged_expert_tier"
         : id == "expert_compute" ? "peer_expert_compute"
         : id == "expert_file"    ? "file_expert_tier"
                                  : nullptr;
}
bool capability_ok(const Components& c, const std::string& id) {
  if (id == "expert_assist") return c.has("staged_expert_tier") || c.has("peer_expert_tier");
  const char* cap = capability_of(id);
  return !cap || c.has(cap);
}

std::string dflt_of(const std::string& id) {
  return component_table()[size_t(component_priority(id) - 1)].dflt;
}

}  // namespace

uint64_t frac_ppm(double fraction) {
  require(fraction >= 0 && fraction <= 1, SS_E_RECIPE, "PL_E_SCHEMA margin_fraction out of [0, 1]");
  // decimal text -> digits D and exponent E (value = D x 10^E), then D x 10^(E+6) rounded half up
  const std::string t = shortest(fraction);
  std::string digits;
  int exp10 = 0;
  bool frac_part = false;
  size_t i = 0;
  for (; i < t.size() && t[i] != 'e' && t[i] != 'E'; ++i) {
    if (t[i] == '.') {
      frac_part = true;
    } else {
      digits += t[i];
      if (frac_part) --exp10;
    }
  }
  if (i < t.size()) exp10 += std::atoi(t.c_str() + i + 1);
  size_t nz = digits.find_first_not_of('0');
  digits = nz == std::string::npos ? "0" : digits.substr(nz);
  const int shift = exp10 + 6;
  if (shift >= 0) {
    uint64_t v = std::strtoull(digits.c_str(), nullptr, 10);
    for (int k = 0; k < shift; ++k) v *= 10;
    return v;
  }
  const size_t drop = size_t(-shift);
  if (drop > digits.size()) return 0;  // < 0.5 ppm (fewer digits than dropped places, leading digit < 5 x 10^-1)
  const std::string keep = digits.substr(0, digits.size() - drop), rest = digits.substr(digits.size() - drop);
  uint64_t v = keep.empty() ? 0 : std::strtoull(keep.c_str(), nullptr, 10);
  if (!rest.empty() && rest[0] >= '5') ++v;  // half up
  return v;
}

PlaceResult place(const PlaceInput& in) {
  PlaceResult res;
  require(in.recipe && in.binding, SS_E_RECIPE, "place: no recipe or binding");
  const Recipe& R = *in.recipe;
  const RRoute* rt = R.route(in.route);
  require(rt != nullptr, SS_E_RECIPE, "place: unknown route \"" + in.route + "\"");
  const std::string anchor_set = rt->devices[0];
  const RPlacement* P = rt->placement(anchor_set);
  const std::string rw = "routes." + rt->id;
  std::vector<std::string>& errs = res.errors;
  auto stop = [&](const char* phase) {
    if (errs.empty()) return false;
    res.phase = phase;
    return true;
  };

  // ---- V1 schema (what needs the descriptor; the recipe parser did the rest) ----------------------------------
  require(rt->execute == Execute::Single, SS_E_RECIPE,
          "PL_E_SCHEMA " + rw + ": a placement has one anchor (execute = \"single\")");
  const Components* C = in.components;
  const RComponent* x_arena = rt->component("arena");
  if (C && C->has_elastic && x_arena && (x_arena->has_gib || x_arena->units >= 0) &&
      P->get("dynamic_slots_per_layer") > 0) {
    const int64_t want = x_arena->units >= 0 ? x_arena->units : int64_t(gib_bytes(x_arena->gib) / C->unit_bytes);
    const int64_t layers = C->per_layer_units ? int64_t(C->units_total / C->per_layer_units) : 0;
    const int64_t legacy = int64_t(P->get("dynamic_slots_per_layer")) * layers;
    if (legacy != want)
      errs.push_back("PL_E_CONFLICT " + rw + ".components.arena.gib (" + num(want) + " active units) vs placement." +
                     anchor_set + ".dynamic_slots_per_layer (" + num(legacy) + " units)");
  }
  if (stop("V1")) return res;

  // ---- V2 binding (S0) ------------------------------------------------------------------------------------------
  std::vector<Dev> devs;
  auto member = [&](const std::string& set) -> const BoundMember* {
    const std::vector<BoundMember>* m = in.binding->of(set);
    return m && !m->empty() ? &(*m)[0] : nullptr;
  };
  auto mem_of = [](const BoundMember* b) {
    return b->memory_bytes >= 0 ? b->memory_bytes : i64(gib_bytes(std::min(b->memory_gib, 1048576.0)));
  };
  {
    Dev a;
    a.set = anchor_set;
    a.anchor = true;
    a.rd = R.device(anchor_set);
    a.bm = member(anchor_set);
    require(a.bm != nullptr, SS_E_RECIPE, "place: the anchor set \"" + anchor_set + "\" is not bound");
    a.bound = a.enabled = a.touched = true;
    a.budget_gib = P->get("device_gib");
    a.margin_gib = P->get("margin_gib");
    a.margin_fraction = P->get("margin_fraction");
    a.margin_default = a.margin_gib == 2.0 && a.margin_fraction == 0.03;
    devs.push_back(a);
  }
  for (const RAssist& as : rt->assists) {  // sorted by set id
    Dev d;
    d.set = as.set;
    d.rd = R.device(as.set);
    d.bm = member(as.set);
    d.bound = d.bm != nullptr;
    d.enabled = d.bound && as.device_gib > 0;
    d.budget_gib = as.device_gib;
    d.margin_gib = as.margin_gib;
    d.margin_fraction = as.margin_fraction;
    d.margin_default = as.margin_gib == 0.5 && as.margin_fraction == 0.03;
    devs.push_back(d);
  }
  for (Dev& d : devs) {
    d.budget = i64(gib_bytes(d.budget_gib));
    d.margin = std::max(i64(gib_bytes(d.margin_gib)), mul(d.budget, i64(frac_ppm(d.margin_fraction))) / 1000000);
    if (!d.bound) continue;
    d.memory = mem_of(d.bm);
    if (d.budget > d.memory)
      errs.push_back("PL_E_BUDGET_ABOVE_DEVICE " + d.set + ": budget " + num(d.budget) + " > device memory " +
                     num(d.memory) + " (" + d.bm->name + ")");
  }
  auto dev_of = [&](const std::string& set) -> Dev* {
    for (Dev& d : devs)
      if (d.set == set) return &d;
    return nullptr;
  };
  Dev& A = devs[0];
  for (const RComponent& c : rt->components) {
    Dev* d = dev_of(c.device);
    if (d && !d->anchor && !d->enabled)
      errs.push_back("PL_E_DEVICE_DISABLED " + c.id + " -> " + c.device +
                     (d->bound ? " (assist device_gib = 0)" : " (the optional set is unbound)"));
  }
  if (stop("V2")) return res;

  // ---- V3 descriptor ------------------------------------------------------------------------------------------
  if (!C) {
    errs.push_back("PL_E_MISSING_DESCRIPTOR " + rw + ": placement mode needs the engine's components-v1");
    stop("V3");
    return res;
  }
  if (C->cache.empty())
    errs.push_back("PL_E_SCHEMA components-v1 config.cache is required in placement mode (\"tierstream\" for the "
                   "TierSite adapter, \"engine\" for the engine cache)");
  if (C->cache_config.is_null())
    errs.push_back("PL_E_SCHEMA components-v1 config.cache_config is required in placement mode (every option that "
                   "sizes or steers the expert cache)");
  const int64_t want_seq = rt->max_sequences > 0 ? rt->max_sequences : int64_t(P->get("max_batch"));
  const int64_t want_rows = rt->max_rows > 0 ? rt->max_rows : 1;
  auto stale = [&](const std::string& what, const std::string& doc, const std::string& here) {
    if (doc != here) errs.push_back("PL_E_STALE " + what + ": components-v1 " + doc + ", recipe " + here);
  };
  stale("config.max_context_tokens", num(C->max_context_tokens), num(int64_t(P->get("max_context_tokens"))));
  stale("config.max_sequences", num(C->max_sequences), num(want_seq));
  stale("config.max_rows", num(C->max_rows), num(want_rows));
  // chunk rows and the other config options are the engine's own: it checks them at load (contract §3.1.3)
  if (!R.engine.empty()) stale("engine", C->engine, R.engine);
  if (!R.model_sha256.empty()) stale("model_sha256", C->model_sha256, R.model_sha256);
  if (in.desc) {
    const std::string t = json_write(desc_to_json(*in.desc));
    stale("desc_sha256", C->desc_sha256, sha256_hex(t.data(), t.size()));
  }
  // touched devices: the anchor and every enabled assist that is an explicit destination
  for (const RComponent& c : rt->components) {
    Dev* d = dev_of(c.device);
    if (d && d->enabled) d->touched = true;
  }
  for (Dev& d : devs) {
    if (!d.touched) continue;
    if (d.bm->arch.empty()) {
      errs.push_back("PL_E_MISSING_CONTEXT " + d.set + ": the probe reports no arch for " + d.bm->id);
      continue;
    }
    auto it = C->context.find(d.bm->arch);
    if (it == C->context.end()) {
      errs.push_back("PL_E_MISSING_CONTEXT " + d.set + ": components-v1 has no context for " + d.bm->arch);
      continue;
    }
    const Components::ContextReceipt* matched = nullptr;
    {  // the allowance applies only with a receipt for exactly this binding: arch, name, driver, platform, build
      const std::string want =
          d.bm->arch + "/" + d.bm->name + "/" + d.bm->driver + "/" + d.bm->platform + "/" + C->build_sha256;
      // the message names the first differing key field of the same-arch receipt with the longest matching prefix
      // in the order arch, device_name, driver, platform, build_sha256 ("arch" when no receipt has this arch)
      static const char* const kFields[] = {"arch", "device_name", "driver", "platform", "build_sha256"};
      int best = 0;
      for (const auto& r : C->context_receipts) {
        const std::string mine[5] = {d.bm->arch, d.bm->name, d.bm->driver, d.bm->platform, C->build_sha256};
        const std::string theirs[5] = {r.arch, r.device_name, r.driver, r.platform, r.build_sha256};
        int k = 0;
        while (k < 5 && mine[k] == theirs[k]) ++k;
        if (k == 5) matched = &r;
        best = std::max(best, k);
      }
      if (!matched) {
        errs.push_back("PL_E_MISSING_CONTEXT " + d.set + ": no context receipt for " + want + " (mismatch: " +
                       kFields[best] + ")");
        continue;
      }
    }
    d.context = i64(it->second);
    d.granularity = i64(matched->alloc_granularity_bytes);  // G: billed per separate allocation (rev 17)
    d.receipt = matched;
    auto g = C->context_growth.find(d.bm->arch);
    d.growth = g == C->context_growth.end() ? 0 : i64(g->second);
    // rev 18, per touched device after the key: the receipt's configuration envelope, then the growth rule
    if (matched->envelope_sha256.empty() || matched->envelope_sha256 != C->envelope_sha256) {
      errs.push_back("PL_E_MISSING_CONTEXT " + d.set +
                     ": receipt qualified on another configuration envelope (mismatch: envelope)");
      continue;
    }
    constexpr int64_t kGrowthFloor = int64_t(64) << 20;  // context_growth >= 2 x max_growth_bytes + 64 MiB (§2.4)
    if (d.growth < add(mul(2, i64(matched->max_growth_bytes)), kGrowthFloor)) {
      errs.push_back("PL_E_MISSING_CONTEXT " + d.set + ": context_growth " + num(d.growth) + " < 2 x max_growth_bytes " +
                     num(i64(matched->max_growth_bytes)) + " + " + num(kGrowthFloor));
      continue;
    }
    // S2 charges max(the engine's allowance, the measured context-v1 value) + growth
    d.allowance = d.context;
    if (in.context)
      if (const ContextEntry* m = in.context->find(C->build_sha256, d.bm->arch, d.bm->name, d.bm->driver,
                                                       d.bm->platform)) {
        const int64_t agg = context_aggregate(*m);  // median of the latest 5 samples; -1 = fewer than 3
        if (agg > mul(2, d.context)) {
          errs.push_back("PL_E_STALE " + d.set + ": context outlier: re-qualify (context-v1 " + num(agg) +
                         " > 2 x the allowance " + num(d.context) + ")");
          continue;
        }
        if (agg >= 0) {
          d.measured = agg;
          d.allowance = std::max(d.allowance, d.measured);
        }
      }
  }
  // rev 18: the `graphs` driver charge (on the anchor: graphs is anchored) is the receipt's qualified graph bytes
  if (errs.empty())
    if (const ComponentDecl* g = C->find("graphs"))
      if (A.receipt && A.receipt->graph_bytes >= 0) {
        int64_t declared = 0;
        for (const Charge& ch : *g->for_role(false))
          if (ch.driver) declared = add(declared, i64(ch.bytes));
        if (declared != A.receipt->graph_bytes)
          errs.push_back("PL_E_STALE graphs: driver charge " + num(declared) + " != receipt graph_bytes " +
                         num(A.receipt->graph_bytes));
      }
  if (stop("V3")) return res;

  // ---- V4 feasibility: S1 destinations ------------------------------------------------------------------------
  struct Placed {
    std::string id, dest, source;  // dest: a set id, host_pinned, host_pageable or file
    const ComponentDecl* decl = nullptr;
  };
  std::vector<Placed> placed;
  for (const ComponentInfo& info : component_table()) {
    const std::string id = info.id;
    const RComponent* x = rt->component(id);
    const ComponentDecl* decl = C->find(id);
    const bool elastic_decl = C->has_elastic && (id == "arena" || id == "spares" || id == "expert_host");
    const bool declared = id == "context" || decl || elastic_decl;
    const char* cap = capability_of(id);
    if (x) {
      if (x->has_layers) {
        errs.push_back("PL_E_CAPABILITY " + id + ": layers needs layer-granular placement (capability \"layers\"), "
                       "not planned in v1");
        continue;
      }
      if ((id == "layer_slots" || id == "layer_host") && (x->has_gib || x->units >= 0)) {
        errs.push_back("PL_E_CAPABILITY " + id + ": needs layer-granular streaming");
        continue;
      }
      if (cap && !capability_ok(*C, id)) {
        errs.push_back("PL_E_CAPABILITY " + id + ": needs " + cap);
        continue;
      }
      if (!declared && !cap) {
        errs.push_back("PL_E_CAPABILITY " + id + ": not declared by the engine");
        continue;
      }
    }
    if (!declared && !(cap && x)) continue;
    if (id == "context") continue;  // billed per touched device, never placed
    const std::string want = x ? x->device : "auto";
    const std::string dflt = info.dflt;
    Placed pl;
    pl.id = id;
    pl.decl = decl;
    pl.source = x && x->device != "auto" ? "recipe" : "default";
    if (want == "auto") {
      if (dflt.empty()) {
        if (x && (x->has_gib || x->units >= 0))
          errs.push_back("PL_E_UNKNOWN_DEVICE " + id + ": auto never chooses an assist; name the assist device");
        continue;  // no default destination: not placed
      }
      pl.dest = dflt == "anchor" ? anchor_set : dflt;
    } else if (want == anchor_set) {
      if (dflt != "anchor" && !(id == "expert_compute")) {
        errs.push_back("PL_E_CAPABILITY " + id + " -> " + want + ": destination not supported (default " +
                       (dflt.empty() ? std::string("an assist") : dflt) + ")");
        continue;
      }
      if (id == "expert_assist" || id == "expert_compute") {
        errs.push_back("PL_E_CAPABILITY " + id + " -> " + want + ": an assist destination");
        continue;
      }
      pl.dest = anchor_set;
    } else if (Dev* d = dev_of(want)) {  // an assist (enabled: V2)
      if (anchored_id(id)) {
        const char* un = unanchor_capability(id);
        if (!un || !C->has(un)) {
          errs.push_back("PL_E_ANCHORED " + id);
          continue;
        }
      } else if (dflt != "anchor" && !dflt.empty()) {
        errs.push_back(id == "engram" ? "PL_E_CAPABILITY engram -> " + want + ": needs engram_cache"
                                      : "PL_E_CAPABILITY " + id + " -> " + want + ": destination not supported");
        continue;
      }
      if (decl && decl->has_variants && !decl->has_assist) {
        errs.push_back("PL_E_ANCHORED " + id);
        continue;
      }
      if (d->rd->kind != SS_BACKEND_CUDA) {
        errs.push_back("PL_E_CAPABILITY " + id + " -> " + want + ": HIP placement is not in v1");
        continue;
      }
      if (d->bm->host != A.bm->host) {
        errs.push_back("PL_E_CAPABILITY " + id + " -> " + want + ": the assist is on host " + d->bm->host +
                       ", the anchor on " + A.bm->host + " (one host in v1)");
        continue;
      }
      if (decl) {
        if (decl->requires_relay && !C->find("relay")) {
          errs.push_back("PL_E_CAPABILITY " + id + " -> " + want + ": needs the relay component");
          continue;
        }
      }
      pl.dest = want;
    } else {  // host_pinned | host_pageable | file
      if (want != dflt) {
        errs.push_back(id == "engram" && want == "host_pinned"
                           ? "PL_E_CAPABILITY engram -> host_pinned: needs engram_cache"
                           : "PL_E_CAPABILITY " + id + " -> " + want + ": destination not supported");
        continue;
      }
      pl.dest = want;
    }
    if (decl && decl->has_variants && !decl->for_role(dev_of(pl.dest) && !dev_of(pl.dest)->anchor)) {
      errs.push_back("PL_E_CAPABILITY " + id + " -> " + pl.dest + ": the engine declares no variant for it");
      continue;
    }
    placed.push_back(pl);
  }
  // A movable component (one with variants) needs a qualification receipt for the arch it lands on, explicit or auto
  // (contract §5.2): the product default (the drafter on the anchor) needs the anchor arch's receipt too.
  for (const Placed& p : placed) {
    const Dev* d = dev_of(p.dest);
    if (!p.decl || !p.decl->has_variants || !d) continue;
    bool ok = false;
    for (const Receipt& r : p.decl->qualified) ok = ok || (r.arch == d->bm->arch && r.build_sha256 == C->build_sha256);
    if (!ok)
      errs.push_back("PL_E_ARCH_UNQUALIFIED " + p.id + " on " + d->set + " (" + d->bm->arch +
                     "): no qualification receipt for build " + C->build_sha256);
  }
  // an engine declaring FILE homes without the capability (G4 I3b); per_layer spare billing is not in v1
  if (C->find("expert_file") && !C->has("file_expert_tier")) {
    const std::string e = "PL_E_CAPABILITY expert_file: needs file_expert_tier";
    if (std::find(errs.begin(), errs.end(), e) == errs.end()) errs.push_back(e);
  }
  if (C->has_elastic && C->device_gated() && C->spare_billing_per_layer)
    errs.push_back("PL_E_CAPABILITY spare_billing: per_layer is not in v1 (placement mode requires elastic.spare_billing "
                   "= \"global\")");
  for (const Placed& p : placed)
    if (Dev* d = dev_of(p.dest)) d->touched = true;
  int n_touched = 0;
  for (const Dev& d : devs) n_touched += d.touched ? 1 : 0;
  if (n_touched > 1 && !C->has("multi_device"))
    errs.push_back("PL_E_CAPABILITY multi_device: the engine has not admitted multi-device processes");
  for (const Dev& d : devs)  // rev 18: no HIP stack_bytes derivation yet, so no HIP anchor or assist in placement mode
    if (d.touched && d.rd->kind == SS_BACKEND_HIP)
      errs.push_back("PL_E_CAPABILITY " + d.set + ": HIP placement is not in v1 (no stack_bytes derivation)");
  if (stop("V4")) return res;

  // ---- S3a quota bounds (before anything is sized) --------------------------------------------------------------
  const bool elastic = C->has_elastic;
  const int64_t unit = elastic ? i64(C->unit_bytes) : 0, total = elastic ? i64(C->units_total) : 0;
  // explicit quotas: `units`, or `gib` -> floor(gib_bytes(gib) / unit_bytes); -1 = none
  auto quota_units = [&](const char* id) -> int64_t {
    const RComponent* x = rt->component(id);
    if (!x || !elastic) return -1;
    if (x->units >= 0) return x->units;
    return x->has_gib ? i64(gib_bytes(x->gib)) / unit : -1;
  };
  const int64_t q_arena = quota_units("arena"), q_assist = quota_units("expert_assist"),
                q_file = quota_units("expert_file"), q_host = quota_units("expert_host");
  if (elastic) {
    int64_t sum = 0;
    for (const auto& q : {std::make_pair("arena", q_arena), std::make_pair("assist", q_assist),
                          std::make_pair("file", q_file)}) {
      if (q.second > total)
        errs.push_back("PL_E_BUDGET coverage: " + std::string(q.first) + " quota " + num(q.second) + " outside [0, " +
                       num(total) + "] experts");
      sum += std::max<int64_t>(q.second, 0);
    }
    if (errs.empty() && sum > total)
      errs.push_back("PL_E_BUDGET coverage: quotas " + num(sum) + " (arena + assist + file) > " + num(total) +
                     " experts");
    // S3 step 0, device-gated adapter (dg_layout :110-134), before any assignment: D >= 1, Q >= 1, and the D spares
    // must find non-resident units after the explicit quotas (the automatic fill leaves them room by construction)
    if (errs.empty() && C->device_gated() && unit % 256)  // dg_layout refuses parts not a multiple of 256 bytes
      errs.push_back("PL_E_SCHEMA device-gated: unit bytes " + num(unit) + " not a multiple of 256");
    if (errs.empty() && C->device_gated()) {
      if (i64(C->spares) < 1 || i64(C->host_spares) < 1) {
        errs.push_back("PL_E_SCHEMA device-gated: spares " + num(i64(C->spares)) + " / host spares " +
                       num(i64(C->host_spares)) + " must be >= 1");
      } else {
        const int64_t left =
            total - std::max<int64_t>(q_file, 0) - std::max<int64_t>(q_arena, 0) - std::max<int64_t>(q_assist, 0);
        if (left < i64(C->spares))
          errs.push_back("PL_E_BUDGET spares need non-resident units: " + num(std::max<int64_t>(left, 0)) +
                         " left for " + num(i64(C->spares)) + " spares");
      }
    }
  }
  if (stop("V4")) return res;

  // ---- S2 reserve -----------------------------------------------------------------------------------------------
  std::vector<Line> host_lines;  // host_pinned, host_pageable, file
  for (const Placed& p : placed) {
    if (!p.decl) continue;
    Dev* d = dev_of(p.dest);
    const bool assist_role = d && !d->anchor;
    Dev& on = d ? *d : A;  // device charges of a host-placed component bill the anchor
    for (const Charge& ch : *p.decl->for_role(assist_role)) {
      Line l;
      l.component = p.id;
      l.name = ch.name;
      l.source = p.source;
      l.at = ch.at;
      l.bytes = i64(ch.bytes);
      l.peak_extra = i64(ch.peak_extra);
      l.phases = ch.phases;
      l.allocs = i64(ch.allocs);
      l.driver = ch.driver;
      if (ch.at == ChargeAt::Device)
        on.add_line(l);
      else
        host_lines.push_back(l);
    }
  }
  if (elastic) {  // device spares: free slots for demand misses, on the anchor, every phase (load too)
    Line l;
    l.component = "spares";
    l.source = "default";
    l.slots = i64(C->spares);
    l.bytes = mul(l.slots, unit);
    A.add_line(l);
  }
  for (Dev& d : devs) {
    if (!d.touched) continue;
    const int64_t ctx = add(d.allowance, d.growth);  // max(context, context-v1) + context_growth, once per device
    // physical allocation overhead (rev 17): G x (allocs of the lines live in the phase + slot_allocs on the anchor);
    // the lines keep their payload bytes
    for (int p = 0; p < kPlacePhases; ++p) {
      int64_t allocs = d.anchor && elastic ? i64(C->slot_allocs) : 0;
      for (const Line& l : d.lines)
        if (l.phases & (1u << p)) allocs = add(allocs, l.allocs);
      const int64_t over = mul(d.granularity, allocs);
      d.overhead = std::max(d.overhead, over);
      d.peak[p] = add(add(d.peak[p], ctx), over);
    }
    if (d.max_peak() > d.budget - d.margin)
      errs.push_back("PL_E_BUDGET " + d.set + ": need " + num(d.max_peak()) + " > budget-margin " +
                     num(d.budget - d.margin) + "; knob: " +
                     (d.anchor ? rw + ".placement." + d.set + ".device_gib" : rw + ".assist." + d.set + ".device_gib") +
                     " (or move a component)");
  }
  if (stop("V4")) return res;

  // ---- S3 elastic sizing ---------------------------------------------------------------------------------------
  int64_t assist_units = 0, file_units = 0, active = 0, spilled = 0, blocks = 0, homes = 0;
  const int64_t host_spares = elastic && C->exclusive ? i64(C->host_spares) : 0;
  const bool explicit_file = q_file >= 0;
  std::string assist_set;
  if (elastic) {
    for (const Placed& p : placed)
      if (p.id == "expert_assist") {
        assist_units = std::max<int64_t>(q_assist, 0);
        Dev& d = *dev_of(p.dest);
        assist_set = d.set;
        Line l;
        l.component = "expert_assist";
        l.source = "recipe";
        l.slots = assist_units;
        l.bytes = mul(assist_units, unit);
        if (l.bytes > d.room())
          errs.push_back("PL_E_BUDGET " + d.set + ": expert_assist " + num(l.bytes) + " > room " + num(d.room()));
        d.add_line(l);
      }
    file_units = std::max<int64_t>(q_file, 0);
    if (stop("V4")) return res;
    // quotas are reserved before the arena fill (S3a before S3b)
    const int64_t avail = total - assist_units - file_units;
    // device-gated adapter: the device spares must find non-resident units (dg_spare_slots <= P); the engine cache
    // holds every expert plus its spare slots (G4 P2)
    const int64_t dev_spares = C->device_gated() ? i64(C->spares) : 0;
    const int64_t room = A.room();
    if (q_arena >= 0) {
      if (mul(q_arena, unit) > room)
        errs.push_back("PL_E_BUDGET " + A.set + ": arena " + num(mul(q_arena, unit)) + " > room " + num(room));
      active = q_arena;
    } else {
      active = std::max<int64_t>(0, std::min(room / unit, avail - dev_spares));
    }
    if (errs.empty() && active + assist_units + file_units > total)
      errs.push_back("PL_E_BUDGET coverage: active " + num(active) + " + file " + num(file_units) +
                     (assist_units ? " + assist " + num(assist_units) : "") + " > " + num(total) + " experts");
    if (stop("V4")) return res;
    Line l;
    l.component = "arena";
    l.source = q_arena >= 0 ? "recipe" : "fill";
    l.slots = active;
    l.bytes = mul(active, unit);
    A.add_line(l);
    homes = C->exclusive ? total - active - assist_units - file_units : total - assist_units - file_units;
  }

  // ---- S4 host checks (S3 step 0 first: the mandatory pinned) -----------------------------------------------
  int64_t other_pinned = 0, pageable = 0;
  // pinned overhead per allocation: the anchor receipt's qualified host granularity (rev 18; default 4 KiB)
  const int64_t hgran = A.receipt ? i64(A.receipt->host_alloc_granularity_bytes) : 4096;
  for (const Line& l : host_lines) {
    if (l.at == ChargeAt::HostPinned)  // payload + one granule per allocation (rev 17/18)
      other_pinned = add(other_pinned, add(l.live(), mul(hgran, l.allocs)));
    if (l.at == ChargeAt::HostPageable) pageable = add(pageable, l.live());
  }
  const int64_t pinned_budget = i64(gib_bytes(P->get("host_pinned_gib")));
  // host blocks are billed at the host stride (round_up(unit_bytes, 4096); equal to unit_bytes for DS)
  const int64_t hstride = elastic ? (unit + 4095) / 4096 * 4096 : 0;
  // expert host blocks are allocated in chunks of at most host_chunk_bytes: one granule per chunk (rev 18)
  const int64_t hchunk = i64(C->host_chunk_bytes);
  auto homes_bytes = [&](int64_t n) {
    const int64_t b = mul(n, hstride);
    return add(b, mul(hgran, ceil_div(b, hchunk)));
  };
  if (add(other_pinned, homes_bytes(host_spares)) > pinned_budget) {
    // nothing elastic is sized on a host that cannot hold its fixed charges (no negative homes)
    errs.push_back("PL_E_HOST_PINNED: mandatory pinned " + num(add(other_pinned, homes_bytes(host_spares))) +
                   " (other " + num(other_pinned) + " + host spares " + num(homes_bytes(host_spares)) + ") > " +
                   num(pinned_budget));
    stop("V4");
    return res;
  }
  blocks = add(homes, host_spares);
  int64_t pinned = add(homes_bytes(blocks), other_pinned);
  if (elastic && pinned > pinned_budget && !explicit_file && C->has("file_expert_tier")) {
    // S3e: VRAM arena -> pinned RAM -> NVMe; the coldest whole units go to the file tier, never the host spares
    // never below the host spares; for a device-gated adapter never below the device spares' needs (D <= P)
    const int64_t floor_d = C->device_gated() ? std::min(homes, i64(C->spares)) : 0;
    const int64_t cap = homes - floor_d;
    spilled = std::min(ceil_div(pinned - pinned_budget, hstride), cap);
    while (spilled < cap && add(homes_bytes(blocks - spilled), other_pinned) > pinned_budget)
      ++spilled;  // the chunk granules can need one more unit than the stride division says
    file_units += spilled;
    homes -= spilled;
    blocks -= spilled;
    pinned = add(homes_bytes(blocks), other_pinned);
  }
  int64_t short_units = 0;  // the fewest blocks whose removal fits the budget
  while (pinned > pinned_budget && short_units < blocks &&
         add(homes_bytes(blocks - short_units), other_pinned) > pinned_budget)
    ++short_units;
  if (pinned > pinned_budget)
    errs.push_back("PL_E_HOST_PINNED: need " + num(pinned) + " > " + num(pinned_budget) + " (expert homes " +
                   num(blocks) + " blocks + other " + num(other_pinned) + "); " + num(short_units) +
                   " units short; knob: " + rw +
                   ".placement." + anchor_set + ".host_pinned_gib" +
                   (C->has("file_expert_tier") ? "" : " (or a file tier: file_expert_tier is not declared)"));
  if (q_host >= 0 && blocks > q_host)
    errs.push_back("PL_E_HOST_PINNED expert_host: " + num(blocks) + " blocks > cap " + num(q_host) + " (" + rw +
                   ".components.expert_host)");
  const int64_t pageable_budget = i64(gib_bytes(P->get("host_pageable_gib")));
  if (pageable_budget > 0 && pageable > pageable_budget)
    errs.push_back("PL_E_HOST_PAGEABLE: need " + num(pageable) + " > " + num(pageable_budget));
  if (elastic) {
    Line h;
    h.component = "expert_host";
    h.at = ChargeAt::HostPinned;
    h.source = "fill";
    h.slots = blocks;
    h.bytes = homes_bytes(blocks);
    host_lines.push_back(h);
    Line f;
    f.component = "expert_file";
    f.at = ChargeAt::File;
    f.source = explicit_file ? "recipe" : "fill";
    f.slots = file_units;
    f.bytes = mul(file_units, unit);
    host_lines.push_back(f);
  }
  res.computed = true;
  if (!errs.empty()) res.phase = "V4";

  // ---- unit assignment (§3.1 usage rank; placement-v1 `units`) ---------------------------------------------------
  Json units_json = Json::array(), seed_json = Json::array();
  std::vector<uint8_t> bucket;  // per global unit: 0 active, 1 host, 2 file, 3 assist
  std::vector<int64_t> layer_units;  // units per layer
  if (elastic) {
    if (in.desc) {
      for (const Layer& l : in.desc->layers)
        if (l.kind == LayerKind::Moe && l.streamable) layer_units.push_back(l.n_units);
    } else if (C->per_layer_units) {
      layer_units.assign(size_t(total / i64(C->per_layer_units)), i64(C->per_layer_units));
    } else {
      layer_units.push_back(total);
    }
    int64_t sum = 0;
    for (int64_t n : layer_units) sum += n;
    if (sum != total || (C->per_layer_units && in.desc &&
                         std::any_of(layer_units.begin(), layer_units.end(),
                                     [&](int64_t n) { return n != i64(C->per_layer_units); }))) {
      errs.push_back("PL_E_STALE desc projection: the descriptor's streamable expert units (" + num(sum) +
                     ") do not match components-v1 elastic (" + num(total) + " units, " +
                     num(i64(C->per_layer_units)) + " per layer)");
      res.phase = "V4";
    } else {
      // rank: usage desc, recency desc (usage-v1); else the descriptor's benefit floor(b x 2^32) desc; else model
      // order; ties by (layer, unit) ascending
      struct Key {
        int64_t a = 0, b = 0, layer = 0, unit = 0, g = 0;
      };
      std::vector<Key> keys;
      std::vector<int64_t> base(layer_units.size(), 0);
      for (size_t L = 1; L < layer_units.size(); ++L) base[L] = base[L - 1] + layer_units[L - 1];
      std::map<std::pair<int64_t, int64_t>, const UsageUnit*> use;
      if (in.usage) {
        // usage-v1 layers with units (dense layers carry none) must be the streamable layers, in order
        std::vector<int64_t> shape;
        for (int64_t n : in.usage->layer_units)
          if (n) shape.push_back(n);
        if (shape != layer_units) {
          errs.push_back("PL_E_STALE usage-v1: its layers with units (" + num(int64_t(shape.size())) +
                         ") do not match the streamable expert layers (" + num(int64_t(layer_units.size())) + ")");
          res.phase = "V4";
        }
        for (const UsageUnit& u : in.usage->units) use[{u.layer, u.unit}] = &u;
      }
      std::vector<const Layer*> moe;
      if (in.desc)
        for (const Layer& l : in.desc->layers)
          if (l.kind == LayerKind::Moe && l.streamable) moe.push_back(&l);
      for (size_t L = 0; L < layer_units.size(); ++L)
        for (int64_t u = 0; u < layer_units[L]; ++u) {
          Key k;
          k.layer = int64_t(L);
          k.unit = u;
          k.g = base[L] + u;
          if (in.usage) {
            auto it = use.find({int64_t(L), u});
            if (it != use.end()) {
              k.a = it->second->selections;
              k.b = it->second->recency;
            }
          } else if (!moe.empty() && !moe[L]->benefit.empty()) {
            const double bf = moe[L]->benefit[size_t(u)];
            k.a = bf > 0 ? int64_t(std::floor(std::min(bf, 1e9) * 4294967296.0)) : 0;
          }
          keys.push_back(k);
        }
      std::stable_sort(keys.begin(), keys.end(), [](const Key& x, const Key& y) {
        if (x.a != y.a) return x.a > y.a;
        if (x.b != y.b) return x.b > y.b;
        return x.layer != y.layer ? x.layer < y.layer : x.unit < y.unit;
      });
      bucket.assign(size_t(total), 1);
      std::vector<int64_t> per_layer(layer_units.size(), 0);
      int64_t left = active;
      std::vector<char> taken(keys.size(), 0);
      for (size_t i = 0; i < keys.size() && left > 0; ++i) {  // hottest -> active (per-layer limit when declared)
        if (C->per_layer_max_active && per_layer[size_t(keys[i].layer)] >= i64(C->per_layer_max_active)) continue;
        bucket[size_t(keys[i].g)] = 0;
        ++per_layer[size_t(keys[i].layer)];
        taken[i] = 1;
        --left;
      }
      int64_t f = file_units, as = assist_units;
      for (size_t i = keys.size(); i-- > 0 && (f > 0 || as > 0);) {  // coldest -> file, then assist
        if (taken[i]) continue;
        bucket[size_t(keys[i].g)] = f > 0 ? 2 : 3;
        if (f > 0)
          --f;
        else
          --as;
      }
      auto ranges = [](const std::vector<int64_t>& ids) {
        Json a = Json::array();
        for (size_t i = 0; i < ids.size();) {
          size_t j = i + 1;
          while (j < ids.size() && ids[j] == ids[j - 1] + 1) ++j;
          Json r = Json::array();
          r.push(Json::number(ids[i]));
          r.push(Json::number(ids[j - 1] + 1));  // half-open
          a.push(r);
          i = j;
        }
        return a;
      };
      for (size_t i = 0; i < keys.size(); ++i)
        if (taken[i]) {
          Json pr = Json::array();
          pr.push(Json::number(keys[i].layer));
          pr.push(Json::number(keys[i].unit));
          seed_json.push(pr);
        }
      for (size_t L = 0; L < layer_units.size(); ++L) {
        std::vector<int64_t> ids[4];
        for (int64_t u = 0; u < layer_units[L]; ++u) ids[bucket[size_t(base[L] + u)]].push_back(u);
        Json e = Json::object();
        e["layer"] = Json::number(int64_t(L));
        e["active"] = ranges(ids[0]);
        e["host"] = ranges(ids[1]);
        e["file"] = ranges(ids[2]);
        e["assist"] = ranges(ids[3]);
        units_json.push(e);
      }
    }
  }

  // ---- S5 library plan: NOT on this base (contract rev 8 §3.2 S5). TierSite opens the device-gated context, admitted
  // by dg_layout (devgate-ts src/dg_layout.cpp), which tierstream d8f22a1 does not have: the compile, its dg.* lines,
  // tiers_plan_id and desc_projection_sha256 land with the rebase onto devgate-ts (impl-notes.md TODO).

  // ---- S5 emit placement-v1 ------------------------------------------------------------------------------------
  Json doc = Json::object();
  doc["schema"] = Json::string(kSchema);
  doc["recipe"] = Json::string(R.name);
  doc["effective_sha256"] = Json::string(in.effective_sha256);
  doc["route"] = Json::string(rt->id);
  doc["engine"] = Json::string(C->engine);
  doc["build_sha256"] = Json::string(C->build_sha256);
  doc["model_sha256"] = Json::string(C->model_sha256);
  doc["desc_sha256"] = Json::string(C->desc_sha256);
  doc["components_sha256"] = Json::string(C->sha256);
  if (in.usage) doc["usage_sha256"] = Json::string(in.usage->sha256);
  if (in.context) doc["context_sha256"] = Json::string(in.context->sha256);
  if (!C->receipts_sha256.empty()) doc["receipts_sha256"] = Json::string(C->receipts_sha256);
  Json cfg = Json::object();
  cfg["max_context_tokens"] = Json::number(C->max_context_tokens);
  cfg["max_sequences"] = Json::number(C->max_sequences);
  cfg["max_rows"] = Json::number(C->max_rows);
  doc["config"] = cfg;
  doc["anchor"] = Json::string(anchor_set);
  auto line_json = [&](const Line& l, bool with_at) {
    Json e = Json::object();
    e["component"] = Json::string(l.component);
    if (!l.name.empty()) e["name"] = Json::string(l.name);
    e["bytes"] = Json::number(l.bytes);
    e["peak_extra"] = Json::number(l.peak_extra);
    if (l.slots < 0) e["allocs"] = Json::number(l.allocs);
    if (l.driver) e["kind"] = Json::string("driver");
    e["phases"] = phases_json(l.phases);
    e["source"] = Json::string(l.source);
    if (l.component == "expert_host") {  // field mapping (§4.3): blocks = homes + host spares
      e["blocks"] = Json::number(l.slots);
      e["homes"] = Json::number(homes);
      e["host_spares"] = Json::number(host_spares);
      e["mode"] = Json::string(C->exclusive ? "exclusive" : "inclusive");
    } else if (l.component == "expert_file") {
      e["units"] = Json::number(l.slots);
      e["spilled"] = Json::number(spilled);
    } else if (l.slots >= 0) {
      e["slots"] = Json::number(l.slots);
    }
    if (with_at) e["at"] = Json::string(charge_at_word(l.at));
    return e;
  };
  Json dj = Json::array();
  for (const Dev& d : devs) {
    if (!d.touched) continue;
    Json e = Json::object();
    e["set"] = Json::string(d.set);
    e["role"] = Json::string(d.anchor ? "anchor" : "assist");
    e["kind"] = Json::string(d.bm->kind);
    e["phys"] = Json::string(d.bm->phys);
    if (!d.bm->uuid.empty()) e["uuid"] = Json::string(d.bm->uuid);
    if (!d.bm->pci.empty()) e["pci"] = Json::string(d.bm->pci);
    e["name"] = Json::string(d.bm->name);
    e["host"] = Json::string(d.bm->host);
    e["arch"] = Json::string(d.bm->arch);
    if (!d.bm->driver.empty()) e["driver"] = Json::string(d.bm->driver);
    if (!d.bm->platform.empty()) e["platform"] = Json::string(d.bm->platform);
    e["memory_bytes"] = Json::number(d.memory);
    e["budget_bytes"] = Json::number(d.budget);
    e["margin_bytes"] = Json::number(d.margin);
    e["margin_source"] = Json::string(d.margin_default ? "default" : "recipe");
    e["context_bytes"] = Json::number(add(d.allowance, d.growth));  // what S2 charged
    e["alloc_overhead_bytes"] = Json::number(d.overhead);           // G x allocations, max over phases (rev 17)
    e["context_allowance_bytes"] = Json::number(d.context);
    if (d.measured >= 0) e["context_measured_bytes"] = Json::number(d.measured);
    e["context_growth_bytes"] = Json::number(d.growth);
    std::vector<Line> ls = d.lines;
    std::stable_sort(ls.begin(), ls.end(), [](const Line& a, const Line& b) {
      return component_priority(a.component) < component_priority(b.component);
    });
    Json la = Json::array();
    for (const Line& l : ls) la.push(line_json(l, false));
    e["lines"] = la;
    Json pk = Json::object();
    for (int p = 0; p < kPlacePhases; ++p) pk[place_phase_word(p)] = Json::number(d.peak[p]);
    e["phase_peak_bytes"] = pk;
    dj.push(e);
  }
  doc["devices"] = dj;
  Json host = Json::object();
  host["pinned_budget_bytes"] = Json::number(pinned_budget);
  host["pageable_budget_bytes"] = Json::number(pageable_budget);
  host["available_floor_bytes"] = Json::number(i64(gib_bytes(P->get("host_available_floor_gib"))));
  host["pinned_total_bytes"] = Json::number(pinned);
  host["other_pinned_bytes"] = Json::number(other_pinned);
  host["pageable_total_bytes"] = Json::number(pageable);
  std::vector<Line> hl = host_lines;
  std::stable_sort(hl.begin(), hl.end(), [](const Line& a, const Line& b) {
    return component_priority(a.component) < component_priority(b.component);
  });
  Json hla = Json::array();
  for (const Line& l : hl) hla.push(line_json(l, true));
  host["lines"] = hla;
  doc["host"] = host;
  if (elastic) {
    doc["units"] = units_json;
    doc["seed"] = seed_json;
  }

  // decisions and the not-chosen alternatives (§3.4: listed with the bytes they would free, never ranked or taken)
  Json dec = Json::array(), alt = Json::array();
  for (const Placed& p : placed)
    dec.push(Json::string(p.id + " -> " + p.dest + " (" + p.source + ")"));
  if (elastic) {
    const int64_t useful = std::min(active, total - assist_units - file_units);
    dec.push(Json::string("arena: " + std::string(q_arena >= 0 ? "requested " + num(q_arena) + ", " : "") +
                          "useful " + num(useful) + ", allocated " + num(active + i64(C->spares)) +
                          " slots (+ window lines); " + (q_arena >= 0 ? "recipe" : "fill")));
    dec.push(Json::string("expert homes: " + num(homes) + " host + " + num(host_spares) + " host spares, " +
                          num(assist_units) + " assist" + (assist_set.empty() ? "" : " on " + assist_set) + ", " +
                          num(file_units) + " file" +
                          (spilled ? " (" + num(spilled) + " spilled, coldest first)" : "") + " of " + num(total) +
                          " units (" + (C->exclusive ? "exclusive" : "inclusive") + "); ranked by " +
                          (in.usage ? "usage-v1" : in.desc ? "the descriptor's benefit" : "model order")));
  }
  for (const Placed& p : placed) {
    if (p.dest != anchor_set || !p.decl || !p.decl->has_variants || !p.decl->has_assist) continue;
    for (const Dev& d : devs) {
      if (d.anchor || !d.enabled) continue;
      int64_t frees = 0, needs = 0;
      for (const Charge& ch : p.decl->anchor)
        if (ch.at == ChargeAt::Device) frees = add(frees, i64(ch.bytes + ch.peak_extra));
      for (const Charge& ch : p.decl->assist)
        if (ch.at == ChargeAt::Device) needs = add(needs, i64(ch.bytes + ch.peak_extra));
      bool qualified = false;
      for (const Receipt& r : p.decl->qualified)
        qualified = qualified || (r.arch == d.bm->arch && r.build_sha256 == C->build_sha256);
      alt.push(Json::string(p.id + " -> " + d.set + ": frees " + num(frees) + " on " + anchor_set + ", needs " +
                            num(needs) + (d.touched ? "" : " + context") + " on " + d.set +
                            (qualified ? "" : " (no qualification receipt for " + d.bm->arch + ")")));
    }
  }
  doc["decisions"] = dec;
  doc["alternatives"] = alt;
  const std::string body = json_write(doc);
  doc["placement_sha256"] = Json::string(sha256_hex(body.data(), body.size()));
  res.doc = doc;

  // ---- explanation (printed from the same numbers) ----------------------------------------------------------
  std::string ex;
  char buf[512];
  ex += "placement " + R.name + " route " + rt->id + " anchor " + anchor_set + " (" + kSchema + ")\n";
  ex += "components " + C->engine + " components_sha256 " + C->sha256 + "\n";
  for (const Dev& d : devs) {
    if (!d.touched) {
      ex += "device " + d.set + ": not touched (" +
            (!d.bound ? "unbound" : !d.enabled ? "device_gib = 0" : "nothing placed on it") + ")\n";
      continue;
    }
    ex += "device " + d.set + " (" + (d.anchor ? "anchor" : "assist") + ", " + d.bm->kind + " " + d.bm->arch + ", " +
          d.bm->name + ", " + d.bm->phys + ")\n";
    std::snprintf(buf, sizeof buf, "  budget %lld (%s), margin %lld (%s), context %lld%s + growth %lld\n",
                  (long long)d.budget, gib_text(d.budget).c_str(), (long long)d.margin,
                  d.margin_default ? "default" : "recipe", (long long)d.allowance,
                  d.measured >= 0 ? (" (allowance " + num(d.context) + ", measured " + num(d.measured) + ")").c_str()
                                  : " (allowance)",
                  (long long)d.growth);
    ex += buf;
    std::vector<Line> ls = d.lines;
    std::stable_sort(ls.begin(), ls.end(), [](const Line& a, const Line& b) {
      return component_priority(a.component) < component_priority(b.component);
    });
    for (const Line& l : ls) {
      std::snprintf(buf, sizeof buf, "  %-14s %14lld + %-11lld %-26s %-8s%s\n", l.component.c_str(),
                    (long long)l.bytes, (long long)l.peak_extra, phases_text(l.phases).c_str(), l.source.c_str(),
                    l.slots >= 0 ? (" " + num(l.slots) + " slots").c_str() : "");
      ex += buf;
    }
    for (int p = 0; p < kPlacePhases; ++p) {
      std::snprintf(buf, sizeof buf, "  peak %-8s %14lld (%s) of %lld\n", place_phase_word(p), (long long)d.peak[p],
                    gib_text(d.peak[p]).c_str(), (long long)(d.budget - d.margin));
      ex += buf;
    }
  }
  std::snprintf(buf, sizeof buf, "host pinned %lld (%s) of %lld, other pinned %lld; pageable %lld of %lld\n",
                (long long)pinned, gib_text(pinned).c_str(), (long long)pinned_budget, (long long)other_pinned,
                (long long)pageable, (long long)pageable_budget);
  ex += buf;
  for (const Line& l : hl) {
    std::snprintf(buf, sizeof buf, "  %-14s %-13s %14lld + %-11lld %-8s%s\n", l.component.c_str(),
                  charge_at_word(l.at), (long long)l.bytes, (long long)l.peak_extra, l.source.c_str(),
                  l.slots >= 0 ? (" " + num(l.slots) + " units").c_str() : "");
    ex += buf;
  }
  ex += "decisions:\n";
  for (const Json& s : dec.arr) ex += "  " + s.text + "\n";
  ex += "alternatives (not chosen; v1 auto never moves a component to an assist):\n";
  if (alt.arr.empty()) ex += "  none\n";
  for (const Json& s : alt.arr) ex += "  " + s.text + "\n";
  for (const auto& e : errs) ex += "refused: " + e + "\n";
  ex += "placement_sha256 " + doc.find("placement_sha256")->text + "\n";
  res.explain = ex;
  return res;
}

namespace {

struct FlagDev {
  std::string id, kind, name, host, uuid, pci;
  Json memory_gib;
  int64_t ordinal = -1;
};

std::string lower(std::string s) {
  for (char& ch : s) ch = char(std::tolower(static_cast<unsigned char>(ch)));
  return s;
}

[[noreturn]] void flag_bad(const std::string& code, const std::string& msg) {
  fail(SS_E_RECIPE, code + " flags: " + msg);
}

Json gib_number(const std::string& flag, const std::string& t) {
  char* end = nullptr;
  const double d = std::strtod(t.c_str(), &end);
  if (t.empty() || !end || *end || t.find_first_not_of("0123456789.") != std::string::npos || !(d >= 0))
    flag_bad("PL_E_SCHEMA", flag + ": \"" + t + "\" is not a number >= 0");
  Json j;
  j.kind = Json::Kind::Number;
  j.text = t;
  return j;
}

std::vector<std::pair<std::string, std::string>> pairs(const std::string& flag, const std::string& v) {
  std::vector<std::pair<std::string, std::string>> out;
  size_t b = 0;
  while (b <= v.size()) {
    size_t e = v.find(',', b);
    const std::string item = v.substr(b, e == std::string::npos ? std::string::npos : e - b);
    const size_t eq = item.find('=');
    if (eq == std::string::npos || eq == 0 || eq + 1 == item.size())
      flag_bad("PL_E_SCHEMA", flag + ": \"" + item + "\" is not KEY=VALUE");
    out.push_back({item.substr(0, eq), item.substr(eq + 1)});
    if (e == std::string::npos) break;
    b = e + 1;
  }
  return out;
}

}  // namespace

PlaceResult place_from_flags(const Json& probe, const Components& c, const std::vector<std::string>& args,
                             std::string* recipe_text) {
  // flags
  std::string anchor_dev;
  std::vector<std::pair<std::string, std::string>> budgets, map;
  std::map<std::string, Json> host;
  Json margin_gib, margin_fraction;
  for (size_t i = 0; i < args.size(); ++i) {
    const std::string a = args[i];
    if (i + 1 >= args.size()) flag_bad("PL_E_SCHEMA", a + " needs a value");
    const std::string v = args[++i];
    if (a == "--anchor")
      anchor_dev = v;
    else if (a == "--gpu-budget-gib")
      for (const auto& p : pairs(a, v)) budgets.push_back(p);
    else if (a == "--device-map")
      for (const auto& p : pairs(a, v)) map.push_back(p);
    else if (a == "--host-pinned-gib" || a == "--host-pageable-gib" || a == "--host-available-floor-gib")
      host[a] = gib_number(a, v);
    else if (a == "--margin-gib")
      margin_gib = gib_number(a, v);
    else if (a == "--margin-fraction")
      margin_fraction = gib_number(a, v);
    else
      flag_bad("PL_E_SCHEMA", "unknown flag " + a);
  }
  // probe rows (the binding re-reads them strictly)
  require(probe.is_object() && probe.find("devices") && probe.find("devices")->is_array(), SS_E_RECIPE,
          "probe: \"devices\" must be an array");
  std::vector<FlagDev> rows;
  for (const Json& e : probe.find("devices")->arr) {
    FlagDev d;
    auto s = [&](const char* k) {
      const Json* v = e.find(k);
      return v && v->is_string() ? v->text : std::string();
    };
    d.id = s("id");
    d.kind = s("kind");
    d.name = s("name");
    d.host = s("host");
    d.uuid = s("uuid").empty() ? "" : normalize_uuid(s("uuid"));
    d.pci = s("pci").empty() ? "" : normalize_pci(s("pci"));
    if (const Json* m = e.find("memory_gib")) d.memory_gib = *m;
    if (const Json* o = e.find("ordinal"))
      if (o->is_number()) d.ordinal = std::atoll(o->text.c_str());
    rows.push_back(d);
  }
  // the backend and the local host: those of runtime ordinal 0 (today's target)
  const FlagDev* zero = nullptr;
  for (const auto& d : rows)
    if (d.ordinal == 0 && (d.kind == "cuda" || d.kind == "hip")) zero = &d;
  if (!zero) flag_bad("PL_E_UNKNOWN_DEVICE", "the probe has no GPU at runtime ordinal 0 (the default anchor)");
  const std::string kind = zero->kind, local = zero->host;
  // visible devices of the backend on the local host, in PCI order (aliases share a PCI address / UUID)
  std::vector<const FlagDev*> vis;
  for (const auto& d : rows) {
    if (d.kind != kind || d.host != local) continue;
    bool dup = false;
    for (const FlagDev* o : vis)
      dup = dup || (!d.pci.empty() && d.pci == o->pci) || (!d.uuid.empty() && d.uuid == o->uuid);
    if (!dup) vis.push_back(&d);
  }
  std::sort(vis.begin(), vis.end(), [](const FlagDev* a, const FlagDev* b) {
    if (a->pci.empty() != b->pci.empty()) return !a->pci.empty();
    return a->pci != b->pci ? a->pci < b->pci : (a->uuid != b->uuid ? a->uuid < b->uuid : a->id < b->id);
  });
  auto resolve = [&](const std::string& dev) -> size_t {
    std::vector<size_t> hit;
    if (dev.compare(0, 3, "gpu") == 0 && dev.size() > 3 && dev.find_first_not_of("0123456789", 3) == std::string::npos) {
      const size_t n = size_t(std::atoll(dev.c_str() + 3));
      if (n < vis.size()) {
        if (vis[n]->pci.empty())
          flag_bad("PL_E_SCHEMA", dev + ": gpuN counts devices in PCI order; the probe has no PCI address for " +
                                      vis[n]->id);
        hit.push_back(n);
      }
    } else if (dev.compare(0, 5, "uuid:") == 0) {
      if (kind != "cuda") flag_bad("PL_E_CAPABILITY", dev + ": uuid: is CUDA-only in v1");
      for (size_t k = 0; k < vis.size(); ++k)
        if (vis[k]->uuid == normalize_uuid(dev.substr(5))) hit.push_back(k);
    } else if (dev.compare(0, 4, "pci:") == 0) {
      for (size_t k = 0; k < vis.size(); ++k)
        if (!vis[k]->pci.empty() && vis[k]->pci == normalize_pci(dev.substr(4))) hit.push_back(k);
    } else if (dev.compare(0, 5, "name:") == 0) {
      for (size_t k = 0; k < vis.size(); ++k)
        if (lower(vis[k]->name).find(lower(dev.substr(5))) != std::string::npos) hit.push_back(k);
    } else {
      flag_bad("PL_E_UNKNOWN_DEVICE", "\"" + dev + "\" is not gpuN, uuid:, pci: or name:");
    }
    if (hit.size() != 1)
      flag_bad("PL_E_UNKNOWN_DEVICE", "\"" + dev + "\" matches " + std::to_string(hit.size()) + " " + kind +
                                          " devices on " + local + " (exactly one needed)");
    return hit[0];
  };
  size_t anchor = 0;
  if (!anchor_dev.empty()) {
    anchor = resolve(anchor_dev);
  } else {
    bool found = false;
    for (size_t k = 0; k < vis.size(); ++k)
      if (vis[k] == zero || (!zero->pci.empty() && vis[k]->pci == zero->pci)) {
        anchor = k;
        found = true;
      }
    if (!found) flag_bad("PL_E_UNKNOWN_DEVICE", "runtime ordinal 0 is not among the visible devices");
  }
  auto set_id = [&](size_t k) { return "gpu" + std::to_string(k); };
  std::map<size_t, Json> budget_of;
  for (const auto& b : budgets) budget_of[resolve(b.first)] = gib_number("--gpu-budget-gib", b.second);
  // the recipe the flags make
  Json m = Json::object();
  m["model"] = probe.find("model") ? *probe.find("model") : Json::string("");
  if (probe.find("engine")) m["engine"] = Json::string(c.engine);
  Json devices = Json::array();
  auto device_json = [&](size_t k, bool optional) {
    Json d = Json::object();
    d["id"] = Json::string(set_id(k));
    d["kind"] = Json::string(kind);
    d["memory_min_gib"] = vis[k]->memory_gib;
    if (!vis[k]->pci.empty())
      d["pci"] = Json::string(vis[k]->pci);
    else if (!vis[k]->uuid.empty() && kind == "cuda")
      d["uuid"] = Json::string(vis[k]->uuid);
    else
      flag_bad("PL_E_SCHEMA", vis[k]->id + ": the probe has neither a PCI address nor a CUDA UUID for it");
    if (optional) d["optional"] = Json::boolean(true);
    return d;
  };
  devices.push(device_json(anchor, false));
  Json assist = Json::object();
  for (const auto& kv : budget_of) {
    if (kv.first == anchor) continue;
    devices.push(device_json(kv.first, true));
    Json a = Json::object();
    a["device_gib"] = kv.second;
    assist[set_id(kv.first)] = a;
  }
  m["devices"] = devices;
  Json pl = Json::object();
  if (budget_of.count(anchor)) pl["device_gib"] = budget_of[anchor];
  if (host.count("--host-pinned-gib")) pl["host_pinned_gib"] = host["--host-pinned-gib"];
  if (host.count("--host-pageable-gib")) pl["host_pageable_gib"] = host["--host-pageable-gib"];
  if (host.count("--host-available-floor-gib")) pl["host_available_floor_gib"] = host["--host-available-floor-gib"];
  if (!margin_gib.is_null()) pl["margin_gib"] = margin_gib;
  if (!margin_fraction.is_null()) pl["margin_fraction"] = margin_fraction;
  pl["max_context_tokens"] = Json::number(c.max_context_tokens);
  pl["max_batch"] = Json::number(c.max_sequences);
  Json comps = Json::object();
  for (const auto& kv : map) {
    Json x = Json::object();
    x["device"] = Json::string(reserved_destination(kv.second) ? kv.second : set_id(resolve(kv.second)));
    comps[kv.first] = x;
  }
  Json rt = Json::object();
  rt["id"] = Json::string("flags");
  rt["phase"] = Json::string("decode");
  Json rd = Json::array();
  rd.push(Json::string(set_id(anchor)));
  rt["devices"] = rd;
  rt["execute"] = Json::string("single");
  rt["max_rows"] = Json::number(c.max_rows);
  rt["max_sequences"] = Json::number(c.max_sequences);
  Json pls = Json::object();
  pls[set_id(anchor)] = pl;
  rt["placement"] = pls;
  if (!assist.obj.empty()) rt["assist"] = assist;
  if (!comps.obj.empty()) rt["components"] = comps;
  Json routes = Json::array();
  routes.push(rt);
  Json doc = Json::object();
  doc["schema"] = Json::string("tierstream/recipe-v1");
  doc["name"] = Json::string("flags");
  doc["match"] = m;
  doc["routes"] = routes;
  const std::string text = json_write(doc);
  if (recipe_text) *recipe_text = text;
  RecipeStack s(text, "<flags>");
  Binding b = recipe_bind(s.effective(), probe);
  PlaceInput in;
  in.recipe = &s.effective();
  in.effective_sha256 = s.effective_sha256();
  in.route = "flags";
  in.binding = &b;
  in.components = &c;
  return place(in);
}

}  // namespace ts
