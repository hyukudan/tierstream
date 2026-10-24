// Hardware binding, admission and compilation of a recipe (docs/recipes.md §3.1, §5.2-5.3, §7, §8).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

#include "errors.hpp"
#include "recipe.hpp"
#include "sha256.hpp"

namespace ss {
namespace {

constexpr long kMaxBindSteps = 1000000;
constexpr double kGiBf = 1073741824.0;

std::string fmtg(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.4g", v);
  return buf;
}

std::string sha_of(const std::string& s) { return sha256_hex(s.data(), s.size()); }

uint64_t add_ck(uint64_t a, uint64_t b, const std::string& what) {
  require(a <= UINT64_MAX - b, SS_E_RECIPE, what + ": byte arithmetic overflows 64 bits");
  return a + b;
}

uint64_t mul_ck(uint64_t a, uint64_t b, const std::string& what) {
  require(a == 0 || b <= UINT64_MAX / a, SS_E_RECIPE, what + ": byte arithmetic overflows 64 bits");
  return a * b;
}

// ---- probe report ------------------------------------------------------------------------------------------------
struct ProbeDev {
  std::string id, kind, name, host, driver;
  double memory_gib = 0, host_ram_gib = -1, disk = -1, h2d = -1;
  int64_t ordinal = -1;
};
struct ProbeLink {
  std::string kind, a, b;
  double gbps = 0;
};

const Json* field(const Json& o, const char* k) { return o.is_object() ? o.find(k) : nullptr; }

std::string pstr(const Json& o, const char* k, const std::string& w, bool required) {
  const Json* v = field(o, k);
  if (!v || v->is_null()) {
    require(!required, SS_E_RECIPE, w + ": missing \"" + k + "\"");
    return "";
  }
  require(v->is_string(), SS_E_RECIPE, w + "." + k + ": expected a string");
  return v->text;
}

double pnum(const Json& o, const char* k, const std::string& w, bool required, double dflt = -1) {
  const Json* v = field(o, k);
  if (!v || v->is_null()) {
    require(!required, SS_E_RECIPE, w + ": missing \"" + k + "\"");
    return dflt;
  }
  char* end = nullptr;
  double d = v->is_number() ? std::strtod(v->text.c_str(), &end) : NAN;
  require(v->is_number() && end && !*end && std::isfinite(d) && d >= 0, SS_E_RECIPE,
          w + "." + k + ": expected a finite number >= 0");
  return d;
}

int64_t pint(const Json& o, const char* k, const std::string& w, bool required, int64_t dflt = -1) {
  const Json* v = field(o, k);
  if (!v || v->is_null()) {
    require(!required, SS_E_RECIPE, w + ": missing \"" + k + "\"");
    return dflt;
  }
  const std::string& t = v->text;
  require(v->is_number() && !t.empty() && t.size() <= 18 && t.find_first_not_of("0123456789") == std::string::npos,
          SS_E_RECIPE, w + "." + k + ": expected an integer >= 0");
  return std::strtoll(t.c_str(), nullptr, 10);
}

bool kind_ok(const std::string& want, const std::string& got) { return want == "other" || want == got; }

const char* backend_word(ss_backend_kind k) {
  return k == SS_BACKEND_CUDA ? "cuda" : k == SS_BACKEND_HIP ? "hip" : "cpu";
}

bool hosts_linked(const std::vector<ProbeLink>& links, const std::string& a, const std::string& b,
                  const std::string& kind, double gbps) {
  if (a == b) return true;
  for (const auto& l : links)
    if (((l.a == a && l.b == b) || (l.a == b && l.b == a)) && kind_ok(kind, l.kind) && l.gbps >= gbps) return true;
  return false;
}

// ---- tiering (docs/recipes.md §5.2) ------------------------------------------------------------------------------
struct UnitRef {
  int64_t g;
  int layer, unit;
};

struct Tiering {
  std::vector<uint8_t> tier;
  std::vector<int> peer;  // per global unit: route set index of a rest = "peer" unit, else -1
  std::vector<std::string> trace;
};

double benefit_of(const Desc& d, const std::vector<double>& benefit, int L, int u) {
  if (!benefit.empty()) return benefit[size_t(d.unit_base[size_t(L)] + u)];
  const Layer& l = d.layers[size_t(L)];
  return l.benefit.empty() ? 1.0 : l.benefit[size_t(u)];
}

bool layer_in(const RGroup& g, int L) {
  if (g.layers.empty()) return true;
  for (const auto& r : g.layers)
    if (L >= r.first && L <= r.second) return true;
  return false;
}

// Units this placement plans (avail), ranked per group; integer counts, fills, rest; then backing.
Tiering tier_units(const Desc& d, const ss_budget& b, const RPlacement& p, const std::vector<double>& benefit,
                   const std::vector<char>& avail, const std::vector<std::string>& route_sets, const std::string& w) {
  const size_t n = size_t(d.total_units);
  Tiering t;
  t.tier.assign(n, SS_TIER_PAIR);
  t.peer.assign(n, -1);
  auto unit_name = [&](int L, int u) { return "unit (layer " + std::to_string(L) + ", " + std::to_string(u) + ")"; };
  std::vector<int> group_of(n, -1);
  std::vector<std::vector<UnitRef>> members(p.groups.size());

  for (size_t gi = 0; gi < p.groups.size(); ++gi) {
    const RGroup& g = p.groups[gi];
    const std::string gw = w + ".units." + g.id;
    for (const auto& r : g.layers)
      for (int L = r.first; L <= r.second; ++L) {
        require(L < int(d.layers.size()), SS_E_RECIPE,
                gw + ".layers: layer " + std::to_string(L) + " is past the descriptor's " +
                    std::to_string(d.layers.size()) + " layers");
        const Layer& l = d.layers[size_t(L)];
        require(l.kind == g.cls && l.streamable && l.n_units > 0, SS_E_RECIPE,
                gw + ".layers: layer " + std::to_string(L) + " is not a streamable " +
                    (g.cls == LayerKind::Moe ? "moe" : "blob") + " layer of the descriptor");
      }
    if (g.rank == Rank::List)
      for (int x : g.list) {
        if (g.cls == LayerKind::Blob) {
          require(layer_in(g, x) && x < int(d.layers.size()) && d.layers[size_t(x)].kind == LayerKind::Blob,
                  SS_E_RECIPE, gw + ".list: layer " + std::to_string(x) + " is not a blob layer of the group");
        } else {
          for (size_t L = 0; L < d.layers.size(); ++L)
            if (d.layers[L].kind == LayerKind::Moe && d.layers[L].streamable && layer_in(g, int(L)))
              require(x < d.layers[L].n_units, SS_E_RECIPE,
                      gw + ".list: expert " + std::to_string(x) + " is past layer " + std::to_string(L) + "'s " +
                          std::to_string(d.layers[L].n_units) + " units");
        }
      }
  }
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const Layer& l = d.layers[L];
    for (int u = 0; u < l.n_units; ++u) {
      const int64_t g = d.unit_base[L] + u;
      if (!l.streamable) {
        t.tier[size_t(g)] = SS_TIER_RESIDENT;
        continue;
      }
      if (!avail[size_t(g)]) continue;
      int found = -1;
      for (size_t gi = 0; gi < p.groups.size(); ++gi)
        if (p.groups[gi].cls == l.kind && layer_in(p.groups[gi], int(L))) {
          if (found >= 0)
            fail(SS_E_RECIPE, w + ": " + unit_name(int(L), u) + " is in groups \"" + p.groups[size_t(found)].id +
                                  "\" and \"" + p.groups[gi].id + "\"");
          found = int(gi);
        }
      require(found >= 0, SS_E_RECIPE, w + ": " + unit_name(int(L), u) + " is in no unit group");
      group_of[size_t(g)] = found;
      members[size_t(found)].push_back({g, int(L), u});
    }
  }

  // Rank order per group (docs/recipes.md §5.2), total: ties end in (layer, unit).
  for (size_t gi = 0; gi < p.groups.size(); ++gi) {
    const RGroup& g = p.groups[gi];
    std::vector<UnitRef>& m = members[gi];
    if (g.rank == Rank::List) {
      std::map<int, size_t> pos;
      for (size_t k = 0; k < g.list.size(); ++k) pos[g.list[k]] = k;
      auto key = [&](const UnitRef& x) {
        auto it = pos.find(g.cls == LayerKind::Moe ? x.unit : x.layer);
        return it == pos.end() ? g.list.size() : it->second;
      };
      std::stable_sort(m.begin(), m.end(), [&](const UnitRef& a, const UnitRef& c) {
        size_t ka = key(a), kc = key(c);
        if (ka != kc) return ka < kc;
        return a.g < c.g;
      });
    } else if (g.rank == Rank::Frequency) {
      bool have = !benefit.empty();
      for (const auto& x : m) have = have || !d.layers[size_t(x.layer)].benefit.empty();
      require(
          have, SS_E_RECIPE,
          w + ".units." + g.id + ".rank: \"frequency\" needs a measured benefit (the caller's or the descriptor's)");
      std::stable_sort(m.begin(), m.end(), [&](const UnitRef& a, const UnitRef& c) {
        double x = benefit_of(d, benefit, a.layer, a.unit) / double(d.unit_stride(a.layer));
        double y = benefit_of(d, benefit, c.layer, c.unit) / double(d.unit_stride(c.layer));
        if (x != y) return x > y;
        return a.g < c.g;
      });
    }  // model: members are already in (layer, unit) order
  }

  enum : char { Open = 0, Res = 1, Pin = 2 };
  std::vector<char> st(n, Open);
  // Steps 1-2: integer counts (moe per layer, blob over the group); never shrunk.
  for (int step = 0; step < 2; ++step)
    for (size_t gi = 0; gi < p.groups.size(); ++gi) {
      const RGroup& g = p.groups[gi];
      const int64_t want = step == 0 ? g.resident : g.pinned;
      if (want == kFill || want == 0) continue;
      const int64_t both = (g.resident == kFill ? 0 : g.resident) + (g.pinned == kFill ? 0 : g.pinned);
      const std::string gw = w + ".units." + g.id;
      std::map<int, int64_t> taken, avail_in;
      for (const auto& x : members[gi]) ++avail_in[g.cls == LayerKind::Moe ? x.layer : -1];
      for (const auto& kv : avail_in)
        require(kv.second >= both, SS_E_RECIPE,
                gw + ": resident + pinned = " + std::to_string(both) + " but " +
                    (kv.first >= 0 ? "layer " + std::to_string(kv.first) : std::string("the group")) + " has " +
                    std::to_string(kv.second) + " units to place");
      for (const auto& x : members[gi]) {
        int key = g.cls == LayerKind::Moe ? x.layer : -1;
        if (st[size_t(x.g)] != Open || taken[key] >= want) continue;
        st[size_t(x.g)] = step == 0 ? Res : Pin;
        ++taken[key];
      }
      t.trace.push_back(gw + ": " + std::to_string(want) + (step == 0 ? " resident" : " pinned") +
                        (g.cls == LayerKind::Moe ? " per layer" : ""));
    }

  // Step 3: resident fill with the planner's arithmetic.
  std::vector<UnitRef> cand;
  for (size_t gi = 0; gi < p.groups.size(); ++gi)
    if (p.groups[gi].resident == kFill)
      for (const auto& x : members[gi])
        if (st[size_t(x.g)] == Open) cand.push_back(x);
  if (!cand.empty()) {
    const uint64_t room = plan_room(d, b);
    // Ring cost per unit size is fixed by the planner (slots per size do not depend on which units stream).
    std::map<uint64_t, uint64_t> ring_cost;
    auto cost_of = [&](const UnitRef& x) {
      const uint64_t s = d.unit_stride(x.layer);
      auto it = ring_cost.find(s);
      if (it == ring_cost.end()) it = ring_cost.emplace(s, plan_ring_bytes(d, b, {x.g})).first;
      return it->second;
    };
    std::set<int64_t> in_cand;
    for (const auto& x : cand) in_cand.insert(x.g);
    uint64_t res_bytes = 0;
    std::map<uint64_t, int64_t> rest_count;  // stride -> non-resident streamed units outside the candidates
    std::vector<UnitRef> all_avail;
    for (const auto& m : members)
      for (const auto& x : m) all_avail.push_back(x);
    std::sort(all_avail.begin(), all_avail.end(), [](const UnitRef& a, const UnitRef& c) { return a.g < c.g; });
    for (const auto& x : all_avail) {
      if (st[size_t(x.g)] == Res) {
        res_bytes = add_ck(res_bytes, d.unit_stride(x.layer), w);
      } else if (!in_cand.count(x.g)) {
        ++rest_count[d.unit_stride(x.layer)];
      }
    }
    for (const auto& x : all_avail) cost_of(x);
    auto rings = [&](const std::map<uint64_t, int64_t>& counts) {
      uint64_t r = 0;
      for (const auto& kv : counts)
        if (kv.second > 0) r += ring_cost.at(kv.first);
      return r;
    };
    if (d.order == Order::Prefix) {
      for (size_t gi = 0; gi < p.groups.size(); ++gi)
        require(p.groups[gi].rank == Rank::Model, SS_E_RECIPE,
                w + ".units." + p.groups[gi].id + ".rank: a prefix descriptor takes only rank = \"model\"");
      std::sort(cand.begin(), cand.end(), [](const UnitRef& a, const UnitRef& c) { return a.g < c.g; });
      // Position of each streamable unit in model order (pair units included: they break a prefix too).
      std::vector<int64_t> streamable;
      for (size_t L = 0; L < d.layers.size(); ++L)
        if (d.layers[L].streamable)
          for (int u = 0; u < d.layers[L].n_units; ++u) streamable.push_back(d.unit_base[L] + u);
      auto pos = [&](int64_t g) {
        return int64_t(std::lower_bound(streamable.begin(), streamable.end(), g) - streamable.begin());
      };
      int64_t n_res1 = 0, max_res1 = -1;
      for (const auto& x : all_avail)
        if (st[size_t(x.g)] == Res) {
          ++n_res1;
          max_res1 = std::max(max_res1, pos(x.g));
        }
      // Cut k: residents = step-1 residents + cand[0, k); rest (for rings) = rest_count + cand[k, end).
      std::map<uint64_t, int64_t> counts = rest_count;
      uint64_t prefix_bytes = 0;
      for (const auto& x : cand) prefix_bytes = add_ck(prefix_bytes, d.unit_stride(x.layer), w);
      size_t chosen = SIZE_MAX;
      for (size_t k = cand.size() + 1; k-- > 0;) {
        if (k < cand.size()) {
          ++counts[d.unit_stride(cand[k].layer)];
          prefix_bytes -= d.unit_stride(cand[k].layer);
        }
        const int64_t max_res = std::max(max_res1, k ? pos(cand[k - 1].g) : int64_t(-1));
        const bool is_prefix = n_res1 + int64_t(k) == max_res + 1;
        if (is_prefix && res_bytes + prefix_bytes + rings(counts) <= room) {
          chosen = k;
          break;
        }
      }
      require(chosen != SIZE_MAX, SS_E_RECIPE,
              w + ": no model-order prefix fits: the explicit resident counts are not a prefix, or they alone do "
                  "not fit the device budget");
      for (size_t k = 0; k < chosen; ++k) st[size_t(cand[k].g)] = Res;
      t.trace.push_back(w + ": resident fill (prefix): " + std::to_string(chosen) + " of " +
                        std::to_string(cand.size()) + " candidates");
    } else {
      uint64_t all = res_bytes;
      for (const auto& x : cand) all = add_ck(all, d.unit_stride(x.layer), w);
      size_t took = 0;
      if (all + rings(rest_count) <= room) {
        for (const auto& x : cand) st[size_t(x.g)] = Res;
        took = cand.size();
      } else {
        std::map<uint64_t, int64_t> counts = rest_count;
        for (const auto& x : cand) ++counts[d.unit_stride(x.layer)];
        const uint64_t ring = rings(counts);
        if (ring <= room) {
          uint64_t used = res_bytes;
          for (const auto& x : cand) {
            const uint64_t s = d.unit_stride(x.layer);
            if (used + s <= room - ring) {
              st[size_t(x.g)] = Res;
              used += s;
              ++took;
            }
          }
        }
      }
      t.trace.push_back(w + ": resident fill: " + std::to_string(took) + " of " + std::to_string(cand.size()) +
                        " candidates");
    }
  }

  // Step 4: pinned fill on the 4 KiB host stride.
  uint64_t host_left = b.host_pinned_bytes;
  for (const auto& h : d.host_fixed) host_left = host_left > h.bytes ? host_left - h.bytes : 0;
  for (size_t g = 0; g < n; ++g)
    if (st[g] == Pin) {
      const uint64_t hb =
          host_stride(d.layers[size_t(std::upper_bound(d.unit_base.begin(), d.unit_base.end(), int64_t(g)) -
                                      d.unit_base.begin() - 1)]
                          .unit_bytes);
      host_left = host_left > hb ? host_left - hb : 0;
    }
  size_t pinned_fill = 0, pinned_cand = 0;
  for (size_t gi = 0; gi < p.groups.size(); ++gi) {
    if (p.groups[gi].pinned != kFill) continue;
    for (const auto& x : members[gi]) {
      if (st[size_t(x.g)] != Open) continue;
      ++pinned_cand;
      const uint64_t hb = host_stride(d.layers[size_t(x.layer)].unit_bytes);
      if (hb <= host_left) {
        st[size_t(x.g)] = Pin;
        host_left -= hb;
        ++pinned_fill;
      }
    }
  }
  if (pinned_cand)
    t.trace.push_back(w + ": pinned fill: " + std::to_string(pinned_fill) + " of " + std::to_string(pinned_cand) +
                      " candidates");

  // Step 5: the rest.
  size_t n_res = 0, n_pin = 0, n_disk = 0, n_peer = 0;
  for (size_t gi = 0; gi < p.groups.size(); ++gi) {
    const RGroup& g = p.groups[gi];
    for (const auto& x : members[gi]) {
      uint8_t& tt = t.tier[size_t(x.g)];
      if (st[size_t(x.g)] == Res) {
        tt = SS_TIER_RESIDENT;
        ++n_res;
      } else if (st[size_t(x.g)] == Pin || g.rest == Rest::Pinned) {
        tt = SS_TIER_PINNED;
        ++n_pin;
      } else if (g.rest == Rest::Disk) {
        tt = SS_TIER_DISK;
        ++n_disk;
      } else {
        tt = SS_TIER_PAIR;
        t.peer[size_t(x.g)] = int(std::find(route_sets.begin(), route_sets.end(), g.peer) - route_sets.begin());
        ++n_peer;
      }
    }
  }
  // Step 6: pinned copies of residents.
  if (b.pinned_backing) plan_backing(d, b, benefit, t.tier);

  // Step 7: descriptor rules (the totals are re-checked by plan_from_tiers).
  if (n_pin)
    require(d.tiers == Tiers::Pinned || d.tiers == Tiers::PinnedDisk, SS_E_RECIPE,
            w + ": pinned units need a descriptor whose tiers allow pinned (it says \"" +
                std::string(tiers_name(d.tiers)) + "\")");
  if (n_disk)
    require(d.tiers == Tiers::Disk || d.tiers == Tiers::PinnedDisk, SS_E_RECIPE,
            w + ": disk units need a descriptor whose tiers allow disk (it says \"" + std::string(tiers_name(d.tiers)) +
                "\")");
  if (d.order == Order::Prefix) {
    bool gap = false;
    for (size_t L = 0; L < d.layers.size(); ++L) {
      if (!d.layers[L].streamable) continue;
      for (int u = 0; u < d.layers[L].n_units; ++u) {
        const uint8_t x = t.tier[size_t(d.unit_base[L] + u)];
        const bool res = x == SS_TIER_RESIDENT || x == SS_TIER_RESIDENT_BACKED;
        require(!(gap && res), SS_E_RECIPE,
                w + ": the descriptor's order is prefix but resident " + unit_name(int(L), u) +
                    " comes after a non-resident unit");
        gap = gap || !res;
      }
    }
  }
  t.trace.push_back(w + ": " + std::to_string(n_res) + " resident, " + std::to_string(n_pin) + " pinned, " +
                    std::to_string(n_disk) + " disk, " + std::to_string(n_peer) + " computed by a peer");
  return t;
}

// Budget of one placement against one descriptor (workspace derived when not set; docs/recipes.md §5.1).
ss_budget budget_for(const RRoute& rt, const RPlacement& p, const Desc& d, const std::string& w, uint64_t* rows_out) {
  ss_budget b{};
  b.struct_size = uint32_t(sizeof b);
  b.device_bytes = gib_bytes(p.get("device_gib"));
  b.host_pinned_bytes = gib_bytes(p.get("host_pinned_gib"));
  b.margin_gib = p.get("margin_gib");
  b.margin_fraction = p.get("margin_fraction");
  b.dynamic_slots_per_layer = uint32_t(p.get("dynamic_slots_per_layer"));
  b.ring_slots = uint32_t(p.get("ring_slots"));
  b.pinned_backing = p.get("pinned_backing") != 0;
  const uint64_t ctx = uint64_t(p.get("max_context_tokens")), batch = uint64_t(p.get("max_batch"));
  uint64_t rows = batch;
  if (rt.phase == Phase::Prefill) {
    uint64_t per_seq = uint64_t(p.get("chunk_tokens"));
    if (!per_seq) {
      per_seq = ctx;
      if (rt.when.prompt_tokens_max != kNoLimit) per_seq = std::min(per_seq, uint64_t(rt.when.prompt_tokens_max));
    }
    rows = mul_ck(batch, std::max<uint64_t>(per_seq, 1), w + ": forward rows");
  }
  if (rows_out) *rows_out = rows;
  if (p.has_workspace) {
    b.workspace_bytes = gib_bytes(p.get("workspace_gib"));
  } else {
    require(d.workspace_bytes_per_row > 0 && d.state_bytes_per_position > 0, SS_E_RECIPE,
            w + ": the descriptor gives no workspace_bytes_per_row / state_bytes_per_position; set workspace_gib");
    b.workspace_bytes =
        add_ck(mul_ck(d.workspace_bytes_per_row, rows, w + ".workspace_gib"),
               mul_ck(mul_ck(d.state_bytes_per_position, ctx, w + ".workspace_gib"), batch, w + ".workspace_gib"),
               w + ".workspace_gib");
  }
  return b;
}

void same_structure(const Desc& a, const Desc& b, const std::string& what) {
  bool same = a.layers.size() == b.layers.size();
  for (size_t L = 0; same && L < a.layers.size(); ++L)
    same = a.layers[L].kind == b.layers[L].kind && a.layers[L].n_units == b.layers[L].n_units &&
           a.layers[L].streamable == b.layers[L].streamable;
  require(same, SS_E_RECIPE, what + ": the descriptors of a hetero route need the same layers, kinds and units");
}

std::string benefit_sha(const std::vector<double>& v) {
  if (v.empty()) return "none";
  std::string s;
  char buf[32];
  for (double x : v) {
    std::snprintf(buf, sizeof buf, "%.17g,", x);
    s += buf;
  }
  return sha_of(s);
}

}  // namespace

// ---- binding (docs/recipes.md §3.1) --------------------------------------------------------------------------------
const std::vector<BoundMember>* Binding::of(const std::string& set) const {
  for (const auto& s : sets)
    if (s.first == set) return &s.second;
  return nullptr;
}

Binding recipe_bind(const Recipe& r, const Json& probe) {
  require(probe.is_object(), SS_E_RECIPE, "probe: expected a JSON object");
  std::vector<std::string> errors;
  const std::string model = pstr(probe, "model", "probe", true);
  if (model != r.model) errors.push_back("match.model: recipe is for \"" + r.model + "\", probed \"" + model + "\"");
  if (!r.model_sha256.empty()) {
    const std::string s = pstr(probe, "model_sha256", "probe", false);
    if (s != r.model_sha256) errors.push_back("match.model_sha256: probed \"" + s + "\"");
  }
  if (!r.engine.empty()) {
    const std::string s = pstr(probe, "engine", "probe", false);
    if (s != r.engine) errors.push_back("match.engine: recipe is for \"" + r.engine + "\", probed \"" + s + "\"");
  }
  std::vector<ProbeDev> devs;
  const Json* pd = field(probe, "devices");
  require(pd && pd->is_array(), SS_E_RECIPE, "probe: \"devices\" must be an array");
  std::set<std::string> seen;
  for (size_t i = 0; i < pd->arr.size(); ++i) {
    const Json& e = pd->arr[i];
    const std::string w = "probe.devices[" + std::to_string(i) + "]";
    ProbeDev x;
    x.id = pstr(e, "id", w, true);
    require(seen.insert(x.id).second, SS_E_RECIPE, w + ": duplicate id \"" + x.id + "\"");
    x.kind = pstr(e, "kind", w, true);
    x.name = pstr(e, "name", w, true);
    x.host = pstr(e, "host", w, true);
    x.driver = pstr(e, "driver", w, false);
    x.memory_gib = pnum(e, "memory_gib", w, true);
    x.host_ram_gib = pnum(e, "host_ram_gib", w, false);
    x.disk = pnum(e, "disk_read_gbytes_ps", w, false);
    x.h2d = pnum(e, "h2d_gbytes_ps", w, false);
    x.ordinal = pint(e, "ordinal", w, false);
    devs.push_back(x);
  }
  std::sort(devs.begin(), devs.end(), [](const ProbeDev& a, const ProbeDev& b) { return a.id < b.id; });
  std::vector<ProbeLink> links;
  if (const Json* pl = field(probe, "links")) {
    require(pl->is_array(), SS_E_RECIPE, "probe: \"links\" must be an array");
    for (size_t i = 0; i < pl->arr.size(); ++i) {
      const Json& e = pl->arr[i];
      const std::string w = "probe.links[" + std::to_string(i) + "]";
      ProbeLink l;
      l.kind = pstr(e, "kind", w, true);
      const Json* h = field(e, "hosts");
      require(h && h->is_array() && h->arr.size() == 2 && h->arr[0].is_string() && h->arr[1].is_string(), SS_E_RECIPE,
              w + ": \"hosts\" must be two host names");
      l.a = h->arr[0].text;
      l.b = h->arr[1].text;
      l.gbps = pnum(e, "gbps", w, true);
      links.push_back(l);
    }
  }

  // Eligible devices per set, with why the others are not.
  std::vector<std::vector<size_t>> elig(r.devices.size());
  for (size_t s = 0; s < r.devices.size(); ++s) {
    const RDevice& rd = r.devices[s];
    std::string why;
    for (size_t i = 0; i < devs.size(); ++i) {
      const ProbeDev& x = devs[i];
      std::string miss;
      if (x.kind != backend_word(rd.kind))
        miss = "kind " + x.kind;
      else if (!rd.name.empty()) {
        std::string a = x.name, b = rd.name;
        for (auto* str : {&a, &b})
          for (char& c : *str) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (a.find(b) == std::string::npos) miss = "name";
      }
      if (miss.empty() && x.memory_gib < rd.memory_min_gib) miss = "memory " + fmtg(x.memory_gib) + " GiB";
      if (miss.empty() && rd.host_ram_min_gib > 0 && x.host_ram_gib < rd.host_ram_min_gib)
        miss = x.host_ram_gib < 0 ? "host RAM not probed" : "host RAM " + fmtg(x.host_ram_gib) + " GiB";
      if (miss.empty() && rd.disk_read_gbytes_ps_min > 0 && x.disk < rd.disk_read_gbytes_ps_min)
        miss = x.disk < 0 ? "disk bandwidth not probed" : "disk " + fmtg(x.disk) + " GB/s";
      if (miss.empty() && rd.h2d_gbytes_ps_min > 0 && x.h2d < rd.h2d_gbytes_ps_min)
        miss = x.h2d < 0 ? "H2D bandwidth not probed" : "H2D " + fmtg(x.h2d) + " GB/s";
      if (miss.empty())
        elig[s].push_back(i);
      else
        why += (why.empty() ? "" : ", ") + x.name + " (" + x.id + "): " + miss;
    }
    if (elig[s].size() < size_t(rd.count)) {
      std::string need = "match.devices." + rd.id + ": needs " + std::to_string(rd.count) + " " +
                         backend_word(rd.kind) + " device(s)" + (rd.name.empty() ? "" : " named \"" + rd.name + "\"") +
                         " with >= " + fmtg(rd.memory_min_gib) + " GiB";
      if (rd.host_ram_min_gib > 0) need += ", host RAM >= " + fmtg(rd.host_ram_min_gib) + " GiB";
      if (rd.disk_read_gbytes_ps_min > 0) need += ", disk >= " + fmtg(rd.disk_read_gbytes_ps_min) + " GB/s";
      if (rd.h2d_gbytes_ps_min > 0) need += ", H2D >= " + fmtg(rd.h2d_gbytes_ps_min) + " GB/s";
      errors.push_back(need + "; eligible " + std::to_string(elig[s].size()) + " of " + std::to_string(devs.size()) +
                       " probed" + (why.empty() ? "" : " (" + why + ")"));
    }
  }
  if (!errors.empty()) {
    std::string m = "the hardware does not match the recipe \"" + r.name + "\":";
    for (const auto& e : errors) m += "\n  " + e;
    fail(SS_E_RECIPE, m);
  }

  // Backtracking over (set, member) slots: members of a set in ascending id, links checked once both ends are bound.
  std::vector<std::vector<size_t>> pick(r.devices.size());
  std::vector<char> used(devs.size(), 0);
  long steps = 0;
  auto set_ok = [&](size_t s) {
    const RDevice& rd = r.devices[s];
    std::set<std::string> hosts;
    for (size_t i : pick[s]) hosts.insert(devs[i].host);
    if (hosts.size() > 1) {
      if (!rd.has_internal_link) return false;
      for (const auto& a : hosts)
        for (const auto& b : hosts)
          if (a < b && !hosts_linked(links, a, b, rd.internal_kind, rd.internal_gbps_min)) return false;
    }
    for (const auto& l : r.links) {
      size_t a = 0, b = 0;
      for (size_t k = 0; k < r.devices.size(); ++k) {
        if (r.devices[k].id == l.a) a = k;
        if (r.devices[k].id == l.b) b = k;
      }
      if ((a != s && b != s) || std::max(a, b) > s) continue;  // checked when the later end is bound
      for (size_t i : pick[a])
        for (size_t j : pick[b])
          if (!hosts_linked(links, devs[i].host, devs[j].host, l.kind, l.gbps_min)) return false;
    }
    return true;
  };
  std::function<bool(size_t)> go = [&](size_t s) -> bool {
    if (s == r.devices.size()) return true;
    if (int(pick[s].size()) == r.devices[s].count) {
      if (!set_ok(s)) return false;
      return go(s + 1);
    }
    for (size_t i : elig[s]) {
      if (used[i] || (!pick[s].empty() && devs[i].id <= devs[pick[s].back()].id)) continue;
      require(++steps <= kMaxBindSteps, SS_E_RECIPE, "match: the binding search is too large (over 10^6 steps)");
      used[i] = 1;
      pick[s].push_back(i);
      if (go(s)) return true;
      pick[s].pop_back();
      used[i] = 0;
    }
    return false;
  };
  if (!go(0)) {
    std::string m = "the hardware does not match the recipe \"" + r.name +
                    "\": no assignment of the probed devices satisfies the links:";
    for (const auto& d : r.devices)
      if (d.has_internal_link)
        m += "\n  match.devices." + d.id + ".internal_link: " + d.internal_kind + " >= " + fmtg(d.internal_gbps_min) +
             " Gb/s between the set's hosts";
    for (const auto& l : r.links)
      m += "\n  match.links." + l.id + ": " + l.kind + " >= " + fmtg(l.gbps_min) + " Gb/s between every host of \"" +
           l.a + "\" and of \"" + l.b + "\"";
    m += "\n  (a set whose members land on two or more hosts needs an internal_link)";
    fail(SS_E_RECIPE, m);
  }
  Binding b;
  for (size_t s = 0; s < r.devices.size(); ++s) {
    std::vector<BoundMember> mem;
    for (size_t i : pick[s]) {
      const ProbeDev& x = devs[i];
      mem.push_back({x.id, x.host, x.driver, x.ordinal, x.disk, x.h2d});
    }
    b.sets.push_back({r.devices[s].id, mem});
  }
  return b;
}

Json binding_json(const Binding& b) {
  Json j = Json::object();
  for (const auto& s : b.sets) {
    Json a = Json::array();
    for (const auto& m : s.second) {
      Json e = Json::object();
      e["id"] = Json::string(m.id);
      e["host"] = Json::string(m.host);
      if (m.ordinal >= 0) e["ordinal"] = Json::number(m.ordinal);
      a.push(e);
    }
    j[s.first] = a;
  }
  return j;
}

// ---- admission (docs/recipes.md §7) ------------------------------------------------------------------------------
Json recipe_select(const Recipe& r, const std::string& eff_sha, const Json& state) {
  require(state.is_object(), SS_E_RECIPE, "state: expected a JSON object");
  const std::string phase_s = pstr(state, "phase", "state", true);
  require(phase_s == "prefill" || phase_s == "decode", SS_E_RECIPE, "state.phase: \"prefill\" or \"decode\"");
  const Phase phase = phase_s == "prefill" ? Phase::Prefill : Phase::Decode;
  const int64_t prompt = pint(state, "prompt_tokens", "state", true);
  const int64_t context = pint(state, "context_tokens", "state", true);
  const int64_t batch = pint(state, "batch", "state", true);
  require(batch >= 1, SS_E_RECIPE, "state.batch: must be >= 1");
  const std::string session = pstr(state, "session_route", "state", false);
  const Json* binding = field(state, "binding");
  const Json* devices = field(state, "devices");
  const Json* links = field(state, "links");
  const Json* hosts = field(state, "hosts");
  const Json* descs = field(state, "descriptors");
  const Json* running = field(state, "running");
  const Json* reqs = field(state, "requirements");
  require(!running || running->is_array(), SS_E_RECIPE, "state.running: expected an array");
  require(!reqs || reqs->is_array(), SS_E_RECIPE, "state.requirements: expected an array");
  const uint64_t floor = gib_bytes(r.host_free_ram_floor_gib);
  const uint64_t commit_floor = gib_bytes(r.host_free_commit_floor_gib);

  auto when_holds = [&](const RRoute& rt) {
    const RWhen& w = rt.when;
    return prompt >= w.prompt_tokens_min && (w.prompt_tokens_max == kNoLimit || prompt <= w.prompt_tokens_max) &&
           batch >= w.batch_min && (w.batch_max == kNoLimit || batch <= w.batch_max);
  };
  // Members of a set from the state's binding: (probe id, host), or a reason.
  auto members_of = [&](const std::string& set, std::vector<std::pair<std::string, std::string>>& out) -> std::string {
    const RDevice* d = r.device(set);
    const Json* b = binding ? field(*binding, set.c_str()) : nullptr;
    if (!b || !b->is_array() || int(b->arr.size()) != d->count) return "no binding for set \"" + set + "\"";
    for (const Json& m : b->arr) {
      const Json* id = field(m, "id");
      const Json* host = field(m, "host");
      if (!id || !id->is_string() || !host || !host->is_string()) return "bad binding for set \"" + set + "\"";
      out.push_back({id->text, host->text});
    }
    return "";
  };
  // One route alone: devices, links, bounds; its counting requirements per member go to `charge` unless credited.
  auto route_check = [&](const RRoute& rt, std::map<std::string, uint64_t>& charge,
                         std::map<std::string, uint64_t>& commit) -> std::string {
    if (!rt.enabled) return "the route is disabled";
    std::vector<std::string> plan_ids;
    std::string ownership;
    std::map<std::string, uint64_t> mine, mine_commit;
    for (const auto& set : rt.devices) {
      std::vector<std::pair<std::string, std::string>> mem;
      std::string why = members_of(set, mem);
      if (!why.empty()) return why;
      for (const auto& m : mem) {
        const Json* dv = devices ? field(*devices, m.first.c_str()) : nullptr;
        const Json* av = dv ? field(*dv, "available") : nullptr;
        if (!av || av->kind != Json::Kind::Bool || !av->b)
          return "device " + m.first + " of \"" + set + "\" is not available";
      }
      const RDevice* d = r.device(set);
      if (d->has_internal_link) {
        const Json* lv = links ? field(*links, (set + ".internal").c_str()) : nullptr;
        const Json* up = lv ? field(*lv, "up") : nullptr;
        if (!up || up->kind != Json::Kind::Bool || !up->b) return "link " + set + ".internal is down";
      }
      const RPlacement* p = rt.placement(set);
      if (context > int64_t(p->get("max_context_tokens")))
        return "context " + std::to_string(context) + " tokens is above max_context_tokens " +
               std::to_string(int64_t(p->get("max_context_tokens"))) + " of \"" + set + "\"";
      if (batch > int64_t(p->get("max_batch")))
        return "batch " + std::to_string(batch) + " is above max_batch " +
               std::to_string(int64_t(p->get("max_batch"))) + " of \"" + set + "\"";
      // Requirements: exactly one counting entry per member.
      const Json* dset = descs ? field(*descs, set.c_str()) : nullptr;
      for (size_t k = 0; k < mem.size(); ++k) {
        const Json* dsha = dset && dset->is_array() && k < dset->arr.size() ? &dset->arr[k] : nullptr;
        if (!dsha || !dsha->is_string())
          return "the state has no descriptor in use for \"" + set + "\" member " + std::to_string(k);
        int count = 0;
        for (const Json& q : reqs ? reqs->arr : std::vector<Json>{}) {
          if (pstr(q, "route", "state.requirements[]", false) != rt.id ||
              pstr(q, "set", "state.requirements[]", false) != set ||
              pint(q, "member", "state.requirements[]", false) != int64_t(k))
            continue;
          if (pstr(q, "effective_sha256", "state.requirements[]", false) != eff_sha ||
              pstr(q, "id", "state.requirements[]", false) != mem[k].first ||
              pstr(q, "descriptor_sha256", "state.requirements[]", false) != dsha->text)
            continue;
          ++count;
          const uint64_t hb = uint64_t(pint(q, "host_bytes", "state.requirements[]", true));
          const uint64_t cb = uint64_t(pint(q, "commit_bytes", "state.requirements[]", true));
          mine[mem[k].second] = add_ck(mine[mem[k].second], hb, "state.requirements");
          mine_commit[mem[k].second] = add_ck(mine_commit[mem[k].second], cb, "state.requirements");
          if (pint(q, "disk_bytes", "state.requirements[]", true) > 0) {
            const Json* dv = devices ? field(*devices, mem[k].first.c_str()) : nullptr;
            const Json* da = dv ? field(*dv, "disk_available") : nullptr;
            if (!da || da->kind != Json::Kind::Bool || !da->b)
              return "the disk of " + mem[k].first + " (\"" + set + "\") is not available and the plan streams from it";
          }
          plan_ids.push_back(pstr(q, "plan_id", "state.requirements[]", true));
          const std::string own = pstr(q, "ownership_sha256", "state.requirements[]", false);
          if (rt.execute == Execute::Hetero) {
            if (own.empty())
              return "requirement of \"" + set + "\" member " + std::to_string(k) + " has no ownership_sha256";
            if (ownership.empty()) ownership = own;
            if (own != ownership) return "the hetero members were compiled against different ownership maps";
          }
        }
        if (count != 1)
          return count == 0
                     ? "not compiled for this recipe: no requirement for \"" + set + "\" member " + std::to_string(k)
                     : "several requirements for \"" + set + "\" member " + std::to_string(k);
      }
    }
    for (const auto& lid : rt.links) {
      const Json* lv = links ? field(*links, lid.c_str()) : nullptr;
      const Json* up = lv ? field(*lv, "up") : nullptr;
      if (!up || up->kind != Json::Kind::Bool || !up->b) return "link " + lid + " is down";
    }
    // Running credit: same effective recipe and exactly these plans.
    bool credited = false;
    for (const Json& e : running ? running->arr : std::vector<Json>{}) {
      if (pstr(e, "route", "state.running[]", false) != rt.id) continue;
      if (pstr(e, "effective_sha256", "state.running[]", false) != eff_sha) continue;
      const Json* ids = field(e, "plan_ids");
      if (!ids || !ids->is_array()) continue;
      std::vector<std::string> got;
      for (const Json& x : ids->arr) got.push_back(x.is_string() ? x.text : "");
      std::vector<std::string> want = plan_ids;
      std::sort(got.begin(), got.end());
      std::sort(want.begin(), want.end());
      credited = credited || got == want;
    }
    if (!credited) {
      for (const auto& kv : mine) charge[kv.first] = add_ck(charge[kv.first], kv.second, "admission");
      for (const auto& kv : mine_commit) commit[kv.first] = add_ck(commit[kv.first], kv.second, "admission");
    }
    return "";
  };
  auto admit = [&](const RRoute& rt, std::map<std::string, uint64_t>& charge) -> std::string {
    std::map<std::string, uint64_t> commit;
    std::string why = route_check(rt, charge, commit);
    if (!why.empty()) return why;
    if (!rt.handoff_route.empty()) {
      const RRoute* h = r.route(rt.handoff_route);
      if (!when_holds(*h)) return "the handoff route \"" + h->id + "\" does not take this workload";
      why = route_check(*h, charge, commit);
      if (!why.empty()) return "handoff route \"" + h->id + "\": " + why;
    }
    if (commit_floor > 0)
      for (const auto& kv : commit) {
        const Json* hv = hosts ? field(*hosts, kv.first.c_str()) : nullptr;
        if (!hv || !field(*hv, "free_commit_gib")) return "no free commit reported for host " + kv.first;
        const double free_gib = pnum(*hv, "free_commit_gib", "state.hosts." + kv.first, true);
        if (free_gib * kGiBf - double(kv.second) < double(commit_floor))
          return "host " + kv.first + ": " + fmtg(free_gib) + " GiB free commit - " + fmtg(double(kv.second) / kGiBf) +
                 " GiB charged is below the commit floor " + fmtg(r.host_free_commit_floor_gib) + " GiB";
      }
    for (const auto& kv : charge) {
      const Json* hv = hosts ? field(*hosts, kv.first.c_str()) : nullptr;
      const Json* fr = hv ? field(*hv, "free_ram_gib") : nullptr;
      if (!fr) return "no free RAM reported for host " + kv.first;
      const double free_gib = pnum(*hv, "free_ram_gib", "state.hosts." + kv.first, true);
      const double left = free_gib * kGiBf - double(kv.second);
      if (left < double(floor))
        return "host " + kv.first + ": " + fmtg(free_gib) + " GiB free - " + fmtg(double(kv.second) / kGiBf) +
               " GiB charged is below the floor " + fmtg(r.host_free_ram_floor_gib) + " GiB";
    }
    return "";
  };

  Json trace = Json::array();
  auto note = [&](const std::string& id, const std::string& verdict, const std::string& why) {
    Json e = Json::object();
    e["route"] = Json::string(id);
    e["verdict"] = Json::string(verdict);
    if (!why.empty()) e["why"] = Json::string(why);
    trace.push(e);
  };
  const RRoute* chosen = nullptr;
  std::map<std::string, uint64_t> chosen_charge;
  if (!session.empty()) {
    const RRoute* s = r.route(session);
    require(s != nullptr, SS_E_RECIPE, "state.session_route: unknown route \"" + session + "\"");
    std::map<std::string, uint64_t> charge;
    std::string why = admit(*s, charge);
    require(why.empty(), SS_E_RECIPE,
            "the session's route \"" + session + "\" cannot continue (" + why +
                "); a running request is never moved: end or restart it");
    note(s->id, "admitted", "session route");
    chosen = s;
    chosen_charge = charge;
  } else {
    std::set<std::string> visited;
    for (const auto& rt : r.routes) {
      if (chosen) break;
      if (rt.phase != phase || visited.count(rt.id)) continue;
      visited.insert(rt.id);
      if (!rt.enabled) {
        note(rt.id, "disabled", "");
        continue;
      }
      if (!when_holds(rt)) {
        note(rt.id, "no match", "");
        continue;
      }
      std::map<std::string, uint64_t> charge;
      std::string why = admit(rt, charge);
      if (why.empty()) {
        note(rt.id, "admitted", "");
        chosen = &rt;
        chosen_charge = charge;
        break;
      }
      note(rt.id, "not admitted", why);
      for (const RRoute* f = rt.fallback.empty() ? nullptr : r.route(rt.fallback); f && !chosen;
           f = f->fallback.empty() ? nullptr : r.route(f->fallback)) {
        if (!visited.insert(f->id).second) break;
        if (!f->enabled) {
          note(f->id, "disabled", "fallback of \"" + rt.id + "\"");
          continue;
        }
        if (!when_holds(*f)) {
          note(f->id, "no match", "fallback of \"" + rt.id + "\"");
          continue;
        }
        std::map<std::string, uint64_t> fc;
        std::string fw = admit(*f, fc);
        if (fw.empty()) {
          note(f->id, "admitted", "fallback of \"" + rt.id + "\"");
          chosen = f;
          chosen_charge = fc;
        } else {
          note(f->id, "not admitted", fw);
        }
      }
    }
  }
  if (!chosen) {
    std::string m = "no route of \"" + r.name + "\" can admit this " + phase_s + " request:";
    for (const Json& e : trace.arr)
      m += "\n  " + e.find("route")->text + ": " + e.find("verdict")->text +
           (e.find("why") ? " (" + e.find("why")->text + ")" : "");
    fail(SS_E_RECIPE, m);
  }
  Json out = Json::object();
  out["schema"] = Json::string("strix-stream/recipe-select-v1");
  out["recipe"] = Json::string(r.name);
  out["effective_sha256"] = Json::string(eff_sha);
  out["route"] = Json::string(chosen->id);
  if (!chosen->handoff_route.empty()) out["handoff_route"] = Json::string(chosen->handoff_route);
  Json ch = Json::object();
  for (const auto& kv : chosen_charge) ch[kv.first] = Json::number(kv.second);
  out["charge"] = ch;
  out["trace"] = trace;
  return out;
}

// ---- compilation (docs/recipes.md §5, §8.1) -----------------------------------------------------------------------
RecipePlan recipe_plan(const Recipe& r, const std::string& eff_sha, const std::string& route, const std::string& set,
                       int member, const std::string& device_id, const Desc& desc, const std::vector<double>& benefit,
                       const std::vector<RefInput>& refs) {
  const RRoute* rt = r.route(route);
  require(rt != nullptr, SS_E_RECIPE, "plan: unknown route \"" + route + "\"");
  auto it = std::find(rt->devices.begin(), rt->devices.end(), set);
  require(it != rt->devices.end(), SS_E_RECIPE, "plan: \"" + set + "\" is not a device set of route \"" + route + "\"");
  const size_t si = size_t(it - rt->devices.begin());
  const RDevice* dev = r.device(set);
  require(member >= 0 && member < dev->count, SS_E_RECIPE,
          "plan: member " + std::to_string(member) + " of \"" + set + "\" (count " + std::to_string(dev->count) + ")");
  require(benefit.empty() || int64_t(benefit.size()) == desc.total_units, SS_E_RECIPE,
          "plan: benefit needs one value per unit of the descriptor (ss_desc_total_units)");
  const RPlacement& p = *rt->placement(set);
  const std::string w = "routes." + route + ".placement." + set;
  if (!r.model.empty())
    require(desc.model.empty() || desc.model == r.model, SS_E_RECIPE,
            "plan: the descriptor is for model \"" + desc.model + "\", the recipe for \"" + r.model + "\"");
  if (!r.engine.empty())
    require(desc.engine == r.engine, SS_E_RECIPE,
            "plan: the descriptor is for engine \"" + desc.engine + "\", the recipe for \"" + r.engine + "\"");

  RecipePlan out;
  uint64_t rows = 0;
  out.budget = budget_for(*rt, p, desc, w, &rows);
  ss_options& o = out.options;
  o.struct_size = uint32_t(sizeof o);
  o.backend = dev->kind;
  o.device = 0;
  o.policy = ss_policy(int(p.get("policy")));
  o.lookahead_budget = uint32_t(p.get("lookahead_units"));
  o.reader_threads = uint32_t(p.get("reader_threads"));
  o.staging_buffers = uint32_t(p.get("staging_buffers"));
  o.staging_bytes = uint64_t(p.get("staging_mib")) << 20;
  o.copy_piece_bytes = uint64_t(p.get("copy_piece_mib")) << 20;
  o.verify_hash = p.get("verify_hash") != 0;
  o.fill_residents = p.get("fill_residents") != 0;
  o.score_window = uint32_t(p.get("score_window"));
  o.pin_planned = p.get("pin_planned") != 0;
  o.direct_io = p.get("direct_io") != 0;
  o.group_units = uint32_t(p.get("group_units"));
  o.spec_persistent = p.get("spec_persistent") != 0;
  require(p.get("tier_mode") == kTierInclusive, SS_E_RECIPE,
          w + ".tier_mode: \"exclusive\" is not supported by this strix-stream build (inclusive tiers only)");
  uint64_t max_align = 4096;
  for (const auto& s : desc.sources) max_align = std::max(max_align, s.align);
  require(
      o.staging_bytes >= 4 * max_align, SS_E_RECIPE,
      w + ".staging_mib: must be >= 4 x the largest source alignment (" + std::to_string(4 * max_align) + " bytes)");

  // Units this member plans: all, or (hetero) the ones its set owns in the route-wide ownership map.
  std::vector<char> avail(size_t(desc.total_units), 1);
  std::string ownership_sha;
  std::vector<std::string> trace;
  if (rt->execute == Execute::Hetero) {
    require(refs.size() == rt->devices.size(), SS_E_RECIPE,
            "plan: a hetero route needs the reference descriptor of every set (" + std::to_string(rt->devices.size()) +
                ")");
    for (size_t k = 0; k < refs.size(); ++k) {
      require(refs[k].desc != nullptr, SS_E_RECIPE, "plan: missing reference descriptor of \"" + rt->devices[k] + "\"");
      same_structure(*refs[0].desc, *refs[k].desc, "plan");
      require(refs[k].benefit.empty() || int64_t(refs[k].benefit.size()) == refs[k].desc->total_units, SS_E_RECIPE,
              "plan: reference benefit of \"" + rt->devices[k] + "\" needs one value per unit");
    }
    same_structure(*refs[0].desc, desc, "plan");
    const Desc& d0 = *refs[0].desc;
    const size_t n = size_t(d0.total_units);
    std::vector<int> state(n, -1), owner(n, -1);  // state: -1 open, else the set it is (or was sent) to
    for (size_t k = 0; k < rt->devices.size(); ++k) {
      const RPlacement& pk = *rt->placement(rt->devices[k]);
      const std::string wk = "routes." + route + ".placement." + rt->devices[k];
      std::vector<char> av(n, 0);
      for (size_t g = 0; g < n; ++g) av[g] = state[g] < 0 || state[g] == int(k);
      ss_budget bk = budget_for(*rt, pk, *refs[k].desc, wk, nullptr);
      Tiering tk = tier_units(*refs[k].desc, bk, pk, refs[k].benefit, av, rt->devices, wk);
      for (size_t g = 0; g < n; ++g) {
        if (!av[g]) continue;
        if (tk.tier[g] == SS_TIER_PAIR && tk.peer[g] >= 0) {
          state[g] = tk.peer[g];
        } else {
          state[g] = int(k);
          owner[g] = int(k);
        }
      }
    }
    std::string map;
    for (size_t L = 0; L < d0.layers.size(); ++L) {
      if (!d0.layers[L].streamable) continue;
      for (int u = 0; u < d0.layers[L].n_units; ++u) {
        const size_t g = size_t(d0.unit_base[L] + u);
        if (owner[g] < 0)
          fail(SS_E_RECIPE,
               "routes." + route + ": unit (layer " + std::to_string(L) + ", " + std::to_string(u) + ") has no owner" +
                   (state[g] >= 0 ? " (sent to \"" + rt->devices[size_t(state[g])] + "\", which does not hold it)"
                                  : std::string()));
        map += char('A' + owner[g]);
      }
    }
    ownership_sha = sha_of(map);
    for (size_t g = 0; g < n; ++g) avail[g] = owner[g] == int(si);
    trace.push_back("routes." + route + ": ownership " + ownership_sha.substr(0, 16) + " (" +
                    std::to_string(std::count(owner.begin(), owner.end(), int(si))) + " units owned by \"" + set +
                    "\")");
  }

  std::vector<uint8_t> tiers;
  if (p.groups.empty()) {
    out.plan = plan_make(desc, out.budget, benefit);
    tiers = out.plan.tier;
    trace.push_back(w + ": no unit groups, planner ss_plan_make");
  } else {
    Tiering t = tier_units(desc, out.budget, p, benefit, avail, rt->devices, w);
    if (rt->execute == Execute::Hetero)
      for (size_t g = 0; g < t.peer.size(); ++g)
        require(!(avail[g] && t.peer[g] >= 0), SS_E_RECIPE,
                w + ": member " + std::to_string(member) + " cannot hold a unit its set owns (global unit " +
                    std::to_string(g) + " would go to a peer); its descriptor differs from member 0's");
    trace.insert(trace.end(), t.trace.begin(), t.trace.end());
    try {
      out.plan = plan_from_tiers(desc, out.budget, t.tier);
    } catch (const Error& e) {
      fail(e.status(), w + ": " + e.what());
    }
    tiers = t.tier;
  }

  // Each stream ring holds (prefetch_layers + 1) layers of its unit size's streamed units.
  const uint32_t pf = uint32_t(p.get("prefetch_layers"));
  for (const RingClass& rc : out.plan.rings) {
    int most = 0;
    for (size_t L = 0; L < desc.layers.size(); ++L) {
      if (!desc.layers[L].streamable || desc.unit_stride(int(L)) != rc.stride) continue;
      int k = 0;
      for (int u = 0; u < desc.layers[L].n_units; ++u) {
        const uint8_t x = tiers[size_t(desc.unit_base[L] + u)];
        k += x == SS_TIER_PINNED || x == SS_TIER_DISK;
      }
      most = std::max(most, k);
    }
    require(uint64_t(rc.slots) >= uint64_t(pf + 1) * uint64_t(most), SS_E_RECIPE,
            w + ".ring_slots: the " + std::to_string(rc.stride) + "-byte ring has " + std::to_string(rc.slots) +
                " slots, below (prefetch_layers + 1) x " + std::to_string(most) + " streamed units per layer");
  }

  // Resources (docs/recipes.md §7.3): everything the library allocates for this member.
  const uint64_t staging_one = (o.staging_bytes + max_align - 1) / max_align * max_align;
  const uint64_t staging = mul_ck(staging_one, o.staging_buffers, w);
  uint64_t tables = 0, ids = 0, dev_tables = 0;
  uint64_t meta = add_ck(65536, mul_ck(16, uint64_t(desc.total_units), w), w);
  for (const RingClass& rc : out.plan.rings) meta = add_ck(meta, 128 * uint64_t(rc.slots), w);
  for (const auto& l : desc.layers) {
    if (l.n_units <= 0) continue;
    const uint64_t nu = uint64_t(l.n_units);
    tables = add_ck(tables, 16 * nu, w);
    if (dev->kind == SS_BACKEND_CUDA) dev_tables = add_ck(dev_tables, 16 * nu, w);
    if (l.kind == LayerKind::Moe)
      ids = add_ck(ids, mul_ck(4, std::max<uint64_t>(1024, mul_ck(rows, uint64_t(std::max(l.top_k, 1)), w)), w), w);
    meta = add_ck(meta, 4096 + 768 * nu, w);
    if (o.policy == SS_POLICY_SCORED)
      meta = add_ck(meta, add_ck(mul_ck(4 * nu, nu, w), mul_ck(uint64_t(o.score_window) + 1, 64 + 8 * nu, w), w), w);
  }
  const uint64_t device_bytes = add_ck(add_ck(out.plan.peak, out.plan.margin, w), dev_tables, w);
  uint64_t host = add_ck(add_ck(add_ck(out.plan.pinned, staging, w), add_ck(tables, ids, w), w), meta, w);
  if (dev->unified()) host = add_ck(host, device_bytes, w);
  // Commit: host memory plus, on a discrete device, its allocations (Windows WDDM bills them to commit).
  const uint64_t commit = dev->unified() ? host : add_ck(host, device_bytes, w);

  const std::string desc_sha = sha_of(json_write(desc_to_json(desc)));
  const std::string ben_sha = benefit_sha(benefit);
  const std::string plan_id = sha_of(eff_sha + "|" + route + "|" + set + "|" + std::to_string(member) + "|" +
                                     device_id + "|" + desc_sha + "|" + ben_sha + "|" + ownership_sha);

  Json j = Json::object();
  j["schema"] = Json::string("strix-stream/recipe-plan-v1");
  j["recipe"] = Json::string(r.name);
  j["effective_sha256"] = Json::string(eff_sha);
  j["route"] = Json::string(route);
  j["set"] = Json::string(set);
  j["member"] = Json::number(int64_t(member));
  j["id"] = Json::string(device_id);
  j["descriptor_sha256"] = Json::string(desc_sha);
  j["benefit_sha256"] = Json::string(ben_sha);
  if (!ownership_sha.empty()) j["ownership_sha256"] = Json::string(ownership_sha);
  j["plan_id"] = Json::string(plan_id);
  Json bj = Json::object();
  bj["device_bytes"] = Json::number(out.budget.device_bytes);
  bj["host_pinned_bytes"] = Json::number(out.budget.host_pinned_bytes);
  bj["workspace_bytes"] = Json::number(out.budget.workspace_bytes);
  bj["margin_gib"] = Json::number(out.budget.margin_gib);
  bj["margin_fraction"] = Json::number(out.budget.margin_fraction);
  bj["dynamic_slots_per_layer"] = Json::number(uint64_t(out.budget.dynamic_slots_per_layer));
  bj["ring_slots"] = Json::number(uint64_t(out.budget.ring_slots));
  bj["pinned_backing"] = Json::boolean(out.budget.pinned_backing != 0);
  j["budget"] = bj;
  Json oj = Json::object();
  oj["backend"] = Json::string(backend_word(o.backend));
  oj["policy"] = Json::string(o.policy == SS_POLICY_LRU ? "lru" : "scored");
  oj["lookahead_budget"] = Json::number(uint64_t(o.lookahead_budget));
  oj["reader_threads"] = Json::number(uint64_t(o.reader_threads));
  oj["staging_buffers"] = Json::number(uint64_t(o.staging_buffers));
  oj["staging_bytes"] = Json::number(o.staging_bytes);
  oj["copy_piece_bytes"] = Json::number(o.copy_piece_bytes);
  oj["verify_hash"] = Json::boolean(o.verify_hash != 0);
  oj["fill_residents"] = Json::boolean(o.fill_residents != 0);
  oj["score_window"] = Json::number(uint64_t(o.score_window));
  oj["pin_planned"] = Json::boolean(o.pin_planned != 0);
  oj["direct_io"] = Json::boolean(o.direct_io != 0);
  oj["group_units"] = Json::number(uint64_t(o.group_units));
  oj["spec_persistent"] = Json::boolean(o.spec_persistent != 0);
  j["options"] = oj;
  Json ej = Json::object();
  ej["prefetch_layers"] = Json::number(uint64_t(pf));
  ej["chunk_tokens"] = Json::number(uint64_t(p.get("chunk_tokens")));
  ej["routed_only_max_tokens"] = Json::number(uint64_t(p.get("routed_only_max_tokens")));
  ej["idle_preload_layers"] = Json::number(uint64_t(p.get("idle_preload_layers")));
  ej["max_copies_per_token"] = Json::number(uint64_t(p.get("max_copies_per_token")));
  ej["max_context_tokens"] = Json::number(uint64_t(p.get("max_context_tokens")));
  ej["max_batch"] = Json::number(uint64_t(p.get("max_batch")));
  ej["forward_rows"] = Json::number(rows);
  j["engine"] = ej;
  Json rj = Json::object();
  rj["device_bytes"] = Json::number(device_bytes);
  rj["host_pinned_bytes"] = Json::number(out.plan.pinned);
  rj["staging_bytes"] = Json::number(staging);
  rj["tables_bytes"] = Json::number(add_ck(tables, ids, w));
  rj["metadata_bytes"] = Json::number(meta);
  rj["host_bytes"] = Json::number(host);
  rj["commit_bytes"] = Json::number(commit);
  rj["disk_bytes"] = Json::number(out.plan.disk);
  j["resources"] = rj;
  j["plan"] = plan_to_json(desc, out.plan);
  Json tj = Json::array();
  for (const auto& s : trace) tj.push(Json::string(s));
  j["trace"] = tj;
  out.doc = j;
  return out;
}

}  // namespace ss
