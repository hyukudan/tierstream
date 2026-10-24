#include "plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include "errors.hpp"

namespace ss {
namespace {

struct U {
  int64_t g;  // global index (model order)
  int layer;
  uint64_t bytes;  // pool stride (what the device holds)
  double benefit;
};

constexpr uint64_t kHostStride = 4096;
constexpr uint64_t kPinnedChunk = uint64_t(1) << 30;

bool has_pinned_copy(uint8_t t) { return t == SS_TIER_PINNED || t == SS_TIER_RESIDENT_BACKED; }

std::string gib(uint64_t n) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.3f", double(n) / double(kGiB));
  return buf;
}

double per_byte(const U& u) {
  return u.bytes ? u.benefit / double(u.bytes) : std::numeric_limits<double>::infinity();
}

bool better(const U& a, const U& b) {
  double x = per_byte(a), y = per_byte(b);
  if (x != y) return x > y;
  return a.g < b.g;
}

std::vector<U> units_of(const Desc& d, const std::vector<double>& benefit) {
  std::vector<U> units;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const Layer& l = d.layers[L];
    if (!l.streamable) continue;
    for (int u = 0; u < l.n_units; ++u) {
      int64_t g = d.unit_base[L] + u;
      double ben = !benefit.empty() ? benefit[size_t(g)] : (!l.benefit.empty() ? l.benefit[size_t(u)] : 1.0);
      require(std::isfinite(ben) && ben >= 0, SS_E_INVALID, "plan: benefit must be finite and >= 0");
      units.push_back({g, int(L), d.unit_stride(int(L)), ben});
    }
  }
  return units;
}

// Upper bound of the dynamic slots (before residents are chosen); finalize computes the exact allocation.
uint64_t dynamic_bound(const Desc& d, const ss_budget& b) {
  uint64_t t = 0;
  if (!b.dynamic_slots_per_layer) return 0;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const Layer& l = d.layers[L];
    if (l.streamable && l.n_units > 0)
      t += uint64_t(std::min<uint32_t>(b.dynamic_slots_per_layer, uint32_t(l.n_units))) * d.unit_stride(int(L));
  }
  return t;
}

// Rings: unit-sized device slots for whatever is not resident, one ring per unit size so a large blob never sizes
// the expert slots. Slots per ring: the budget's, else the descriptor's, else 2 x the most units any layer of that
// size has (two layers in flight).
std::vector<RingClass> rings_for(const Desc& d, const ss_budget& b, const std::vector<U>& streamed) {
  std::vector<RingClass> out;
  for (const U& u : streamed) {
    if (d.tiers == Tiers::Pair || !u.bytes) continue;
    auto it = std::find_if(out.begin(), out.end(), [&](const RingClass& c) { return c.stride == u.bytes; });
    if (it != out.end()) continue;
    uint32_t slots = b.ring_slots ? b.ring_slots : d.ring_slots;
    if (!slots)
      for (size_t L = 0; L < d.layers.size(); ++L)
        if (d.layers[L].streamable && d.layers[L].n_units && d.unit_stride(int(L)) == u.bytes)
          slots = std::max(slots, uint32_t(2 * d.layers[L].n_units));
    out.push_back({u.bytes, slots});
  }
  std::sort(out.begin(), out.end(), [](const RingClass& x, const RingClass& y) { return x.stride < y.stride; });
  return out;
}

uint64_t ring_bytes(const std::vector<RingClass>& r) {
  uint64_t t = 0;
  for (const RingClass& c : r) t += c.stride * c.slots;
  return t;
}

// Totals and the last checks, from the tiers alone (shared by the planner and by externally decided tiers). Every
// figure is what the context allocates.
void finalize(const Desc& d, const ss_budget& b, Plan& p) {
  std::vector<U> streamed;
  p.resident = p.pinned = p.disk = p.dynamic = 0;
  p.n_resident = p.n_pinned = p.n_disk = p.n_pair = p.n_backed = 0;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const Layer& l = d.layers[L];
    int planned = 0;
    for (int u = 0; u < l.n_units; ++u) {
      int64_t g = d.unit_base[L] + u;
      uint8_t t = p.tier[size_t(g)];
      require(t <= SS_TIER_RESIDENT_BACKED, SS_E_INVALID, "plan: bad tier value");
      if (!l.streamable) {
        require(t == SS_TIER_RESIDENT, SS_E_INVALID, "plan: a non-streamable layer's units must be resident");
        continue;
      }
      if (t == SS_TIER_RESIDENT || t == SS_TIER_RESIDENT_BACKED) {
        p.resident += d.unit_stride(int(L));
        ++p.n_resident;
        ++planned;
        if (t == SS_TIER_RESIDENT_BACKED) ++p.n_backed;
        continue;
      }
      // Pair units are computed by another device and never fetched: they take no stream ring.
      if (t != SS_TIER_PAIR) streamed.push_back({g, int(L), d.unit_stride(int(L)), 0.0});
      if (t == SS_TIER_PINNED) {
        ++p.n_pinned;
      } else if (t == SS_TIER_DISK) {
        p.disk += l.unit_bytes;
        ++p.n_disk;
      } else {
        ++p.n_pair;
      }
    }
    // Dynamic slots as the context allocates them: never more than the layer's non-resident units.
    if (l.streamable && l.n_units > 0 && b.dynamic_slots_per_layer)
      p.dynamic += uint64_t(std::min<uint32_t>(b.dynamic_slots_per_layer, uint32_t(l.n_units - planned))) *
                   d.unit_stride(int(L));
  }
  p.pinned = pinned_layout(d, p.tier).total;
  for (const auto& n : d.host_fixed) p.pinned += n.bytes;  // sum validated not to overflow (desc_validate)
  require(p.pinned >= pinned_layout(d, p.tier).total, SS_E_INVALID, "plan: host byte total overflows");
  p.rings = rings_for(d, b, streamed);
  p.ring = ring_bytes(p.rings);
  require(p.pinned <= b.host_pinned_bytes, SS_E_CAPACITY,
          d.engine + ": " + gib(p.pinned) + " GiB must be pinned on the host, above the host budget " +
              gib(b.host_pinned_bytes) + " GiB; raise the cap or the host budget");
  p.peak = p.fixed + p.dynamic + p.ring + p.resident;
  require(p.peak + p.margin <= p.cap, SS_E_CAPACITY,
          d.engine + ": predicted peak " + gib(p.peak) + " GiB above the cap " + gib(p.cap) + " GiB - margin " +
              gib(p.margin) + " GiB");
}

Plan start(const Desc& d, const ss_budget& b) {
  require(b.device_bytes > 0, SS_E_INVALID, "plan: device budget must be > 0");
  Plan p;
  p.budget = b;
  p.cap = b.device_bytes;
  p.margin = plan_margin(b);
  p.tier.assign(size_t(d.total_units), SS_TIER_RESIDENT);
  p.dynamic_slots_per_layer = b.dynamic_slots_per_layer;
  p.fixed = d.fixed_total() + b.workspace_bytes;
  p.dynamic = dynamic_bound(d, b);
  require(p.cap >= p.margin + p.fixed + p.dynamic, SS_E_CAPACITY,
          d.engine + ": fixed device memory " + gib(p.fixed) + " GiB + dynamic slots " + gib(p.dynamic) +
              " GiB + margin " + gib(p.margin) + " GiB exceeds the " + gib(p.cap) + " GiB cap");
  return p;
}

}  // namespace

uint64_t host_stride(uint64_t bytes) { return (bytes + kHostStride - 1) / kHostStride * kHostStride; }

uint64_t plan_room(const Desc& d, const ss_budget& b) {
  Plan p = start(d, b);
  return p.cap - p.margin - p.fixed - p.dynamic;
}

uint64_t plan_ring_bytes(const Desc& d, const ss_budget& b, const std::vector<int64_t>& streamed) {
  std::vector<U> units;
  units.reserve(streamed.size());
  for (int64_t g : streamed) {
    const int L = int(std::upper_bound(d.unit_base.begin(), d.unit_base.end(), g) - d.unit_base.begin()) - 1;
    units.push_back({g, L, d.unit_stride(L), 0.0});
  }
  return ring_bytes(rings_for(d, b, units));
}

void plan_backing(const Desc& d, const ss_budget& b, const std::vector<double>& benefit, std::vector<uint8_t>& tier) {
  // Pinned copies for residents with the host budget left after host_fixed and the pinned tier, the likeliest to be
  // evicted (lowest benefit per byte) first, stopping at the first that does not fit.
  uint64_t host_left = b.host_pinned_bytes;
  for (const auto& n : d.host_fixed) host_left = host_left > n.bytes ? host_left - n.bytes : 0;
  std::vector<U> res;
  for (const U& u : units_of(d, benefit)) {
    const uint8_t t = tier[size_t(u.g)];
    if (t == SS_TIER_PINNED) {
      const uint64_t hb = host_stride(d.layers[size_t(u.layer)].unit_bytes);
      host_left = host_left > hb ? host_left - hb : 0;
    } else if (t == SS_TIER_RESIDENT) {
      res.push_back(u);
    }
  }
  std::stable_sort(res.begin(), res.end(), [](const U& x, const U& y) { return better(y, x); });
  for (const U& u : res) {
    const uint64_t hb = host_stride(d.layers[size_t(u.layer)].unit_bytes);
    if (hb > host_left) break;
    tier[size_t(u.g)] = SS_TIER_RESIDENT_BACKED;
    host_left -= hb;
  }
}

PinnedLayout pinned_layout(const Desc& d, const std::vector<uint8_t>& tier) {
  PinnedLayout pl;
  pl.chunk_of.assign(size_t(d.total_units), -1);
  pl.offset_of.assign(size_t(d.total_units), 0);
  uint64_t used = 0;
  bool open = false;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const uint64_t stride = host_stride(d.layers[L].unit_bytes);
    for (int u = 0; u < d.layers[L].n_units; ++u) {
      int64_t g = d.unit_base[L] + u;
      if (!has_pinned_copy(tier[size_t(g)])) continue;
      if (!open || (used > 0 && used + stride > kPinnedChunk)) {
        if (open) pl.chunk_bytes.push_back(used);
        used = 0;
        open = true;
      }
      pl.chunk_of[size_t(g)] = int32_t(pl.chunk_bytes.size());
      pl.offset_of[size_t(g)] = used;
      used += stride;
    }
  }
  if (open) pl.chunk_bytes.push_back(used);
  for (uint64_t c : pl.chunk_bytes) {
    require(pl.total <= UINT64_MAX - c, SS_E_INVALID, "plan: pinned layout overflows");
    pl.total += c;
  }
  return pl;
}

bool same_budget(const ss_budget& a, const ss_budget& b) {
  return a.device_bytes == b.device_bytes && a.host_pinned_bytes == b.host_pinned_bytes &&
         a.workspace_bytes == b.workspace_bytes && a.margin_gib == b.margin_gib &&
         a.margin_fraction == b.margin_fraction && a.dynamic_slots_per_layer == b.dynamic_slots_per_layer &&
         a.ring_slots == b.ring_slots && a.pinned_backing == b.pinned_backing;
}

uint64_t plan_margin(const ss_budget& b) {
  double m = std::max(b.margin_gib * double(kGiB), b.margin_fraction * double(b.device_bytes));
  return uint64_t(m);
}

Plan plan_make(const Desc& d, const ss_budget& b, const std::vector<double>& benefit) {
  require(benefit.empty() || int64_t(benefit.size()) == d.total_units, SS_E_INVALID,
          "plan: benefit needs one value per unit (ss_desc_total_units)");
  Plan p = start(d, b);
  std::vector<U> units = units_of(d, benefit);
  uint64_t room = p.cap - p.margin - p.fixed - p.dynamic;

  std::vector<char> chosen(units.size(), 0);
  uint64_t total = 0;
  for (const U& u : units) total += u.bytes;
  bool ok = false;
  if (d.order == Order::Prefix) {
    // Longest prefix whose bytes plus the rings for the remaining units fit.
    std::vector<U> rest;
    for (size_t k = units.size() + 1; k-- > 0;) {
      rest.assign(units.begin() + std::ptrdiff_t(k), units.end());
      if (total + ring_bytes(rings_for(d, b, rest)) <= room) {
        for (size_t i = 0; i < k; ++i) chosen[i] = 1;
        ok = true;
        break;
      }
      if (k) total -= units[k - 1].bytes;
    }
  } else if (total <= room) {
    std::fill(chosen.begin(), chosen.end(), char(1));
    ok = true;
  } else {
    // Something streams: reserve the rings for every unit size first, then best benefit per byte (ties in model
    // order). Greedy is exact for equal-sized units, which is every layer's case.
    uint64_t ring = ring_bytes(rings_for(d, b, units));
    if (ring <= room) {
      room -= ring;
      std::vector<size_t> idx(units.size());
      for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
      std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t c) { return better(units[a], units[c]); });
      uint64_t used = 0;
      for (size_t i : idx) {
        if (used + units[i].bytes <= room) {
          chosen[i] = 1;
          used += units[i].bytes;
        }
      }
      ok = true;
    }
  }
  require(ok, SS_E_CAPACITY,
          d.engine + ": nothing fits under the " + gib(p.cap) + " GiB cap: fixed " + gib(p.fixed) + " GiB + margin " +
              gib(p.margin) + " GiB + the stream rings exceed it");

  // Host tiers: pinned-only units, then spillable units best benefit per byte while the host budget lasts. Host
  // bytes are the 4 KiB-strided bytes the pinned tier allocates.
  uint64_t host_left = b.host_pinned_bytes;
  for (const auto& n : d.host_fixed) host_left = host_left > n.bytes ? host_left - n.bytes : 0;
  std::vector<U> spill;
  for (size_t i = 0; i < units.size(); ++i) {
    if (chosen[i]) continue;
    const U& u = units[i];
    uint8_t& t = p.tier[size_t(u.g)];
    switch (d.tiers) {
      case Tiers::Pinned: {
        t = SS_TIER_PINNED;
        uint64_t hb = host_stride(d.layers[size_t(u.layer)].unit_bytes);
        host_left = host_left > hb ? host_left - hb : 0;
        break;
      }
      case Tiers::Disk: t = SS_TIER_DISK; break;
      case Tiers::Pair: t = SS_TIER_PAIR; break;
      case Tiers::PinnedDisk: spill.push_back(u); break;
    }
  }
  std::stable_sort(spill.begin(), spill.end(), better);
  for (const U& u : spill) {
    const uint64_t hb = host_stride(d.layers[size_t(u.layer)].unit_bytes);
    if (hb <= host_left) {
      p.tier[size_t(u.g)] = SS_TIER_PINNED;
      host_left -= hb;
    } else {
      p.tier[size_t(u.g)] = SS_TIER_DISK;
    }
  }
  if (b.pinned_backing) plan_backing(d, b, benefit, p.tier);
  finalize(d, b, p);
  return p;
}

Plan plan_from_tiers(const Desc& d, const ss_budget& b, const std::vector<uint8_t>& tiers) {
  require(int64_t(tiers.size()) == d.total_units, SS_E_INVALID, "plan: one tier per unit (ss_desc_total_units)");
  Plan p = start(d, b);
  p.tier = tiers;
  finalize(d, b, p);
  return p;
}

Json plan_to_json(const Desc& d, const Plan& p) {
  Json j = Json::object();
  j["schema"] = Json::string("strix-stream/plan-v1");
  j["engine"] = Json::string(d.engine);
  j["cap_bytes"] = Json::number(p.cap);
  j["margin_bytes"] = Json::number(p.margin);
  j["fixed_bytes"] = Json::number(p.fixed);
  j["dynamic_bytes"] = Json::number(p.dynamic);
  j["dynamic_slots_per_layer"] = Json::number(uint64_t(p.dynamic_slots_per_layer));
  j["ring_bytes"] = Json::number(p.ring);
  Json rings = Json::array();
  for (const RingClass& c : p.rings) {
    Json r = Json::object();
    r["slot_bytes"] = Json::number(c.stride);
    r["slots"] = Json::number(uint64_t(c.slots));
    rings.push(r);
  }
  j["rings"] = rings;
  j["resident_bytes"] = Json::number(p.resident);
  j["pinned_bytes"] = Json::number(p.pinned);
  j["disk_bytes"] = Json::number(p.disk);
  j["peak_bytes"] = Json::number(p.peak);
  j["n_resident"] = Json::number(p.n_resident);
  j["n_pinned"] = Json::number(p.n_pinned);
  j["n_disk"] = Json::number(p.n_disk);
  j["n_pair"] = Json::number(p.n_pair);
  j["n_backed"] = Json::number(p.n_backed);
  // Per layer, one letter per unit: R resident, P pinned, D disk, X pair, B resident with a pinned copy.
  Json layers = Json::array();
  for (size_t L = 0; L < d.layers.size(); ++L) {
    std::string s;
    for (int u = 0; u < d.layers[L].n_units; ++u) s += "RPDXB"[p.tier[size_t(d.unit_base[L] + u)]];
    layers.push(Json::string(s));
  }
  j["tiers"] = layers;
  return j;
}

std::vector<uint8_t> tiers_from_json(const Desc& d, const Json& j) {
  static const char kLetters[] = "RPDXB";
  const Json* t = j.find("tiers");
  require(t && t->is_array() && t->arr.size() == d.layers.size(), SS_E_DESC,
          "plan json: \"tiers\" needs one string per layer");
  std::vector<uint8_t> out;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const std::string s = json_str(t->arr[L], "plan.tiers");
    require(s.size() == size_t(d.layers[L].n_units), SS_E_DESC,
            "plan json: tiers[" + std::to_string(L) + "] needs one letter per unit");
    for (char c : s) {
      const char* k = c ? std::strchr(kLetters, c) : nullptr;
      require(k != nullptr, SS_E_DESC, "plan json: tier letters are R P D X B");
      out.push_back(uint8_t(k - kLetters));
    }
  }
  return out;
}

}  // namespace ss
