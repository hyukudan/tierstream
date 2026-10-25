#include "resident_set.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "errors.hpp"

namespace ts {

ResidentSet::ResidentSet(int n_units, int capacity, ss_policy policy, uint32_t window)
    : n_units_(n_units), policy_(policy), window_(window ? window : 16) {
  require(capacity >= 0 && capacity <= n_units, SS_E_INVALID, "resident set: capacity must be 0..n_units");
  slot_of_.assign(size_t(n_units), -1);
  unit_in_.assign(size_t(capacity), -1);
  pinned_.assign(size_t(capacity), 0);
  last_use_.assign(size_t(n_units), 0);
  if (policy_ == SS_POLICY_SCORED) {
    trans_.assign(size_t(n_units) * size_t(n_units), 0);
    trans_n_.assign(size_t(n_units), 0);
    freq_.assign(size_t(n_units), 0);
  }
}

int ResidentSet::place(int unit, bool pin) {
  require(unit >= 0 && unit < n_units_, SS_E_INVALID, "resident set: unit out of range");
  if (slot_of_[size_t(unit)] >= 0) {
    if (pin) pinned_[size_t(slot_of_[size_t(unit)])] = 1;
    return slot_of_[size_t(unit)];
  }
  for (size_t s = 0; s < unit_in_.size(); ++s) {
    if (unit_in_[s] < 0) {
      unit_in_[s] = unit;
      pinned_[s] = pin;
      slot_of_[size_t(unit)] = int32_t(s);
      ++n_resident_;
      return int(s);
    }
  }
  return -1;
}

void ResidentSet::drop(int unit) {
  int s = slot_of_[size_t(unit)];
  if (s < 0) return;
  unit_in_[size_t(s)] = -1;
  pinned_[size_t(s)] = 0;
  slot_of_[size_t(unit)] = -1;
  --n_resident_;
}

double ResidentSet::score(int unit) const {
  if (policy_ != SS_POLICY_SCORED) return 0.0;
  // 0.7 x one-step transition prediction from the previous route + 0.2 x window frequency + 0.1 x 1/(1+age).
  double trans = 0.0;
  if (!prev_.empty()) {
    for (int32_t p : prev_) {
      uint32_t n = trans_n_[size_t(p)];
      if (n) trans += double(trans_[size_t(p) * size_t(n_units_) + size_t(unit)]) / double(n);
    }
    trans /= double(prev_.size());
  }
  double freq = recent_.empty() ? 0.0 : double(freq_[size_t(unit)]) / double(recent_.size());
  uint64_t lu = last_use_[size_t(unit)];
  double age = lu ? double(tick_ - lu) : std::numeric_limits<double>::infinity();
  double rec = std::isinf(age) ? 0.0 : 1.0 / (1.0 + age);
  return 0.7 * trans + 0.2 * freq + 0.1 * rec;
}

int ResidentSet::victim_lru(const std::vector<char>& keep) const {
  int best = -1;
  for (size_t s = 0; s < unit_in_.size(); ++s) {
    int32_t u = unit_in_[s];
    if (u < 0 || pinned_[s] || keep[size_t(u)]) continue;
    if (best < 0) {
      best = int(s);
      continue;
    }
    int32_t b = unit_in_[size_t(best)];
    if (last_use_[size_t(u)] < last_use_[size_t(b)] || (last_use_[size_t(u)] == last_use_[size_t(b)] && u < b))
      best = int(s);
  }
  return best;
}

Admission ResidentSet::admit(const std::vector<int32_t>& routed_in, const std::vector<char>& busy,
                             bool admit_misses, int max_evictions) {
  Admission a;
  std::vector<int32_t> routed = routed_in;
  std::sort(routed.begin(), routed.end());
  routed.erase(std::unique(routed.begin(), routed.end()), routed.end());
  std::vector<char> keep(size_t(n_units_), 0);
  std::vector<int32_t> misses;
  for (int32_t u : routed) {
    require(u >= 0 && u < n_units_, SS_E_INVALID, "routes: unit id out of range");
    keep[size_t(u)] = 1;
    if (slot_of_[size_t(u)] >= 0) a.hits.push_back(u);
    else misses.push_back(u);
  }
  for (size_t u = 0; u < busy.size() && u < keep.size(); ++u)
    if (busy[u]) keep[u] = 1;
  if (!admit_misses || unit_in_.empty()) {
    a.transient = misses;
    return a;
  }

  std::vector<int32_t> take;  // misses that get a slot, in admission order
  if (policy_ == SS_POLICY_LRU) {
    take = misses;
  } else {
    // Scored cut: the unpinned residents outside this route and the misses compete for the rows they can hold.
    struct C {
      int32_t unit;
      double score;
      uint64_t last;
      bool resident;
    };
    std::vector<C> comp;
    int free_slots = 0;
    for (size_t s = 0; s < unit_in_.size(); ++s) {
      int32_t u = unit_in_[s];
      if (u < 0) ++free_slots;
      else if (!pinned_[s] && !keep[size_t(u)]) comp.push_back({u, score(u), last_use_[size_t(u)], true});
    }
    size_t rows = size_t(free_slots) + comp.size();
    for (int32_t u : misses) comp.push_back({u, score(u), last_use_[size_t(u)], false});
    std::sort(comp.begin(), comp.end(), [](const C& x, const C& y) {
      if (x.score != y.score) return x.score > y.score;
      if (x.last != y.last) return x.last > y.last;
      return x.unit < y.unit;
    });
    for (size_t i = 0; i < comp.size() && i < rows; ++i)
      if (!comp[i].resident) take.push_back(comp[i].unit);
    // Residents below the cut are the eviction candidates; only as many go as the admitted misses need.
    std::vector<char> below(size_t(n_units_), 0);
    for (size_t i = rows; i < comp.size(); ++i)
      if (comp[i].resident) below[size_t(comp[i].unit)] = 1;
    for (size_t u = 0; u < keep.size(); ++u)
      if (!below[u] && slot_of_[u] >= 0) keep[u] = 1;  // above the cut: not evictable now
  }

  std::vector<char> admitted(size_t(n_units_), 0);
  for (int32_t u : take) {
    int s = -1;
    for (size_t k = 0; k < unit_in_.size(); ++k)
      if (unit_in_[k] < 0) {
        s = int(k);
        break;
      }
    if (s < 0) {
      if (max_evictions >= 0 && int(a.evicted.size()) >= max_evictions) {
        ++a.deferred;  // no free slot and no eviction allowed: this miss (and every later one) stays transient
        continue;
      }
      s = victim_lru(keep);
      if (s < 0) break;  // everything resident is pinned or in use this forward
      int32_t old = unit_in_[size_t(s)];
      slot_of_[size_t(old)] = -1;
      a.evicted.push_back(old);
      a.evicted_slot.push_back(s);
      --n_resident_;
    }
    unit_in_[size_t(s)] = u;
    pinned_[size_t(s)] = 0;
    slot_of_[size_t(u)] = s;
    keep[size_t(u)] = 1;
    admitted[size_t(u)] = 1;
    ++n_resident_;
    a.admitted.push_back(u);
    a.slot_of.push_back(s);
  }
  for (int32_t u : misses)
    if (!admitted[size_t(u)]) a.transient.push_back(u);
  // Report admitted units ascending with their slots.
  std::vector<size_t> ord(a.admitted.size());
  for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
  std::sort(ord.begin(), ord.end(), [&](size_t x, size_t y) { return a.admitted[x] < a.admitted[y]; });
  Admission sorted = a;
  for (size_t i = 0; i < ord.size(); ++i) {
    sorted.admitted[i] = a.admitted[ord[i]];
    sorted.slot_of[i] = a.slot_of[ord[i]];
  }
  std::vector<size_t> ev(a.evicted.size());
  for (size_t i = 0; i < ev.size(); ++i) ev[i] = i;
  std::sort(ev.begin(), ev.end(), [&](size_t x, size_t y) { return a.evicted[x] < a.evicted[y]; });
  for (size_t i = 0; i < ev.size(); ++i) {
    sorted.evicted[i] = a.evicted[ev[i]];
    sorted.evicted_slot[i] = a.evicted_slot[ev[i]];
  }
  return sorted;
}

void ResidentSet::observe(const std::vector<int32_t>& routed_in, const std::vector<int32_t>& misses) {
  std::vector<int32_t> routed = routed_in;
  std::sort(routed.begin(), routed.end());
  routed.erase(std::unique(routed.begin(), routed.end()), routed.end());
  ++tick_;
  if (policy_ == SS_POLICY_LRU) {
    for (int32_t u : routed) last_use_[size_t(u)] = tick_;
    return;
  }
  // SCORED: hits do not refresh recency; a miss (admitted or transient) and a first use do.
  for (int32_t u : misses) last_use_[size_t(u)] = tick_;
  for (int32_t u : routed)
    if (last_use_[size_t(u)] == 0) last_use_[size_t(u)] = tick_;
  for (int32_t p : prev_) {
    ++trans_n_[size_t(p)];
    for (int32_t u : routed) ++trans_[size_t(p) * size_t(n_units_) + size_t(u)];
  }
  prev_ = routed;
  recent_.push_back(routed);
  for (int32_t u : routed) ++freq_[size_t(u)];
  if (recent_.size() > window_) {
    for (int32_t u : recent_.front()) --freq_[size_t(u)];
    recent_.pop_front();
  }
}

}  // namespace ts
