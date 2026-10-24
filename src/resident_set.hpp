// Per-layer resident set (docs/design.md §3): which units hold a persistent device slot, and the admission decision
// for a forward's routed units. Pure host logic, deterministic (every tie broken by ascending unit id).
#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "strix_stream/ss.h"

namespace ss {

struct Admission {
  std::vector<int32_t> hits;       // routed and resident (ascending)
  std::vector<int32_t> admitted;   // misses that take a persistent slot (ascending)
  std::vector<int32_t> slot_of;    // slot for each admitted unit (parallel to admitted)
  std::vector<int32_t> evicted;    // units that lost their slot to an admitted miss (ascending)
  std::vector<int32_t> evicted_slot;  // the slot each evicted unit held (parallel to evicted)
  std::vector<int32_t> transient;  // misses served from the stream ring this forward (ascending)
  int deferred = 0;                // misses that would have evicted, refused by max_evictions (in transient)
};

class ResidentSet {
 public:
  // capacity: persistent slots of this layer; n_units: units of the layer.
  ResidentSet(int n_units, int capacity, ss_policy policy, uint32_t window);

  int capacity() const { return int(unit_in_.size()); }
  int slot(int unit) const { return slot_of_[size_t(unit)]; }
  bool pinned(int unit) const { int s = slot(unit); return s >= 0 && pinned_[size_t(s)]; }
  int n_resident() const { return n_resident_; }

  // Place a unit in a free slot at load (planner residents, seed). Returns the slot or -1 when full.
  int place(int unit, bool pin);

  // Remove a unit from its slot (its read was cancelled or failed).
  void drop(int unit);

  // Admission for one forward. routed: distinct routed units (any order); busy: units that must not be evicted this
  // forward (already planned / in flight), may be empty. admit_misses = false serves every miss from the ring
  // (prefill with static residents). max_evictions >= 0 caps the evictions (exclusive tier mode: one free block per
  // victim); misses past the cap that need a victim are served from the ring and nothing is evicted for them.
  Admission admit(const std::vector<int32_t>& routed, const std::vector<char>& busy, bool admit_misses,
                  int max_evictions = -1);

  // Record the forward's routed set and which of them missed (recency, transition counts, frequency window).
  void observe(const std::vector<int32_t>& routed, const std::vector<int32_t>& misses);

  // Score of a unit for the next admission (SCORED policy; exposed for tests).
  double score(int unit) const;

 private:
  int victim_lru(const std::vector<char>& keep) const;

  int n_units_;
  ss_policy policy_;
  uint32_t window_;
  std::vector<int32_t> slot_of_;   // unit -> slot or -1
  std::vector<int32_t> unit_in_;   // slot -> unit or -1
  std::vector<char> pinned_;       // slot -> pinned
  std::vector<uint64_t> last_use_; // unit -> tick of last use (0 = never)
  uint64_t tick_ = 0;
  int n_resident_ = 0;
  // SCORED state.
  std::vector<uint32_t> trans_;      // n_units^2 counts: previous route -> current route
  std::vector<uint32_t> trans_n_;    // per unit: times it appeared in a route that had a successor
  std::vector<int32_t> prev_;        // previous route
  std::deque<std::vector<int32_t>> recent_;
  std::vector<uint32_t> freq_;       // occurrences in the window
};

}  // namespace ss
