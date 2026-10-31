// Device-gated decode mode: layer leases (dg_core.hpp, ss.h "Layer leases"). A large-piece prefill reads every expert
// of one gated layer from the device-gated arena: the units already on the device in place, the others from a bounded
// prefill ring of borrowed persistent slots, without ever moving a decode unit (the ring is reserved once, between
// tokens, and returned afterwards).

#include <algorithm>
#include <chrono>
#include <deque>
#include <string>
#include <tuple>

#include "dg_core.hpp"
#include "errors.hpp"

namespace ts {

namespace {

const char* const kPoisoned = "device-gated: the cache failed (a reset or a make_resident); close the poisoned context";

uint64_t ns_since(std::chrono::steady_clock::time_point t0) {
  return uint64_t(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace

void DgTransport::record_on(void*, void*) {
  fail(SS_E_UNSUPPORTED, "device-gated: this transport cannot record on a foreign stream");
}
void DgTransport::upload_lease_map(uint32_t, const int32_t*, uint32_t) {
  fail(SS_E_UNSUPPORTED, "device-gated: this transport has no lease maps");
}

void DgCore::no_ring(const char* what) const {
  require(!ring_on_, SS_E_INVALID,
          std::string(what) + ": a prefill ring is reserved (release every lease and call ss_dg_return_ring first)");
}

uint32_t DgCore::ring_free() const {
  uint32_t n = 0;
  for (const RingSlot& r : ring_) n += r.lease == kNone ? 1u : 0u;
  return n;
}

uint32_t DgCore::leases_active() const {
  uint32_t n = 0;
  for (const LeaseRec& l : leases_) n += l.active ? 1u : 0u;
  return n;
}

// Release events whose consumers finished: forgotten by every holder, back to the pool.
void DgCore::prune_releases() {
  for (size_t i = 0; i < releases_.size();) {
    void* ev = releases_[i];
    if (!t_->done(ev)) {
      ++i;
      continue;
    }
    auto drop = [ev](std::vector<void*>& v) { v.erase(std::remove(v.begin(), v.end(), ev), v.end()); };
    for (RingSlot& r : ring_) drop(r.rel);
    for (LeaseRec& l : leases_) drop(l.rel);
    event_pool_.push_back(ev);
    releases_.erase(releases_.begin() + std::ptrdiff_t(i));
  }
}

DgCore::RingSizing DgCore::ring_sizing(uint32_t depth, const std::vector<uint32_t>& kept) const {
  std::vector<uint8_t> keep(keys_, 0);
  for (uint32_t k : kept)
    if (k < keys_) keep[k] = 1;
  RingSizing r;
  size_t evicting = 0, eligible = 0, eligible_clean = 0, discarded = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    if (x.home == Home::Evicting) ++evicting;
    else if (x.home == Home::Device && !keep[k]) {
      ++eligible;
      eligible_clean += x.backed ? 1u : 0u;
    }
  }
  for (const Writeback& w : wbs_) discarded += w.discard ? 1u : 0u;
  const size_t free_after = free_slots_.size() + evicting;
  size_t evictable = eligible;
  if (!c_.disk_tier)  // a dirty victim needs a free block for its write-back
    evictable = std::min(eligible, eligible_clean + free_blocks_.size() + discarded);
  r.reservable = uint32_t(std::min<size_t>(free_after + evictable, c_.persistent_slots));
  r.free_slots = uint32_t(free_slots_.size());
  std::vector<uint32_t> need(gated(), 0);
  for (uint32_t g = 0; g < gated(); ++g) {
    r.max_layer_units = std::max(r.max_layer_units, units(g));
    for (uint32_t u = 0; u < units(g); ++u) {
      const Home h = e_[key(g, u)].home;
      need[g] += h != Home::Device && h != Home::Coming ? 1u : 0u;
    }
    r.max_layer_need = std::max(r.max_layer_need, need[g]);
  }
  const uint32_t d = std::max<uint32_t>(1, std::min<uint32_t>(depth, gated()));
  for (uint32_t g0 = 0; g0 + d <= gated(); ++g0) {
    uint64_t s = 0;
    for (uint32_t g = g0; g < g0 + d; ++g) s += need[g];
    r.plan = uint32_t(std::max<uint64_t>(r.plan, s));
  }
  return r;
}

// ss_dg_reserve_ring (dg_core.hpp). Every refusal comes before any action; then a drain (every read, copy and write-back
// finished and collected), the victims in the policy's order with make_resident's lower-tier rules, the table uploaded.
DgCore::RingResult DgCore::reserve_ring(uint32_t n, const std::vector<uint32_t>& kept) {
  const auto t0 = std::chrono::steady_clock::now();
  require(!reset_failed_, SS_E_BACKEND, kPoisoned);
  require(!ring_on_, SS_E_INVALID, "ss_dg_reserve_ring: a ring is already reserved (ss_dg_return_ring first)");
  require(lm_g_ == kNone, SS_E_INVALID, "ss_dg_reserve_ring: a layer token is open");
  std::vector<uint8_t> keep(keys_, 0);
  for (uint32_t k : kept) {
    require(k < keys_, SS_E_INVALID, "ss_dg_reserve_ring: kept key out of range");
    keep[k] = 1;
  }
  size_t evicting = 0, eligible_n = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    require(x.home != Home::Bank, SS_E_BACKEND, "ss_dg_reserve_ring: a bank unit between tokens");
    if (x.home == Home::Evicting) ++evicting;
    else if (x.home == Home::Device && !keep[k]) ++eligible_n;
  }
  const size_t free_after = free_slots_.size() + evicting;
  require(n <= free_after + eligible_n, SS_E_CAPACITY,
          "ss_dg_reserve_ring: " + std::to_string(n) + " ring slots, but only " + std::to_string(free_after) +
              " free slots and " + std::to_string(eligible_n) +
              " evictable units (the rest hold kept units or prefetches in flight)");
  try {  // drain point: a device failure here poisons the core (nothing of this call changed yet)
    drain();
    for (const Writeback& w : wbs_) t_->event_sync(w.done);
    collect();
    require(wbs_.empty(), SS_E_BACKEND, "ss_dg_reserve_ring: write-backs not collected");
  } catch (...) {
    reset_failed_ = true;
    throw;
  }
  auto eligible = [&](uint32_t k) {
    const Entry& x = e_[k];
    return x.home == Home::Device && !x.wb_pending && !keep[k];
  };
  const size_t n_vict = n > free_slots_.size() ? n - free_slots_.size() : 0;
  std::vector<uint32_t> victims;
  if (n_vict && c_.scored) {
    select_scored(n_vict, eligible, false);
    for (const ScoreCand& cd : select_heap_) victims.push_back(cd.key);
  } else {
    for (uint32_t k = tail_; k != kNone && victims.size() < n_vict; k = e_[k].prev)
      if (eligible(k)) victims.push_back(k);
  }
  require(victims.size() == n_vict, SS_E_BACKEND, "ss_dg_reserve_ring: fewer victims than the capacity check counted");
  if (!c_.disk_tier) {  // two-tier: every dirty victim's write-back needs a free block (refused, nothing evicted)
    size_t dirty = 0;
    for (uint32_t v : victims) dirty += e_[v].backed ? 0u : 1u;
    require(dirty <= free_blocks_.size(), SS_E_CAPACITY,
            "ss_dg_reserve_ring: " + std::to_string(dirty) + " victims to write back but " +
                std::to_string(free_blocks_.size()) + " free pinned blocks");
  }
  RingResult r;
  std::vector<uint8_t> hold(keys_, 0);  // this call's written-back victims are never dropped to their file here
  std::vector<int32_t> wb;              // slots of committed write-backs on kD2h (free once kD2h is synced)
  try {
    for (uint32_t v : victims) {
      Entry& x = e_[v];
      // a clean copy (its block is its home again) or, three-tier with no block to free, its file: no copy
      const bool no_copy = x.backed || (free_blocks_.empty() && !drop_host_lru(&hold, false));
      int32_t b = -1;
      if (!no_copy) {
        b = free_blocks_.back();
        t_->d2h(x.slot, b);  // the dirty unit's only copy: written back before its slot is reused
        free_blocks_.pop_back();
      }
      ++r.evicted;
      unlink(v);
      --on_device_;
      table_[v] = -1;
      if (no_copy) {
        free_slots_.push_back(x.slot);
        to_lower(v);
        continue;
      }
      wb.push_back(x.slot);
      x.home = Home::Host;
      x.slot = -1;
      x.block = b;
      hpush(v);
      hold[v] = 1;
    }
    if (!wb.empty()) {
      t_->sync(DgTransport::kD2h);
      free_slots_.insert(free_slots_.end(), wb.begin(), wb.end());
      wb.clear();
    }
    if (r.evicted) t_->upload_table(table_.data(), keys_);  // (syncs kDem) the victims' -1
  } catch (...) {
    // a device failure: the host records stay one consistent ownership, what the device holds is unknown
    reset_failed_ = true;
    free_slots_.insert(free_slots_.end(), wb.begin(), wb.end());
    st_.lease_reserve_evicted += r.evicted;
    throw;
  }
  require(free_slots_.size() >= n, SS_E_BACKEND, "ss_dg_reserve_ring: the evictions freed too few slots");
  ring_.assign(n, RingSlot{});
  for (uint32_t i = 0; i < n; ++i) {
    ring_[i].slot = free_slots_.back();
    free_slots_.pop_back();
  }
  ring_of_.assign(keys_, -1);
  ring_age_ = 0;
  ring_on_ = true;
  r.slots = n;
  ++st_.lease_reserves;
  st_.lease_reserve_evicted += r.evicted;
  st_.lease_ns += ns_since(t0);
  return r;
}

void DgCore::return_ring() {
  const auto t0 = std::chrono::steady_clock::now();
  require(!reset_failed_, SS_E_BACKEND, kPoisoned);
  require(ring_on_, SS_E_INVALID, "ss_dg_return_ring: no ring is reserved");
  require(leases_active() == 0, SS_E_INVALID, "ss_dg_return_ring: release every lease first (ss_dg_release_layer)");
  try {  // every consumer of a lease finished (its residents may move again), every lease copy landed
    for (void* ev : releases_) t_->event_sync(ev);
    t_->sync(DgTransport::kDem);
  } catch (...) {
    reset_failed_ = true;
    throw;
  }
  st_.lease_return_wait_ns += ns_since(t0);
  event_pool_.insert(event_pool_.end(), releases_.begin(), releases_.end());
  releases_.clear();
  for (LeaseRec& l : leases_) l.rel.clear();
  for (const RingSlot& r : ring_) free_slots_.push_back(r.slot);
  ring_.clear();
  ring_of_.clear();
  ring_on_ = false;
  ++st_.lease_returns;
}

// ss_dg_lease_layer (dg_core.hpp). Refusals first (nothing changed); then hits and copies in unit order, the map, ready.
DgCore::LeaseResult DgCore::lease_layer(uint32_t g, const std::vector<uint32_t>& ids, bool all, uint32_t cap) {
  const auto t0 = std::chrono::steady_clock::now();
  require(!reset_failed_, SS_E_BACKEND, kPoisoned);
  require(ring_on_, SS_E_INVALID, "ss_dg_lease_layer: no ring is reserved (ss_dg_reserve_ring first)");
  require(g < gated(), SS_E_INVALID, "ss_dg_lease_layer: gated layer out of range");
  uint32_t index = kNone;
  for (uint32_t i = 0; i < kMaxLeases; ++i) {
    require(!(leases_[i].active && leases_[i].g == g), SS_E_INVALID,
            "ss_dg_lease_layer: gated layer " + std::to_string(g) + " is already leased (release it first)");
    if (!leases_[i].active && index == kNone) index = i;
  }
  require(index != kNone, SS_E_INVALID,
          "ss_dg_lease_layer: " + std::to_string(kMaxLeases) + " leases are active (release one first)");
  std::vector<uint32_t> want;
  if (all) {
    for (uint32_t u = 0; u < units(g); ++u) want.push_back(u);
  } else {
    std::vector<uint8_t> seen(units(g), 0);
    for (uint32_t u : ids) {
      require(u < units(g), SS_E_INVALID, "ss_dg_lease_layer: id out of range");
      require(!seen[u], SS_E_INVALID, "ss_dg_lease_layer: duplicate id " + std::to_string(u));
      seen[u] = 1;
      want.push_back(u);
    }
  }
  prune_releases();
  // classify: in place (on the device: nothing moves a device unit while the ring is reserved), a ring hit, a miss
  std::vector<uint32_t> hits, misses;  // units
  for (uint32_t u : want) {
    const uint32_t k = key(g, u);
    const Entry& x = e_[k];
    require(x.home != Home::Bank && x.home != Home::Evicting, SS_E_BACKEND,
            "ss_dg_lease_layer: a unit in flight while a ring is reserved");
    if (x.home == Home::Device || x.home == Home::Coming) continue;
    const int32_t ri = ring_of_[k];
    if (ri >= 0 && ring_[size_t(ri)].lease == kNone) hits.push_back(u);
    else misses.push_back(u);
  }
  // the slots misses may take: not held, not hit now; untagged first, then without a release in flight, then the
  // least recently released (a slot whose readers may still run is the last choice: its copy waits for them)
  std::vector<uint32_t> cand;
  std::vector<uint8_t> claimed(ring_.size(), 0);
  for (uint32_t u : hits) claimed[size_t(ring_of_[key(g, u)])] = 1;
  for (uint32_t i = 0; i < ring_.size(); ++i)
    if (ring_[i].lease == kNone && !claimed[i]) cand.push_back(i);
  auto rank = [&](uint32_t i) {
    const RingSlot& r = ring_[i];
    return std::make_tuple(r.rel.empty() ? 0 : 1, r.key == kNone ? 0 : 1, r.age, i);
  };
  std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return rank(a) < rank(b); });
  require(misses.size() <= cand.size(), SS_E_CAPACITY,
          "ss_dg_lease_layer: " + std::to_string(misses.size()) + " units to copy into the ring but " +
              std::to_string(cand.size()) + " ring slots are not held by active leases (ring " +
              std::to_string(ring_.size()) + ")");
  require(cap == 0 || hits.size() + misses.size() <= cap, SS_E_CAPACITY,
          "ss_dg_lease_layer: the lease needs " + std::to_string(hits.size() + misses.size()) +
              " ring slots, above its ring_slots " + std::to_string(cap));

  LeaseRec& L = leases_[index];
  LeaseResult res;
  res.index = index;
  res.g = g;
  res.leased = uint32_t(want.size());
  L.map.assign(units(g), -1);
  L.ring.clear();
  std::vector<uint32_t> taken;  // ring indices this call holds (given back on a failure): hits first, then misses
  for (uint32_t u : want) {
    const Entry& x = e_[key(g, u)];
    if (x.home == Home::Device || x.home == Home::Coming) {
      L.map[u] = x.slot;
      ++res.resident;
    }
  }
  for (uint32_t u : hits) {
    const uint32_t ri = uint32_t(ring_of_[key(g, u)]);
    ring_[ri].lease = index;
    taken.push_back(ri);
    L.map[u] = ring_[ri].slot;
    ++res.hits;
  }
  struct Read {
    uint32_t key, ri;
    int32_t stage;
    void* handle;
  };
  std::deque<Read> reads;  // file reads in flight, oldest first, each in its own staging buffer
  const size_t depth =
      c_.disk_tier ? std::max<size_t>(1, std::min<size_t>(std::max<uint32_t>(c_.disk_queue, 1), c_.disk_staging)) : 1;
  std::vector<void*> waited;  // release events kDem already waits for in this call
  uint64_t disk_reads = 0;
  bool io_error = false;
  auto wait_rel = [&](std::vector<void*>& rel) {
    bool any = false;
    for (void* ev : rel)
      if (!t_->done(ev)) {
        any = true;
        if (std::find(waited.begin(), waited.end(), ev) == waited.end()) {
          t_->wait(DgTransport::kDem, ev);  // device-side: the copies after it wait for the old consumer
          waited.push_back(ev);
        }
      }
    rel.clear();
    return any;
  };
  auto end_read = [&] {
    const Read rd = reads.front();
    reads.pop_front();
    try {
      t_->read_unit_end(rd.handle);
    } catch (...) {
      io_error = true;
      throw;
    }
    t_->h2d(DgTransport::kDem, ring_[rd.ri].slot, rd.stage);
    void*& ev = stage_ev_[uint32_t(rd.stage) - c_.blocks];  // the staging buffer is reusable once this copy landed
    if (!ev) ev = event();
    t_->record(ev, DgTransport::kDem);
    ring_[rd.ri].key = rd.key;  // tagged once its copy is enqueued
    ring_of_[rd.key] = int32_t(rd.ri);
  };
  try {
    size_t ci = 0;
    for (uint32_t u : misses) {
      const uint32_t k = key(g, u);
      const uint32_t ri = cand[ci++];
      RingSlot& r = ring_[ri];
      if (r.key != kNone) {  // its old unit is gone (untagged before anything overwrites it)
        ring_of_[r.key] = -1;
        r.key = kNone;
      }
      if (wait_rel(r.rel)) ++res.waits;
      r.lease = index;
      taken.push_back(ri);
      L.map[u] = r.slot;
      ++res.misses;
      const Entry& x = e_[k];
      if (x.home == Home::Host) {  // its pinned block (quiescent: nothing moves while the ring is reserved)
        t_->h2d(DgTransport::kDem, r.slot, x.block);
        r.key = k;
        ring_of_[k] = int32_t(ri);
        continue;
      }
      require(x.home == Home::File, SS_E_BACKEND, "ss_dg_lease_layer: a unit without a pinned block or a file");
      // its file, through the next staging buffer (its last copy landed first), as a demand miss
      if (reads.size() >= depth) end_read();
      const uint32_t si = stage_next_;
      stage_next_ = (si + 1) % c_.disk_staging;
      if (stage_ev_[si] && !t_->done(stage_ev_[si])) t_->event_sync(stage_ev_[si]);
      const int32_t stage = int32_t(c_.blocks + si);
      void* h = nullptr;
      try {
        h = t_->read_unit_begin(k, stage);
      } catch (...) {
        io_error = true;
        throw;
      }
      reads.push_back({k, ri, stage, h});
      ++disk_reads;
    }
    while (!reads.empty()) end_read();
    wait_rel(L.rel);  // the map buffer's last readers
    t_->upload_lease_map(index, L.map.data(), units(g));
    if (!L.ready) L.ready = event();
    t_->record(L.ready, DgTransport::kDem);
  } catch (...) {
    for (const Read& rd : reads) {  // reads not ended: their slots stay untagged
      try {
        t_->read_unit_end(rd.handle);
      } catch (...) {
      }
    }
    // the slots taken go back to the ring: a hit's keeps its unit, a miss's is untagged (its copy may not have run)
    for (size_t i = 0; i < taken.size(); ++i) {
      RingSlot& r = ring_[taken[i]];
      r.lease = kNone;
      if (i >= hits.size() && r.key != kNone) {
        ring_of_[r.key] = -1;
        r.key = kNone;
      }
    }
    L.map.clear();
    L.ring.clear();
    if (!io_error) reset_failed_ = true;  // a device failure: what the device holds is unknown
    st_.lease_ns += ns_since(t0);
    throw;
  }
  L.ring = taken;
  L.g = g;
  L.active = true;
  res.ready = L.ready;
  res.map = L.map.data();
  ++st_.leases;
  st_.lease_units += res.leased;
  st_.lease_resident += res.resident;
  st_.lease_ring_hits += res.hits;
  st_.lease_ring_misses += res.misses;
  st_.lease_bytes += uint64_t(res.misses) * unit_bytes_;
  st_.lease_disk_reads += disk_reads;
  st_.lease_disk_bytes += disk_reads * unit_bytes_;
  st_.lease_ring_waits += res.waits;
  st_.lease_ns += ns_since(t0);
  return res;
}

void DgCore::release_layer(uint32_t index, void* stream) {
  require(index < kMaxLeases && leases_[index].active, SS_E_INVALID, "ss_dg_release_layer: not an active lease");
  LeaseRec& L = leases_[index];
  void* ev = nullptr;
  if (stream) {
    ev = event();
    try {
      t_->record_on(ev, stream);
    } catch (...) {
      event_pool_.push_back(ev);  // the lease stays active (a retry may succeed)
      throw;
    }
    releases_.push_back(ev);
    L.rel.push_back(ev);
  }
  for (uint32_t ri : L.ring) {
    RingSlot& r = ring_[ri];
    r.lease = kNone;
    r.age = ++ring_age_;
    if (ev) r.rel.push_back(ev);
  }
  L.ring.clear();
  L.map.clear();
  L.active = false;
}

}  // namespace ts
