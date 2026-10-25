// Device-gated decode mode: the cache state machine (dg_core.hpp).

#include "dg_core.hpp"

#include <algorithm>

#include "errors.hpp"

namespace ts {

namespace {

[[noreturn]] void broken(const std::string& what) { fail(SS_E_BACKEND, "device-gated cache invariant: " + what); }

}  // namespace

DgCore::DgCore(const DgConfig& cfg, DgTransport* t) : c_(cfg), t_(t) {
  base_.assign(1, 0);
  uint64_t n = 0;
  for (uint32_t u : c_.n_units) {
    n += u;
    require(n < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 gated units");
    base_.push_back(uint32_t(n));
  }
  keys_ = uint32_t(n);
  require(c_.persistent_slots >= 1 && c_.spare_target <= c_.persistent_slots, SS_E_INVALID,
          "device-gated: spare target above the persistent slots");
  const uint64_t slots = uint64_t(c_.persistent_slots) + 2ull * c_.bank_slots;
  require(slots < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 slots");
  slots_ = uint32_t(slots);
  bank_first_ = c_.persistent_slots;
  e_.assign(keys_, Entry{});
  table_.assign(keys_, -1);
  bank_key_.assign(2u * size_t(c_.bank_slots), kNone);
  pf_layer_.assign(gated(), 0);
  layer_missed_.assign(gated(), 0);
  lc_.assign(gated(), DgLayerCounts{});
  if (c_.prefetch_reserve_auto) c_.prefetch_reserve = std::min<uint32_t>(24, c_.spare_target / 4);
}

uint32_t DgCore::gated_of(uint32_t k) const {
  return uint32_t(std::upper_bound(base_.begin(), base_.end(), k) - base_.begin()) - 1;
}

DgCore::~DgCore() {
  for (void* ev : all_events_) t_->event_destroy(ev);
}

void DgCore::place(const std::vector<uint32_t>& residents) {
  // homes: the residents (by priority, then nothing else) on the device, every other key in a block
  const uint32_t dev_max = c_.persistent_slots - c_.spare_target;
  std::vector<uint8_t> placed(keys_, 0);
  std::vector<uint32_t> order;
  for (uint32_t k : residents) {
    require(k < keys_, SS_E_INVALID, "device-gated: resident key out of range");
    if (!placed[k] && order.size() < dev_max) {
      placed[k] = 1;
      order.push_back(k);
    }
  }
  const uint32_t dev0 = uint32_t(order.size());
  require(uint64_t(keys_) - dev0 <= c_.blocks, SS_E_CAPACITY, "device-gated: fewer host blocks than non-resident units");
  int32_t next_block = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    Entry& x = e_[k];
    x = Entry{};
    if (!placed[k]) {
      x.home = Home::Host;
      x.block = next_block++;
    }
  }
  for (uint32_t i = 0; i < dev0; ++i) {
    Entry& x = e_[order[i]];
    x.home = Home::Device;
    x.slot = int32_t(i);
    table_[order[i]] = x.slot;
  }
  head_ = tail_ = kNone;
  for (uint32_t i = dev0; i-- > 0;) push_front(order[i]);  // the first resident most recent
  on_device_ = dev0;
  free_slots_.clear();
  free_blocks_.clear();
  for (uint32_t s = c_.persistent_slots; s-- > dev0;) free_slots_.push_back(int32_t(s));
  for (int32_t b = int32_t(c_.blocks); b-- > next_block;) free_blocks_.push_back(b);
}

// ---- recency list ----
void DgCore::unlink(uint32_t k) {
  Entry& x = e_[k];
  if (!x.in_list) return;
  if (x.prev != kNone) e_[x.prev].next = x.next; else head_ = x.next;
  if (x.next != kNone) e_[x.next].prev = x.prev; else tail_ = x.prev;
  x.prev = x.next = kNone;
  x.in_list = false;
}

void DgCore::push_front(uint32_t k) {
  Entry& x = e_[k];
  x.prev = kNone;
  x.next = head_;
  if (head_ != kNone) e_[head_].prev = k; else tail_ = k;
  head_ = k;
  x.in_list = true;
}

void* DgCore::event() {
  if (!event_pool_.empty()) {
    void* ev = event_pool_.back();
    event_pool_.pop_back();
    return ev;
  }
  void* ev = t_->event_create();
  all_events_.push_back(ev);
  return ev;
}

void DgCore::write_entry(DgTransport::Stream s, uint32_t k, int32_t v) {
  table_[k] = v;
  t_->write_entry(s, k, v);
}

// Finished write-backs, in order: the key's home becomes its block and the slot is free; a discarded one (the key
// was resurrected) only returns its block. Callable inside a token: the entries were reset at the boundary that
// started the write-back.
uint32_t DgCore::collect() {
  uint32_t n = 0;
  while (!wbs_.empty()) {
    Writeback& w = wbs_.front();
    if (!t_->done(w.done)) break;
    Entry& x = e_[w.key];
    x.wb_pending = false;
    if (w.discard) {
      free_blocks_.push_back(w.block);
      if (mut_ == kFreeResurrectedSlot) free_slots_.push_back(w.slot);
    } else {
      x.home = Home::Host;
      x.block = w.block;
      x.slot = -1;
      free_slots_.push_back(w.slot);
    }
    ++n;
    event_pool_.push_back(w.done);
    wbs_.pop_front();
  }
  st_.writebacks_done += n;
  return n;
}

// A demand bank slot of gated layer g: bank g & 1, reused two gated layers later (the device finished layer g - 2
// before layer g's export, which triggered this fetch). The previous alias of the slot is unpublished first.
int32_t DgCore::take_bank_slot(uint32_t g) {
  const uint32_t b = g & 1u;
  if (bank_seq_[b] != seq_ || bank_layer_[b] != g) {
    bank_seq_[b] = seq_;
    bank_layer_[b] = g;
    bank_next_[b] = 0;
  }
  if (bank_next_[b] >= c_.bank_slots) return -1;
  const uint32_t idx = b * c_.bank_slots + bank_next_[b]++;
  const int32_t slot = int32_t(bank_first_ + idx);
  const uint32_t old = bank_key_[idx];
  if (old != kNone && e_[old].home == Home::Bank && e_[old].slot == slot) {
    e_[old].home = Home::Host;
    e_[old].slot = -1;
    if (mut_ != kNoBankUnpublish) write_entry(DgTransport::kDem, old, -1);
  }
  bank_key_[idx] = kNone;
  return slot;
}

void DgCore::begin_token(uint32_t seq) {
  seq_ = seq;
  pf_token_ = 0;
  demand_free_used_ = 0;
  std::fill(pf_layer_.begin(), pf_layer_.end(), 0);
  std::fill(layer_missed_.begin(), layer_missed_.end(), uint8_t(0));
  pred_.clear();
  pred_layer_ = kNone;
}

bool DgCore::on_device_or_coming(uint32_t g, uint32_t unit) const {
  const Home h = e_[key(g, unit)].home;
  return h == Home::Device || h == Home::Coming || h == Home::Bank;
}

bool DgCore::fetch(uint32_t g, uint32_t unit, bool prefetch) {
  const uint32_t k = key(g, unit);
  Entry& x = e_[k];
  // a prefetch over budget does nothing at all (no copy and no resurrection)
  if (prefetch && (pf_layer_[g] >= c_.prefetch_per_layer || pf_token_ >= c_.prefetch_per_token)) return false;
  switch (x.home) {
    case Home::Device:
    case Home::Coming:
    case Home::Bank:
      return false;
    case Home::Evicting: {  // resurrect: the slot still holds the bytes (the write-back only reads them)
      for (Writeback& w : wbs_)
        if (w.key == k && !w.discard) {
          w.discard = true;
          break;
        }
      x.home = Home::Device;
      x.block = -1;  // the destination block now belongs to the discarded write-back
      write_entry(DgTransport::kDem, k, x.slot);
      push_front(k);
      ++on_device_;
      ++st_.resurrected;
      return true;
    }
    case Home::Host:
      break;
  }
  int32_t slot = -1;
  bool bank = false;
  const size_t reserve = prefetch ? c_.prefetch_reserve : 0u;
  if (free_slots_.size() <= reserve || (pressure_ && !prefetch)) st_.collected_mid += collect();
  const bool capped = !prefetch && pressure_ && demand_free_used_ >= pressure_free_;
  if (!capped && free_slots_.size() > reserve) {
    if (!prefetch) ++demand_free_used_;
    slot = free_slots_.back();
    free_slots_.pop_back();
  } else if (prefetch) {
    return false;
  } else {
    slot = take_bank_slot(g);
    if (slot >= 0) {
      bank = true;
    } else {  // a layer has at most bank_slots distinct misses: only reachable through an internal error
      while (free_slots_.empty() && !wbs_.empty()) {
        t_->event_sync(wbs_.front().done);
        collect();
        ++st_.stalled;
      }
      if (free_slots_.empty()) fail(SS_E_BACKEND, "device-gated: no free slot, no bank slot, no write-back to wait for");
      slot = free_slots_.back();
      free_slots_.pop_back();
    }
  }
  const DgTransport::Stream s = prefetch ? DgTransport::kPf : DgTransport::kDem;
  if (mut_ == kEntryBeforeCopy) {
    write_entry(s, k, slot);
    t_->h2d(s, slot, x.block);
  } else {
    t_->h2d(s, slot, x.block);
    write_entry(s, k, slot);
  }
  x.slot = slot;
  x.via_prefetch = prefetch;
  x.used = false;
  if (prefetch) {
    x.pf_done = event();
    t_->record(x.pf_done, s);
    if (mut_ == kEarlyEventRecycle) event_pool_.push_back(x.pf_done);
  }
  if (bank) {
    x.home = Home::Bank;
    bank_key_[size_t(slot) - bank_first_] = k;
    banked_.push_back(k);
    ++st_.bank_serves;
    ++lc_[g].bank_serves;
  } else {
    x.home = Home::Coming;
    if (mut_ == kEarlySourceFree) free_blocks_.push_back(x.block);
    coming_.push_back(k);
    push_front(k);
    ++on_device_;
  }
  if (prefetch) {
    ++st_.prefetched;
    ++lc_[g].prefetched;
    ++pf_layer_[g];
    ++pf_token_;
  } else {
    ++st_.fetched;
    ++lc_[g].fetched;
  }
  return true;
}

void DgCore::cover(uint32_t g, uint32_t unit) {
  const Entry& x = e_[key(g, unit)];
  // only this key's copies land before ready
  if (x.home == Home::Coming && x.via_prefetch && x.pf_done && mut_ != kNoCover) {
    t_->wait(DgTransport::kDem, x.pf_done);
    ++st_.covered_waits;
  }
}

void DgCore::use(uint32_t g, uint32_t unit, bool was_miss) {
  const uint32_t k = key(g, unit);
  Entry& x = e_[k];
  ++st_.lookups;
  ++lc_[g].lookups;
  if (was_miss) {
    ++st_.misses;
    ++lc_[g].misses;
    if (!layer_missed_[g]) {
      layer_missed_[g] = 1;
      ++st_.miss_layers;
    }
  }
  if (x.via_prefetch && !x.used) {
    x.used = true;
    ++st_.prefetch_used;
    ++lc_[g].prefetch_used;
  }
  if (x.in_list) {
    unlink(k);
    push_front(k);
  }
}

void DgCore::serve_demand(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask) {
  const uint32_t nu = units(g);
  for (uint32_t i = 0; i < n; ++i)  // before any key is computed (the export kernel already aborted the token)
    require(ids[i] < nu, SS_E_INVALID,
            "device-gated: layer " + std::to_string(g) + " exported id " + std::to_string(ids[i]) + " of " +
                std::to_string(nu));
  if (diag_ && mask) t_->diag_mark(g, 0);
  // true misses first (the device waits for them), per masked occurrence in export order: a repeated id is fetched at
  // its first occurrence and covered at the later ones (site.cu)
  for (uint32_t i = 0; i < n; ++i) {
    if (!(mask >> i & 1u)) continue;
    if (on_device_or_coming(g, ids[i])) {
      cover(g, ids[i]);
      ++st_.covered;
    } else {
      fetch(g, ids[i], false);
    }
  }
  if (mask) signal(g);
  if (diag_ && mask) t_->diag_mark(g, 1);
}

void DgCore::serve_use(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask) {
  const bool have_pred = pred_layer_ == g;
  uniq_.assign(ids, ids + n);
  std::sort(uniq_.begin(), uniq_.end());
  uniq_.erase(std::unique(uniq_.begin(), uniq_.end()), uniq_.end());
  for (uint32_t id : uniq_) {
    bool miss = false;
    for (uint32_t i = 0; i < n; ++i)
      if (ids[i] == id && (mask >> i & 1u)) miss = true;
    use(g, id, miss);
    if (have_pred && std::find(pred_.begin(), pred_.end(), id) != pred_.end()) ++st_.predicted_true;
  }
  pred_.clear();  // a prediction is for the next layer only (the replica clears it at every layer)
  pred_layer_ = kNone;
}

void DgCore::serve_prediction(uint32_t g_next, const uint32_t* pred, uint32_t n) {
  if (g_next >= gated()) return;
  const uint32_t nu = units(g_next);
  uniq_.clear();
  for (uint32_t i = 0; i < n; ++i) {  // keep the order: the budget keeps the head of the list
    if (pred[i] >= nu) {  // advice: dropped before any key is computed
      ++st_.pred_invalid;
      continue;
    }
    if (std::find(uniq_.begin(), uniq_.end(), pred[i]) == uniq_.end()) uniq_.push_back(pred[i]);
  }
  st_.predicted += uniq_.size();  // counted whatever the budget lets through (the replica's definition)
  pred_.assign(uniq_.begin(), uniq_.end());
  pred_layer_ = g_next;
  for (uint32_t id : uniq_) fetch(g_next, id, true);
}

void DgCore::end_token(const std::vector<uint32_t>& keep) {
  // every copy and table write of the token has landed before anything is invalidated
  t_->sync(DgTransport::kDem);
  t_->sync(DgTransport::kPf);
  for (uint32_t k : coming_) {  // admitted: device homes now, their source blocks are free
    Entry& x = e_[k];
    if (x.home != Home::Coming) continue;
    x.home = Home::Device;
    x.via_prefetch = false;
    if (x.pf_done) {
      event_pool_.push_back(x.pf_done);
      x.pf_done = nullptr;
    }
    free_blocks_.push_back(x.block);
    x.block = -1;
  }
  coming_.clear();
  for (uint32_t k : banked_) {  // bank units: back to their blocks (the upload below resets the entry)
    Entry& x = e_[k];
    if (x.home == Home::Bank) {
      x.home = Home::Host;
      x.slot = -1;
      table_[k] = -1;
    }
    if (x.pf_done) {  // never: a bank serve is a demand
      event_pool_.push_back(x.pf_done);
      x.pf_done = nullptr;
    }
    x.via_prefetch = false;
  }
  banked_.clear();
  std::fill(bank_key_.begin(), bank_key_.end(), kNone);
  bank_seq_[0] = bank_seq_[1] = 0;
  bank_layer_[0] = bank_layer_[1] = kNone;
  collect();
  // victims: least recently used first, until free + in-flight write-backs reach the target
  std::vector<uint8_t> keep_mask;
  if (!keep.empty()) {
    keep_mask.assign(keys_, 0);
    for (uint32_t k : keep)
      if (k < keys_) keep_mask[k] = 1;
  }
  size_t pending = 0;
  for (const Writeback& w : wbs_) pending += w.discard ? 0u : 1u;
  std::vector<uint32_t> victims;
  for (uint32_t k = tail_; k != kNone && free_slots_.size() + pending + victims.size() < c_.spare_target;
       k = e_[k].prev) {
    const Entry& x = e_[k];
    if (x.home != Home::Device || (x.wb_pending && mut_ != kReevictPending) || (!keep_mask.empty() && keep_mask[k]))
      continue;
    victims.push_back(k);
  }
  if (pressure_ && pressure_recent_) {  // the most recently used too (they come back soon)
    uint32_t n = 0;
    for (uint32_t k = head_; k != kNone && n < pressure_recent_; k = e_[k].next) {
      const Entry& x = e_[k];
      if (x.home != Home::Device || (x.wb_pending && mut_ != kReevictPending) ||
          (!keep_mask.empty() && keep_mask[k]))
        continue;
      if (std::find(victims.begin(), victims.end(), k) != victims.end()) continue;
      victims.push_back(k);
      ++n;
    }
  }
  size_t n_evict = 0;
  for (uint32_t k : victims) {
    if (free_blocks_.empty()) {  // stays on the device: coverage kept
      ++st_.evict_short;
      break;
    }
    Entry& x = e_[k];
    const int32_t b = free_blocks_.back();
    free_blocks_.pop_back();
    unlink(k);
    --on_device_;
    x.home = Home::Evicting;
    x.block = b;
    x.wb_pending = true;
    table_[k] = -1;
    ++n_evict;
  }
  // The table first, then the write-backs, so the upload never queues behind them on a shared copy engine (as the
  // reference, cache.cpp:552-566). The device is done with the token, and an evicting slot is reused only after its
  // write-back's event.
  t_->upload_table(table_.data(), keys_);  // the device table = the host mirror
  for (size_t i = 0; i < n_evict; ++i) {
    const uint32_t k = victims[i];
    Entry& x = e_[k];
    t_->d2h(x.slot, x.block);
    if (mut_ == kEarlyWritebackBlock) free_blocks_.push_back(x.block);
    void* ev = event();
    t_->record(ev, DgTransport::kD2h);
    wbs_.push_back({k, x.slot, x.block, ev, false});
    ++st_.evicted;
    ++lc_[gated_of(k)].evicted;
  }
  ++st_.tokens;
}

uint32_t DgCore::seed(const std::vector<uint32_t>& keys) {
  collect();
  uint32_t n = 0;
  for (uint32_t k : keys) {
    require(k < keys_, SS_E_INVALID, "ss_seed: unit out of range");
    Entry& x = e_[k];
    if (x.home != Home::Host || free_slots_.size() <= c_.spare_target) continue;
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    t_->h2d(DgTransport::kDem, slot, x.block);
    free_blocks_.push_back(x.block);
    x.block = -1;
    x.slot = slot;
    x.home = Home::Device;
    table_[k] = slot;
    push_front(k);
    ++on_device_;
    ++n;
  }
  if (n) t_->upload_table(table_.data(), keys_);  // syncs kDem: the copies landed before the blocks are reused
  return n;
}

void DgCore::drain() {
  t_->sync(DgTransport::kDem);
  t_->sync(DgTransport::kPf);
  t_->sync(DgTransport::kD2h);
}

void DgCore::set_pressure(uint32_t free_per_token, uint32_t evict_recent) {
  pressure_ = free_per_token != 0 || evict_recent != 0;
  pressure_free_ = free_per_token;
  pressure_recent_ = evict_recent;
}

void DgCore::set_spare_target(uint32_t n) {
  require(n <= c_.persistent_slots, SS_E_INVALID, "device-gated: spare target above the persistent slots");
  c_.spare_target = n;
  if (c_.prefetch_reserve_auto) c_.prefetch_reserve = std::min<uint32_t>(24, n / 4);  // cache.cpp:171
}

void DgCore::set_prefetch_budget(uint32_t per_layer, uint32_t per_token) {
  c_.prefetch_per_layer = per_layer;
  c_.prefetch_per_token = per_token;
}

void DgCore::check() const {
  std::vector<uint8_t> slot_used(slots_, 0), block_used(c_.blocks, 0), ev_used(all_events_.size(), 0);
  auto take = [](std::vector<uint8_t>& v, int64_t i, const char* what) {
    if (i < 0 || size_t(i) >= v.size() || v[size_t(i)]) broken(what);
    v[size_t(i)] = 1;
  };
  auto ev_index = [&](void* ev) -> int64_t {
    for (size_t i = 0; i < all_events_.size(); ++i)
      if (all_events_[i] == ev) return int64_t(i);
    return -1;
  };
  uint32_t listed = 0, dev = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    switch (x.home) {
      case Home::Host:
        take(block_used, x.block, "host block");
        if (table_[k] != -1) broken("a host key is mapped");
        if (x.in_list) broken("a host key is in the recency list");
        break;
      case Home::Device:
        take(slot_used, x.slot, "device slot");
        ++dev;
        if (table_[k] != x.slot) broken("table != device slot");
        if (x.block != -1) broken("a device key holds a block");
        break;
      case Home::Coming:
        take(slot_used, x.slot, "coming slot");
        take(block_used, x.block, "coming source block");
        ++dev;
        if (table_[k] != x.slot) broken("table != coming slot");
        break;
      case Home::Bank:
        take(slot_used, x.slot, "bank slot");
        take(block_used, x.block, "bank home block");
        if (uint32_t(x.slot) < bank_first_) broken("a bank key outside the banks");
        if (table_[k] != x.slot) broken("table != bank slot");
        if (x.in_list) broken("a bank key is in the recency list");
        break;
      case Home::Evicting:
        take(slot_used, x.slot, "evicting slot");
        take(block_used, x.block, "write-back destination");
        if (table_[k] != -1) broken("an evicting key is mapped");
        if (!x.wb_pending) broken("an evicting key without a write-back");
        if (x.in_list) broken("an evicting key is in the recency list");
        break;
    }
    if (x.in_list) ++listed;
    if (x.pf_done) take(ev_used, ev_index(x.pf_done), "prefetch event");
  }
  for (const Writeback& w : wbs_) {
    const Entry& x = e_[w.key];
    if (!x.wb_pending) broken("a write-back of a key without wb_pending");
    if (w.discard) {
      if (x.home != Home::Device && x.home != Home::Coming) broken("a discarded write-back of a key not on the device");
      if (x.slot != w.slot) broken("a resurrected key left its slot");
      take(block_used, w.block, "discarded write-back block");
    } else if (x.home != Home::Evicting || x.slot != w.slot || x.block != w.block) {
      broken("a live write-back does not match its key");
    }
    take(ev_used, ev_index(w.done), "write-back event");
  }
  for (int32_t s : free_slots_) {
    if (uint32_t(s) >= bank_first_) broken("a bank slot in the free list");
    take(slot_used, s, "free slot");
  }
  for (int32_t b : free_blocks_) take(block_used, b, "free block");
  for (void* ev : event_pool_) take(ev_used, ev_index(ev), "pooled event");
  for (uint32_t b = 0; b < c_.blocks; ++b)
    if (!block_used[b]) broken("a host block is lost");
  for (uint32_t s = 0; s < bank_first_; ++s)
    if (!slot_used[s]) broken("a slot is lost");
  for (size_t i = 0; i < ev_used.size(); ++i)
    if (!ev_used[i]) broken("an event is lost");
  if (dev != on_device_ || listed != on_device_) broken("device count / recency list");
}

}  // namespace ts
