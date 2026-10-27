// Device-gated decode mode: the cache state machine (dg_core.hpp).

#include "dg_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <tuple>

#include "dg_proto.hpp"
#include "dg_score.hpp"
#include "errors.hpp"

namespace ts {

namespace {

[[noreturn]] void broken(const std::string& what) { fail(SS_E_BACKEND, "device-gated cache invariant: " + what); }

double us_since(std::chrono::steady_clock::time_point& t) {
  const auto now = std::chrono::steady_clock::now();
  const double us = std::chrono::duration<double, std::micro>(now - t).count();
  t = now;
  return us;
}

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
  require(t_->parts() >= 1, SS_E_INVALID, "device-gated: a unit without parts");
  for (uint32_t p = 0; p < t_->parts(); ++p) unit_bytes_ += t_->part_bytes(p);
  apply_wb_defer();
  if (c_.scored) {  // scored eviction: every table allocated now (the service never allocates)
    require(c_.w_trans <= 1000 && c_.w_freq <= 1000 && c_.w_rec <= 1000 &&
                c_.w_trans + c_.w_freq + c_.w_rec == 1000,
            SS_E_INVALID, "device-gated: score weights must each be 0..1000 and sum to 1000");
    require(c_.max_ids >= 1 && c_.max_ids <= 64, SS_E_INVALID, "device-gated: score unions need max ids 1..64");
    (void)dg_score_cells(c_.n_units);  // checked: units <= 65536 per layer, the cell count fits
    sc_.resize(gated());
    for (uint32_t g = 0; g < gated(); ++g) {
      LayerScore& l = sc_[g];
      const size_t nu = units(g);
      l.T.assign(nu * nu, 0);
      l.rowsum.assign(nu, 0);
      l.cnt.assign(nu, 0);
      l.ring.assign(size_t(kDgScoreWindow) * c_.max_ids, 0);
      l.ring_len.assign(kDgScoreWindow, 0);
      l.prev.reserve(c_.max_ids);
      l.cur.reserve(c_.max_ids);
      l.list.reserve(std::min<size_t>(nu, c_.persistent_slots));
    }
    heap_.reserve(gated());
    heap_pos_.assign(gated(), 0);
    select_heap_.reserve(c_.persistent_slots);
  }
  victims_.reserve(c_.persistent_slots);
}

// The derived write-back scheduling state of c_.wb_defer (at construction, and after set_wb_defer / reset_open).
void DgCore::apply_wb_defer() {
  if (c_.wb_defer) {
    wb_stream_ = c_.wb_on_d2h ? DgTransport::kD2h : DgTransport::kDem;
    layer_bytes_ = c_.wb_layer_bytes ? c_.wb_layer_bytes : t_->part_bytes(0);
    max_outstanding_ = c_.wb_max_outstanding ? c_.wb_max_outstanding : 2 * unit_bytes_;
    require(layer_bytes_ <= uint64_t(INT64_MAX) / 4, SS_E_INVALID, "device-gated: dg_wb_layer_bytes too large");
    if (c_.wb_on_d2h && !post_ready_ev_) post_ready_ev_ = event();
  } else {
    wb_stream_ = DgTransport::kDem;
    layer_bytes_ = 0;
    max_outstanding_ = 0;
  }
}

// Between tokens (persist: one load serves A/B arms with and without wbsched). Turning deferral off first submits
// every deferred write-back (they become ordinary recorded write-backs, collected as usual); either way the per-token
// credit / carry restart as after open. Placement only: the same bytes and arithmetic.
std::vector<uint32_t> DgCore::residency() const {
  std::vector<uint32_t> out;
  out.reserve(on_device_);
  for (uint32_t k = head_; k != kNone; k = e_[k].next)
    if (e_[k].home == Home::Device) out.push_back(k);
  return out;
}

void DgCore::set_wb_defer(bool on) {
  if (on == c_.wb_defer) return;
  if (!on)
    while (!deferred_.empty()) finish_deferred_front();
  c_.wb_defer = on;
  apply_wb_defer();
  credit_ = 0;
  prev_evict_ = 0;
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
  open_c_ = c_;
  open_e_ = e_;
  open_table_ = table_;
  open_free_slots_ = free_slots_;
  open_free_blocks_ = free_blocks_;
  open_head_ = head_;
  open_tail_ = tail_;
  open_on_device_ = on_device_;
  open_wp_ = wp_;
  open_rate_gbps_ = rate_gbps_;
  open_rate_set_ = rate_set_;
  open_stamp_ = stamp_;
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
  if (c_.scored) {  // unique stamps: stamp_room() keeps the counter below kStampLimit before every pushing phase
    if (stamp_ == UINT64_MAX) broken("the recency stamp counter wrapped");
    x.lru_stamp = ++stamp_;
  }
}

void DgCore::stamp_room(const char* what) const {
  if (c_.scored && stamp_ > kStampLimit)
    fail(SS_E_BACKEND, std::string("device-gated: ") + what +
                           ": the scored eviction's recency stamp counter is exhausted (reopen the context)");
}

size_t DgCore::pending_writebacks() const {
  size_t pending = deferred_.size();
  for (const Writeback& w : wbs_) pending += w.discard ? 0u : 1u;
  return pending;
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
// started the write-back. Every record here has its event recorded after its last submitted copy (a deferred one
// enters wbs_ only then), so done() never answers for a stale or never-recorded event.
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
    outstanding_ -= w.bytes;
    st_.wb_completed_bytes += w.bytes;
    event_pool_.push_back(w.done);
    wbs_.pop_front();
  }
  st_.writebacks_done += n;
  return n;
}

// ---- wb_defer: the deferred write-backs (Unsubmitted -> Partial -> recorded, in wbs_) ----
// One more part of a deferred record; the cursor moves only once the copy was accepted.
void DgCore::submit_part(Deferred& d) {
  const uint32_t p = d.parts_done;
  t_->d2h_part(wb_stream_, d.slot, d.block, p);
  ++d.parts_done;
  const uint64_t b = t_->part_bytes(p);
  d.bytes += b;
  outstanding_ += b;
  st_.wb_issued_bytes += b;
}

// The front's completion event, recorded after its submitted parts on the same stream; the record moves to wbs_.
// discard: the key was resurrected (it keeps its slot, the block returns at collection, unsent parts are never sent).
void* DgCore::record_deferred_front(bool discard) {
  Deferred& d = deferred_.front();
  void* ev = event();
  try {
    t_->record(ev, wb_stream_);
  } catch (...) {
    event_pool_.push_back(ev);  // the record stays deferred: its block is never recycled
    throw;
  }
  wbs_.push_back({d.key, d.slot, d.block, ev, discard, d.bytes});
  deferred_.pop_front();
  return ev;
}

void* DgCore::finish_deferred_front() {
  Deferred& d = deferred_.front();
  const uint32_t P = t_->parts();
  if (mut_ == kRecordBeforeLastPart && P > 1) {  // the event before the last copy: collect may promote a torn block
    while (d.parts_done + 1 < P) submit_part(d);
    void* ev = event();
    t_->record(ev, wb_stream_);
    submit_part(d);
    wbs_.push_back({d.key, d.slot, d.block, ev, false, d.bytes});
    deferred_.pop_front();
    return ev;
  }
  while (d.parts_done < P) submit_part(d);
  return record_deferred_front(false);
}

void DgCore::issue_writebacks(uint32_t g, const std::function<bool()>& ahead) {
  if (deferred_.empty()) {
    credit_ = 0;
    return;
  }
  if (g + 1 >= gated()) return;
  const int64_t grant = int64_t(layer_bytes_);
  credit_ = std::min(credit_ + grant, grant);  // a debt carries over, unused credit (also under the cap) does not
  st_.collected_mid += collect();
  bool after_ready = false;
  while (credit_ > 0 && !deferred_.empty() && outstanding_ < max_outstanding_ && !(ahead && ahead())) {
    if (wb_stream_ == DgTransport::kD2h && !after_ready) {  // kD2h: behind the ready (and copies) already on kDem
      t_->record(post_ready_ev_, DgTransport::kDem);
      t_->wait(DgTransport::kD2h, post_ready_ev_);
      after_ready = true;
    }
    Deferred& d = deferred_.front();
    credit_ -= int64_t(t_->part_bytes(d.parts_done));
    if (d.parts_done + 1 == t_->parts()) finish_deferred_front();
    else submit_part(d);
  }
}

// A demand without a free slot while victims are deferred: wait for the oldest live write-back already in flight, or,
// with none, finish the deferred front and wait for it; again until a persistent slot is free (a completed discarded
// write-back returns only its block), then the usual fallback. Bypasses the credit and the outstanding cap.
void DgCore::force_free_slot() {
  while (free_slots_.empty() && !deferred_.empty()) {
    void* ev = nullptr;
    for (const Writeback& w : wbs_)
      if (!w.discard) {
        ev = w.done;
        break;
      }
    if (!ev) ev = finish_deferred_front();
    ++st_.wb_forced;
    const auto t0 = std::chrono::steady_clock::now();
    t_->event_sync(ev);
    st_.wb_forced_wait_ns += uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    st_.collected_mid += collect();
  }
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
  require(!reset_failed_, SS_E_BACKEND, "device-gated: reset failed; close the poisoned context");
  if (c_.scored) {  // the token's epoch (the first is 1); one token cannot push more than 2^40 stamps
    stamp_room("begin_token");
    require(epoch_ < (uint64_t(1) << 63), SS_E_BACKEND,
            "device-gated: the scored eviction's epoch counter is exhausted");
    ++epoch_;
  }
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

bool DgCore::fetch(uint32_t g, uint32_t unit, bool prefetch) { return fetch_impl(g, unit, prefetch, true); }

// budgeted = false: a winpf prior prefetch between tokens (its own caps and byte budget; the per-token budgets are
// reset at begin_token).
bool DgCore::fetch_impl(uint32_t g, uint32_t unit, bool prefetch, bool budgeted) {
  const uint32_t k = key(g, unit);
  Entry& x = e_[k];
  // a prefetch over budget does nothing at all (no copy and no resurrection)
  if (prefetch && budgeted && (pf_layer_[g] >= c_.prefetch_per_layer || pf_token_ >= c_.prefetch_per_token))
    return false;
  switch (x.home) {
    case Home::Device:
    case Home::Coming:
    case Home::Bank:
      return false;
    case Home::Evicting: {  // resurrect: the slot still holds the bytes (the write-back only reads them)
      // wb_defer, a deferred record of k. Nothing submitted: cancelled (the record goes first, then its block returns
      // unread). Partly submitted (only the front): its event after the submitted parts, discarded; the unsent parts
      // are never needed (a discarded block never becomes a home). Either way published on kDem below, as any
      // resurrection, also for a prefetch: a later demand covers a Device key without a wait, kDem orders it.
      for (auto it = deferred_.begin(); it != deferred_.end(); ++it) {
        if (it->key != k) continue;
        if (it->parts_done == 0) {
          const int32_t b = it->block;
          deferred_.erase(it);
          free_blocks_.push_back(b);
          x.wb_pending = false;
          ++st_.wb_cancelled;
          st_.wb_cancelled_bytes += unit_bytes_;
        } else {
          require(it == deferred_.begin(), SS_E_BACKEND, "device-gated: parts submitted behind the deferred front");
          st_.wb_cancelled_bytes += unit_bytes_ - it->bytes;
          record_deferred_front(true);  // wb_pending stays until its collection
        }
        x.home = Home::Device;
        x.block = -1;
        write_entry(DgTransport::kDem, k, x.slot);
        push_front(k);
        ++on_device_;
        ++st_.resurrected;
        return true;
      }
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
  if (!prefetch && !capped && free_slots_.empty()) force_free_slot();  // wb_defer only (else nothing is deferred)
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
      force_free_slot();
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
  x.pf_kind = 0;
  x.pf_late = false;
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
  Entry& x = e_[key(g, unit)];
  // only this key's copies land before ready
  if (x.home == Home::Coming && x.via_prefetch && x.pf_done && mut_ != kNoCover) {
    t_->wait(DgTransport::kDem, x.pf_done);
    ++st_.covered_waits;
    if (x.pf_kind && !x.pf_late) {  // winpf: the demand met the copy in flight (it may have landed meanwhile)
      x.pf_late = true;
      ++(x.pf_kind == 1 ? st_.prior_late : st_.l1_late);
    }
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
  if (c_.scored) x.demand_epoch = epoch_;  // a demand use (never a prefetch, a resurrection, a seed)
  if (x.via_prefetch && !x.used) {
    x.used = true;
    ++st_.prefetch_used;
    ++lc_[g].prefetch_used;
    if (x.pf_kind) ++(x.pf_kind == 1 ? st_.prior_used : st_.l1_used);
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
  if (wp_.enable && !win_w_.empty() && (!wp_.top_k || n % wp_.top_k != 0))  // winpf: a hinted window's export
    fail(SS_E_BACKEND, "winpf: gated layer " + std::to_string(g) + " exported " + std::to_string(n) +
                           " ids: not rows x top_k (" + std::to_string(wp_.top_k) + ")");
  if (diag_ && mask) t_->diag_mark(g, 0);
  bool first_marked = false;
  // true misses first (the device waits for them), per masked occurrence in export order: a repeated id is fetched at
  // its first occurrence and covered at the later ones (site.cu)
  for (uint32_t i = 0; i < n; ++i) {
    if (!(mask >> i & 1u)) continue;
    if (on_device_or_coming(g, ids[i])) {
      cover(g, ids[i]);
      ++st_.covered;
    } else {
      fetch(g, ids[i], false);
      if (diag_ && !first_marked) {
        t_->diag_mark(g, 2);
        first_marked = true;
      }
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
  if (c_.scored) score_history(g);  // the union saved before serve_prediction reuses uniq_
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

void DgCore::serve_prediction(uint32_t g_next, const uint32_t* pred, uint32_t n, uint32_t n_export, bool ahead) {
  if (g_next >= gated()) return;
  const uint32_t nu = units(g_next);
  if (wp_.enable && !win_w_.empty()) {
    // winpf: the window's row predictions ranked by weighted row votes, then by the best rank inside a row (the
    // predictor's order), then by unit (deterministic); the head is admitted under the l+1 cap, the vote floor, the
    // bucket precision floor, the generic budgets and the reserve.
    const uint32_t rows = wp_.top_k ? n_export / wp_.top_k : 0;
    if (!wp_.top_k || n_export % wp_.top_k != 0 || rows == 0 || n % rows != 0)
      fail(SS_E_BACKEND, "winpf: gated layer " + std::to_string(g_next - 1) + " exported " + std::to_string(n_export) +
                             " ids and " + std::to_string(n) + " predicted ids: not rows x top_k (" +
                             std::to_string(wp_.top_k) + ") and rows x K");
    const uint32_t per = n / rows;
    cand_.clear();
    for (uint32_t i = 0; i < n; ++i) {
      if (pred[i] >= nu) {  // advice: dropped before any key is computed
        ++st_.pred_invalid;
        continue;
      }
      const uint32_t r = i / per, q = i % per;
      bool dup = false;  // an id twice in one row votes once
      for (uint32_t j = i - q; j < i && !dup; ++j) dup = pred[j] == pred[i];
      if (dup) continue;
      const float w = r < win_w_.size() ? win_w_[r] : 1.0f;
      bool found = false;
      for (Cand& c : cand_)
        if (c.unit == pred[i]) {
          c.votes += w;
          c.rank = std::min(c.rank, q);
          found = true;
          break;
        }
      if (!found) cand_.push_back({w, q, pred[i]});
    }
    st_.predicted += cand_.size();
    std::sort(cand_.begin(), cand_.end(), [](const Cand& a, const Cand& b) {
      if (a.votes != b.votes) return a.votes > b.votes;
      if (a.rank != b.rank) return a.rank < b.rank;
      return a.unit < b.unit;
    });
    pred_.clear();
    for (const Cand& c : cand_) pred_.push_back(c.unit);
    pred_layer_ = g_next;
    st_.l1_candidates += cand_.size();
    if (ahead) {  // the device ran ahead: layer g_next's demand is (being) served; a copy now only adds FIFO bytes
      ++st_.l1_skipped_ahead;
      return;
    }
    uint32_t issued = 0;
    for (const Cand& c : cand_) {
      if (issued >= wp_.l1_cap || c.votes < wp_.l1_min_votes - 1e-6f) break;
      if (on_device_or_coming(g_next, c.unit)) continue;
      const uint32_t b = uint32_t(std::min(4L, std::lround(c.votes))) * 3u + (c.rank <= 1 ? 0u : c.rank <= 3 ? 1u : 2u);
      if (wp_.l1_pmin > 0 && (bkt_hit_[b] + 1) / (bkt_n_[b] + 1) < wp_.l1_pmin && ++explore_ % 64u != 0) {
        ++st_.l1_priced_out;
        continue;
      }
      if (!fetch(g_next, c.unit, true)) break;  // over a budget or the reserve
      Entry& x = e_[key(g_next, c.unit)];
      if (x.home == Home::Coming && x.via_prefetch) {  // (a resurrection costs no copy: not counted)
        x.pf_kind = 2;
        x.pf_bucket = uint8_t(b);
        ++st_.l1_issued;
        ++issued;
      }
    }
    if (issued) t_->flush(DgTransport::kPf);  // the copies start now, ahead of the next layer's demand
    return;
  }
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

void DgCore::end_token(const std::vector<uint32_t>& keep, uint32_t rows) {
  require(rows > 0, SS_E_INVALID, "device-gated: boundary rows must be positive");
  // every copy and table write of the token has landed before anything is invalidated
  DgBoundaryTimes bt = bt_;  // published in one step at the end (a throw leaves the last complete one)
  std::chrono::steady_clock::time_point t;
  if (diag_) t = std::chrono::steady_clock::now();
  t_->sync(DgTransport::kDem);
  t_->sync(DgTransport::kPf);
  if (diag_) {
    bt.sync_us = us_since(t);
    t_->diag_boundary();
  }
  const bool hinted = !win_w_.empty();  // winpf: this token was a hinted window
  for (uint32_t k : coming_) {  // admitted: device homes now, their source blocks are free
    Entry& x = e_[k];
    if (x.home != Home::Coming) continue;
    if (x.pf_kind == 2) {  // winpf: the (votes, rank) bucket's online precision
      double& n = bkt_n_[x.pf_bucket];
      double& h = bkt_hit_[x.pf_bucket];
      n += 1;
      h += x.used ? 1 : 0;
      if (n > 4096) {
        n /= 2;
        h /= 2;
      }
    }
    if (wp_.enable && wp_.revert_unused && hinted && x.via_prefetch && !x.used) {
      // winpf: an unused prefetch goes back to its source block, which still holds the bytes (a Coming unit's block
      // is freed only here, at admission): no write-back. The device is done with the token and kPf is synced; the
      // table upload below unpublishes it.
      if (x.pf_kind) ++(x.pf_kind == 1 ? st_.prior_reverted : st_.l1_reverted);
      x.home = Home::Host;
      free_slots_.push_back(x.slot);
      x.slot = -1;
      if (mut_ != kRevertKeepsEntry) table_[k] = -1;
      unlink(k);
      --on_device_;
      x.via_prefetch = false;
      x.pf_kind = 0;
      x.pf_late = false;
      if (x.pf_done) {
        event_pool_.push_back(x.pf_done);
        x.pf_done = nullptr;
      }
      continue;
    }
    x.pf_kind = 0;
    x.pf_late = false;
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
  // Victims: one acceptance loop fed by the policy's order (the recency tail, or the merged score lists), then the
  // pressure override from the recency head. Eligible = a device home, no write-back in flight, not kept (Coming units
  // were admitted or reverted above). A victim is accepted with a free block, until free + pending (in flight or
  // deferred) + accepted reaches the target; at the first victim without a block nothing more is accepted (it stays on
  // the device, coverage kept). Only the accepted list feeds the write-backs below.
  std::vector<uint8_t> keep_mask;
  if (!keep.empty()) {
    keep_mask.assign(keys_, 0);
    for (uint32_t k : keep)
      if (k < keys_) keep_mask[k] = 1;
  }
  if (probe_) probe_();
  const size_t base = free_slots_.size() + pending_writebacks();
  victims_.clear();
  bool short_blocks = false;
  auto eligible = [&](uint32_t k) {
    const Entry& x = e_[k];
    return x.home == Home::Device && !(x.wb_pending && mut_ != kReevictPending) &&
           !(!keep_mask.empty() && keep_mask[k] && mut_ != kIgnoreKeep);
  };
  auto accept = [&](uint32_t k) {
    if (free_blocks_.empty()) {
      ++st_.evict_short;
      short_blocks = true;
      return false;
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
    victims_.push_back(k);
    return true;
  };
  const bool want_score = c_.scored && (!c_.scored_auto || rows > 1);
  bool by_score = want_score;
  for (uint32_t g = 0; by_score && g < gated(); ++g) by_score = sc_[g].list_epoch == epoch_;
  if (want_score && !by_score) ++st_.score_fallbacks;  // invalid score metadata: this boundary uses the recency order
  if (by_score && c_.boundary_score) {
    // Same victims as the public sorted lists + merge: only eligibility can change after ranking, never a key's
    // score / epochs / stamp. At the boundary score only eligible keys and retain the needed smallest candidates.
    const auto start = std::chrono::steady_clock::now();
    const size_t need = c_.spare_target > base ? c_.spare_target - base : 0;
    auto less = [this](const ScoreCand& a, const ScoreCand& b) { return cand_less(a, b); };
    select_heap_.clear();
    for (uint32_t g = 0; need && g < gated(); ++g) {
      uint64_t D = 0;
      for (uint32_t r : sc_[g].cur) D += sc_[g].rowsum[r];
      ++st_.score_layers;
      for (uint32_t u = 0; u < units(g); ++u) {
        if (!eligible(base_[g] + u)) continue;
        ++st_.score_candidates;
        const ScoreCand cd = score_candidate(g, u, D, select_heap_.size() == need ? &select_heap_.front() : nullptr);
        if (select_heap_.size() < need) {
          select_heap_.push_back(cd);
          std::push_heap(select_heap_.begin(), select_heap_.end(), less);
        } else if (less(cd, select_heap_.front())) {
          std::pop_heap(select_heap_.begin(), select_heap_.end(), less);
          select_heap_.back() = cd;
          std::push_heap(select_heap_.begin(), select_heap_.end(), less);
        }
      }
    }
    std::sort_heap(select_heap_.begin(), select_heap_.end(), less);
    for (const ScoreCand& cd : select_heap_) {
      if (!accept(cd.key)) break;
      ++st_.scored_victims;
    }
    st_.score_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
  } else if (by_score) {
    // heap merge of the per-layer lists (each sorted by the same total order), stopped at the accepted target
    auto head = [&](uint32_t g) -> const ScoreCand& { return sc_[g].list[heap_pos_[g]]; };
    auto after = [&](uint32_t a, uint32_t b) { return cand_less(head(b), head(a)); };
    heap_.clear();
    for (uint32_t g = 0; g < gated(); ++g) {
      heap_pos_[g] = 0;
      if (!sc_[g].list.empty()) heap_.push_back(g);
    }
    std::make_heap(heap_.begin(), heap_.end(), after);
    while (!heap_.empty() && base + victims_.size() < c_.spare_target) {
      std::pop_heap(heap_.begin(), heap_.end(), after);
      const uint32_t g = heap_.back();
      const uint32_t k = head(g).key;
      if (++heap_pos_[g] < sc_[g].list.size()) std::push_heap(heap_.begin(), heap_.end(), after);
      else heap_.pop_back();
      if (!eligible(k)) continue;
      if (!accept(k)) break;
      ++st_.scored_victims;
    }
  } else {
    for (uint32_t k = tail_; k != kNone && base + victims_.size() < c_.spare_target;) {
      const uint32_t prev = e_[k].prev;
      if (eligible(k) && !accept(k)) break;
      k = prev;
    }
  }
  if (pressure_ && pressure_recent_ && !short_blocks) {  // the most recently used too (they come back soon)
    uint32_t n = 0;
    for (uint32_t k = head_; k != kNone && n < pressure_recent_;) {
      const uint32_t next = e_[k].next;
      if (eligible(k)) {
        if (!accept(k)) break;
        ++n;
      }
      k = next;
    }
  }
  const size_t n_evict = victims_.size();
  // The table first, then the write-backs, so the upload never queues behind them on a shared copy engine (as the
  // reference, cache.cpp:552-566). The device is done with the token, and an evicting slot is reused only after its
  // write-back's event.
  if (diag_) bt.select_us = us_since(t);
  t_->upload_table(table_.data(), keys_);  // the device table = the host mirror
  if (diag_) bt.upload_us = us_since(t);
  // wb_defer: a victim selected now is deferred into the next token, behind what the last token left. Of those older
  // records, as many as the larger of this and the last boundary's victim counts are carried along (pending counted
  // them above; the forced path guards a demand against starvation); the older excess is submitted now, in order,
  // after the upload (it may not delay it). Deferred victims are persistent slots only (a bank unit is
  // never in the recency list), and a keep list protects selection: it never cancels an older deferred victim.
  // (The new victims are queued first, behind the old ones: a failed copy leaves every victim with its record.)
  // File homes, when merged, are discards: no block, no write-back, outside pending and the credit.
  const size_t n_old = deferred_.size(), carry = std::max(n_evict, prev_evict_);
  const size_t n_flush = n_old > carry ? n_old - carry : 0;
  prev_evict_ = n_evict;
  credit_ = 0;
  for (size_t i = 0; c_.wb_defer && i < n_evict; ++i) {
    const uint32_t k = victims_[i];
    Entry& x = e_[k];
    deferred_.push_back({k, x.slot, x.block, 0, 0});
    if (mut_ == kEarlyDeferredSlot) free_slots_.push_back(x.slot);
    ++st_.wb_deferred;
    ++st_.evicted;
    ++lc_[gated_of(k)].evicted;
  }
  size_t n_issued = 0;
  const uint64_t issued0 = st_.wb_issued_bytes;
  for (; n_issued < n_flush; ++n_issued) finish_deferred_front();
  st_.wb_boundary_flushed += n_issued;
  st_.wb_boundary_flushed_bytes += st_.wb_issued_bytes - issued0;
  for (size_t i = 0; !c_.wb_defer && i < n_evict; ++i, ++n_issued) {
    const uint32_t k = victims_[i];
    Entry& x = e_[k];
    t_->d2h(x.slot, x.block);
    if (mut_ == kEarlyWritebackBlock) free_blocks_.push_back(x.block);
    void* ev = event();
    t_->record(ev, DgTransport::kD2h);
    wbs_.push_back({k, x.slot, x.block, ev, false, unit_bytes_});
    outstanding_ += unit_bytes_;
    st_.wb_issued_bytes += unit_bytes_;
    ++st_.evicted;
    ++lc_[gated_of(k)].evicted;
  }
  if (diag_) {
    if (n_issued && !c_.wb_defer) t_->diag_writebacks_issued();  // deferred: not on its stream, not measured
    bt.issue_us = us_since(t);
    bt.writebacks = uint32_t(n_evict);
    ++bt.boundaries;
    bt_ = bt;
  }
  win_sym_.clear();  // winpf: the window is done; the next hint starts the next window's rows
  win_w_.clear();
  prior_bytes_ = 0;
  ++st_.tokens;
}

// ---- scored eviction (docs/design.md §3.8) ----
// serve_use of gated layer g: its unique sorted union is copied into cur first (serve_prediction reuses uniq_); then
// every (r in the previous token's union, e in this one) cell takes one saturating increment, its row sum only with
// it; the union enters the ring of the last kDgScoreWindow unions (ring counts follow); it becomes the previous one.
void DgCore::score_history(uint32_t g) {
  LayerScore& l = sc_[g];
  const uint32_t n = units(g);
  require(uniq_.size() <= c_.max_ids, SS_E_BACKEND, "device-gated: a score union above max ids");
  l.cur.assign(uniq_.begin(), uniq_.end());
  l.served = epoch_;
  for (uint32_t r : l.prev) {
    uint32_t* row = l.T.data() + size_t(r) * n;
    for (uint32_t e : l.cur) {
      if (row[e] == UINT32_MAX) {
        ++st_.score_saturated;
        continue;
      }
      ++row[e];
      ++l.rowsum[r];
    }
  }
  if (l.ring_n == kDgScoreWindow) {  // the oldest union leaves the window
    const uint32_t* old = l.ring.data() + size_t(l.ring_head) * c_.max_ids;
    for (uint32_t i = 0; i < l.ring_len[l.ring_head]; ++i) --l.cnt[old[i]];
    l.ring_head = (l.ring_head + 1) % kDgScoreWindow;
    --l.ring_n;
  }
  const uint32_t at = (l.ring_head + l.ring_n) % kDgScoreWindow;
  std::copy(l.cur.begin(), l.cur.end(), l.ring.begin() + ptrdiff_t(size_t(at) * c_.max_ids));
  l.ring_len[at] = uint8_t(l.cur.size());
  ++l.ring_n;
  for (uint32_t e : l.cur) ++l.cnt[e];
  l.prev.assign(l.cur.begin(), l.cur.end());
}

// For each device or coming unit e of layer g (wb_pending ones too: eligibility is the boundary's), cur = this token's
// union of g:
//   N = sum_{r in cur} T[r][e], D = sum_{r in cur} rowsum[r], P = ((|cur| N) << 16) / max(1, D)
//   F = (cnt[e] << 16) / 16, Rec = (1 << 16) / (1 + epoch - demand_epoch), S = w_t P + w_f F + w_r Rec
// (unsigned floor divisions; N < 2^38, D < 2^54, (|cur| N) << 16 < 2^60, S < 2^33), sorted by (S, demand epoch,
// stamp) and tagged with this epoch. A layer's union, epochs and stamps are final here: each layer is served once per
// token, its prefetches and resurrections come before its use, collection and admission never push.
void DgCore::score_layer(uint32_t g) {
  if (!c_.scored) return;
  LayerScore& l = sc_[g];
  if (l.served != epoch_) return;  // not served this token: no list (the boundary falls back and counts it)
  if (mut_ == kSkipRank && g == 1 && epoch_ % 3 == 0) return;  // a layer left unranked: the boundary must fall back
  if (mut_ == kStaleList && g == 1 && (epoch_ & 1u) == 0 && l.list_epoch) {  // a list re-tagged, not re-ranked
    l.list_epoch = epoch_;
    return;
  }
  if (c_.boundary_score) {  // the service only closes the layer; AUTO's row count arrives at the boundary
    l.list_epoch = epoch_;
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  const uint32_t n = units(g);
  std::vector<uint32_t> scratch;  // kNoUnionSave: whatever the shared scratch holds now (in range)
  if (mut_ == kNoUnionSave)
    for (uint32_t r : uniq_)
      if (r < n) scratch.push_back(r);
  const std::vector<uint32_t>& cur = mut_ == kNoUnionSave ? scratch : l.cur;
  uint64_t D = 0;
  for (uint32_t r : cur) D += l.rowsum[r];
  l.list.clear();
  for (uint32_t u = 0; u < n; ++u) {
    const uint32_t k = base_[g] + u;
    const Entry& x = e_[k];
    if (x.home != Home::Device && x.home != Home::Coming) continue;
    // Keep the shared-scratch mutation meaningful in the reference path.
    if (mut_ == kNoUnionSave) l.cur.swap(scratch);
    l.list.push_back(score_candidate(g, u, D));
    if (mut_ == kNoUnionSave) l.cur.swap(scratch);
  }
  std::sort(l.list.begin(), l.list.end(), [this](const ScoreCand& a, const ScoreCand& b) { return cand_less(a, b); });
  l.list_epoch = epoch_;
  ++st_.score_layers;
  st_.score_candidates += l.list.size();
  st_.score_ns += uint64_t(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

DgCore::ScoreCand DgCore::score_candidate(uint32_t g, uint32_t u, uint64_t D, const ScoreCand* cutoff) const {
  const LayerScore& l = sc_[g];
  const Entry& x = e_[base_[g] + u];
  const uint64_t F = (uint64_t(l.cnt[u]) << 16) / kDgScoreWindow;
  const uint64_t R = (uint64_t(1) << 16) / (1 + (epoch_ - x.demand_epoch));
  ScoreCand cd{uint64_t(c_.w_freq) * F + uint64_t(c_.w_rec) * R, x.demand_epoch, x.lru_stamp, base_[g] + u};
  // All terms are nonnegative: if even F + Rec cannot enter the heap, skip the transition row reads entirely.
  if (cutoff && !cand_less(cd, *cutoff)) return cd;
  uint64_t N = 0;
  for (uint32_t r : l.cur) N += l.T[size_t(r) * units(g) + u];
  const uint64_t P = ((uint64_t(l.cur.size()) * N) << 16) / std::max<uint64_t>(1, D);
  cd.s += uint64_t(c_.w_trans) * P;
  return cd;
}

void DgCore::set_policy(uint32_t policy) {
  require(policy <= 2, SS_E_INVALID, "device-gated: policy must be LRU, SCORED or SCORED_AUTO");
  require(!policy || !sc_.empty(), SS_E_INVALID, "device-gated: scoring must be provisioned at open");
  drain();
  collect();
  c_.scored = policy != 0;
  c_.scored_auto = policy == 2;
  reset_score();
  for (Entry& x : e_) x.demand_epoch = x.lru_stamp = 0;
  stamp_ = 0;
  for (uint32_t k = tail_; k != kNone; k = e_[k].prev) e_[k].lru_stamp = ++stamp_;
}

void DgCore::load_profile(const void* data, uint64_t len, const std::string& identity) {
  require(c_.scored, SS_E_INVALID, "device-gated: a score profile needs policy SCORED");
  const DgProfileView v = dg_profile_check(data, len, c_.n_units, identity);
  const unsigned char* p = v.cells;
  for (uint32_t g = 0; g < gated(); ++g) {
    LayerScore& l = sc_[g];
    const uint32_t n = units(g);
    for (size_t i = 0; i < l.T.size(); ++i, p += 4)
      l.T[i] = uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
    for (uint32_t r = 0; r < n; ++r) {
      uint64_t sum = 0;
      for (uint32_t e = 0; e < n; ++e) sum += l.T[size_t(r) * n + e];
      l.rowsum[r] = sum;
    }
    l.open_T = l.T;  // the reset_open state
  }
  prov_ = v.provenance;
}

std::vector<uint32_t> DgCore::score_window(uint32_t g) const {
  const LayerScore& l = sc_[g];
  std::vector<uint32_t> v;
  for (uint32_t i = 0; i < l.ring_n; ++i) {
    const uint32_t a = (l.ring_head + i) % kDgScoreWindow;
    const auto first = l.ring.begin() + ptrdiff_t(size_t(a) * c_.max_ids);
    v.insert(v.end(), first, first + l.ring_len[a]);
    v.push_back(kNone);
  }
  return v;
}

void DgCore::reset_score() {
  for (uint32_t g = 0; g < uint32_t(sc_.size()); ++g) {
    LayerScore& l = sc_[g];
    const uint32_t n = units(g);
    if (l.open_T.empty()) std::fill(l.T.begin(), l.T.end(), 0u);
    else std::copy(l.open_T.begin(), l.open_T.end(), l.T.begin());
    for (uint32_t r = 0; r < n; ++r) {
      uint64_t sum = 0;
      for (uint32_t e = 0; e < n && !l.open_T.empty(); ++e) sum += l.T[size_t(r) * n + e];
      l.rowsum[r] = sum;
    }
    std::fill(l.cnt.begin(), l.cnt.end(), uint16_t(0));
    std::fill(l.ring.begin(), l.ring.end(), 0u);
    std::fill(l.ring_len.begin(), l.ring_len.end(), uint8_t(0));
    l.ring_head = l.ring_n = 0;
    l.prev.clear();
    l.cur.clear();
    l.served = 0;
    l.list.clear();
    l.list_epoch = 0;
  }
  epoch_ = 0;
  stamp_ = open_stamp_;
}

void DgCore::set_winpf(const DgWinpf& w) {
  if (!rate_set_ || w.copy_gbps != wp_.copy_gbps) {  // a learned rate survives phase toggles with the same seed rate
    rate_gbps_ = w.copy_gbps;
    rate_set_ = true;
  }
  wp_ = w;
}

void DgCore::observe_copy(double unit_ms) {
  if (!(unit_ms > 0) || !unit_bytes_) return;
  const double g = double(unit_bytes_) / (unit_ms * 1e6);
  if (g < 1 || g > 200) return;  // a stamp that cannot be a copy
  rate_gbps_ = 0.9 * rate_gbps_ + 0.1 * g;
}

void DgCore::window_hint(const int32_t* symbols, const float* weights, uint32_t rows) {
  stamp_room("window_hint");
  ++st_.winpf_hints;
  st_.winpf_hint_rows += rows;
  for (uint32_t r = 0; r < rows; ++r) {  // at most 64 rows per window (the context refuses more)
    win_sym_.push_back(symbols[r]);
    const float w = weights ? weights[r] : 1.0f;
    win_w_.push_back(w > 0 ? std::min(w, 1.0f) : 0.0f);
  }
  const uint64_t issued0 = st_.prior_issued;
  issue_prior();
  if (st_.prior_issued != issued0) t_->flush(DgTransport::kPf);  // the prior copies start now (the draft runs)
}

void DgCore::issue_prior() {
  const uint32_t P = std::min(std::min(wp_.prior_layers, prior_.layers), gated());
  if (!wp_.enable || !wp_.prior_cap || !P) return;
  st_.collected_mid += collect();  // finished write-backs leave the ledger
  const uint64_t budget = uint64_t(double(wp_.prior_window_us) * rate_gbps_ * 1000.0);  // us x GB/s = bytes
  for (uint32_t g = 0; g < P; ++g) {
    const uint32_t nu = units(g);
    pcand_.clear();  // votes = P(no hinted row uses the unit), rank unused
    for (size_t i = 0; i < win_sym_.size(); ++i) {
      const auto it = prior_.row.find(win_sym_[i]);
      if (it == prior_.row.end()) continue;
      const size_t base = (size_t(it->second) * prior_.layers + g) * prior_.k;
      for (uint32_t j = 0; j < prior_.k; ++j) {
        const uint32_t u = prior_.unit[base + j];
        if (u >= nu) continue;
        const float p = std::min(1.0f, prior_.p[base + j] * win_w_[i]);
        bool found = false;
        for (Cand& x : pcand_)
          if (x.unit == u) {
            x.votes *= 1.0f - p;
            found = true;
            break;
          }
        if (!found) pcand_.push_back({1.0f - p, 0, u});
      }
    }
    std::sort(pcand_.begin(), pcand_.end(),
              [](const Cand& a, const Cand& b) { return a.votes != b.votes ? a.votes < b.votes : a.unit < b.unit; });
    uint32_t issued = 0;
    for (const Cand& x : pcand_) {
      if (issued >= wp_.prior_cap || 1.0f - x.votes < wp_.prior_pmin - 1e-6f) break;
      if (on_device_or_coming(g, x.unit)) continue;
      // the copy-engine budget of the window: every byte already queued (write-backs submitted and not collected:
      // conservative) comes first on a FIFO engine; a resurrection (Evicting) copies nothing
      if (e_[key(g, x.unit)].home != Home::Evicting && outstanding_ + prior_bytes_ + unit_bytes_ > budget) {
        ++st_.prior_budget_stops;
        return;
      }
      if (!fetch_impl(g, x.unit, true, false)) return;  // the reserve
      Entry& e = e_[key(g, x.unit)];
      if (e.home == Home::Coming && e.via_prefetch) {
        e.pf_kind = 1;
        prior_bytes_ += unit_bytes_;
        ++st_.prior_issued;
        ++issued;
      }
    }
  }
}

uint32_t DgCore::seed(const std::vector<uint32_t>& keys) {
  // winpf: prior prefetches issued between tokens are Coming with their entries already in the mirror; their bytes
  // land before the upload below publishes the mirror
  stamp_room("ss_seed");
  if (!coming_.empty() && mut_ != kSeedNoPfSync) t_->sync(DgTransport::kPf);
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
  while (!deferred_.empty()) finish_deferred_front();
  t_->sync(DgTransport::kDem);
  t_->sync(DgTransport::kPf);
  t_->sync(DgTransport::kD2h);
}

bool DgCore::same_open_state(const DgCore& b) const {
  auto cfg = [](const DgConfig& c) {
    return std::tie(c.n_units, c.persistent_slots, c.bank_slots, c.blocks, c.spare_target, c.prefetch_reserve,
                    c.prefetch_reserve_auto, c.prefetch_per_layer, c.prefetch_per_token, c.wb_defer, c.wb_on_d2h,
                    c.wb_layer_bytes, c.wb_max_outstanding);
  };
  auto wp = [](const DgWinpf& w) {
    return std::tie(w.enable, w.prior_layers, w.prior_cap, w.prior_pmin, w.l1_cap, w.l1_min_votes, w.l1_pmin,
                    w.revert_unused, w.top_k, w.prior_window_us, w.copy_gbps);
  };
  auto entry = [](const Entry& x) {
    return std::tie(x.home, x.slot, x.block, x.prev, x.next, x.in_list, x.wb_pending, x.via_prefetch, x.used,
                    x.pf_kind, x.pf_late, x.pf_bucket, x.demand_epoch, x.lru_stamp);
  };
  if (cfg(c_) != cfg(b.c_) || wp(wp_) != wp(b.wp_) || e_.size() != b.e_.size()) return false;
  for (size_t k = 0; k < e_.size(); ++k)
    if (entry(e_[k]) != entry(b.e_[k])) return false;
  if (std::tie(table_, free_slots_, free_blocks_, head_, tail_, on_device_, bank_key_, pf_layer_, pf_token_,
               coming_, banked_, pred_, pred_layer_, layer_missed_, pressure_, pressure_free_, pressure_recent_,
               demand_free_used_, credit_, outstanding_, prev_evict_, rate_gbps_, rate_set_, explore_,
               win_sym_, win_w_, prior_bytes_, layer_bytes_, max_outstanding_, wb_stream_) !=
      std::tie(b.table_, b.free_slots_, b.free_blocks_, b.head_, b.tail_, b.on_device_, b.bank_key_, b.pf_layer_, b.pf_token_,
               b.coming_, b.banked_, b.pred_, b.pred_layer_, b.layer_missed_, b.pressure_, b.pressure_free_, b.pressure_recent_,
               b.demand_free_used_, b.credit_, b.outstanding_, b.prev_evict_, b.rate_gbps_, b.rate_set_, b.explore_,
               b.win_sym_, b.win_w_, b.prior_bytes_, b.layer_bytes_, b.max_outstanding_, b.wb_stream_)) return false;
  for (unsigned i = 0; i < 2; ++i)
    if (bank_layer_[i] != b.bank_layer_[i] || bank_next_[i] != b.bank_next_[i]) return false;
  for (unsigned i = 0; i < 15; ++i)
    if (bkt_n_[i] != b.bkt_n_[i] || bkt_hit_[i] != b.bkt_hit_[i]) return false;
  if (std::tie(c_.scored, c_.scored_auto, c_.boundary_score, c_.w_trans, c_.w_freq, c_.w_rec, epoch_, stamp_) !=
          std::tie(b.c_.scored, b.c_.scored_auto, b.c_.boundary_score, b.c_.w_trans, b.c_.w_freq, b.c_.w_rec,
                   b.epoch_, b.stamp_) ||
      sc_.size() != b.sc_.size())
    return false;
  for (size_t g = 0; g < sc_.size(); ++g)
    if (score_window(uint32_t(g)) != b.score_window(uint32_t(g)) ||
        std::tie(sc_[g].T, sc_[g].rowsum, sc_[g].cnt, sc_[g].prev, sc_[g].served, sc_[g].list_epoch) !=
            std::tie(b.sc_[g].T, b.sc_[g].rowsum, b.sc_[g].cnt, b.sc_[g].prev, b.sc_[g].served, b.sc_[g].list_epoch))
      return false;
  return wbs_.empty() && b.wbs_.empty() && deferred_.empty() && b.deferred_.empty() &&
         prior_.layers == b.prior_.layers && prior_.k == b.prior_.k && prior_.row == b.prior_.row &&
         prior_.unit == b.prior_.unit && prior_.p == b.prior_.p;
}

void DgCore::reset_open() {
  require(!reset_failed_, SS_E_BACKEND, "device-gated: reset failed; close the poisoned context");
  require(open_e_.size() == keys_, SS_E_INVALID, "device-gated: reset before place");
  try {
    drain();
    // drain submits deferred parts but deliberately does not collect. Wait for every record, including discards.
    for (const Writeback& w : wbs_) t_->event_sync(w.done);
    collect();
    require(wbs_.empty(), SS_E_BACKEND, "device-gated: reset could not collect write-backs");
    check();

    // Coming and bank copies retain an authoritative, immutable source block. After drain we can revert them
    // without a synthetic boundary (which would select fresh victims). Recycle their completed events exactly once.
    for (Entry& x : e_) {
      if (x.pf_done) {
        t_->event_sync(x.pf_done);
        event_pool_.push_back(x.pf_done);
        x.pf_done = nullptr;
      }
    }
    const size_t P = c_.persistent_slots, N = P + c_.blocks, stage = N;
    std::vector<size_t> current(keys_), target(keys_);
    std::vector<uint32_t> occupant(N + 1, kNone), wanted(N, kNone);
    for (uint32_t k = 0; k < keys_; ++k) {
      const Entry& x = e_[k];
      require(x.home != Home::Evicting, SS_E_BACKEND, "device-gated: reset left an evicting key");
      current[k] = x.home == Home::Device ? size_t(x.slot) : P + size_t(x.block);
      const Entry& o = open_e_[k];
      target[k] = o.home == Home::Device ? size_t(o.slot) : P + size_t(o.block);
      occupant[current[k]] = k;
      wanted[target[k]] = k;
    }
    auto block = [&](size_t loc) { return loc == stage ? -1 : int32_t(loc - P); };
    auto move = [&](uint32_t k, size_t dst) {
      const size_t src = current[k];
      require(occupant[dst] == kNone, SS_E_BACKEND, "device-gated: reset overwrote a live source");
      if (src < P && dst < P) t_->d2d(int32_t(src), int32_t(dst));
      else if (src < P) {
        t_->d2h(int32_t(src), block(dst));
        t_->sync(DgTransport::kD2h);
      } else if (dst < P) {
        t_->h2d(DgTransport::kDem, int32_t(dst), block(src));
        t_->sync(DgTransport::kDem);
      } else t_->h2h(block(src), block(dst));
      occupant[src] = kNone;
      occupant[dst] = k;
      current[k] = dst;
    };
    // Paths first: fill each target hole from its current source, propagating the hole. Each move fixes one key.
    for (size_t loc = 0; loc < N; ++loc) {
      size_t hole = loc;
      while (occupant[hole] == kNone && wanted[hole] != kNone) {
        const uint32_t k = wanted[hole];
        const size_t next = current[k];
        move(k, hole);
        hole = next;
      }
    }
    // What remains consists of disjoint cycles, even with zero free slots/blocks. Save one unit in the existing
    // dedicated host staging buffer, rotate into the hole, then close the cycle. Never overwrite an unread source.
    for (uint32_t k = 0; k < keys_; ++k) {
      if (current[k] == target[k]) continue;
      size_t hole = current[k];
      move(k, stage);
      while (hole != target[k]) {
        const uint32_t next = wanted[hole];
        require(next != kNone, SS_E_BACKEND, "device-gated: reset cycle has no predecessor");
        const size_t src = current[next];
        move(next, hole);
        hole = src;
      }
      move(k, hole);
    }

    c_ = open_c_;
    apply_wb_defer();  // the open's scheduling mode (a runtime set_wb_defer may have changed it)
    std::copy(open_e_.begin(), open_e_.end(), e_.begin());
    reset_score();  // scored eviction's learning state, as winpf's: the open state (the profile's counts, if any)
    std::copy(open_table_.begin(), open_table_.end(), table_.begin());
    free_slots_ = open_free_slots_;
    free_blocks_ = open_free_blocks_;
    head_ = open_head_;
    tail_ = open_tail_;
    on_device_ = open_on_device_;
    coming_.clear();
    banked_.clear();
    std::fill(bank_key_.begin(), bank_key_.end(), kNone);
    // Empty bank ownership is invalidated by layer, independently of the retained token sequence domain.
    bank_seq_[0] = bank_seq_[1] = 0;
    bank_layer_[0] = bank_layer_[1] = kNone;
    bank_next_[0] = bank_next_[1] = 0;
    std::fill(pf_layer_.begin(), pf_layer_.end(), 0);
    pf_token_ = 0;
    pred_.clear();
    pred_layer_ = kNone;
    std::fill(layer_missed_.begin(), layer_missed_.end(), uint8_t(0));
    pressure_ = false;
    pressure_free_ = pressure_recent_ = demand_free_used_ = 0;
    prev_evict_ = 0;
    credit_ = 0;
    outstanding_ = 0;
    wp_ = open_wp_;
    rate_gbps_ = open_rate_gbps_;
    rate_set_ = open_rate_set_;
    std::fill(std::begin(bkt_n_), std::end(bkt_n_), 0);
    std::fill(std::begin(bkt_hit_), std::end(bkt_hit_), 0);
    explore_ = 0;
    win_sym_.clear();
    win_w_.clear();
    prior_bytes_ = 0;
    bt_ = DgBoundaryTimes{};
    t_->diag_reset();
    t_->upload_table(table_.data(), keys_);
    t_->sync(DgTransport::kDem);
    check();
  } catch (...) {
    reset_failed_ = true;
    throw;
  }
}

uint32_t DgCore::abandon_writebacks() noexcept {
  const uint32_t n = uint32_t(deferred_.size());
  deferred_.clear();
  return n;
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
  // write-back records <-> wb_pending keys, a bijection: each record (recorded in wbs_, or deferred) has its own key,
  // every key with wb_pending has one, and an evicting key's is live (not discarded). Deferred: parts submitted only
  // at the front (all of them only after a failed record), persistent slots only; the bytes add up.
  std::vector<uint8_t> rec_of(keys_, 0), live_of(keys_, 0);
  uint64_t out = 0;
  for (const Writeback& w : wbs_) {
    if (rec_of[w.key]++) broken("two write-back records of one key");
    live_of[w.key] = !w.discard;
    out += w.bytes;
  }
  if (!c_.wb_defer && !deferred_.empty()) broken("a deferred write-back without wb_defer");
  for (size_t i = 0; i < deferred_.size(); ++i) {
    const Deferred& d = deferred_[i];
    if (d.key >= keys_) broken("a deferred write-back of no key");
    const Entry& x = e_[d.key];
    if (x.home != Home::Evicting || x.slot != d.slot || x.block != d.block)
      broken("a deferred write-back does not match its evicting key");
    if (uint32_t(d.slot) >= bank_first_) broken("a deferred write-back of a bank slot");
    if (d.parts_done > t_->parts()) broken("a deferred write-back past its last part");
    if (i > 0 && d.parts_done != 0) broken("parts submitted behind the deferred front");
    uint64_t b = 0;
    for (uint32_t p = 0; p < d.parts_done; ++p) b += t_->part_bytes(p);
    if (b != d.bytes) broken("a deferred write-back's bytes != its submitted parts");
    if (rec_of[d.key]++) broken("a deferred key with another write-back record");
    live_of[d.key] = 1;
    out += d.bytes;
  }
  if (out != outstanding_) broken("outstanding write-back bytes != the records'");
  for (uint32_t k = 0; k < keys_; ++k) {
    if (e_[k].wb_pending != (rec_of[k] != 0)) broken("wb_pending without a write-back record, or a record without it");
    if (e_[k].home == Home::Evicting && !live_of[k]) broken("an evicting key without a live write-back");
  }
  if (post_ready_ev_) take(ev_used, ev_index(post_ready_ev_), "post-ready event");
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
  if (c_.scored) {  // stamps strictly decrease from the head; every row sum / ring count equals its recount
    uint64_t last = UINT64_MAX;
    for (uint32_t k = head_; k != kNone; k = e_[k].next) {
      if (e_[k].lru_stamp >= last || e_[k].lru_stamp == 0 || e_[k].lru_stamp > stamp_) broken("recency stamps");
      last = e_[k].lru_stamp;
    }
    for (uint32_t g = 0; g < gated(); ++g) {
      const LayerScore& l = sc_[g];
      const uint32_t n = units(g);
      std::vector<uint16_t> cnt(n, 0);
      for (uint32_t e : score_window(g))
        if (e != kNone) ++cnt[e];
      if (cnt != l.cnt || l.ring_n > kDgScoreWindow) broken("score ring counts");
      for (uint32_t r = 0; r < n; ++r) {
        uint64_t sum = 0;
        for (uint32_t e = 0; e < n; ++e) sum += l.T[size_t(r) * n + e];
        if (sum != l.rowsum[r]) broken("score row sum != its cells");
      }
      for (uint64_t ep : {l.served, l.list_epoch})
        if (ep > epoch_) broken("a score epoch in the future");
    }
  }
}

}  // namespace ts
