// Device-gated decode mode: the cache state machine (dg_core.hpp).

#include "dg_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
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
  run_.assign(gated(), 1);
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
  require(!c_.disk_tier || c_.disk_staging >= 1, SS_E_INVALID, "device-gated: the disk tier needs staging buffers");
  require(c_.disk_tier || (!c_.disk_prefetch && !c_.disk_staging && !c_.home_blocks), SS_E_INVALID,
          "device-gated: disk options without the disk tier");
  require(c_.inclusive < c_.blocks || c_.inclusive == 0, SS_E_INVALID,
          "device-gated: the inclusive share must leave a block for write-backs");
  if (c_.disk_tier) stage_ev_.assign(c_.disk_staging, nullptr);
  // dg_no_admit: every distinct miss of a layer takes a bank slot (an export holds at most max_ids ids)
  require(!c_.no_admit || c_.bank_slots >= c_.max_ids, SS_E_CAPACITY,
          "device-gated: dg_no_admit serves every miss through a bank slot: it needs " + std::to_string(c_.max_ids) +
              " bank slots per bank (dg_max_ids, two banks), " + std::to_string(c_.bank_slots) + " configured");
  rs_.assign(keys_, 0.0);
  rhits_.assign(keys_, 0);
}

void DgTransport::read_unit(uint32_t, int32_t) {
  fail(SS_E_UNSUPPORTED, "device-gated: this transport cannot read a unit from its file");
}

// ---- three-tier: the pinned homes' recency list ----
void DgCore::hpush(uint32_t k) {
  ++on_host_;
  if (!c_.disk_tier) return;
  Entry& x = e_[k];
  x.hprev = kNone;
  x.hnext = hhead_;
  if (hhead_ != kNone) e_[hhead_].hprev = k; else htail_ = k;
  hhead_ = k;
  x.in_hlist = true;
}

void DgCore::hunlink(uint32_t k) {
  --on_host_;
  if (!c_.disk_tier) return;
  Entry& x = e_[k];
  if (!x.in_hlist) broken("a pinned home outside the recency list");
  if (x.hprev != kNone) e_[x.hprev].hnext = x.hnext; else hhead_ = x.hnext;
  if (x.hnext != kNone) e_[x.hnext].hprev = x.hprev; else htail_ = x.hprev;
  x.hprev = x.hnext = kNone;
  x.in_hlist = false;
}

std::vector<uint32_t> DgCore::host_order() const {
  std::vector<uint32_t> out;
  for (uint32_t k = hhead_; k != kNone; k = e_[k].hnext) out.push_back(k);
  return out;
}

// A key that leaves the device / a bank without a write-back: its block (it held one: a clean copy or a bank home)
// is its home again, else its file (three-tier). Its slot and entry are the caller's.
void DgCore::to_lower(uint32_t k) {
  Entry& x = e_[k];
  x.slot = -1;
  if (x.backed) {
    x.backed = false;
    --backed_;
  }
  if (x.block >= 0) {
    x.home = Home::Host;
    hpush(k);
  } else {
    if (!c_.disk_tier) broken("a key without a block left the device without a write-back");
    x.home = Home::File;
    ++on_file_;
  }
}

// RAM -> drop: the least recently homed pinned key goes to its file (no copy: the file holds its bytes); its block is
// free at once (a Host key's block is quiescent: no copy reads or writes it).
bool DgCore::drop_host_lru(const std::vector<uint8_t>* skip, bool count) {
  if (!c_.disk_tier) return false;
  uint32_t k = htail_;
  while (k != kNone && skip && (*skip)[k]) k = e_[k].hprev;
  if (k == kNone) return false;
  Entry& x = e_[k];
  hunlink(k);
  free_blocks_.push_back(x.block);
  x.block = -1;
  x.home = Home::File;
  ++on_file_;
  if (count) ++st_.disk_drops;
  return true;
}

// The next staging buffer of the file reads: its previous copy must have landed before the read overwrites it.
// Drain point: a buffer whose read is still in flight has its copy enqueued first (the records up to its own). The wait
// that follows is on that buffer's own last copy (recorded when its record was ended): no other read in flight holds
// the buffer or that event, so the others stay in flight.
int32_t DgCore::take_stage() {
  const uint32_t i = stage_next_;
  stage_next_ = (i + 1) % c_.disk_staging;
  const int32_t block = int32_t(c_.blocks + i);
  for (size_t j = 0; j < pending_.size(); ++j)
    if (pending_[j].stage == block) {
      drain_pending(j + 1);
      break;
    }
  if (stage_ev_[i] && !t_->done(stage_ev_[i]) && mut_ != kStageNoWait) {
    t_->event_sync(stage_ev_[i]);
    ++st_.disk_stage_waits;
  }
  return block;
}

// ---- three-tier: overlapped file misses ----
// A file miss's fetch takes its slot and staging buffer and updates every host structure as a synchronous fetch does
// (homes, the mirror table_, lists, counters); only the device work that needs the read's bytes waits here: the copy,
// the entry, the staging buffer's event and a prefetch's event, enqueued in fetch order once the read ended. A read in
// flight holds exactly: its key (Coming / Bank, the mirror's entry set, no write-back record: it came from its file),
// its slot (out of the free list, or its bank slot), its staging buffer, and no event yet. An operation drains only
// the reads whose key, slot, staging buffer or event it touches (targeted), and the demand reads at a public entry
// point; everything else it does is on other keys, slots, blocks and events, so it sees what it saw with synchronous
// reads (the host state is the same, and the device work it enqueues or waits on is independent of the copies not
// enqueued yet). The device reads a layer's entries only after its ready, which ends the layer's reads first.
void DgCore::copy_in(DgTransport::Stream s, uint32_t k, int32_t slot, int32_t src) {
  if (mut_ == kEntryBeforeCopy) {
    write_entry(s, k, slot);
    t_->h2d(s, slot, src);
  } else {
    t_->h2d(s, slot, src);
    write_entry(s, k, slot);
  }
}

void DgCore::finish_pending(const Pending& p) {
  Entry& x = e_[p.key];
  copy_in(p.s, p.key, p.slot, p.stage);
  void*& ev = stage_ev_[uint32_t(p.stage) - c_.blocks];  // the staging buffer is reusable once this copy landed
  if (!ev) ev = event();
  t_->record(ev, p.s);
  if (p.prefetch) {
    x.pf_done = event();
    t_->record(x.pf_done, p.s);
    if (mut_ == kEarlyEventRecycle) event_pool_.push_back(x.pf_done);
  }
}

// A read that failed: the key goes back to its file, its slot to where it came from (a bank slot stays used for the
// layer, unpublished), nothing was enqueued for it. Its stage keeps its last landed copy's event.
void DgCore::undo_pending(const Pending& p) {
  Entry& x = e_[p.key];
  const uint32_t g = gated_of(p.key);
  table_[p.key] = -1;
  if (p.bank) {
    bank_key_[size_t(p.slot) - bank_first_] = kNone;
    banked_.erase(std::find(banked_.begin(), banked_.end(), p.key));
    --st_.bank_serves;
    --lc_[g].bank_serves;
  } else {
    unlink(p.key);
    --on_device_;
    coming_.erase(std::find(coming_.begin(), coming_.end(), p.key));
    free_slots_.push_back(p.slot);
    if (p.free_counted) --demand_free_used_;
  }
  x.home = Home::File;
  x.slot = -1;
  x.via_prefetch = x.used = x.pf_late = false;
  x.pf_kind = 0;
  ++on_file_;
  if (p.prefetch) {
    --st_.prefetched;
    --lc_[g].prefetched;
  } else {
    --st_.fetched;
    --lc_[g].fetched;
  }
}

void DgCore::drain_pending(size_t n) {
  std::exception_ptr first;
  bool device_failed = false;  // a failed copy: the later records are ended and undone, nothing more is enqueued
  bool pf = false;
  for (; n > 0 && !pending_.empty(); --n) {
    const Pending p = pending_.front();
    pending_.pop_front();
    if (p.entry != entries_) {  // ended in a later call than its fetch's (it overlapped that call's caller)
      ++st_.disk_carried;
      st_.disk_carried_bytes += unit_bytes_;
    }
    const auto t0 = std::chrono::steady_clock::now();
    bool read_ok = true;
    try {
      st_.disk_read_ns += t_->read_unit_end(p.handle);
    } catch (...) {
      read_ok = false;
      if (!first) first = std::current_exception();
    }
    st_.disk_wait_ns += uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    if (!read_ok || device_failed) {
      undo_pending(p);
      continue;
    }
    ++st_.disk_reads;
    st_.disk_read_bytes += unit_bytes_;
    try {
      finish_pending(p);
      pf = pf || p.prefetch;
    } catch (...) {
      if (!first) first = std::current_exception();
      device_failed = true;  // (this record stays fetched, as a synchronous fetch whose copy failed)
      n = SIZE_MAX;          // every later record is ended and undone in this call
    }
  }
  if (pf && !first) t_->flush(DgTransport::kPf);  // prefetch copies enqueued outside their call: submitted now
  if (first) std::rethrow_exception(first);
}

// Prefetch reads in flight across calls: a public entry point ends the demand reads (the records up to the last demand
// one, oldest first); a prefetch's read ends when its key is touched again (drain_key: a fetch or a cover), when its
// layer is done (drain_layer: before its ready), or at a drain of everything (the end of the token, slot / stage
// pressure, write-backs submitted between tokens, a seed, a reset, a setting, destruction).
void DgCore::drain_through(const std::function<bool(const Pending&)>& f) {
  for (size_t i = pending_.size(); i-- > 0;)
    if (f(pending_[i])) {
      drain_pending(i + 1);
      return;
    }
}

void DgCore::drain_demand() {
  drain_through([](const Pending& p) { return !p.prefetch; });
}

void DgCore::drain_key(uint32_t k) {
  drain_through([k](const Pending& p) { return p.key == k; });
}

void DgCore::drain_layer(uint32_t g) {
  drain_through([this, g](const Pending& p) { return !p.prefetch || gated_of(p.key) == g; });
}

void DgCore::enter() {
  ++entries_;
  drain_demand();
}

void DgCore::poll_reads() {
  enter();
  size_t n = 0;
  while (n < pending_.size() && t_->read_unit_done(pending_[n].handle)) ++n;
  if (n) drain_pending(n);
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
  drain_pending();  // drain point: write-backs submitted (every read ended first)
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

// Destruction: the reads in flight are ended (their errors dropped) before anything they write into can be freed;
// their copies are never enqueued (the owner drains or abandons the streams).
DgCore::~DgCore() {
  for (const Pending& p : pending_) {
    try {
      t_->read_unit_end(p.handle);
    } catch (...) {
    }
  }
  for (void* ev : all_events_) t_->event_destroy(ev);
}

void DgCore::place(const std::vector<uint32_t>& residents, const std::vector<uint8_t>* disk) {
  drain_pending();  // drain point: the open placement
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
  const bool three = c_.disk_tier;
  const std::vector<uint8_t> none(three && !disk ? keys_ : 0, 0);  // no flags: no plan-disk key
  if (three && !disk) disk = &none;
  require(!three || disk->size() == keys_, SS_E_INVALID, "device-gated: the disk flags do not cover the keys");
  // three-tier: which non-device keys get a block (plan-pinned first, then by priority), as flags in key order
  std::vector<uint8_t> blocked(keys_, 0);
  std::vector<uint32_t> host_rank;  // the pinned homes by priority (the recency order)
  if (three) {
    std::vector<uint8_t> seen(keys_, 0);
    std::vector<uint32_t> prio;
    for (uint32_t k : residents)
      if (k < keys_ && !seen[k]) {
        seen[k] = 1;
        prio.push_back(k);
      }
    for (uint32_t k = 0; k < keys_; ++k)
      if (!seen[k]) prio.push_back(k);
    const uint32_t cap = std::min(c_.home_blocks, c_.blocks);  // exact: 0 = no pinned home (all spares)
    for (int pass = 0; pass < 2; ++pass)
      for (uint32_t k : prio)
        if (!placed[k] && (*disk)[k] == uint8_t(pass) && host_rank.size() < cap) {
          blocked[k] = 1;
          host_rank.push_back(k);
        }
  } else {
    require(uint64_t(keys_) - dev0 <= c_.blocks, SS_E_CAPACITY, "device-gated: fewer host blocks than non-resident units");
  }
  int32_t next_block = 0;
  on_host_ = on_file_ = backed_ = 0;
  hhead_ = htail_ = kNone;
  for (uint32_t k = 0; k < keys_; ++k) {
    Entry& x = e_[k];
    x = Entry{};
    if (!placed[k] && (!three || blocked[k])) {
      x.home = Home::Host;
      x.block = next_block++;
      if (!three) ++on_host_;
    } else if (!placed[k]) {
      x.home = Home::File;
      ++on_file_;
    }
  }
  for (size_t i = host_rank.size(); i-- > 0;) hpush(host_rank[i]);  // the first by priority most recent
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
  open_hhead_ = hhead_;
  open_htail_ = htail_;
  open_on_host_ = on_host_;
  open_on_file_ = on_file_;
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
      hpush(w.key);
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
  enter();
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
    drain_key(old);  // (its read, if one were still in flight: ended before the alias goes)
    to_lower(old);  // its block (two-tier: always), or its file (fetched from it)
    if (mut_ != kNoBankUnpublish) write_entry(DgTransport::kDem, old, -1);
  }
  bank_key_[idx] = kNone;
  return slot;
}

void DgCore::begin_token(uint32_t seq, const std::vector<uint8_t>* run) {
  require(!reset_failed_, SS_E_BACKEND,
          "device-gated: the cache failed (a reset or a make_resident); close the poisoned context");
  require(!run || run->size() == gated(), SS_E_INVALID, "device-gated: the exported layers do not cover the gated layers");
  enter();
  if (run) std::copy(run->begin(), run->end(), run_.begin());
  else std::fill(run_.begin(), run_.end(), uint8_t(1));
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
// Overlapped file misses: a fetch that does nothing (over budget, a key on the device, a file key a prefetch may not
// read) reads one home and changes nothing, so it keeps the reads in flight; a fetch of a coming key ends that key's
// read if it is in flight. Every other path of a miss keeps them too (pressure mode takes these on every miss):
//  - resurrection: its key is Evicting (a read's key never is) and its slot is that key's; it cancels / records its
//    own write-back and publishes its own entry;
//  - a copy from a pinned block: another key, its own slot and block (a host block, never a staging buffer);
//  - collect(): retires recorded write-backs only, whose keys are Evicting or resurrected (never a read's key: a key
//    read from its file has no write-back record, check()), their slots (each held once) and host blocks back to the
//    free lists, their events (recorded, never a read's) to the pool; done() is a host query;
//  - force_free_slot(): submits / waits on deferred write-backs (victims chosen at the last boundary, after every read
//    ended: Evicting keys, their slots, host blocks), then collect();
//  - a bank slot: unpublishes the slot's previous alias, a key of layer g - 2 (its reads ended at its ready; a
//    targeted drain covers it anyway).
// Drain points here: the oldest past the queue depth, take_stage (a staging buffer reused), the bank slot's alias,
// the no-slot fallback (an internal error), and everything with synchronous reads (disk_queue 1).
bool DgCore::fetch_impl(uint32_t g, uint32_t unit, bool prefetch, bool budgeted) {
  const uint32_t k = key(g, unit);
  Entry& x = e_[k];
  // a prefetch over budget does nothing at all (no copy and no resurrection); dg_no_admit issues none
  if (prefetch && budgeted && (pf_layer_[g] >= c_.prefetch_per_layer || pf_token_ >= c_.prefetch_per_token))
    return false;
  if (prefetch && c_.no_admit) return false;
  switch (x.home) {
    case Home::Coming:
      drain_key(k);  // (its own prefetch read in flight: ended; nothing else to do)
      return false;
    case Home::Device:
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
    case Home::File:  // three-tier: a file read below; a prefetch only with disk_prefetch
      if (prefetch && !c_.disk_prefetch) return false;
      break;
  }
  int32_t slot = -1;
  bool bank = false, free_counted = false;
  const size_t reserve = prefetch ? c_.prefetch_reserve : 0u;
  if (c_.no_admit) {  // a demand (prefetches return above): a bank slot, released at the boundary, its home unchanged
    slot = take_bank_slot(g);
    if (slot < 0) fail(SS_E_BACKEND, "device-gated: dg_no_admit: no bank slot for a miss of gated layer " +
                                         std::to_string(g) + " (more distinct misses than dg_max_ids)");
    bank = true;
  } else if (free_slots_.size() <= reserve || (pressure_ && !prefetch)) {
    st_.collected_mid += collect();
  }
  const bool capped = !prefetch && pressure_ && demand_free_used_ >= pressure_free_;
  if (!bank && !prefetch && !capped && free_slots_.empty()) force_free_slot();  // wb_defer only (else none deferred)
  if (bank) {
    // (dg_no_admit: taken above)
  } else if (!capped && free_slots_.size() > reserve) {
    if (!prefetch) ++demand_free_used_;
    free_counted = !prefetch;
    slot = free_slots_.back();
    free_slots_.pop_back();
  } else if (prefetch) {
    return false;
  } else {
    slot = take_bank_slot(g);
    if (slot >= 0) {
      bank = true;
    } else {  // a layer has at most bank_slots distinct misses: only reachable through an internal error
      drain_pending();
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
  const bool from_file = x.home == Home::File;
  if (from_file) {  // its bytes from the file into a staging buffer, then the same copy (once the read ended)
    const size_t depth = std::max<uint32_t>(c_.disk_queue, 1);
    int32_t stage = -1;
    void* h = nullptr;
    try {
      if (pending_.size() >= depth) drain_pending(pending_.size() + 1 - depth);  // the queue depth
      stage = take_stage();
      const auto t0 = std::chrono::steady_clock::now();
      h = t_->read_unit_begin(k, stage);
      st_.disk_read_ns += uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    } catch (...) {  // nothing begun for this key: its slot goes back (a bank slot stays used, unpublished)
      if (!bank) {
        free_slots_.push_back(slot);
        if (free_counted) --demand_free_used_;
      }
      throw;
    }
    if (!pending_.empty()) ++st_.disk_overlapped;
    pending_.push_back({k, slot, stage, s, prefetch, bank, free_counted, h, entries_});
    st_.disk_max_in_flight = std::max<uint64_t>(st_.disk_max_in_flight, pending_.size());
    table_[k] = slot;  // the mirror now (as a synchronous fetch), the device's entry with the copy
  } else {
    copy_in(s, k, slot, x.block);
  }
  if (x.home == Home::Host) hunlink(k);
  else if (x.home == Home::File) --on_file_;
  x.slot = slot;
  x.via_prefetch = prefetch;
  x.used = false;
  x.pf_kind = 0;
  x.pf_late = false;
  if (prefetch && !from_file) {  // (a file read's: with its copy)
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
    if (mut_ == kEarlySourceFree && x.block >= 0) free_blocks_.push_back(x.block);
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
  if (from_file && c_.disk_queue <= 1) drain_pending();  // synchronous reads: the copy now, as before
  return true;
}

void DgCore::cover(uint32_t g, uint32_t unit) {
  Entry& x = e_[key(g, unit)];
  // only this key's copies land before ready; drain point: this key's read if in flight (a file prefetch's event is
  // recorded with its copy). The wait is on this key's event only: the other reads stay in flight.
  if (x.home == Home::Coming && x.via_prefetch && mut_ != kNoCover) drain_key(key(g, unit));
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
  enter();
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

// Drain points: the start (a public entry point: the demand reads), cover() of a prefetch (the demand reads and that
// key's), signal() (this layer's reads, its misses' and its prefetches', in flight until its ready).
void DgCore::serve_demand(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask) {
  enter();
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
        drain_demand();  // (dg_diag: the mark follows the first fetched unit's copies)
        t_->diag_mark(g, 2);
        first_marked = true;
      }
    }
  }
  if (mask) signal(g);
  else drain_layer(g);  // (no miss: the layer's prefetch reads still end before it is done)
  if (diag_ && mask) t_->diag_mark(g, 1);
}

void DgCore::signal(uint32_t g) {
  drain_layer(g);  // drain point: this layer's copies and entries (demand, prefetch) enqueued before its ready
  t_->signal_ready(g, seq_);
}

void DgCore::serve_use(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask) {
  enter();
  const bool have_pred = pred_layer_ == g;
  uniq_.assign(ids, ids + n);
  std::sort(uniq_.begin(), uniq_.end());
  uniq_.erase(std::unique(uniq_.begin(), uniq_.end()), uniq_.end());
  if (c_.scored) score_history(g);  // the union saved before serve_prediction reuses uniq_
  for (uint32_t id : uniq_) {  // placement rounds: the unit's exports since the last round (saturating)
    uint32_t& h = rhits_[key(g, id)];
    if (h != UINT32_MAX) ++h;
  }
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

// Drain point: the start (a public entry point). Its prefetches' file reads stay in flight past the call: they end when
// layer g_next is demanded (cover of the key, at the latest its ready), or at a drain of everything.
void DgCore::serve_prediction(uint32_t g_next, const uint32_t* pred, uint32_t n, uint32_t n_export, bool ahead) {
  enter();
  serve_prediction_impl(g_next, pred, n, n_export, ahead);
}

void DgCore::serve_prediction_impl(uint32_t g_next, const uint32_t* pred, uint32_t n, uint32_t n_export, bool ahead) {
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
      if (!c_.disk_prefetch && e_[key(g_next, c.unit)].home == Home::File) continue;  // demand only (three-tier)
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
    if (issued) t_->flush(DgTransport::kPf);  // the copies start now (a file read's: flushed when it ends)
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
  drain_pending();  // drain point: the end of the token (before its streams are synchronized; victims are chosen
                    // below with no read in flight)
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
  std::vector<uint8_t> keep_mask;
  if (!keep.empty()) {
    keep_mask.assign(keys_, 0);
    for (uint32_t k : keep)
      if (k < keys_) keep_mask[k] = 1;
  }
  auto kept = [&](uint32_t k) { return !keep_mask.empty() && keep_mask[k] && mut_ != kIgnoreKeep; };
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
    if (wp_.enable && wp_.revert_unused && hinted && x.via_prefetch && !x.used && !kept(k)) {
      // winpf: an unused prefetch goes back to its source block, which still holds the bytes (a Coming unit's block
      // is freed only here, at admission): no write-back. The device is done with the token and kPf is synced; the
      // table upload below unpublishes it.
      if (x.pf_kind) ++(x.pf_kind == 1 ? st_.prior_reverted : st_.l1_reverted);
      free_slots_.push_back(x.slot);
      to_lower(k);  // its block, or (read from its file) its file
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
    if (x.block >= 0) {  // (a unit read from its file holds none)
      if (backed_ < c_.inclusive) {  // inclusive share: the source block stays, a clean copy of the slot's bytes
        x.backed = true;
        ++backed_;
        ++st_.inclusive_kept;
      } else {
        free_blocks_.push_back(x.block);
        x.block = -1;
      }
    }
  }
  coming_.clear();
  for (uint32_t k : banked_) {  // bank units: back to their blocks (the upload below resets the entry)
    Entry& x = e_[k];
    if (x.home == Home::Bank) {
      to_lower(k);
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
  // the device, coverage kept). Only the accepted list feeds the write-backs below. (A kept unused prefetch was admitted
  // above, not reverted: a kept unit on the device stays there.)
  // dg_no_admit: nothing was admitted, no boundary evicts (placement changes only through rounds / make_resident)
  const bool select = !c_.no_admit;
  const bool want_score = select && c_.scored && (!c_.scored_auto || rows > 1);
  // A layer this token did not export (begin_token's run) was never served or ranked: it is ranked here from its
  // unchanged history (the bounded selector only needs it closed), so every list is this token's as after a full one.
  for (uint32_t g = 0; c_.scored && g < gated(); ++g)
    if (!run_[g]) {
      if (c_.boundary_score) sc_[g].list_epoch = epoch_;
      else rank_layer(g);
    }
  if (probe_) probe_();
  const size_t base = free_slots_.size() + pending_writebacks();
  victims_.clear();
  size_t got = 0;  // accepted victims: write-backs (victims_) and three-tier / inclusive drops (no copy)
  bool short_blocks = false;
  auto eligible = [&](uint32_t k) {
    const Entry& x = e_[k];
    return x.home == Home::Device && !(x.wb_pending && mut_ != kReevictPending) && !kept(k);
  };
  auto accept = [&](uint32_t k) {
    Entry& x = e_[k];
    if (x.backed || (free_blocks_.empty() && !drop_host_lru() && c_.disk_tier)) {
      // a clean copy (its block is its home again) or, three-tier with no block to free, its file: no copy
      if (x.backed) ++st_.inclusive_drops;
      else ++st_.disk_discards;
      unlink(k);
      --on_device_;
      free_slots_.push_back(x.slot);
      to_lower(k);
      table_[k] = -1;
      ++st_.evicted;
      ++lc_[gated_of(k)].evicted;
      ++got;
      return true;
    }
    if (free_blocks_.empty()) {
      ++st_.evict_short;
      short_blocks = true;
      return false;
    }
    const int32_t b = free_blocks_.back();
    free_blocks_.pop_back();
    unlink(k);
    --on_device_;
    x.home = Home::Evicting;
    x.block = b;
    x.wb_pending = true;
    table_[k] = -1;
    victims_.push_back(k);
    ++got;
    return true;
  };
  bool by_score = want_score;
  for (uint32_t g = 0; by_score && g < gated(); ++g) by_score = sc_[g].list_epoch == epoch_;
  if (want_score && !by_score) ++st_.score_fallbacks;  // invalid score metadata: this boundary uses the recency order
  if (by_score && c_.boundary_score) {
    // Same victims as the public sorted lists + merge: only eligibility can change after ranking, never a key's
    // score / epochs / stamp. At the boundary score only eligible keys and retain the needed smallest candidates.
    const auto start = std::chrono::steady_clock::now();
    const size_t need = c_.spare_target > base ? c_.spare_target - base : 0;
    select_scored(need, eligible, true);
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
    while (!heap_.empty() && base + got < c_.spare_target) {
      std::pop_heap(heap_.begin(), heap_.end(), after);
      const uint32_t g = heap_.back();
      const uint32_t k = head(g).key;
      if (++heap_pos_[g] < sc_[g].list.size()) std::push_heap(heap_.begin(), heap_.end(), after);
      else heap_.pop_back();
      if (!eligible(k)) continue;
      if (!accept(k)) break;
      ++st_.scored_victims;
    }
  } else if (select) {
    for (uint32_t k = tail_; k != kNone && base + got < c_.spare_target;) {
      const uint32_t prev = e_[k].prev;
      if (eligible(k) && !accept(k)) break;
      k = prev;
    }
  }
  if (select && pressure_ && pressure_recent_ && !short_blocks) {  // the most recently used too (they come back soon)
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
  enter();
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
  rank_layer(g);
}

void DgCore::rank_layer(uint32_t g) {
  LayerScore& l = sc_[g];
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

// The bounded selection (the boundary's, and make_resident's under scoring): the `need` smallest eligible candidates of
// every gated layer by the public total order, ascending in select_heap_. The same victims as the per-layer lists +
// merge: a layer's candidates are scored from its current union, epochs and stamps.
void DgCore::select_scored(size_t need, const std::function<bool(uint32_t)>& eligible, bool count) {
  auto less = [this](const ScoreCand& a, const ScoreCand& b) { return cand_less(a, b); };
  select_heap_.clear();
  for (uint32_t g = 0; need && g < gated(); ++g) {
    uint64_t D = 0;
    for (uint32_t r : sc_[g].cur) D += sc_[g].rowsum[r];
    if (count) ++st_.score_layers;
    for (uint32_t u = 0; u < units(g); ++u) {
      if (!eligible(base_[g] + u)) continue;
      if (count) ++st_.score_candidates;
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
  drain_pending();  // drain point: between tokens, the policy (everything settled)
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
  drain_pending();  // drain point: open only
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
  drain_pending();  // drain point: a setting (between tokens)
  if (!rate_set_ || w.copy_gbps != wp_.copy_gbps) {  // a learned rate survives phase toggles with the same seed rate
    rate_gbps_ = w.copy_gbps;
    rate_set_ = true;
  }
  wp_ = w;
}

void DgCore::observe_copy(double unit_ms) {
  drain_pending();  // drain point: a setting (between tokens)
  if (!(unit_ms > 0) || !unit_bytes_) return;
  const double g = double(unit_bytes_) / (unit_ms * 1e6);
  if (g < 1 || g > 200) return;  // a stamp that cannot be a copy
  rate_gbps_ = 0.9 * rate_gbps_ + 0.1 * g;
}

// Drain point: the start (a public entry point). issue_prior's file reads stay in flight into the token (they end
// when their layer is demanded, or at a drain of everything).
void DgCore::window_hint(const int32_t* symbols, const float* weights, uint32_t rows) {
  enter();
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
      if (!c_.disk_prefetch && e_[key(g, x.unit)].home == Home::File) continue;  // demand only (three-tier)
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
  drain_pending();  // drain point: between tokens (copies, then an upload)
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
    hunlink(k);
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

// ss_dg_make_resident (dg_core.hpp). Every refusal comes before any action; the rest is synchronous: every copy landed
// and the table is uploaded when it returns. A victim's slot is taken only after its write-back landed (kD2h synced), a
// copy's source block is freed only after kDem is synced, a staging buffer is reused only after its last copy's event.
DgCore::ResidentResult DgCore::make_resident(uint32_t g, const std::vector<uint32_t>& ids,
                                             const std::vector<uint32_t>& other_kept) {
  const auto t0 = std::chrono::steady_clock::now();
  require(!reset_failed_, SS_E_BACKEND,
          "device-gated: the cache failed (a reset or a make_resident); close the poisoned context");
  require(g < gated(), SS_E_INVALID, "ss_dg_make_resident: gated layer out of range");
  std::vector<uint8_t> want(keys_, 0), other(keys_, 0);
  for (uint32_t u : ids) {
    require(u < units(g), SS_E_INVALID, "ss_dg_make_resident: id out of range");
    require(!want[key(g, u)], SS_E_INVALID, "ss_dg_make_resident: duplicate id " + std::to_string(u));
    want[key(g, u)] = 1;
  }
  size_t n_other = 0;
  for (uint32_t k : other_kept)  // (g's own current keep included: protected until the call succeeded)
    if (k < keys_ && !other[k] && !want[k]) {
      other[k] = 1;
      ++n_other;
    }
  // capacity: the kept units of every layer leave the spare target free for the boundaries
  const uint64_t cap = uint64_t(c_.persistent_slots) - c_.spare_target;
  require(ids.size() + n_other <= cap, SS_E_CAPACITY,
          "ss_dg_make_resident: " + std::to_string(ids.size()) + " units of gated layer " + std::to_string(g) + " + " +
              std::to_string(n_other) + " other kept units + the spare target " + std::to_string(c_.spare_target) +
              " exceed the " + std::to_string(c_.persistent_slots) + " persistent slots");
  // the slots it can use: free ones, the ones write-backs in flight will free, eligible victims (prior prefetches still
  // coming and the other layers' kept units are not)
  size_t need = 0, eligible_n = 0, evicting = 0, discarded = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    require(x.home != Home::Bank, SS_E_BACKEND, "ss_dg_make_resident: a bank unit between tokens");
    if (x.home == Home::Evicting) ++evicting;
    if (want[k]) need += x.home != Home::Device && x.home != Home::Coming ? 1u : 0u;
    else if (x.home == Home::Device && !other[k]) ++eligible_n;
  }
  for (const Writeback& w : wbs_) discarded += w.discard ? 1u : 0u;
  const size_t free_after = free_slots_.size() + evicting;
  require(need <= free_after + eligible_n, SS_E_CAPACITY,
          "ss_dg_make_resident: " + std::to_string(need) + " units to copy in, " + std::to_string(free_after) +
              " free slots and " + std::to_string(eligible_n) +
              " evictable units (the rest hold other layers' kept units or prefetches in flight)");
  require(c_.disk_tier || need <= free_after || free_after + free_blocks_.size() + discarded > 0, SS_E_CAPACITY,
          "ss_dg_make_resident: no free slot and no free pinned block to start the write-backs");
  stamp_room("ss_dg_make_resident");

  // drain point: everything settled (reads, copies, deferred and recorded write-backs collected); a device failure
  // here poisons the core (nothing of this call changed yet)
  try {
    drain();
    for (const Writeback& w : wbs_) t_->event_sync(w.done);
    collect();
    require(wbs_.empty(), SS_E_BACKEND, "ss_dg_make_resident: write-backs not collected");
  } catch (...) {
    reset_failed_ = true;
    throw;
  }
  ResidentResult r;
  res_victims_.clear();
  std::vector<uint32_t> in;  // the units to copy in, in the caller's order
  for (uint32_t u : ids) {
    const uint32_t k = key(g, u);
    Entry& x = e_[k];
    if (x.home == Home::Coming) {  // a prior prefetch (its copy landed: kPf synced): admitted now, never reverted
      x.home = Home::Device;
      x.via_prefetch = false;
      x.pf_kind = 0;
      x.pf_late = false;
      if (x.pf_done) {
        event_pool_.push_back(x.pf_done);
        x.pf_done = nullptr;
      }
      if (x.block >= 0) {
        free_blocks_.push_back(x.block);
        x.block = -1;
      }
    } else if (x.home != Home::Device) {
      in.push_back(k);
    }
  }
  auto eligible = [&](uint32_t k) {
    const Entry& x = e_[k];
    return x.home == Home::Device && !x.wb_pending && !want[k] && !other[k];
  };
  std::vector<uint32_t> victims;
  const size_t n_vict = in.size() > free_slots_.size() ? in.size() - free_slots_.size() : 0;
  if (n_vict && c_.scored) {
    select_scored(n_vict, eligible, false);
    for (const ScoreCand& cd : select_heap_) victims.push_back(cd.key);
  } else {
    for (uint32_t k = tail_; k != kNone && victims.size() < n_vict; k = e_[k].prev)
      if (eligible(k)) victims.push_back(k);
  }
  require(victims.size() == n_vict, SS_E_BACKEND, "ss_dg_make_resident: fewer victims than the capacity check counted");

  std::vector<uint8_t> hold = want;  // never dropped to a file here: the units to copy in, this call's write-backs
  // Transactions: a copy changes no home until the transport accepted it (the slot back to the free list and the
  // mirror entry back to -1 otherwise: the source keeps its home and its block); a source block is released, and a
  // written-back victim's slot reused, only after the stream that reads it was synced.
  std::vector<int32_t> released;  // source blocks of committed copies on kDem (free once kDem is synced)
  std::vector<int32_t> wb;        // slots of committed write-backs on kD2h (free once kD2h is synced)
  struct Read {
    uint32_t key;
    int32_t slot, stage;
    void* handle;
  };
  std::deque<Read> reads;  // file reads in flight, oldest first, each in its own staging buffer
  const size_t depth = c_.disk_tier ? std::max<size_t>(1, std::min<size_t>(std::max<uint32_t>(c_.disk_queue, 1),
                                                                          c_.disk_staging))
                                    : 1;
  bool io_error = false;  // the failure was a file read's: the device is fine, the cache is re-established
  auto commit = [&](uint32_t k, int32_t slot) {
    Entry& x = e_[k];
    x.home = Home::Device;
    x.slot = slot;
    x.block = -1;
    push_front(k);  // the most recent
    ++on_device_;
    ++r.copied;
  };
  auto transfer = [&](uint32_t k, int32_t slot, int32_t src) {  // the copy and the entry, or nothing owned changes
    try {
      copy_in(DgTransport::kDem, k, slot, src);
    } catch (...) {
      table_[k] = -1;
      free_slots_.push_back(slot);
      throw;
    }
  };
  auto flush_dem = [&] {
    t_->sync(DgTransport::kDem);
    free_blocks_.insert(free_blocks_.end(), released.begin(), released.end());
    released.clear();
  };
  auto flush_d2h = [&] {
    if (wb.empty()) return;
    t_->sync(DgTransport::kD2h);
    free_slots_.insert(free_slots_.end(), wb.begin(), wb.end());
    wb.clear();
  };
  auto end_read = [&] {
    const Read rd = reads.front();
    reads.pop_front();
    try {
      t_->read_unit_end(rd.handle);
    } catch (...) {
      io_error = true;
      free_slots_.push_back(rd.slot);
      throw;
    }
    transfer(rd.key, rd.slot, rd.stage);
    --on_file_;
    commit(rd.key, rd.slot);
    void*& ev = stage_ev_[uint32_t(rd.stage) - c_.blocks];  // the staging buffer is reusable once this copy landed
    if (!ev) ev = event();
    t_->record(ev, DgTransport::kDem);
  };
  auto account = [&] {
    st_.resident_units += r.copied;
    st_.resident_evicted += r.evicted;
    st_.resident_bytes += uint64_t(r.copied) * unit_bytes_;
    st_.resident_ns += uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
  };
  try {
    size_t next = 0, vi = 0;
    while (next < in.size()) {
      // no free slot: evict the next victims (as many as the remaining units need and the blocks allow)
      const bool evict = free_slots_.empty();
      while (evict && vi < victims.size() && free_slots_.size() + wb.size() < in.size() - next) {
        const uint32_t v = victims[vi];
        Entry& x = e_[v];
        if (!x.backed && free_blocks_.empty() && !released.empty()) {  // the blocks of the copies already enqueued
          flush_dem();
          continue;
        }
        if (!x.backed && free_blocks_.empty() && !c_.disk_tier) break;  // (two-tier: progress is guaranteed above)
        // a clean copy (its block is its home again) or, three-tier with no block to free, its file: no copy
        const bool no_copy = x.backed || (free_blocks_.empty() && !drop_host_lru(&hold, false));
        int32_t b = -1;
        if (!no_copy) {
          b = free_blocks_.back();
          t_->d2h(x.slot, b);  // the dirty unit's only copy: written back before its slot is reused
          free_blocks_.pop_back();
        }
        ++vi;
        res_victims_.push_back(v);
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
      flush_d2h();
      require(!free_slots_.empty(), SS_E_BACKEND, "ss_dg_make_resident: no slot for a unit");
      const int32_t slot = free_slots_.back();
      free_slots_.pop_back();
      const uint32_t k = in[next];
      Entry& x = e_[k];
      if (x.home == Home::Host) {
        const int32_t b = x.block;
        transfer(k, slot, b);
        ++next;
        hunlink(k);
        released.push_back(b);
        commit(k, slot);
        continue;
      }
      // its file, through the next staging buffer (its last copy landed first)
      try {
        if (reads.size() >= depth) end_read();
        const uint32_t i = stage_next_;
        stage_next_ = (i + 1) % c_.disk_staging;
        if (stage_ev_[i] && !t_->done(stage_ev_[i])) t_->event_sync(stage_ev_[i]);
        const int32_t stage = int32_t(c_.blocks + i);
        void* h = nullptr;
        try {
          h = t_->read_unit_begin(k, stage);
        } catch (...) {
          io_error = true;
          throw;
        }
        reads.push_back({k, slot, stage, h});
        ++next;
      } catch (...) {
        free_slots_.push_back(slot);
        throw;
      }
    }
    while (!reads.empty()) end_read();
    flush_d2h();
    flush_dem();
    t_->upload_table(table_.data(), keys_);  // (syncs kDem) every entry, the victims' -1 included
  } catch (...) {
    // reads not ended give their slots back (their keys never changed home); the units moved so far stay
    for (const Read& rd : reads) {
      try {
        t_->read_unit_end(rd.handle);
      } catch (...) {
      }
      free_slots_.push_back(rd.slot);
    }
    reads.clear();
    bool ok = false;
    if (io_error) {  // the device is fine: what was enqueued lands, the table goes up
      try {
        flush_d2h();
        flush_dem();
        t_->upload_table(table_.data(), keys_);
        ok = true;
      } catch (...) {
      }
    }
    if (!ok) {
      // a device failure (or one while re-establishing): nothing more is enqueued. The host records stay one
      // consistent ownership (every slot, block and event held once), but what the device holds is unknown: the core
      // is poisoned, and so is its context.
      reset_failed_ = true;
      free_slots_.insert(free_slots_.end(), wb.begin(), wb.end());
      wb.clear();
      free_blocks_.insert(free_blocks_.end(), released.begin(), released.end());
      released.clear();
    }
    account();
    throw;
  }
  account();
  return r;
}

// ss_dg_round (dg_core.hpp, docs/design.md §3.10). Every move is synchronous: its copy and its stream synced before the
// home changes, so a cancellation, a read failure or a device failure finds every unit in exactly one home.
DgRoundOut DgCore::round(const DgRoundIn& in, const std::vector<uint32_t>& kept_keys) {
  const auto t0 = std::chrono::steady_clock::now();
  last_round_ = DgRoundOut{};
  require(!reset_failed_, SS_E_BACKEND,
          "device-gated: the cache failed (a reset or a make_resident); close the poisoned context");
  require(std::isfinite(in.decay) && in.decay >= 0 && in.decay <= 1, SS_E_INVALID, "ss_dg_round: decay must be 0..1");
  require(std::isfinite(in.hysteresis) && in.hysteresis >= 1 && in.hysteresis <= 1e6, SS_E_INVALID,
          "ss_dg_round: hysteresis must be 1..1e6");
  require(in.flags == 0, SS_E_INVALID, "ss_dg_round: unknown flags");
  std::vector<uint8_t> kept(keys_, 0);
  size_t n_kept = 0;
  for (uint32_t k : kept_keys)
    if (k < keys_ && !kept[k]) {
      kept[k] = 1;
      ++n_kept;
    }
  for (uint32_t k = 0; k < keys_; ++k)
    require(e_[k].home != Home::Bank, SS_E_BACKEND, "ss_dg_round: a bank unit between tokens");
  stamp_room("ss_dg_round");
  round_cancel_.store(false, std::memory_order_release);
  // drain point: everything settled (reads, copies, deferred and recorded write-backs collected); a device failure
  // here poisons the core (nothing of this call changed yet)
  try {
    drain();
    for (const Writeback& w : wbs_) t_->event_sync(w.done);
    collect();
    require(wbs_.empty(), SS_E_BACKEND, "ss_dg_round: write-backs not collected");
  } catch (...) {
    reset_failed_ = true;
    throw;
  }
  for (uint32_t k = 0; k < keys_; ++k) {  // the decayed counts
    rs_[k] = in.decay * rs_[k] + double(rhits_[k]);
    rhits_[k] = 0;
  }

  DgRoundOut r;
  const uint64_t ub = unit_bytes_;
  uint64_t spent = 0;
  uint32_t moved = 0;
  auto fits = [&](uint64_t b) { return !in.budget_bytes || (spent <= in.budget_bytes && b <= in.budget_bytes - spent); };
  auto cancelled = [&] {
    if (round_cancel_.load(std::memory_order_acquire)) r.cancelled = 1;
    return r.cancelled != 0;
  };
  auto boundary = [&] {  // a unit boundary: the cancellation check (the test probe runs first)
    if (round_probe_) round_probe_(moved);
    return cancelled();
  };
  // tie-breaks: the recency positions (least recent first) of the device units and of the pinned homes
  std::vector<uint32_t> rank(keys_, 0), hrank(keys_, 0);
  {
    uint32_t i = 0;
    for (uint32_t k = tail_; k != kNone; k = e_[k].prev) rank[k] = i++;
  }
  auto hotter = [&](uint32_t a, uint32_t b) { return rs_[a] != rs_[b] ? rs_[a] > rs_[b] : a < b; };
  auto colder_dev = [&](uint32_t a, uint32_t b) {
    if (rs_[a] != rs_[b]) return rs_[a] < rs_[b];
    return rank[a] != rank[b] ? rank[a] < rank[b] : a < b;
  };
  auto colder_host = [&](uint32_t a, uint32_t b) {
    if (rs_[a] != rs_[b]) return rs_[a] < rs_[b];
    return hrank[a] != hrank[b] ? hrank[a] < hrank[b] : a < b;
  };
  std::vector<uint32_t> dev, host, file;
  size_t n_coming = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    if (x.home == Home::Coming) ++n_coming;
    if (kept[k] || x.wb_pending) continue;  // never moved
    if (x.home == Home::Device) dev.push_back(k);
    else if (x.home == Home::Host) host.push_back(k);
    else if (x.home == Home::File) file.push_back(k);
  }
  std::sort(dev.begin(), dev.end(), colder_dev);
  std::sort(host.begin(), host.end(), hotter);
  std::sort(file.begin(), file.end(), hotter);
  // the device's share of the rank: persistent slots - spares - kept units (- units coming from a prior prefetch)
  int64_t room = int64_t(c_.persistent_slots) - int64_t(c_.spare_target) - int64_t(n_kept) - int64_t(n_coming) -
                 int64_t(dev.size());
  const DgTransport::Stream ws = c_.wb_on_d2h ? DgTransport::kD2h : DgTransport::kDem;

  struct Rd {
    uint32_t key;
    int32_t block;
    void* handle;
  };
  std::deque<Rd> reads;  // (b) file reads in flight, oldest first, each into its own block
  bool io_error = false;
  auto account = [&] {
    r.ns = uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    ++st_.round_count;
    st_.round_units += uint64_t(r.promoted_to_device) + r.promoted_to_pinned + r.demoted_from_device + r.dropped_to_file;
    st_.round_bytes += r.bytes_read + r.bytes_h2d + r.bytes_written_back;
    st_.round_ns += r.ns;
    st_.round_cancelled += r.cancelled;
    last_round_ = r;
  };
  // A pinned unit's copy into `slot` (owned by the caller): committed once the copy and its entry were accepted and
  // kDem synced; a cancellation seen then drops it (the slot freed, the entry back to -1, the unit keeps its block).
  auto promote = [&](uint32_t k, int32_t slot) {
    Entry& x = e_[k];
    const int32_t b = x.block;
    try {
      copy_in(DgTransport::kDem, k, slot, b);
      t_->sync(DgTransport::kDem);
    } catch (...) {
      table_[k] = -1;
      free_slots_.push_back(slot);
      throw;
    }
    spent += ub;
    r.bytes_h2d += ub;
    if (cancelled()) {
      table_[k] = -1;
      free_slots_.push_back(slot);
      return false;
    }
    hunlink(k);
    free_blocks_.push_back(b);  // (kDem synced: the copy that read it landed)
    x.block = -1;
    x.slot = slot;
    x.home = Home::Device;
    push_front(k);  // the most recent
    ++on_device_;
    ++r.promoted_to_device;
    return true;
  };
  // A block for a dirty device unit's write-back: a free one, else (three-tier) the coldest pinned unit's, dropped to
  // its file (never `except`, the unit being promoted). -1: none.
  size_t hback = host.size();
  auto wb_block = [&](uint32_t except) -> int32_t {
    if (!free_blocks_.empty()) {
      const int32_t b = free_blocks_.back();
      free_blocks_.pop_back();
      return b;
    }
    if (!c_.disk_tier) return -1;
    while (hback > 0) {
      const uint32_t p = host[--hback];
      Entry& y = e_[p];
      if (p == except || y.home != Home::Host) continue;
      hunlink(p);
      const int32_t b = y.block;
      y.block = -1;
      y.home = Home::File;
      ++on_file_;
      ++r.dropped_to_file;
      return b;
    }
    return -1;
  };
  auto end_read = [&] {
    const Rd rd = reads.front();
    reads.pop_front();
    try {
      t_->read_unit_end(rd.handle);
    } catch (...) {
      io_error = true;
      free_blocks_.push_back(rd.block);
      throw;
    }
    r.bytes_read += ub;
    if (cancelled()) {  // dropped: the block is free again, the unit stays in its file
      free_blocks_.push_back(rd.block);
      return;
    }
    Entry& x = e_[rd.key];
    --on_file_;
    x.home = Home::Host;
    x.block = rd.block;
    hpush(rd.key);  // the most recently homed
    ++r.promoted_to_pinned;
  };
  try {
    // (a) pinned -> device
    size_t j = 0;
    for (size_t i = 0; i < host.size() && moved < in.max_units; ++i) {
      const uint32_t c = host[i];
      if (!(rs_[c] > 0)) break;
      if (e_[c].home != Home::Host) continue;  // (dropped to its file for a write-back block)
      if (boundary()) break;
      if (room > 0 && free_slots_.size() > c_.spare_target) {  // the device holds less than its share
        if (!fits(ub)) break;
        const int32_t slot = free_slots_.back();
        free_slots_.pop_back();
        if (!promote(c, slot)) break;
        --room;
        ++moved;
        continue;
      }
      if (j >= dev.size()) break;
      const uint32_t v = dev[j];
      if (!(rs_[c] > in.hysteresis * rs_[v])) break;
      Entry& y = e_[v];
      if (!fits(ub + (y.backed ? 0 : ub))) break;
      const int32_t slot = y.slot;
      if (y.backed) {  // a clean copy: its block is its home again, no write
        unlink(v);
        --on_device_;
        table_[v] = -1;
        to_lower(v);
      } else {
        const int32_t b = wb_block(c);
        if (b < 0) break;
        try {
          if (ws == DgTransport::kD2h) {
            t_->d2h(slot, b);
          } else {
            for (uint32_t p = 0; p < t_->parts(); ++p) t_->d2h_part(DgTransport::kDem, slot, b, p);
          }
          t_->sync(ws);
        } catch (...) {
          free_blocks_.push_back(b);  // the unit keeps its slot (its only copy)
          throw;
        }
        unlink(v);
        --on_device_;
        table_[v] = -1;
        y.home = Home::Host;
        y.slot = -1;
        y.block = b;
        hpush(v);
        spent += ub;
        r.bytes_written_back += ub;
      }
      ++j;
      ++r.demoted_from_device;
      if (!promote(c, slot)) break;  // (cancelled: the demoted unit's slot is free)
      ++moved;
    }
    // (b) three-tier: file -> pinned, in place of the coldest pinned units
    if (c_.disk_tier && !r.cancelled) {
      std::vector<uint32_t> pinned;
      for (uint32_t k = 0; k < keys_; ++k)
        if (e_[k].home == Home::Host && !kept[k] && !e_[k].wb_pending) pinned.push_back(k);
      uint32_t hi = 0;
      for (uint32_t k = htail_; k != kNone; k = e_[k].hprev) hrank[k] = hi++;  // after (a)'s pushes
      std::sort(pinned.begin(), pinned.end(), colder_host);
      const size_t depth = std::max<uint32_t>(c_.disk_queue, 1);
      size_t pv = 0;
      for (size_t i = 0; i < file.size() && moved < in.max_units && pv < pinned.size(); ++i) {
        const uint32_t c = file[i];
        if (!(rs_[c] > 0)) break;
        if (e_[c].home != Home::File) continue;
        if (boundary()) break;
        const uint32_t p = pinned[pv];
        if (!(rs_[c] > in.hysteresis * rs_[p])) break;
        if (!fits(ub)) break;
        if (reads.size() >= depth) end_read();
        if (r.cancelled) break;
        ++pv;
        Entry& y = e_[p];  // dropped to its file (no copy): its block is quiescent
        hunlink(p);
        const int32_t b = y.block;
        y.block = -1;
        y.home = Home::File;
        ++on_file_;
        ++r.dropped_to_file;
        void* h = nullptr;
        try {
          h = t_->read_unit_begin(c, b);
        } catch (...) {
          io_error = true;
          free_blocks_.push_back(b);
          throw;
        }
        reads.push_back({c, b, h});
        spent += ub;
        ++moved;
      }
      while (!reads.empty()) end_read();
    }
    t_->upload_table(table_.data(), keys_);  // (syncs kDem) every entry, the demoted units' -1 included
  } catch (...) {
    for (const Rd& rd : reads) {  // reads not ended: their blocks free again, their units still in their files
      try {
        t_->read_unit_end(rd.handle);
      } catch (...) {
      }
      free_blocks_.push_back(rd.block);
    }
    reads.clear();
    bool ok = false;
    if (io_error) {  // the device is fine (every copy was synced): the table goes up
      try {
        t_->upload_table(table_.data(), keys_);
        ok = true;
      } catch (...) {
      }
    }
    if (!ok) reset_failed_ = true;  // a device failure: the host records are one ownership, the device unknown
    account();
    throw;
  }
  account();
  return r;
}

void DgCore::seed_round_counts(const std::vector<uint32_t>& keys, const std::vector<float>& counts) {
  require(keys.size() == counts.size(), SS_E_INVALID, "ss_dg_round_seed_counts: keys and counts differ in length");
  for (size_t i = 0; i < keys.size(); ++i) {
    require(keys[i] < keys_, SS_E_INVALID, "ss_dg_round_seed_counts: unit out of range");
    require(std::isfinite(counts[i]) && counts[i] >= 0, SS_E_INVALID,
            "ss_dg_round_seed_counts: counts must be finite and >= 0");
  }
  for (size_t i = 0; i < keys.size(); ++i) rs_[keys[i]] = double(counts[i]);
}

void DgCore::drain() {
  drain_pending();  // drain point: a public entry point (between tokens)
  while (!deferred_.empty()) finish_deferred_front();
  t_->sync(DgTransport::kDem);
  t_->sync(DgTransport::kPf);
  t_->sync(DgTransport::kD2h);
}

bool DgCore::same_open_state(const DgCore& b) const {
  auto cfg = [](const DgConfig& c) {
    return std::tie(c.n_units, c.persistent_slots, c.bank_slots, c.blocks, c.spare_target, c.prefetch_reserve,
                    c.prefetch_reserve_auto, c.prefetch_per_layer, c.prefetch_per_token, c.wb_defer, c.wb_on_d2h,
                    c.wb_layer_bytes, c.wb_max_outstanding, c.disk_tier, c.disk_prefetch, c.home_blocks,
                    c.disk_staging, c.inclusive, c.disk_queue);
  };
  auto wp = [](const DgWinpf& w) {
    return std::tie(w.enable, w.prior_layers, w.prior_cap, w.prior_pmin, w.l1_cap, w.l1_min_votes, w.l1_pmin,
                    w.revert_unused, w.top_k, w.prior_window_us, w.copy_gbps);
  };
  auto entry = [](const Entry& x) {
    return std::tie(x.home, x.slot, x.block, x.prev, x.next, x.in_list, x.wb_pending, x.via_prefetch, x.used,
                    x.pf_kind, x.pf_late, x.pf_bucket, x.demand_epoch, x.lru_stamp, x.backed, x.in_hlist, x.hprev,
                    x.hnext);
  };
  if (cfg(c_) != cfg(b.c_) || wp(wp_) != wp(b.wp_) || e_.size() != b.e_.size()) return false;
  if (std::tie(hhead_, htail_, on_host_, on_file_, backed_) !=
      std::tie(b.hhead_, b.htail_, b.on_host_, b.on_file_, b.backed_))
    return false;
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
  if (rs_ != b.rs_ || rhits_ != b.rhits_ || c_.no_admit != b.c_.no_admit) return false;
  return wbs_.empty() && b.wbs_.empty() && deferred_.empty() && b.deferred_.empty() && pending_.empty() &&
         b.pending_.empty() &&
         prior_.layers == b.prior_.layers && prior_.k == b.prior_.k && prior_.row == b.prior_.row &&
         prior_.unit == b.prior_.unit && prior_.p == b.prior_.p;
}

void DgCore::reset_open() {
  require(!reset_failed_, SS_E_BACKEND,
          "device-gated: the cache failed (a reset or a make_resident); close the poisoned context");
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
    for (void* ev : stage_ev_)  // three-tier: every staging copy landed (Coming keys from files have no block)
      if (ev) t_->event_sync(ev);
    // Coming keys fetched from their file have no source block: their persistent slot is their only copy now (they
    // were drained), so they take part as device keys; clean copies are dropped (the open holds none).
    const size_t P = c_.persistent_slots, N = P + c_.blocks, stage = N, file = N + 1;
    std::vector<size_t> current(keys_), target(keys_);
    std::vector<uint32_t> occupant(N + 2, kNone), wanted(N + 2, kNone);
    for (uint32_t k = 0; k < keys_; ++k) {
      const Entry& x = e_[k];
      require(x.home != Home::Evicting, SS_E_BACKEND, "device-gated: reset left an evicting key");
      // (a bank slot is not a persistent location: a bank key read from its file goes back to its file)
      const bool on_slot = x.home == Home::Device || (x.block < 0 && x.home == Home::Coming);
      const bool from_file = x.home == Home::File || (x.block < 0 && x.home == Home::Bank);
      current[k] = from_file ? file : on_slot ? size_t(x.slot) : P + size_t(x.block);
      const Entry& o = open_e_[k];
      target[k] = o.home == Home::File ? file : o.home == Home::Device ? size_t(o.slot) : P + size_t(o.block);
      if (current[k] != file) occupant[current[k]] = k;
      if (target[k] != file) wanted[target[k]] = k;
    }
    for (uint32_t k = 0; k < keys_; ++k)  // a key going back to its file vacates its location (no copy)
      if (target[k] == file && current[k] != file) {
        occupant[current[k]] = kNone;
        current[k] = file;
      }
    auto block = [&](size_t loc) { return loc == stage ? -1 : int32_t(loc - P); };
    auto move = [&](uint32_t k, size_t dst) {
      const size_t src = current[k];
      require(dst < file && occupant[dst] == kNone, SS_E_BACKEND, "device-gated: reset overwrote a live source");
      if (src == file && dst < P) {  // read into the reset staging buffer, then to the slot
        t_->read_unit(k, -1);
        t_->h2d(DgTransport::kDem, int32_t(dst), -1);
        t_->sync(DgTransport::kDem);
      } else if (src == file) t_->read_unit(k, block(dst));
      else if (src < P && dst < P) t_->d2d(int32_t(src), int32_t(dst));
      else if (src < P) {
        t_->d2h(int32_t(src), block(dst));
        t_->sync(DgTransport::kD2h);
      } else if (dst < P) {
        t_->h2d(DgTransport::kDem, int32_t(dst), block(src));
        t_->sync(DgTransport::kDem);
      } else t_->h2h(block(src), block(dst));
      if (src != file) occupant[src] = kNone;
      occupant[dst] = k;
      current[k] = dst;
    };
    // Paths first: fill each target hole from its current source, propagating the hole. Each move fixes one key.
    // (A key read from its file ends its path: nothing was vacated.)
    for (size_t loc = 0; loc < N; ++loc) {
      size_t hole = loc;
      while (hole < N && occupant[hole] == kNone && wanted[hole] != kNone) {
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
      require(current[k] != file, SS_E_BACKEND, "device-gated: reset left a file read behind");
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
    hhead_ = open_hhead_;
    htail_ = open_htail_;
    on_host_ = open_on_host_;
    on_file_ = open_on_file_;
    backed_ = 0;
    stage_next_ = 0;
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
    std::fill(run_.begin(), run_.end(), uint8_t(1));
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
    std::fill(rs_.begin(), rs_.end(), 0.0);  // placement rounds: the counts start over (seed them again)
    std::fill(rhits_.begin(), rhits_.end(), 0u);
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
  drain_pending();  // drain point: a setting (between tokens)
  pressure_ = free_per_token != 0 || evict_recent != 0;
  pressure_free_ = free_per_token;
  pressure_recent_ = evict_recent;
}

void DgCore::set_spare_target(uint32_t n) {
  drain_pending();  // drain point: a setting (between tokens)
  require(n <= c_.persistent_slots, SS_E_INVALID, "device-gated: spare target above the persistent slots");
  c_.spare_target = n;
  if (c_.prefetch_reserve_auto) c_.prefetch_reserve = std::min<uint32_t>(24, n / 4);  // cache.cpp:171
}

void DgCore::set_prefetch_budget(uint32_t per_layer, uint32_t per_token) {
  drain_pending();  // drain point: a setting (between tokens)
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
  uint32_t listed = 0, dev = 0, hosts = 0, files = 0, backed = 0, hlisted = 0;
  for (uint32_t k = 0; k < keys_; ++k) {
    const Entry& x = e_[k];
    if (x.backed && x.home != Home::Device) broken("a clean copy of a key not on the device");
    if (x.in_hlist != (c_.disk_tier && x.home == Home::Host)) broken("the pinned recency list != the pinned homes");
    hlisted += x.in_hlist ? 1u : 0u;
    switch (x.home) {
      case Home::Host:
        take(block_used, x.block, "host block");
        ++hosts;
        if (table_[k] != -1) broken("a host key is mapped");
        if (x.in_list) broken("a host key is in the recency list");
        break;
      case Home::File:
        ++files;
        if (!c_.disk_tier) broken("a file home without the disk tier");
        if (x.slot != -1 || x.block != -1 || table_[k] != -1 || x.in_list || x.wb_pending || x.pf_done)
          broken("a file home holds a slot, a block, an entry or a copy");
        break;
      case Home::Device:
        take(slot_used, x.slot, "device slot");
        ++dev;
        if (table_[k] != x.slot) broken("table != device slot");
        if (x.backed) {
          take(block_used, x.block, "clean copy block");
          ++backed;
        } else if (x.block != -1) {
          broken("a device key holds a block");
        }
        break;
      case Home::Coming:
        take(slot_used, x.slot, "coming slot");
        if (x.block >= 0 || !c_.disk_tier) take(block_used, x.block, "coming source block");
        ++dev;
        if (table_[k] != x.slot) broken("table != coming slot");
        break;
      case Home::Bank:
        take(slot_used, x.slot, "bank slot");
        if (x.block >= 0 || !c_.disk_tier) take(block_used, x.block, "bank home block");
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
  // reads in flight: each a Coming / Bank key on its slot, mapped in the mirror, without a prefetch event yet, on its
  // own staging buffer
  std::vector<uint8_t> stage_used(c_.disk_staging, 0);
  for (const Pending& p : pending_) {
    const Entry& x = e_[p.key];
    if ((x.home != Home::Coming && x.home != Home::Bank) || x.slot != p.slot || x.block != -1 || x.pf_done ||
        table_[p.key] != p.slot || p.bank != (x.home == Home::Bank) || p.prefetch != x.via_prefetch)
      broken("a file read in flight does not match its key");
    take(stage_used, int64_t(p.stage) - int64_t(c_.blocks), "a staging buffer with two reads in flight");
    // collect / force_free_slot / eviction never touch it: no write-back record of its key or slot (the slot itself is
    // held once: slot_used above), and a staging buffer is never in a free list
    if (x.wb_pending) broken("a key read from its file with a write-back record");
    for (const Writeback& w : wbs_)
      if (w.key == p.key || w.slot == p.slot) broken("a write-back record on a file read in flight");
    for (const Deferred& d : deferred_)
      if (d.key == p.key || d.slot == p.slot) broken("a deferred write-back on a file read in flight");
  }
  for (int32_t b : free_blocks_)
    if (b < 0 || uint32_t(b) >= c_.blocks) broken("a staging buffer in the free blocks");
  if (pending_.size() > std::max<uint32_t>(c_.disk_queue, 1)) broken("more file reads in flight than the queue depth");
  for (int32_t s : free_slots_) {
    if (uint32_t(s) >= bank_first_) broken("a bank slot in the free list");
    take(slot_used, s, "free slot");
  }
  for (int32_t b : free_blocks_) take(block_used, b, "free block");
  for (void* ev : event_pool_) take(ev_used, ev_index(ev), "pooled event");
  for (void* ev : stage_ev_)
    if (ev) take(ev_used, ev_index(ev), "staging event");
  for (uint32_t b = 0; b < c_.blocks; ++b)
    if (!block_used[b]) broken("a host block is lost");
  for (uint32_t s = 0; s < bank_first_; ++s)
    if (!slot_used[s]) broken("a slot is lost");
  for (size_t i = 0; i < ev_used.size(); ++i)
    if (!ev_used[i]) broken("an event is lost");
  if (dev != on_device_ || listed != on_device_) broken("device count / recency list");
  if (hosts != on_host_ || files != on_file_ || backed != backed_ || backed_ > c_.inclusive)
    broken("pinned / file / clean-copy counts");
  if (c_.disk_tier) {  // the pinned recency list: every Host key once, links consistent
    uint32_t n = 0, prev = kNone;
    for (uint32_t k = hhead_; k != kNone; k = e_[k].hnext) {
      if (++n > keys_ || e_[k].home != Home::Host || e_[k].hprev != prev) broken("the pinned recency list");
      prev = k;
    }
    if (n != hlisted || prev != htail_) broken("the pinned recency list's length / tail");
  }
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
