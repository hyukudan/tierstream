// Device-gated decode mode: the cache state machine (docs/design.md §3.7), free of any device API. A transport
// performs the copies, publications and events; the CUDA one is dg_cuda.cu, the tests script their own.
//
// Every gated unit (key = unit_base[g] + unit) has one authoritative home: a slot of one arena shared by all gated
// layers, or a pinned block. Inside a token the core only ADDS table entries (-1 -> slot), on the stream that copied
// the bytes; evictions happen at the boundary, and so do the D2H write-backs unless they are deferred (wb_defer: the
// service issues them part by part after served layers, the next boundary issues what is left). Port of
// wdsrtx's StreamCache (ds41/rtx-replica a8cccdc7, src/cuda/ds41_stream_cache.cpp).
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ts {

// What the core asks of the device. Streams are the core's own; an event is an opaque handle from event_create.
class DgTransport {
 public:
  enum Stream { kDem = 0, kPf = 1, kD2h = 2 };
  virtual ~DgTransport() = default;
  virtual void h2d(Stream s, int32_t slot, int32_t block) = 0;  // every part, block -> slot
  virtual void d2h(int32_t slot, int32_t block) = 0;            // every part, slot -> block, on kD2h
  // Reset movement. Block -1 is a dedicated one-unit host staging buffer, also accepted by h2d/d2h.
  // d2d and h2h complete before returning; the core synchronizes h2d/d2h before reusing a source.
  virtual void d2d(int32_t src, int32_t dst) = 0;
  virtual void h2h(int32_t src, int32_t dst) = 0;
  virtual void d2h_part(Stream s, int32_t slot, int32_t block, uint32_t part) = 0;  // one part, slot -> block
  virtual uint32_t parts() const = 0;                     // parts per unit (>= 1)
  virtual uint64_t part_bytes(uint32_t part) const = 0;  // a part's bytes (write-back accounting)
  // slot_of[key] = v after everything enqueued on s so far (a release: system fence, then the write).
  virtual void write_entry(Stream s, uint32_t key, int32_t v) = 0;
  virtual void signal_ready(uint32_t g, uint32_t seq) = 0;  // ready[g] = seq on kDem, a release too
  virtual void* event_create() = 0;
  virtual void event_destroy(void* ev) = 0;
  virtual void record(void* ev, Stream s) = 0;
  virtual void wait(Stream s, void* ev) = 0;  // device-side
  virtual bool done(void* ev) = 0;            // host query: the last record completed
  virtual void event_sync(void* ev) = 0;
  virtual void sync(Stream s) = 0;
  virtual void upload_table(const int32_t* table, uint32_t n) = 0;  // the whole table on kDem, then sync
  // Diagnostics (dg_diag): a timing mark on kDem, which = 0 before the first demand copy of layer g, 2 after the
  // first fetched unit's copies (if any), 1 after its ready. Only called for a layer with a non-zero mask. Default:
  // nothing.
  virtual void diag_mark(uint32_t g, int which) {
    (void)g;
    (void)which;
  }
  // dg_diag at the boundary (no device work may be added there): after the token's streams are synced; then once
  // the boundary's write-backs (n > 0) are enqueued on kD2h. Default: nothing.
  virtual void diag_reset() {}
  // dg_flush_ready at runtime (between tokens; default: not supported by this transport, ignored)
  virtual void set_flush_ready(bool) {}
  virtual void diag_boundary() {}
  virtual void diag_writebacks_issued() {}
  // winpf: push the stream's queued work to the GPU now (under WDDM a copy can otherwise sit in the driver's batch until
  // a later submission, i.e. until after the demand it was meant to precede). Never waits. Default: nothing.
  virtual void flush(Stream s) { (void)s; }
};

// dg_diag: the host time of the last boundary's phases (end_token), microseconds.
struct DgBoundaryTimes {
  double sync_us = 0;    // kDem + kPf synchronize
  double select_us = 0;  // admissions, collect, victim selection
  double upload_us = 0;  // the table upload (includes its synchronize)
  double issue_us = 0;   // enqueueing the write-backs
  uint32_t writebacks = 0;
  uint64_t boundaries = 0;  // boundaries timed since open
};

struct DgConfig {
  std::vector<uint32_t> n_units;   // per gated layer
  uint32_t persistent_slots = 0;   // R + spare: slots [0, persistent_slots)
  uint32_t bank_slots = 0;         // B per bank: slots [persistent_slots, persistent_slots + 2B)
  uint32_t blocks = 0;
  uint32_t spare_target = 0;
  uint32_t prefetch_reserve = 0;
  bool prefetch_reserve_auto = false;  // the reserve follows the spare target: min(24, spare_target / 4)
  uint32_t prefetch_per_layer = 4;
  uint32_t prefetch_per_token = 48;
  // dg_wb_defer: write-backs deferred into the next token, issued on kDem right after a layer's ready (or on kD2h
  // behind a post-ready event: wb_on_d2h), wb_layer_bytes of credit per served layer (0 = the first part's bytes),
  // while fewer than wb_max_outstanding bytes are in flight (0 = two units' bytes).
  bool wb_defer = false;
  bool wb_on_d2h = false;
  uint64_t wb_layer_bytes = 0;
  uint64_t wb_max_outstanding = 0;
  // Scored eviction (SS_POLICY_SCORED, docs/design.md §3.8): the boundary's victims in score order instead of the
  // recency tail. Integer weights out of 1000 (sum 1000); max_ids = the most ids one export can hold (the unions).
  bool scored = false;
  uint32_t w_trans = 700, w_freq = 200, w_rec = 100;
  uint32_t max_ids = 64;
};

struct DgStats {
  uint64_t tokens = 0, lookups = 0, misses = 0, covered = 0, fetched = 0, prefetched = 0, prefetch_used = 0;
  uint64_t resurrected = 0, bank_serves = 0, collected_mid = 0, evicted = 0, evict_short = 0, writebacks_done = 0;
  uint64_t stalled = 0, pred_invalid = 0;
  // design rev 3.4 §11.6 (the replica's definitions): covered = masked occurrences on the device or coming at service
  // time; covered_waits = the demand stream's waits on a prefetch's copies; predicted = deduplicated valid predicted
  // ids; predicted_true = unique ids of a layer found in the prediction made at the layer before; miss_layers =
  // gated layers with >= 1 miss, per token.
  uint64_t covered_waits = 0, predicted = 0, predicted_true = 0, miss_layers = 0;
  // wb_defer: victims deferred at a boundary; cancelled before any part was submitted (also counted in resurrected);
  // finished early for a demand without a free slot; still (partly) unsubmitted at the next boundary. Bytes: every
  // write-back copy submitted (both modes), completed (collected), never copied (cancelled, or the unsent parts of a
  // partly submitted victim resurrected), submitted by a boundary's flush; the host's waits of the forced path.
  uint64_t wb_deferred = 0, wb_cancelled = 0, wb_forced = 0, wb_boundary_flushed = 0;
  uint64_t wb_issued_bytes = 0, wb_completed_bytes = 0, wb_cancelled_bytes = 0, wb_boundary_flushed_bytes = 0;
  uint64_t wb_forced_wait_ns = 0;
  // winpf: window hints; prior (gated layers < P, between tokens) and l+1 (in-token) prefetch
  // copies issued, used by their token, late (a demand of their token waited for their copies), reverted (unused, back
  // to the host block at the boundary); l+1 candidates ranked, priced out by their bucket's precision.
  uint64_t winpf_hints = 0, winpf_hint_rows = 0, prior_issued = 0, prior_used = 0, prior_late = 0, prior_reverted = 0;
  uint64_t l1_candidates = 0, l1_issued = 0, l1_used = 0, l1_late = 0, l1_reverted = 0, l1_priced_out = 0;
  uint64_t l1_skipped_ahead = 0;  // l+1 predictions not served: layer l+1 had already exported (its demand covers)
  uint64_t prior_budget_stops = 0;  // window hints whose prior prefetches stopped at the copy-engine byte budget
  // scored eviction: boundaries that fed the recency order instead (invalid score metadata: should never happen);
  // victims taken in score order; layers scored and the candidates they ranked; transition increments a saturated
  // cell refused; the service's scoring time
  uint64_t score_fallbacks = 0, scored_victims = 0, score_layers = 0, score_candidates = 0, score_saturated = 0;
  uint64_t score_ns = 0;
  // the service: layers whose ranking waited for a later layer (the device was ahead), and the ranking time spent
  // after the last gated layer was served (on the terminal path)
  uint64_t score_deferred = 0, score_last_ns = 0;
};

// Window prefetch (winpf, opt-in). Off = today's behaviour byte for byte.
struct DgWinpf {
  bool enable = false;
  uint32_t prior_layers = 4;   // gated layers 0..P-1 served from the token prior at a window hint
  uint32_t prior_cap = 1;      // prior prefetches per gated layer per hint call (0 = no prior prefetch)
  float prior_pmin = 0.7f;     // admit when P(used by the hinted rows) = 1 - prod(1 - min(1, w p)) >= this
  uint32_t l1_cap = 1;         // l+1 prefetches per gated layer per token (the generic budgets still apply)
  float l1_min_votes = 1.0f;   // weighted row votes an l+1 candidate needs
  float l1_pmin = 0.0f;        // > 0: a (votes, rank) bucket whose online precision is below this is not admitted
  bool revert_unused = true;   // an unused prefetch goes back to its host block at the boundary (no write-back)
  uint32_t top_k = 0;          // ids per row in an export (the engine's top_k); rows = n / top_k, required when on
  uint32_t prior_window_us = 1500;  // copy-engine time the draft step leaves before the next step's first export
  double copy_gbps = 30.0;     // initial copy rate; dg_diag measurements replace it (observe_copy)
};
// tierstream/hint-prior-v1, parsed: per symbol, for gated layers 0..layers-1, k (unit, P(used)) ranked entries.
struct DgPrior {
  uint32_t layers = 0, k = 0;
  std::unordered_map<int32_t, uint32_t> row;  // symbol -> row index
  std::vector<uint32_t> unit;                  // [rows][layers][k], 0xffffffff = none
  std::vector<float> p;                        // [rows][layers][k]
};

// Per gated layer (folded into the context's per-layer stats at each successful boundary).
struct DgLayerCounts {
  uint64_t lookups = 0, misses = 0, fetched = 0, bank_serves = 0, prefetched = 0, prefetch_used = 0, evicted = 0;
};

class DgCore {
 public:
  enum class Home : uint8_t { Host, Device, Coming, Bank, Evicting };

  DgCore(const DgConfig& cfg, DgTransport* t);
  ~DgCore();
  DgCore(const DgCore&) = delete;
  DgCore& operator=(const DgCore&) = delete;

  // Open: `residents` (keys, by priority) take slots 0.., every other key a block. The caller fills the bytes
  // (slot_of / block_of) and uploads table() before the first token.
  void place(const std::vector<uint32_t>& residents);
  uint32_t keys() const { return keys_; }
  uint32_t key(uint32_t g, uint32_t unit) const { return base_[g] + unit; }
  uint32_t gated() const { return uint32_t(base_.size() - 1); }
  uint32_t units(uint32_t g) const { return base_[g + 1] - base_[g]; }
  const std::vector<uint32_t>& unit_base() const { return base_; }
  int32_t slot_of(uint32_t k) const { return e_[k].slot; }
  int32_t block_of(uint32_t k) const { return e_[k].block; }
  Home home(uint32_t k) const { return e_[k].home; }
  const std::vector<int32_t>& table() const { return table_; }
  uint32_t on_device() const { return on_device_; }
  uint32_t free_slots() const { return uint32_t(free_slots_.size()); }
  const DgStats& stats() const { return st_; }
  const std::vector<DgLayerCounts>& layer_counts() const { return lc_; }
  uint32_t free_blocks() const { return uint32_t(free_blocks_.size()); }
  uint32_t deferred() const { return uint32_t(deferred_.size()); }  // victims whose write-back is not recorded yet
  uint32_t deferred_parts_issued() const { return deferred_.empty() ? 0u : deferred_.front().parts_done; }
  uint64_t wb_outstanding_bytes() const { return outstanding_; }  // submitted write-back bytes not collected yet
  uint32_t gated_of(uint32_t k) const;  // the gated layer of a key

  // ---- inside a token (the service thread) ----
  void begin_token(uint32_t seq);
  bool on_device_or_coming(uint32_t g, uint32_t unit) const;
  // Makes (g, unit) arrive: a copy into a free slot (or a bank slot for a demand), or a resurrection. A prefetch over
  // budget does nothing. Returns true if something was enqueued.
  bool fetch(uint32_t g, uint32_t unit, bool prefetch);
  // A demand miss found coming from a prefetch: the demand stream waits for that unit's copies before ready.
  void cover(uint32_t g, uint32_t unit);
  // A true use (recency + stats); was_miss = its export saw -1.
  void use(uint32_t g, uint32_t unit, bool was_miss);
  void signal(uint32_t g) { t_->signal_ready(g, seq_); }
  // wb_defer: after gated layer g was served (its ready and its prefetches already enqueued), the next deferred
  // write-back parts under the layer's byte credit (a part larger than the credit left still goes: the excess is a
  // debt the next layers repay) and the outstanding cap (finished write-backs are collected first). Nothing after the
  // last gated layer (the boundary's demand-stream sync would wait for it). ahead(), asked before every part: a
  // later layer already exported (the device ran ahead): no more parts, its demand comes first.
  void issue_writebacks(uint32_t g, const std::function<bool()>& ahead = nullptr);
  // The service's work for one exported layer (StreamSite::serve): serve_demand = the masked occurrences in export
  // order (cover a unit on the device or coming, else fetch it), then ready if any; serve_use = recency and stats over
  // the unique ids. serve = both. An id outside the layer throws SS_E_INVALID before any key is computed (the export
  // kernel already aborted the token for it).
  void serve_demand(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask);
  void serve_use(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask);
  void serve(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask) {
    serve_demand(g, ids, n, mask);
    serve_use(g, ids, n, mask);
  }
  // The prediction exported at layer g for layer g + 1: deduplicated, in order, under the prefetch budgets. With
  // winpf on and a window hint for this token: n_export = layer g's exported ids (rows = n_export / top_k, both n's
  // must be multiples: SS_E_BACKEND otherwise), ranked by weighted row votes, then by the best rank; ahead = layer g + 1
  // already exported (nothing is issued: its demand covers).
  void serve_prediction(uint32_t g_next, const uint32_t* pred, uint32_t n, uint32_t n_export = 0, bool ahead = false);
  // Test-only fault injection (tests/test_dg.cpp, the model checker): each must be caught.
  enum Mutation { kNoMutation = 0, kEntryBeforeCopy, kNoCover, kEarlyEventRecycle, kNoBankUnpublish,
                  kEarlySourceFree, kFreeResurrectedSlot, kEarlyWritebackBlock, kReevictPending,
                  kEarlyDeferredSlot, kRecordBeforeLastPart, kRevertKeepsEntry, kSeedNoPfSync,
                  kIgnoreKeep, kStaleList, kNoUnionSave, kSkipRank };
  void set_mutation(int m) { mut_ = m; }

  // ---- boundary (the engine thread, the device done with the token) ----
  // Drains the token's copies, returns bank units to their blocks, collects write-backs, evicts LRU victims (never a
  // key in keep) until free + pending (in flight or deferred) reaches spare_target, uploads the table, then issues the
  // write-backs: the ones still deferred from the last boundary, then this boundary's (or defers these).
  void end_token(const std::vector<uint32_t>& keep);
  // Between tokens: HOST keys into free slots (no eviction, no bank), synchronously. Returns how many moved.
  uint32_t seed(const std::vector<uint32_t>& keys);
  // Between tokens: every deferred write-back submitted, then every stream synchronized.
  void drain();
  // Between tokens, with exclusive ownership. Restore the immutable place() blueprint; copies only misplaced units.
  // Counters remain lifetime counters. A failed reset poisons the core; the owning context must also be poisoned.
  void reset_open();
  // Between tokens: write-back deferral on / off (dg_wb_defer at runtime; off submits the deferred ones first).
  void set_wb_defer(bool on);
  // The device-resident keys (home Device) in recency order, most recent first (persist: a usage / residency file
  // written at shutdown seeds the next open's placement).
  std::vector<uint32_t> residency() const;
  bool wb_defer() const { return c_.wb_defer; }
  bool reset_failed() const { return reset_failed_; }
  // Test-only semantic comparison: excludes sequence domain, event handles, dead scratch and lifetime counters.
  bool same_open_state(const DgCore& other) const;
  // Teardown only (noexcept, no device call): drops the deferred write-backs nothing was submitted for yet. A partly
  // submitted one's copies are drained by the caller's stream syncs before any memory is freed. The core is not
  // used afterwards (its evicting keys have no record).
  uint32_t abandon_writebacks() noexcept;
  // Every key in exactly one home, every slot / block / event accounted once, table == homes, LRU == device set.
  // Throws ts::Error(SS_E_BACKEND) naming the first broken rule.
  void check() const;

  // Test-only (qualification of the rare paths): demand misses take at most free_per_token free slots per token and
  // every boundary also evicts the evict_recent most recently used keys. 0, 0 = off.
  void set_pressure(uint32_t free_per_token, uint32_t evict_recent);
  // Between tokens (design rev 3.4 §11.8): the spare target takes effect at the next boundary, the budgets from the
  // next token's prefetches. spare_target <= persistent_slots (the caller bounds it by the billed spares).
  void set_spare_target(uint32_t n);
  void set_prefetch_budget(uint32_t per_layer, uint32_t per_token);
  uint32_t spare_target() const { return c_.spare_target; }
  // ---- window prefetch (winpf), between tokens ----
  void set_winpf(const DgWinpf& w);
  const DgWinpf& winpf() const { return wp_; }
  void set_prior(DgPrior&& p) { prior_ = std::move(p); }
  const DgPrior& prior() const { return prior_; }
  // The next window's rows (symbol = token id, weight in [0, 1]): appended to the rows hinted since the last boundary;
  // issues the prior prefetches of gated layers 0..P-1 now.
  void window_hint(const int32_t* symbols, const float* weights, uint32_t rows);
  // dg_diag: the device time of one unit's copies (the least queued first copy of a token): the copy-rate estimate the
  // prior budget uses (exponential average).
  void observe_copy(double unit_ms);
  uint32_t hinted_rows() const { return uint32_t(win_sym_.size()); }
  uint32_t coming() const {  // units whose copy is enqueued and not admitted yet (test hook)
    uint32_t n = 0;
    for (uint32_t k : coming_) n += e_[k].home == Home::Coming ? 1u : 0u;
    return n;
  }
  double copy_gbps() const { return rate_gbps_; }
  // The l+1 buckets' learning state lives as long as the context (ss_stats_reset does not touch it), as scored
  // eviction's tables (scored.md 8.4).
  uint32_t prefetch_reserve() const { return c_.prefetch_reserve; }
  void set_diag(bool on) { diag_ = on; }
  const DgBoundaryTimes& boundary_times() const { return bt_; }

  // ---- scored eviction (c_.scored; docs/design.md §3.8) ----
  // Open only, after place() and before the first token: the profile's transition counts (dg_score.hpp format,
  // checked against units and identity; SS_E_INVALID otherwise). They are also the reset_open state.
  void load_profile(const void* data, uint64_t len, const std::string& identity);
  const std::string& profile_provenance() const { return prov_; }
  // The service, right after gated layer g was served (ready, prediction, write-back slice): ranks layer g's device
  // and coming units by (score, demand epoch, recency stamp) from this token's union of g. Nothing when not scored.
  void score_layer(uint32_t g);
  bool scored() const { return c_.scored; }
  // The service's ranking schedule (counters only): a layer deferred behind a later one, the terminal ranking time.
  void note_score_deferred() { ++st_.score_deferred; }
  void add_score_last_ns(uint64_t ns) { st_.score_last_ns += ns; }
  struct ScoreCand {
    uint64_t s, epoch, stamp;
    uint32_t key;
  };
  // Test hooks: the last boundary's accepted victims (both policies, in acceptance order), a call at the boundary right
  // before victim selection (after admissions, bank returns and collection), and the scored state.
  const std::vector<uint32_t>& last_victims() const { return victims_; }
  void set_boundary_probe(std::function<void()> f) { probe_ = std::move(f); }
  size_t pending_writebacks() const;  // deferred + non-discarded recorded write-backs (the boundary's pending)
  bool wb_pending(uint32_t k) const { return e_[k].wb_pending; }
  uint64_t lru_stamp(uint32_t k) const { return e_[k].lru_stamp; }
  uint64_t demand_epoch(uint32_t k) const { return e_[k].demand_epoch; }
  uint64_t epoch() const { return epoch_; }
  uint64_t stamp_counter() const { return stamp_; }
  void set_stamp_counter(uint64_t v) { stamp_ = v; }  // test only: start near the limit
  static constexpr uint64_t kStampLimit = UINT64_MAX - (uint64_t(1) << 40);  // checked before every phase that pushes
  uint32_t score_cell(uint32_t g, uint32_t r, uint32_t e) const { return sc_[g].T[size_t(r) * units(g) + e]; }
  uint64_t score_rowsum(uint32_t g, uint32_t r) const { return sc_[g].rowsum[r]; }
  const std::vector<ScoreCand>& score_list(uint32_t g) const { return sc_[g].list; }
  uint64_t score_list_epoch(uint32_t g) const { return sc_[g].list_epoch; }

 private:
  static constexpr uint32_t kNone = 0xffffffffu;
  struct Entry {
    Home home = Home::Host;
    int32_t slot = -1;   // Device / Coming / Bank / Evicting
    int32_t block = -1;  // Host / Coming (source, held) / Bank (home) / Evicting (write-back destination)
    uint32_t prev = kNone, next = kNone;
    bool in_list = false;
    bool wb_pending = false;    // a write-back of its slot is in flight (Evicting, or resurrected)
    bool via_prefetch = false;  // Coming on the prefetch stream: a demand covers it through pf_done
    bool used = false;          // prefetched and then used this token (stats)
    uint8_t pf_kind = 0;        // winpf: 0 = generic / demand, 1 = prior, 2 = l+1
    bool pf_late = false;       // winpf: a demand waited for this prefetch's copies
    uint8_t pf_bucket = 0;      // winpf: the l+1 (votes, rank) bucket
    uint64_t demand_epoch = 0;  // scored: the epoch of its last demand use (0 = never; place / seed / prefetch: 0)
    uint64_t lru_stamp = 0;     // scored: the recency list position (unique; larger = nearer the head)
    void* pf_done = nullptr;
  };
  struct Writeback {
    uint32_t key;
    int32_t slot, block;
    void* done;
    bool discard;  // resurrected: the key kept its slot, the block goes back to the free pool
    uint64_t bytes;  // submitted (a discarded partial write-back: its submitted parts only)
  };
  // wb_defer: a victim whose write-back event is not recorded yet, in victim order. Unsubmitted (parts_done 0) or
  // partly submitted (only the front: parts_done parts accepted, possibly all of them if the record failed). Its
  // completion event exists only once it moved to wbs_: collect() never queries a deferred record.
  struct Deferred {
    uint32_t key;
    int32_t slot, block;
    uint32_t parts_done;
    uint64_t bytes;  // submitted so far
  };

  void unlink(uint32_t k);
  void push_front(uint32_t k);
  void* event();
  uint32_t collect();
  void submit_part(Deferred& d);
  void* record_deferred_front(bool discard);  // its event after its submitted parts; into wbs_
  void* finish_deferred_front();             // the front's remaining parts, then record_deferred_front(false)
  void force_free_slot();
  int32_t take_bank_slot(uint32_t g);
  void write_entry(DgTransport::Stream s, uint32_t k, int32_t v);
  bool fetch_impl(uint32_t g, uint32_t unit, bool prefetch, bool budgeted);
  void apply_wb_defer();
  void issue_prior();  // window_hint's prior prefetches
  void stamp_room(const char* what) const;  // scored: the stamp counter may take one more phase of pushes
  void score_history(uint32_t g);           // scored: serve_use's history update of layer g (union saved first)
  void reset_score();                       // scored: the open state of every table (reset_open)
  std::vector<uint32_t> score_window(uint32_t g) const;  // the ring's unions in order, each closed by kNone
  bool cand_less(const ScoreCand& a, const ScoreCand& b) const {
    if (a.s != b.s) return a.s < b.s;
    if (a.epoch != b.epoch) return a.epoch < b.epoch;
    return a.stamp < b.stamp;
  }

  // Captured once by place(), after configuration resolution. No device allocations are replaced at reset.
  DgConfig open_c_;
  std::vector<Entry> open_e_;
  std::vector<int32_t> open_table_, open_free_slots_, open_free_blocks_;
  uint32_t open_head_ = kNone, open_tail_ = kNone, open_on_device_ = 0;
  DgWinpf open_wp_;
  double open_rate_gbps_ = 30;
  bool open_rate_set_ = false, reset_failed_ = false;
  DgConfig c_;
  DgTransport* t_;
  std::vector<uint32_t> base_;  // unit_base, n_gated + 1
  uint32_t keys_ = 0, slots_ = 0, bank_first_ = 0;
  std::vector<Entry> e_;
  std::vector<int32_t> table_;
  std::vector<int32_t> free_slots_, free_blocks_;
  std::deque<Writeback> wbs_;
  std::deque<Deferred> deferred_;
  DgTransport::Stream wb_stream_ = DgTransport::kDem;  // where deferred parts go
  uint64_t unit_bytes_ = 0, layer_bytes_ = 0, max_outstanding_ = 0, outstanding_ = 0;
  int64_t credit_ = 0;               // issue_writebacks' byte credit (< 0: a debt)
  size_t prev_evict_ = 0;            // victims chosen at the last boundary (the carry bound)
  void* post_ready_ev_ = nullptr;    // wb_on_d2h: kD2h waits for the layer's ready through it
  std::vector<void*> event_pool_;
  std::vector<void*> all_events_;
  std::vector<uint32_t> bank_key_;
  uint32_t bank_seq_[2] = {0, 0}, bank_layer_[2] = {kNone, kNone}, bank_next_[2] = {0, 0};
  std::vector<uint32_t> pf_layer_;
  uint32_t pf_token_ = 0;
  bool pressure_ = false;
  uint32_t pressure_free_ = 0, pressure_recent_ = 0, demand_free_used_ = 0;
  uint32_t head_ = kNone, tail_ = kNone, on_device_ = 0;
  std::vector<uint32_t> coming_, banked_;
  std::vector<uint32_t> pred_;     // the deduplicated valid prediction for gated layer pred_layer_
  uint32_t pred_layer_ = kNone;
  std::vector<uint8_t> layer_missed_;  // per gated layer, this token
  bool diag_ = false;
  DgBoundaryTimes bt_;
  uint32_t seq_ = 0;
  DgStats st_;
  std::vector<DgLayerCounts> lc_;
  int mut_ = kNoMutation;
  std::vector<uint32_t> uniq_;
  // winpf
  DgWinpf wp_;
  DgPrior prior_;
  std::vector<int32_t> win_sym_;
  std::vector<float> win_w_;
  uint64_t prior_bytes_ = 0;  // prior bytes issued for the window being hinted
  double rate_gbps_ = 30.0;
  bool rate_set_ = false;
  // l+1 buckets: rounded votes (0..4) x best rank (0-1, 2-3, 4+); online precision = (hit + 1) / (n + 1)
  double bkt_n_[15] = {}, bkt_hit_[15] = {};
  uint32_t explore_ = 0;  // every 64th priced-out candidate is still admitted (keeps a bucket's estimate alive)
  struct Cand {
    float votes;
    uint32_t rank, unit;
  };
  std::vector<Cand> cand_, pcand_;
  // scored eviction, per gated layer: T (uint32 saturating cells, row-major [r][e]), exact uint64 row sums, ring counts
  // of the last kDgScoreWindow unions, the ring, the previous and this token's union, the ranked candidates
  struct LayerScore {
    std::vector<uint32_t> T, open_T;  // open_T: the profile's cells (empty = zeros)
    std::vector<uint64_t> rowsum;
    std::vector<uint16_t> cnt;
    std::vector<uint32_t> ring;       // kDgScoreWindow x max_ids
    std::vector<uint8_t> ring_len;    // kDgScoreWindow
    uint32_t ring_head = 0, ring_n = 0;
    std::vector<uint32_t> prev, cur;  // unique sorted unions (capacity max_ids)
    uint64_t served = 0;              // the epoch whose union cur holds
    std::vector<ScoreCand> list;      // capacity min(units, persistent slots)
    uint64_t list_epoch = 0;          // the epoch it was ranked in (0 = none)
  };
  std::vector<LayerScore> sc_;
  uint64_t epoch_ = 0, stamp_ = 0, open_stamp_ = 0;
  std::string prov_;
  std::vector<uint32_t> victims_;
  std::vector<uint32_t> heap_, heap_pos_;
  std::function<void()> probe_;
};

}  // namespace ts
