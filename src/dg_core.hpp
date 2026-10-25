// Device-gated decode mode: the cache state machine (docs/design.md §3.7), free of any device API. A transport
// performs the copies, publications and events; the CUDA one is dg_cuda.cu, the tests script their own.
//
// Every gated unit (key = unit_base[g] + unit) has one authoritative home: a slot of one arena shared by all gated
// layers, or a pinned block. Inside a token the core only ADDS table entries (-1 -> slot), on the stream that copied
// the bytes; evictions and D2H write-backs happen at the boundary. Port of wdsrtx's StreamCache (ds41/rtx-replica
// a8cccdc7, src/cuda/ds41_stream_cache.cpp).
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace ts {

// What the core asks of the device. Streams are the core's own; an event is an opaque handle from event_create.
class DgTransport {
 public:
  enum Stream { kDem = 0, kPf = 1, kD2h = 2 };
  virtual ~DgTransport() = default;
  virtual void h2d(Stream s, int32_t slot, int32_t block) = 0;  // every part, block -> slot
  virtual void d2h(int32_t slot, int32_t block) = 0;            // every part, slot -> block, on kD2h
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
};

struct DgConfig {
  std::vector<uint32_t> n_units;   // per gated layer
  uint32_t persistent_slots = 0;   // R + spare: slots [0, persistent_slots)
  uint32_t bank_slots = 0;         // B per bank: slots [persistent_slots, persistent_slots + 2B)
  uint32_t blocks = 0;
  uint32_t spare_target = 0;
  uint32_t prefetch_reserve = 0;
  uint32_t prefetch_per_layer = 4;
  uint32_t prefetch_per_token = 48;
};

struct DgStats {
  uint64_t tokens = 0, lookups = 0, misses = 0, covered = 0, fetched = 0, prefetched = 0, prefetch_used = 0;
  uint64_t resurrected = 0, bank_serves = 0, collected_mid = 0, evicted = 0, evict_short = 0, writebacks_done = 0;
  uint64_t stalled = 0, pred_invalid = 0;
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
  // The service's work for one exported layer (StreamSite::serve): the true misses first (fetch, or cover a unit
  // coming from a prefetch), then ready if any, then recency. An id outside the layer throws SS_E_INVALID before any
  // key is computed (the export kernel already aborted the token for it).
  void serve(uint32_t g, const uint32_t* ids, uint32_t n, uint64_t mask);
  // The prediction exported at layer g for layer g + 1: deduplicated, in order, under the prefetch budgets.
  void serve_prediction(uint32_t g_next, const uint32_t* pred, uint32_t n);
  // Test-only fault injection (tests/test_dg.cpp, the model checker): each must be caught.
  enum Mutation { kNoMutation = 0, kEntryBeforeCopy, kNoCover, kEarlyEventRecycle, kNoBankUnpublish,
                  kEarlySourceFree, kFreeResurrectedSlot, kEarlyWritebackBlock, kReevictPending };
  void set_mutation(int m) { mut_ = m; }

  // ---- boundary (the engine thread, the device done with the token) ----
  // Drains the token's copies, returns bank units to their blocks, collects write-backs, evicts LRU victims (never a
  // key in keep) until free + pending reaches spare_target, uploads the table.
  void end_token(const std::vector<uint32_t>& keep);
  // Between tokens: HOST keys into free slots (no eviction, no bank), synchronously. Returns how many moved.
  uint32_t seed(const std::vector<uint32_t>& keys);
  void drain();
  // Every key in exactly one home, every slot / block / event accounted once, table == homes, LRU == device set.
  // Throws ts::Error(SS_E_BACKEND) naming the first broken rule.
  void check() const;

  // Test-only (qualification of the rare paths): demand misses take at most free_per_token free slots per token and
  // every boundary also evicts the evict_recent most recently used keys. 0, 0 = off.
  void set_pressure(uint32_t free_per_token, uint32_t evict_recent);

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
    void* pf_done = nullptr;
  };
  struct Writeback {
    uint32_t key;
    int32_t slot, block;
    void* done;
    bool discard;  // resurrected: the key kept its slot, the block goes back to the free pool
  };

  void unlink(uint32_t k);
  void push_front(uint32_t k);
  void* event();
  uint32_t collect();
  int32_t take_bank_slot(uint32_t g);
  void write_entry(DgTransport::Stream s, uint32_t k, int32_t v);

  DgConfig c_;
  DgTransport* t_;
  std::vector<uint32_t> base_;  // unit_base, n_gated + 1
  uint32_t keys_ = 0, slots_ = 0, bank_first_ = 0;
  std::vector<Entry> e_;
  std::vector<int32_t> table_;
  std::vector<int32_t> free_slots_, free_blocks_;
  std::deque<Writeback> wbs_;
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
  uint32_t seq_ = 0;
  DgStats st_;
  std::vector<DgLayerCounts> lc_;
  int mut_ = kNoMutation;
  std::vector<uint32_t> uniq_;
};

}  // namespace ts
