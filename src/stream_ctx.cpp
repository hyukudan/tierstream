#include "stream_ctx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <set>

#include "errors.hpp"
#include "xxh64.hpp"

namespace ts {
namespace {

constexpr uint64_t kHostAlign = 4096;

// Copies an error message if it can; a failed copy (out of memory) leaves it empty rather than throwing.
void keep_message(std::string& out, const char* msg) noexcept {
  try {
    out = msg;
  } catch (...) {
  }
}

bool resident_tier(uint8_t t) { return t == SS_TIER_RESIDENT || t == SS_TIER_RESIDENT_BACKED; }

uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

char* at(void* base, uint64_t off) { return static_cast<char*>(base) + off; }

uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

void add(ss_layer_stats& a, const ss_layer_stats& b) {
  a.routed += b.routed;
  a.hits += b.hits;
  a.admitted += b.admitted;
  a.transient += b.transient;
  a.evicted += b.evicted;
  a.prefetched += b.prefetched;
  a.spec_issued += b.spec_issued;
  a.spec_used += b.spec_used;
  a.spec_dropped += b.spec_dropped;
  a.bytes_pinned += b.bytes_pinned;
  a.bytes_disk += b.bytes_disk;
  a.wait_ns += b.wait_ns;
}

}  // namespace

Ctx::Ctx(const Desc& d, const ss_budget& b, const Plan& p, const ss_options& o) : d_(d), plan_(p), o_(o) {
  (void)b;
  try {
    init();
  } catch (...) {
    cleanup();
    throw;
  }
}

Ctx::~Ctx() { cleanup(); }

void Ctx::init() {
  require(int64_t(plan_.tier.size()) == d_.total_units, SS_E_INVALID, "open: plan does not match the descriptor");
  excl_ = plan_.budget.tier_mode == SS_TIER_MODE_EXCLUSIVE;
  require(!(excl_ && o_.backend == SS_BACKEND_HIP), SS_E_UNSUPPORTED,
          "open: the exclusive tier mode needs a discrete device (on unified memory a slot and a pinned block are the "
          "same memory)");
  // A resident's slot is the only copy an eviction writes back: it must hold the unit's bytes from open.
  require(!(excl_ && !o_.fill_residents), SS_E_INVALID, "open: the exclusive tier mode needs fill_residents = 1");
  // A descriptor whose tiers are "pinned" keeps every non-resident unit off its file: a discard would file-home one.
  require(!(excl_ && o_.exclusive_discard && d_.tiers == Tiers::Pinned), SS_E_INVALID,
          "open: exclusive_discard needs a descriptor whose tiers allow disk (tiers = \"pinned\" refuses it)");
  switch (o_.backend) {
    case SS_BACKEND_CPU: be_ = make_cpu_backend(); break;
    case SS_BACKEND_CUDA: be_ = make_cuda_backend(o_.device); break;
    case SS_BACKEND_HIP: be_ = make_hip_backend(o_.device); break;
    default: fail(SS_E_INVALID, "open: unknown backend");
  }
  if (!o_.reader_threads) o_.reader_threads = 2;
  if (!o_.staging_buffers) o_.staging_buffers = 4;
  if (!o_.staging_bytes) o_.staging_bytes = uint64_t(256) << 20;
  if (!o_.copy_piece_bytes) o_.copy_piece_bytes = uint64_t(64) << 20;
  uint64_t max_align = kHostAlign;
  for (const auto& s : d_.sources) max_align = std::max(max_align, s.align);
  require(o_.staging_bytes >= 4 * max_align, SS_E_INVALID, "open: staging_bytes must be >= 4 x the source alignment");
  o_.staging_bytes = round_up(o_.staging_bytes, max_align);

  for (const auto& s : d_.sources) files_.push_back(std::make_unique<File>(s.path, o_.direct_io != 0));

  // Staging ring (pinned host): file reads land here, then go to the device (non-unified) or to the slot.
  for (uint32_t i = 0; i < o_.staging_buffers; ++i) {
    Staging s;
    s.host = be_->alloc_pinned(size_t(o_.staging_bytes));
    s.ev = be_->event_create();
    staging_.push_back(s);
    staging_free_.push_back(i);
  }

  // Pinned host tier (PINNED units and pinned copies of RESIDENT_BACKED units): the planner's exact layout.
  pinned_of_.assign(size_t(d_.total_units), nullptr);
  {
    PinnedLayout pl = pinned_layout(d_, plan_.tier);
    for (uint64_t bytes : pl.chunk_bytes) pinned_chunks_.push_back(be_->alloc_pinned(size_t(bytes)));
    for (int64_t g = 0; g < d_.total_units; ++g)
      if (pl.chunk_of[size_t(g)] >= 0)
        pinned_of_[size_t(g)] = at(pinned_chunks_[size_t(pl.chunk_of[size_t(g)])], pl.offset_of[size_t(g)]);
  }

  // Exclusive tier mode: block classes (one per host stride), spare blocks, write-back stream and events.
  if (excl_) {
    wb_stream_ = be_->writeback_stream();
    blocks_ev_ = be_->event_create();
    wb_mark_ev_ = be_->event_create();
    wb_pending_.assign(size_t(d_.total_units), nullptr);
    for (const SpareClass& c : plan_.spares) block_bytes_.push_back(c.block_bytes);
    free_blocks_.resize(block_bytes_.size());
    layer_class_.assign(d_.layers.size(), -1);
    std::vector<uint64_t> class_total(block_bytes_.size(), 0);
    for (size_t L = 0; L < d_.layers.size(); ++L) {
      const Layer& l = d_.layers[L];
      if (!l.streamable || !l.n_units) continue;
      const uint64_t hs = host_stride(l.unit_bytes);
      const auto it = std::find(block_bytes_.begin(), block_bytes_.end(), hs);
      require(it != block_bytes_.end(), SS_E_INVALID, "open: exclusive plan without a block class (internal)");
      layer_class_[L] = int(it - block_bytes_.begin());
      for (int u = 0; u < l.n_units; ++u)
        if (pinned_of_[size_t(gidx(int(L), u))]) ++class_total[size_t(layer_class_[L])];
    }
    for (size_t k = 0; k < plan_.spares.size(); ++k) {
      const uint64_t bytes = plan_.spares[k].block_bytes;
      const uint64_t per_chunk = std::max<uint64_t>(1, (uint64_t(1) << 30) / bytes);
      class_total[k] += plan_.spares[k].blocks;
      // Every block of the class can be free at once: the free list never reallocates after open.
      free_blocks_[k].reserve(size_t(class_total[k]));
      std::vector<void*> blocks;
      for (uint64_t left = plan_.spares[k].blocks; left > 0;) {
        const uint64_t n = std::min(per_chunk, left);
        void* chunk = be_->alloc_pinned(size_t(n * bytes));
        spare_chunks_.push_back(chunk);
        for (uint64_t i = 0; i < n; ++i) blocks.push_back(at(chunk, i * bytes));
        left -= n;
      }
      for (size_t i = blocks.size(); i-- > 0;) free_blocks_[k].push_back(blocks[i]);  // lowest address taken first
      n_blocks_ += class_total[k];
      spares_of_.push_back(plan_.spares[k].blocks);
      xst_.free_blocks += plan_.spares[k].blocks;
    }
    xst_.min_free_blocks = xst_.free_blocks;
    wb_units_.reserve(size_t(d_.total_units));
  }

  // Layers: resident pool, release event, unit tables.
  L_.resize(d_.layers.size());
  for (size_t L = 0; L < d_.layers.size(); ++L) {
    const Layer& l = d_.layers[L];
    LayerState& s = L_[L];
    s.release_ev = be_->event_create();
    s.ready_ev = be_->event_create();
    s.ids_ev = be_->event_create();
    if (l.n_units == 0) continue;
    int planned = 0;
    for (int u = 0; u < l.n_units; ++u)
      if (resident_tier(plan_.tier[size_t(gidx(int(L), u))])) ++planned;
    int cap = l.streamable ? std::min(l.n_units, planned + int(plan_.dynamic_slots_per_layer)) : l.n_units;
    s.rs = std::make_unique<ResidentSet>(l.n_units, cap, o_.policy, o_.score_window);
    if (cap) s.pool = be_->alloc_device(size_t(uint64_t(cap) * d_.unit_stride(int(L))));
    for (int v = 0; v < 2; ++v) {
      s.table_host[v] = static_cast<uint64_t*>(be_->alloc_pinned(sizeof(uint64_t) * size_t(l.n_units)));
      std::memset(s.table_host[v], 0, sizeof(uint64_t) * size_t(l.n_units));
      s.table_dev[v] = be_->unified() ? static_cast<void*>(s.table_host[v])
                                      : be_->alloc_device(sizeof(uint64_t) * size_t(l.n_units));
      s.table_ev[v] = be_->event_create();
    }
    s.loc.assign(size_t(l.n_units), None);
    s.slot.assign(size_t(l.n_units), -1);
    s.job.assign(size_t(l.n_units), nullptr);
    s.spec_unclaimed.assign(size_t(l.n_units), 0);
    s.count.assign(size_t(l.n_units), 0);
  }

  // Stream ring: unit-sized slots shared by every layer.
  for (const RingClass& rc : plan_.rings) {
    if (!rc.slots) continue;
    void* base = be_->alloc_device(size_t(rc.stride * rc.slots));
    ring_classes_.push_back({rc.stride, ring_.size(), rc.slots, 0, base});
    for (uint32_t i = 0; i < rc.slots; ++i) {
      RingSlot slot;
      slot.dev = at(base, rc.stride * i);
      slot.stride = rc.stride;
      ring_.push_back(slot);
    }
  }
  ring_owner_ev_.assign(ring_.size(), nullptr);

  stats_.assign(d_.layers.size(), ss_layer_stats{});
  fwd_stats_.assign(d_.layers.size(), ss_layer_stats{});
  snap_.layers.assign(d_.layers.size(), ss_layer_stats{});

  // Host read-ahead (design §2/§9): R pinned buffers of the largest unit that can be read from its file, each with its
  // own event. None when no unit can be (every streamable unit pinned or computed by a peer).
  if (o_.readahead_units) {
    require(o_.readahead_units <= 64, SS_E_INVALID, "open: readahead_units must be <= 64");
    ra_bytes_ = ra_buffer_bytes(d_, plan_.tier);
    require(ra_bytes_ <= uint64_t(SIZE_MAX), SS_E_INVALID, "open: read-ahead buffer bytes overflow size_t");
    if (ra_bytes_) {
      require(ra_bytes_ <= UINT64_MAX / o_.readahead_units &&
                  ra_bytes_ * o_.readahead_units <= uint64_t(SIZE_MAX),
              SS_E_INVALID, "open: readahead_units x buffer bytes overflow");
      ra_free_.reserve(o_.readahead_units);
      ra_ev_.reserve(o_.readahead_units);  // reserved first: a created resource is always registered for cleanup
      ra_host_.reserve(o_.readahead_units);
      ra_jobs_.reserve(size_t(d_.total_units));
      for (uint32_t i = 0; i < o_.readahead_units; ++i) {
        ra_ev_.push_back(be_->event_create());
        ra_host_.push_back(be_->alloc_pinned(size_t(ra_bytes_)));
        ra_free_.push_back(i);
      }
      for (size_t L = 0; L < d_.layers.size(); ++L)
        if (L_[L].rs) L_[L].ra.assign(size_t(d_.layers[L].n_units), nullptr);
    }
  }
  start_workers();

  // Fill: the pinned tier first (host), then residents (from their pinned copy when they have one, else the file).
  std::vector<JobPtr> fill;
  for (size_t L = 0; L < d_.layers.size(); ++L) {
    for (int u = 0; u < d_.layers[L].n_units; ++u) {
      const void* pa = pinned_of_[size_t(gidx(int(L), u))];
      if (!pa) continue;
      JobPtr j = std::make_shared<Job>();
      j->layer = int(L);
      j->unit = u;
      j->dst = const_cast<void*>(pa);
      j->host_dst = true;
      fill.push_back(j);
      submit(j);
    }
  }
  for (const JobPtr& j : fill) wait_job(*j);
  fill.clear();
  for (size_t L = 0; L < d_.layers.size(); ++L) {
    const Layer& l = d_.layers[L];
    LayerState& s = L_[L];
    if (!s.rs) continue;
    for (int u = 0; u < l.n_units; ++u) {
      bool res = !l.streamable || resident_tier(plan_.tier[size_t(gidx(int(L), u))]);
      if (!res) continue;
      int slot = s.rs->place(u, !l.streamable || o_.pin_planned != 0);
      require(slot >= 0, SS_E_CAPACITY, "open: resident pool full (internal)");
      if (o_.fill_residents) {
        JobPtr j = make_job(int(L), u, at(s.pool, uint64_t(slot) * d_.unit_stride(int(L))), false);
        fill.push_back(j);
        submit(j);
      }
    }
  }
  for (const JobPtr& j : fill) wait_job(*j);
  for (const JobPtr& j : fill)
    if (j->done_ev) ev_pool_.push_back(j->done_ev);
  be_->stream_sync(nullptr);
}

void Ctx::cleanup() {
  if (ra_on()) {  // shutdown sweep (also with no forward begun and on a poisoned context): nothing waits on a buffer
    std::lock_guard<std::mutex> lk(q_mu_);
    ra_frozen_ = true;
    ra_sweep_locked();
  }
  stop_workers();
  // Workers are joined: reconcile what a close without ss_end left, then break the entry <-> job references.
  if (ra_on()) {
    ra_reconcile();
    ra_teardown();
  }
  if (!be_) return;
  try {
    be_->stream_sync(nullptr);
  } catch (const Error&) {
    // A sticky device error: free what we can.
  }
  for (auto& s : L_) {
    if (s.pool) be_->free_device(s.pool);
    if (s.release_ev) be_->event_destroy(s.release_ev);
    if (s.ready_ev) be_->event_destroy(s.ready_ev);
    if (s.ids_ev) be_->event_destroy(s.ids_ev);
    if (s.ids_host) be_->free_pinned(s.ids_host);
    for (int v = 0; v < 2; ++v) {
      if (s.table_host[v]) be_->free_pinned(s.table_host[v]);
      if (s.table_dev[v] && !be_->unified()) be_->free_device(s.table_dev[v]);
      if (s.table_ev[v]) be_->event_destroy(s.table_ev[v]);
    }
  }
  try {
    if (wb_stream_) be_->stream_sync(wb_stream_);
  } catch (const Error&) {
  }
  for (const RingRange& r : ring_classes_) be_->free_device(r.base);
  for (void* c : pinned_chunks_) be_->free_pinned(c);
  for (void* c : spare_chunks_) be_->free_pinned(c);
  for (void* b : ra_host_) be_->free_pinned(b);  // quarantined ones too: the streams were synchronized above
  for (void* e : ra_ev_) be_->event_destroy(e);
  ra_host_.clear();
  ra_ev_.clear();
  for (void* e : wb_evs_) be_->event_destroy(e);
  if (blocks_ev_) be_->event_destroy(blocks_ev_);
  if (wb_mark_ev_) be_->event_destroy(wb_mark_ev_);
  spare_chunks_.clear();
  wb_evs_.clear();
  blocks_ev_ = wb_mark_ev_ = nullptr;
  for (auto& s : staging_) {
    be_->free_pinned(s.host);
    be_->event_destroy(s.ev);
  }
  for (void* e : ev_pool_) be_->event_destroy(e);
  for (const JobPtr& j : fwd_jobs_)
    if (j->done_ev) be_->event_destroy(j->done_ev);
  L_.clear();
  ring_classes_.clear();
  pinned_chunks_.clear();
  staging_.clear();
  ev_pool_.clear();
  fwd_jobs_.clear();
  be_.reset();
}

// ---- workers -------------------------------------------------------------------------------------------------------

void Ctx::start_workers() {
  for (uint32_t i = 0; i < o_.reader_threads; ++i) workers_.emplace_back([this] { worker(); });
}

void Ctx::stop_workers() {
  {
    std::lock_guard<std::mutex> lk(q_mu_);
    stop_ = true;
  }
  q_cv_.notify_all();
  for (auto& t : workers_) t.join();
  workers_.clear();
}

void Ctx::worker() {
  try {
    be_->bind_thread();
  } catch (const std::exception&) {
    // The first job reports the device error.
  }
  while (true) {
    JobPtr j;
    {
      std::unique_lock<std::mutex> lk(q_mu_);
      // Priority: demands (ready read-ahead landings included) > read-ahead reads > speculative reads. Unattached
      // read-ahead reads run on at most max(1, readers - 1) readers; a speculative read runs only while no demand is
      // queued and at most one demand is running.
      const int ra_cap = std::max(1, int(o_.reader_threads) - 1);
      q_cv_.wait(lk, [&] {
        return stop_ || (!hold_ && (!demand_q_.empty() || (!ra_q_.empty() && ra_running_ < ra_cap) ||
                                    (!spec_q_.empty() && demand_running_ <= 1)));
      });
      if (stop_ && demand_q_.empty() && spec_q_.empty() && ra_q_.empty()) return;
      if (!demand_q_.empty()) {
        j = demand_q_.front();
        demand_q_.pop_front();
        ++demand_running_;
        j->ran_as = Job::AsDemand;
      } else if (!ra_q_.empty() && (ra_running_ < ra_cap || stop_)) {
        j = ra_q_.front();
        ra_q_.pop_front();
        ++ra_running_;
        j->ran_as = Job::AsRa;
      } else if (!spec_q_.empty() && demand_running_ <= 1) {
        j = spec_q_.front();
        spec_q_.pop_front();
        j->ran_as = Job::AsSpec;
      } else {
        continue;
      }
      int expect = Job::Queued;
      if (!j->state.compare_exchange_strong(expect, Job::Running)) {
        if (j->ran_as == Job::AsDemand) --demand_running_;
        else if (j->ran_as == Job::AsRa) --ra_running_;
        if (j->ra_child && j->ra) ra_child_epilogue_locked(*j);  // cancelled after it was queued: its buffer goes back
        done_cv_.notify_all();
        continue;  // cancelled while queued
      }
    }
    int final_state = Job::Done;
    try {
      run_job(*j);
    } catch (const Error& e) {
      j->err_status = e.status();
      try {
        j->err = e.what();
      } catch (...) {
      }
      final_state = Job::Failed;
    } catch (const std::bad_alloc&) {
      j->err_status = SS_E_NOMEM;
      final_state = Job::Failed;
    } catch (const std::exception& e) {
      j->err_status = SS_E_BACKEND;
      try {
        j->err = e.what();
      } catch (...) {
      }
      final_state = Job::Failed;
    }
    {
      std::lock_guard<std::mutex> lk(q_mu_);
      j->state.store(final_state);
      if (j->ran_as == Job::AsDemand) --demand_running_;
      else if (j->ran_as == Job::AsRa) --ra_running_;
      // Read-ahead: the terminal state and its consequences (landing queued, buffer back) in one section.
      if (j->ra_producer) ra_producer_epilogue_locked(*j);
      else if (j->ra_child) ra_child_epilogue_locked(*j);
    }
    done_cv_.notify_all();
    q_cv_.notify_all();
  }
}

void Ctx::submit(const JobPtr& j) {
  if (!be_->unified() && j->from_pinned && !j->host_dst) {
    // Pinned tier -> device: issue the pieces now on the copy stream (no host wait, no reader thread).
    const uint64_t n = d_.layers[size_t(j->layer)].unit_bytes;
    try {
      std::lock_guard<std::mutex> lk(copy_mu_);
      be_->stream_wait(nullptr, j->dst_release_ev);
      be_->stream_wait(nullptr, j->dst_wb_ev);
#if !defined(SS_MUTATION_NO_SRC_WAIT)  // test-only mutation build: proves the CUDA test catches its absence
      be_->stream_wait(nullptr, j->src_wait_ev);
#else
      if (j->src_wait_ev)  // the mutation reports what it skipped and what else orders this copy
        std::fprintf(stderr, "mutation: source wait skipped (src %p, dst_release %p, dst_wb %p)\n", j->src_wait_ev,
                     j->dst_release_ev, j->dst_wb_ev);
#endif
      for (uint64_t off = 0; off < n; off += o_.copy_piece_bytes)
        be_->copy_h2d(at(j->dst, off), static_cast<const char*>(j->host_src) + off,
                      size_t(std::min<uint64_t>(o_.copy_piece_bytes, n - off)), nullptr);
      be_->event_record(j->done_ev, nullptr);
    } catch (const Error& e) {
      fail_job(*j, e.status(), e.what());
      throw;
    } catch (const std::exception& e) {
      fail_job(*j, SS_E_BACKEND, e.what());
      throw;
    }
    j->state.store(Job::Done);
    return;
  }
  try {
    std::lock_guard<std::mutex> lk(q_mu_);
    (j->spec ? spec_q_ : demand_q_).push_back(j);
  } catch (const std::exception& e) {
    fail_job(*j, SS_E_NOMEM, e.what());
    throw;
  }
  q_cv_.notify_all();
}

void Ctx::fail_job(Job& j, ss_status st, const char* msg) noexcept {
  // The terminal state is published even if copying the message fails (it then stays empty).
  {
    std::lock_guard<std::mutex> lk(q_mu_);
    j.err_status = st;
    try {
      j.err = msg;
    } catch (...) {
    }
    j.state.store(Job::Failed);
  }
  done_cv_.notify_all();
}

void Ctx::promote(const JobPtr& j) {
  // A speculative read that is now a demand jumps to the front of the demand queue if it has not started.
  std::lock_guard<std::mutex> lk(q_mu_);
  if (j->state.load() != Job::Queued || !j->spec) return;
  auto it = std::find(spec_q_.begin(), spec_q_.end(), j);
  if (it == spec_q_.end()) return;
  try {
    demand_q_.push_front(j);
  } catch (...) {
    return;  // stays queued as speculative: still owned by a queue, still runs
  }
  spec_q_.erase(std::find(spec_q_.begin(), spec_q_.end(), j));
  j->spec = false;
  q_cv_.notify_one();
}

void Ctx::wait_job(Job& j) {
  std::unique_lock<std::mutex> lk(q_mu_);
  done_cv_.wait(lk, [&] {
    int s = j.state.load();
    return s == Job::Done || s == Job::Cancelled || s == Job::Failed;
  });
  if (j.state.load() == Job::Failed) fail(j.err_status, j.err);
}

void Ctx::run_job(Job& j) {
  if (j.ra_child) {
    run_ra_child(j);
    return;
  }
  if (j.ra_producer) {
    // The buffer's last H2D out has finished. Test-only mutation 7 (must make the CUDA test fail): skip it once.
    if (!(test_inject_.load() == 7 && test_inject_.exchange(0) == 7)) be_->event_sync(ra_ev_[size_t(j.ra->buf)]);
    if (test_inject_.load() == 5 && test_inject_.exchange(0) == 5) fail(SS_E_IO, "read-ahead: injected read failure");
  }
  const Layer& l = d_.layers[size_t(j.layer)];
  bool want_hash = o_.verify_hash && !l.hash.empty();
  uint64_t digest = 0;
  if (j.host_dst) {
    // A producer's bytes_read counts every completed part, also when a later part or the hash check fails.
    read_unit_host(j.layer, j.unit, j.dst, want_hash, &digest, j.ra_producer ? &j.bytes_read : nullptr);
    if (j.ra_producer && test_inject_.load() == 9 && test_inject_.exchange(0) == 9) digest = ~digest;
  } else if (be_->unified()) {
    be_->event_sync(j.dst_release_ev);  // the slot's previous reader is done
    be_->event_sync(j.dst_wb_ev);
    if (j.from_pinned) {
      std::memcpy(j.dst, j.host_src, size_t(l.unit_bytes));
      if (want_hash) digest = xxh64(j.dst, size_t(l.unit_bytes));
    } else {
      read_unit_host(j.layer, j.unit, j.dst, want_hash, &digest);
    }
  } else {
    read_unit_device(j);
    return;
  }
  if (want_hash && digest != l.hash[size_t(j.unit)])
    fail(SS_E_HASH, "unit (" + std::to_string(j.layer) + ", " + std::to_string(j.unit) +
                        ") bytes differ from the descriptor hash");
}

// A staging buffer for one chunk: taken from the free list (its last H2D done), always given back.
class Ctx::StagingLease {
 public:
  explicit StagingLease(Ctx& c) : c_(c) {
    {
      std::unique_lock<std::mutex> lk(c_.staging_mu_);
      c_.staging_cv_.wait(lk, [&] { return !c_.staging_free_.empty(); });
      i_ = c_.staging_free_.back();
      c_.staging_free_.pop_back();
    }
    try {
      c_.be_->event_sync(c_.staging_[i_].ev);
    } catch (...) {
      give_back();
      throw;
    }
  }
  ~StagingLease() { give_back(); }
  StagingLease(const StagingLease&) = delete;
  StagingLease& operator=(const StagingLease&) = delete;
  void* host() const { return c_.staging_[i_].host; }
  void* ev() const { return c_.staging_[i_].ev; }

 private:
  void give_back() {
    if (done_) return;
    done_ = true;
    {
      std::lock_guard<std::mutex> lk(c_.staging_mu_);
      c_.staging_free_.push_back(i_);
    }
    c_.staging_cv_.notify_one();
  }
  Ctx& c_;
  size_t i_ = 0;
  bool done_ = false;
};

void Ctx::read_unit_host(int layer, int unit, void* dst, bool hash, uint64_t* digest, uint64_t* progress) {
  const Layer& l = d_.layers[size_t(layer)];
  Xxh64 h;
  uint64_t pos = 0;
  for (const Part& part : l.parts) {
    const File& f = *files_[size_t(part.source)];
    const uint64_t al = d_.sources[size_t(part.source)].align;
    const uint64_t F = part.offset + uint64_t(unit) * part.stride;
    char* out = at(dst, pos);
    require(F + part.bytes <= f.size(), SS_E_IO, "unit (" + std::to_string(layer) + ", " + std::to_string(unit) +
                                                     ") part " + part.name + " is past the end of its source");
    bool in_place = !f.direct() || (F % al == 0 && part.bytes % al == 0 && uintptr_t(out) % al == 0);
    if (in_place) {
      require(f.pread(out, part.bytes, F) == part.bytes, SS_E_IO, "short read");
    } else {
      // Covering aligned range through a staging buffer.
      const uint64_t cap = (o_.staging_bytes - 2 * al) / al * al;
      for (uint64_t c = 0; c < part.bytes;) {
        uint64_t len = std::min(cap, part.bytes - c);
        uint64_t a0 = (F + c) / al * al, delta = F + c - a0;
        uint64_t a1 = round_up(F + c + len, al);
        StagingLease buf(*this);
        require(f.pread(buf.host(), a1 - a0, a0) >= delta + len, SS_E_IO, "short read");
        std::memcpy(out + c, at(buf.host(), delta), size_t(len));
        c += len;
      }
    }
    if (hash) h.update(out, size_t(part.bytes));
    pos += part.bytes;
    if (progress) {
      *progress = pos;
      if (test_inject_.load() == 8 && test_inject_.exchange(0) == 8)
        fail(SS_E_IO, "read-ahead: injected failure after a part");
    }
  }
  if (digest) *digest = h.digest();
}

void Ctx::read_unit_device(Job& j) {
  const Layer& l = d_.layers[size_t(j.layer)];
  bool want_hash = o_.verify_hash && !l.hash.empty();
  Xxh64 h;
  uint64_t pos = 0;
  bool waited = false;
  for (const Part& part : l.parts) {
    const File& f = *files_[size_t(part.source)];
    const uint64_t al = f.direct() ? d_.sources[size_t(part.source)].align : 1;
    const uint64_t F = part.offset + uint64_t(j.unit) * part.stride;
    require(F + part.bytes <= f.size(), SS_E_IO, "unit (" + std::to_string(j.layer) + ", " + std::to_string(j.unit) +
                                                     ") part " + part.name + " is past the end of its source");
    const uint64_t cap = (o_.staging_bytes - 2 * std::max<uint64_t>(al, kHostAlign)) / kHostAlign * kHostAlign;
    for (uint64_t c = 0; c < part.bytes;) {
      uint64_t len = std::min(cap, part.bytes - c);
      uint64_t a0 = (F + c) / al * al, delta = F + c - a0;
      uint64_t a1 = round_up(F + c + len, al);
      StagingLease buf(*this);  // waits for the previous H2D out of this buffer
      require(f.pread(buf.host(), a1 - a0, a0) >= delta + len, SS_E_IO, "short read");
      const char* src = at(buf.host(), delta);
      if (want_hash) h.update(src, size_t(len));
      std::lock_guard<std::mutex> lk(copy_mu_);
      if (!waited) {
        be_->stream_wait(nullptr, j.dst_release_ev);
        be_->stream_wait(nullptr, j.dst_wb_ev);
        waited = true;
      }
      be_->copy_h2d(at(j.dst, pos + c), src, size_t(len), nullptr);
      be_->event_record(buf.ev(), nullptr);
      c += len;
    }
    pos += part.bytes;
  }
  if (want_hash && h.digest() != l.hash[size_t(j.unit)])
    fail(SS_E_HASH, "unit (" + std::to_string(j.layer) + ", " + std::to_string(j.unit) +
                        ") bytes differ from the descriptor hash");
  std::lock_guard<std::mutex> lk(copy_mu_);
  be_->event_record(j.done_ev, nullptr);
}

// ---- jobs and placement --------------------------------------------------------------------------------------------

JobPtr Ctx::make_job(int layer, int unit, void* dst, bool spec) {
  JobPtr j = std::make_shared<Job>();
  j->layer = layer;
  j->unit = unit;
  j->spec = spec;
  j->dst = dst;
  j->dst_release_ev = L_[size_t(layer)].release_ev;
  j->host_src = pinned_of_[size_t(gidx(layer, unit))];
  j->from_pinned = j->host_src != nullptr;
  if (excl_ && j->from_pinned) {
    j->src_wait_ev = wb_pending_[size_t(gidx(layer, unit))];
    if (j->src_wait_ev) ++xst_.rereads;
  }
  if (!be_->unified()) {
    if (!ev_pool_.empty()) {
      j->done_ev = ev_pool_.back();
      ev_pool_.pop_back();
    } else {
      j->done_ev = be_->event_create();
    }
  }
  return j;
}

void Ctx::fetch_into_persistent(int layer, int unit, int slot, bool spec, void* dst_wb_ev) {
  LayerState& s = L_[size_t(layer)];
  if (ra_on() && spec && s.ra[size_t(unit)]) {  // a speculative admission takes the ordinary path, as in the ring
    std::lock_guard<std::mutex> lk(q_mu_);
    ra_abandon_locked(*s.ra[size_t(unit)]);
  }
  JobPtr j = make_job(layer, unit, at(s.pool, uint64_t(slot) * d_.unit_stride(layer)), spec);
  j->dst_wb_ev = dst_wb_ev;
  fwd_jobs_.push_back(j);  // tracked before anything points at it (end() drains and rolls back what it tracks)
  s.loc[size_t(unit)] = Persistent;
  s.slot[size_t(unit)] = slot;
  s.job[size_t(unit)] = j;
  // Read-ahead: as in fetch_into_ring, the copy out of the buffer counts in neither bytes_disk nor bytes_pinned (the
  // file bytes are counted once, at ss_end); the landing keeps both destination waits (release, victim write-back).
  if (ra_on() && !spec && ra_land(layer, unit, j)) return;
  (j->from_pinned ? fwd_stats_[size_t(layer)].bytes_pinned : fwd_stats_[size_t(layer)].bytes_disk) +=
      d_.layers[size_t(layer)].unit_bytes;
  submit(j);
}

int Ctx::block_class(int layer) const { return excl_ ? layer_class_[size_t(layer)] : -1; }

// ---- host read-ahead ---------------------------------------------------------------------------------------------------

void Ctx::test_ra_quiesce() {
  for (const JobPtr& p : ra_jobs_) wait_job_quiet(*p);
}

void Ctx::fail_job_locked(Job& j, ss_status st, const char* msg) noexcept {
  j.err_status = st;
  try {
    j.err = msg;
  } catch (...) {
  }
  j.state.store(Job::Failed);
}

// The one way a buffer goes back: consumes the handle, so a second call is a no-op. Quarantined buffers never return.
void Ctx::ra_take_buf_locked(RaEntry& e) {
  if (e.buf < 0 || e.quarantined) return;
  ra_free_.push_back(uint32_t(e.buf));  // capacity = every buffer, reserved at open: never allocates
  e.buf = -1;
}

void Ctx::ra_abandon_locked(RaEntry& e) {
  switch (e.st) {
    case RaEntry::Waiting:
      e.st = RaEntry::Abandoned;
      break;
    case RaEntry::Ready:  // a retained result nobody will land
      ra_take_buf_locked(e);
      e.st = RaEntry::Abandoned;
      break;
    case RaEntry::Reading: {
      int expect = Job::Queued;
      if (e.producer->state.compare_exchange_strong(expect, Job::Cancelled)) {
        auto it = std::find(ra_q_.begin(), ra_q_.end(), e.producer);
        if (it != ra_q_.end()) ra_q_.erase(it);
        ra_take_buf_locked(e);
      }  // Running: its epilogue sees Abandoned and takes the buffer back
      e.st = RaEntry::Abandoned;
      break;
    }
    default:  // Landing (its child decides), Consumed, Failed, Abandoned
      break;
  }
}

void Ctx::ra_producer_epilogue_locked(Job& j) {
  RaEntry& e = *j.ra;
  if (j.state.load() == Job::Done) {
    if (e.st == RaEntry::Landing && e.child && !e.child_queued) {
      if (e.child->state.load() == Job::Queued) {
        try {
          if (test_inject_.load() == 2 && test_inject_.exchange(0) == 2) throw std::bad_alloc();
          demand_q_.push_front(e.child);
          e.child_queued = true;
        } catch (...) {
          fail_job_locked(*e.child, SS_E_NOMEM, "read-ahead: queueing a landing failed");
          ra_take_buf_locked(e);
          e.st = RaEntry::Failed;
        }
      } else {  // the child was cancelled while it waited for this read
        ra_take_buf_locked(e);
      }
    } else if (e.st == RaEntry::Abandoned) {
      ra_take_buf_locked(e);
    } else if (e.st == RaEntry::Reading) {
      e.st = RaEntry::Ready;  // kept for a later landing
    }
  } else {  // Failed: a failed buffer is never copied
    if (e.child && e.child->state.load() == Job::Queued) fail_job_locked(*e.child, j.err_status, j.err.c_str());
    ra_take_buf_locked(e);
    if (e.st != RaEntry::Abandoned) e.st = RaEntry::Failed;
  }
}

void Ctx::ra_child_epilogue_locked(Job& j) {
  RaEntry& e = *j.ra;
  const int st = j.state.load();
  if (st == Job::Failed && j.issued_piece) {  // copies out of the buffer may still be in flight: never reuse it
    e.quarantined = true;
    e.st = RaEntry::Failed;
    return;
  }
  // Done (its buffer event is recorded), failed before any copy, or cancelled: it no longer reads the buffer.
  const int ps = e.producer ? e.producer->state.load() : Job::Done;  // cleared at reconcile: it had ended
  if (ps == Job::Done || ps == Job::Failed || ps == Job::Cancelled) ra_take_buf_locked(e);
  e.st = st == Job::Done ? RaEntry::Consumed : st == Job::Failed ? RaEntry::Failed : RaEntry::Abandoned;
}

void Ctx::ra_sweep_locked() {
  for (auto& s : L_)
    for (auto& e : s.ra)
      if (e) ra_abandon_locked(*e);
  ra_wait_.clear();
}

// After the forward's producers and jobs ended: count each entry once, then drop the entries (a job that still holds
// one keeps it alive).
void Ctx::ra_drain_and_reconcile() {
  ra_reconcile();
  ra_teardown();
}

// Once per entry, after its producer and child ended: requests, reads, landings, and the file bytes actually read
// (a read whose hash failed still read its bytes).
void Ctx::ra_reconcile() {
  for (size_t L = 0; L < L_.size(); ++L) {
    for (auto& e : L_[L].ra) {
      if (!e) continue;
      ++ra_st_.requested;
      const int ps = e->producer ? e->producer->state.load() : Job::Cancelled;
      const uint64_t b = e->producer ? e->producer->bytes_read : 0;
      ra_st_.bytes += b;
      fwd_stats_[L].bytes_disk += b;
      if (ps == Job::Done) {
        ++ra_st_.read;
        if (e->st == RaEntry::Consumed) ++ra_st_.landed;
        else ++ra_st_.unused;
      } else if (ps == Job::Failed) {
        ++ra_st_.failed;
      }
    }
  }
}

void Ctx::ra_teardown() {
  {
    // Break the entry <-> job references (a cycle otherwise); every producer and child has ended.
    std::lock_guard<std::mutex> lk(q_mu_);
    for (auto& s : L_)
      for (auto& e : s.ra)
        if (e) {
          e->producer.reset();
          e->child.reset();
        }
  }
  for (auto& s : L_)
    for (auto& e : s.ra) e.reset();
  ra_jobs_.clear();
}

bool Ctx::ra_eligible(int layer, int unit, bool pending_ok) const {
  const LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  if (!s.rs || !l.streamable || unit < 0 || unit >= l.n_units || s.ra[size_t(unit)]) return false;
  const int64_t g = gidx(layer, unit);
  if (plan_.tier[size_t(g)] == SS_TIER_PAIR || pinned_of_[size_t(g)] || s.rs->slot(unit) >= 0) return false;
  // Never a unit larger than the buffers (sized for the units the plan can leave without a pinned source). Defensive:
  // an exclusive discard needs more victims than free blocks, which only resident or disk units of the block size
  // (that is, of a size the buffers hold) can cause; a block size of only pinned units has a free block per victim.
  if (host_stride(l.unit_bytes) > ra_bytes_) return false;
  if (s.job[size_t(unit)]) return false;
  if (s.loc[size_t(unit)] != None) return pending_ok && s.loc[size_t(unit)] == Ring;  // waiting for a ring slot
  return true;
}

void Ctx::ra_request(int layer, int unit, bool pending_ok) {
  if (!ra_on() || ra_frozen_ || !ra_eligible(layer, unit, pending_ok)) return;
  auto e = std::make_shared<RaEntry>();
  e->layer = layer;
  e->unit = unit;
  L_[size_t(layer)].ra[size_t(unit)] = e;  // a tombstone from here to ss_end: never requested twice this forward
  ra_wait_.push_back(e);
}

// Starts waiting requests while buffers are free (engine thread, at its calls; never inside ss_end).
void Ctx::ra_pump() {
  if (!ra_on() || ra_frozen_) return;
  while (!ra_wait_.empty()) {
    std::shared_ptr<RaEntry> e = ra_wait_.front();
    const LayerState& s = L_[size_t(e->layer)];
    const int64_t g = gidx(e->layer, e->unit);
    const bool miss = s.rs->slot(e->unit) < 0 && !s.job[size_t(e->unit)] && !pinned_of_[size_t(g)];
    if (e->st != RaEntry::Waiting || !miss) {
      {
        std::lock_guard<std::mutex> lk(q_mu_);
        if (e->st == RaEntry::Waiting) e->st = RaEntry::Abandoned;
      }
      ra_wait_.pop_front();
      continue;
    }
    JobPtr p = std::make_shared<Job>();  // allocated outside the lock
    p->layer = e->layer;
    p->unit = e->unit;
    p->host_dst = true;
    p->ra_producer = true;
    p->ra = e;
    bool started = false, failed = false;
    {
      std::lock_guard<std::mutex> lk(q_mu_);
      if (ra_free_.empty()) return;  // no buffer: the next engine call tries again
      e->buf = int32_t(ra_free_.back());
      ra_free_.pop_back();
      p->dst = ra_host_[size_t(e->buf)];
      e->producer = p;
      e->st = RaEntry::Reading;
      ra_jobs_.push_back(p);  // reserved for every unit at open: never reallocates
      try {
        if (test_inject_.load() == 1 && test_inject_.exchange(0) == 1) throw std::bad_alloc();
        ra_q_.push_back(p);
        started = true;
        ++ra_started_;
      } catch (...) {  // never left nonterminal: the unit's fetch reads its file
        fail_job_locked(*p, SS_E_NOMEM, "read-ahead: queueing a read failed");
        ra_take_buf_locked(*e);
        e->st = RaEntry::Failed;
        failed = true;
      }
    }
    ra_wait_.pop_front();
    if (started) q_cv_.notify_all();
    if (failed) done_cv_.notify_all();
  }
}

// The unit got a device slot (ring or persistent; `child` is its job): land from its read-ahead buffer when it has one
// (true), else the caller reads the file (false).
bool Ctx::ra_land(int layer, int unit, const JobPtr& child) {
  std::shared_ptr<RaEntry> ep = L_[size_t(layer)].ra[size_t(unit)];
  if (!ep) return false;
  RaEntry& e = *ep;
  bool attached = false, failed_read = false, failed_child = false;
  {
    std::lock_guard<std::mutex> lk(q_mu_);
    switch (e.st) {
      case RaEntry::Waiting:  // not started: the ordinary read starts as early
        e.st = RaEntry::Abandoned;
        break;
      case RaEntry::Ready:
        child->ra_child = true;
        child->ra = ep;
        e.child = child;
        try {
          if (test_inject_.load() == 2 && test_inject_.exchange(0) == 2) throw std::bad_alloc();
          demand_q_.push_front(child);
          e.child_queued = true;
          e.st = RaEntry::Landing;
          attached = true;
        } catch (...) {  // as in the producer's epilogue: the landing fails terminally (no timing-dependent fallback)
          fail_job_locked(*child, SS_E_NOMEM, "read-ahead: queueing a landing failed");
          ra_take_buf_locked(e);
          e.st = RaEntry::Failed;
          attached = true;
          failed_child = true;
        }
        break;
      case RaEntry::Reading:
        child->ra_child = true;
        child->ra = ep;
        e.child = child;
        e.st = RaEntry::Landing;
        attached = true;  // the producer's epilogue queues the child
        if (e.producer->state.load() == Job::Queued) {  // a demand now waits behind it: promote it
          auto it = std::find(ra_q_.begin(), ra_q_.end(), e.producer);
          if (it != ra_q_.end()) {
            try {
              demand_q_.push_front(e.producer);
              ra_q_.erase(std::find(ra_q_.begin(), ra_q_.end(), e.producer));
            } catch (...) {
              // stays in the read-ahead queue: still runs, only later
            }
          }
        }
        break;
      case RaEntry::Failed:
        failed_read = true;
        break;
      default:
        break;
    }
  }
  if (attached) q_cv_.notify_all();
  if (failed_child) done_cv_.notify_all();
  if (failed_read) ++ra_st_.fallback;
  return attached;
}

// FromRA copy: the RA buffer into the slot, after the slot's previous reader (ring owner / layer release) and, in a
// persistent slot of the exclusive mode, after the write-back of the victim that held it.
void Ctx::run_ra_child(Job& j) {
  const Layer& l = d_.layers[size_t(j.layer)];
  const size_t b = size_t(j.ra->buf);  // stable while this job runs: only a terminal child gives it back
  const char* src = static_cast<const char*>(ra_host_[b]);
  if (test_inject_.load() == 3 && test_inject_.exchange(0) == 3) fail(SS_E_IO, "read-ahead: injected copy failure");
  // Test-only mutation 10 (must make the CUDA test fail): skip the wait for the victim's write-back once.
  const bool skip_wb = j.dst_wb_ev && test_inject_.load() == 10 && test_inject_.exchange(0) == 10;
  if (skip_wb)  // the mutation reports that it acted (a test that cannot observe it says so, not a silent pass)
    std::fprintf(stderr, "mutation 10: write-back wait skipped (unit %d, %d)\n", j.layer, j.unit);
  if (be_->unified()) {
    // Test-only mutation 6 (must make the GPU test fail): skip this landing's wait for the slot's previous reader.
    if (!(test_inject_.load() == 6 && test_inject_.exchange(0) == 6)) be_->event_sync(j.dst_release_ev);
    if (!skip_wb) be_->event_sync(j.dst_wb_ev);
    j.issued_piece = true;
    std::memcpy(j.dst, src, size_t(l.unit_bytes));
    if (test_inject_.load() == 4 && test_inject_.exchange(0) == 4) fail(SS_E_IO, "read-ahead: injected copy failure");
    return;
  }
  std::lock_guard<std::mutex> lk(copy_mu_);
  // Test-only mutation 6 (must make the CUDA test fail): skip this landing's wait for the slot's previous reader.
  if (!(test_inject_.load() == 6 && test_inject_.exchange(0) == 6)) be_->stream_wait(nullptr, j.dst_release_ev);
  if (!skip_wb) be_->stream_wait(nullptr, j.dst_wb_ev);
  for (uint64_t off = 0; off < l.unit_bytes; off += o_.copy_piece_bytes) {
    be_->copy_h2d(at(j.dst, off), src + off, size_t(std::min<uint64_t>(o_.copy_piece_bytes, l.unit_bytes - off)),
                  nullptr);
    j.issued_piece = true;
    if (test_inject_.load() == 4 && test_inject_.exchange(0) == 4) fail(SS_E_IO, "read-ahead: injected copy failure");
  }
  be_->event_record(ra_ev_[b], nullptr);  // the next read into this buffer host-waits it
  be_->event_record(j.done_ev, nullptr);
}

// Exclusive tier mode: free blocks an admission may write victims back into. A speculative admission
// (spec_persistent) leaves half of the class's spares to demands.
uint64_t Ctx::writeback_blocks(int layer, bool speculative) const {
  const int cls = block_class(layer);
  if (cls < 0) return 0;
  uint64_t f = free_blocks_[size_t(cls)].size();
  if (speculative) {
    const uint64_t reserve = (spares_of_[size_t(cls)] + 1) / 2;
    f = f > reserve ? f - reserve : 0;
  }
  return f;
}

// Exclusive tier mode: evictions this admission may make, one free block per victim; a demand admission is uncapped
// with exclusive_discard (a victim past the free blocks is discarded to its file). A speculative admission keeps the
// cap: a guess never discards a resident.
int Ctx::eviction_cap(int layer, bool speculative) const {
  if (!excl_) return -1;
  if (block_class(layer) < 0) return 0;
  if (o_.exclusive_discard && !speculative) return -1;
  return int(std::min<uint64_t>(writeback_blocks(layer, speculative), uint64_t(INT32_MAX)));
}

// Exclusive tier mode: the admission's victims leave their slots for free blocks (device to host on the write-back
// stream) before anything is written into those slots; admission capped the evictions at the free blocks. With
// exclusive_discard a demand admission's evictions are uncapped: the first *n_wb victims are written back while free
// blocks last, the rest are discarded (no copy, no block: weights are immutable and every unit's bytes are in its
// file, so the unit is file-homed from here, exactly like a disk unit). Returns the event the admitted units' writes into the written-back
// victims' slots wait for (nullptr on synchronous backends or when nothing is written back).
void* Ctx::write_back(int layer, const Admission& a, bool speculative, size_t* n_wb) {
  const LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  const int cls = block_class(layer);
  const uint64_t room = writeback_blocks(layer, speculative);
  if (cls < 0 || (!o_.exclusive_discard && room < a.evicted.size()))
    fail(SS_E_BACKEND, "exclusive: more victims than free blocks (internal)");
  const size_t nw = size_t(std::min<uint64_t>(room, a.evicted.size()));
  *n_wb = nw;
  for (size_t i = 0; i < a.evicted.size(); ++i)
    if (pinned_of_[size_t(gidx(layer, a.evicted[i]))] != nullptr)
      fail(SS_E_BACKEND, "exclusive: an evicted resident still owns a block (internal)");
  xst_.discards += a.evicted.size() - nw;  // a discard is no write-back: no bytes, no block, no event
  if (!nw) return nullptr;
  void* ev = nullptr;
  if (!be_->unified()) {
    // Blocks freed by earlier forwards: their last H2D reader is before blocks_ev_ on the copy stream, and so is the
    // last copy into any slot that holds a resident of an earlier forward (every victim).
    if (wb_gate_) {
      be_->stream_wait(wb_stream_, blocks_ev_);
      wb_gate_ = false;
    }
    if (!ev_pool_.empty()) {
      ev = ev_pool_.back();
      ev_pool_.pop_back();
    } else {
      ev = be_->event_create();
    }
    try {
      wb_evs_.push_back(ev);
    } catch (...) {
      be_->event_destroy(ev);
      throw;
    }
  }
  for (size_t i = 0; i < nw; ++i) {
    const int64_t g = gidx(layer, a.evicted[i]);
    void* b = free_blocks_[size_t(cls)].back();
    free_blocks_[size_t(cls)].pop_back();
    pinned_of_[size_t(g)] = b;
    be_->copy_d2h(b, at(s.pool, uint64_t(a.evicted_slot[i]) * d_.unit_stride(layer)), size_t(l.unit_bytes),
                  wb_stream_);
    ++xst_.writebacks;
    xst_.writeback_bytes += l.unit_bytes;
  }
  if (ev) be_->event_record(ev, wb_stream_);
  // Published after the record: a read of one of these blocks in the same forward waits for it.
  for (size_t i = 0; i < nw; ++i) {
    const int64_t g = gidx(layer, a.evicted[i]);
    wb_pending_[size_t(g)] = ev;
    wb_units_.push_back(g);  // reserved for every unit at open: never reallocates
  }
  return ev;
}

void Ctx::fetch_admitted(int layer, const Admission& a, bool spec) {
  size_t n_wb = 0;
  void* ev = excl_ && !a.evicted.empty() ? write_back(layer, a, spec, &n_wb) : nullptr;
  for (size_t k = 0; k < a.admitted.size(); ++k) {
    void* wait = nullptr;  // only a write into a written-back victim's slot waits for the victims' write-back
    if (ev)
      for (size_t i = 0; i < n_wb && !wait; ++i)
        if (a.evicted_slot[i] == a.slot_of[k]) wait = ev;
    fetch_into_persistent(layer, a.admitted[k], a.slot_of[k], spec, wait);
  }
}

// A unit that landed in a slot from its block no longer owns the block (its H2D was issued; the block's next writer
// waits for that copy through blocks_ev_ or a stream sync).
void Ctx::free_block(int layer, int unit) {
  const int64_t g = gidx(layer, unit);
  void* b = const_cast<void*>(pinned_of_[size_t(g)]);
  if (!b) return;
  free_blocks_[size_t(block_class(layer))].push_back(b);  // capacity reserved at open
  pinned_of_[size_t(g)] = nullptr;
  ++xst_.promoted;
}

bool Ctx::fetch_into_ring(int layer, int unit, bool spec) {
  const uint64_t stride = d_.unit_stride(layer);
  auto rc = std::find_if(ring_classes_.begin(), ring_classes_.end(),
                         [&](const RingRange& r) { return r.stride == stride; });
  if (rc == ring_classes_.end()) return false;
  for (size_t k = 0; k < rc->count; ++k) {
    size_t i = rc->first + (rc->cursor + k) % rc->count;
    if (ring_[i].owner >= 0) continue;
    rc->cursor = (i - rc->first + 1) % rc->count;
    LayerState& s = L_[size_t(layer)];
    JobPtr j = make_job(layer, unit, ring_[i].dev, spec);
    fwd_jobs_.push_back(j);
    // The slot's previous reader is the layer that released it last (its latest record is at or after that use).
    j->dst_release_ev = ring_owner_ev_[i];
    ring_[i].owner = layer;
    ring_[i].job = j;
    s.loc[size_t(unit)] = Ring;
    s.slot[size_t(unit)] = int32_t(i);
    s.job[size_t(unit)] = j;
    // Read-ahead: the unit's bytes are (being) read into a host buffer; the job copies them from there. Its file bytes
    // are counted once, at ss_end; the copy counts in neither bytes_disk nor bytes_pinned.
    if (ra_on() && !spec && ra_land(layer, unit, j)) return true;
    (j->from_pinned ? fwd_stats_[size_t(layer)].bytes_pinned : fwd_stats_[size_t(layer)].bytes_disk) +=
        d_.layers[size_t(layer)].unit_bytes;
    submit(j);
    return true;
  }
  return false;
}

void Ctx::dispatch_pending() {
  for (size_t L = 0; L < L_.size(); ++L) {
    LayerState& s = L_[L];
    if (s.pending.empty()) continue;
    size_t k = 0;
    while (k < s.pending.size() && fetch_into_ring(int(L), s.pending[k], false)) ++k;
    s.pending.erase(s.pending.begin(), s.pending.begin() + std::ptrdiff_t(k));
    if (!s.pending.empty()) return;  // keep layer order: a later layer never overtakes an earlier one
  }
}

void* Ctx::unit_addr(int layer, int unit) const {
  const LayerState& s = L_[size_t(layer)];
  if (s.loc[size_t(unit)] == Ring) return ring_[size_t(s.slot[size_t(unit)])].dev;
  int slot = s.rs->slot(unit);
  if (slot < 0) return nullptr;
  return at(s.pool, uint64_t(slot) * d_.unit_stride(layer));
}

// ---- forward API -----------------------------------------------------------------------------------------------------

void Ctx::check_stream(void* stream, const char* what) {
  // Every event the library records or waits on sits on one engine stream per forward (design.md §6.1).
  if (!fwd_stream_set_) {
    fwd_stream_ = stream;
    fwd_stream_set_ = true;
    return;
  }
  require(stream == fwd_stream_, SS_E_INVALID,
          std::string(what) + ": a forward must use one engine stream (join other streams into it first)");
}

void Ctx::check_layer(int layer, const char* what) const {
  require(layer >= 0 && layer < n_layers(), SS_E_INVALID, std::string(what) + ": layer out of range");
}

void Ctx::open_forward() {
  if (fwd_open_) return;
  for (size_t L = 0; L < L_.size(); ++L) {
    LayerState& s = L_[L];
    std::fill(s.loc.begin(), s.loc.end(), int8_t(None));
    std::fill(s.slot.begin(), s.slot.end(), -1);
    std::fill(s.job.begin(), s.job.end(), nullptr);
    std::fill(s.spec_unclaimed.begin(), s.spec_unclaimed.end(), char(0));
    std::fill(s.count.begin(), s.count.end(), 0u);
    s.routed.clear();
    s.pending.clear();
    s.hits.clear();
    s.groups.clear();
    s.group_view.clear();
    s.routes_given = s.acquired = s.released = s.skip = false;
    s.ids_n = -1;
  }
  std::fill(fwd_stats_.begin(), fwd_stats_.end(), ss_layer_stats{});
  fwd_open_ = true;
}

void Ctx::seed(int layer, const int32_t* units, int n) {
  check_layer(layer, "seed");
  require(!in_forward_ && !fwd_open_, SS_E_INVALID, "seed: only between forwards");
  LayerState& s = L_[size_t(layer)];
  require(s.rs != nullptr, SS_E_INVALID, "seed: layer has no units");
  require(n >= 0 && (units || n == 0), SS_E_INVALID, "seed: bad unit list");
  for (int i = 0; i < n; ++i)  // the whole list is valid before anything is placed
    require(units[i] >= 0 && units[i] < d_.layers[size_t(layer)].n_units, SS_E_INVALID, "seed: unit out of range");
  // Bookkeeping is reserved before anything is placed, so recording a placement never throws.
  std::vector<JobPtr> jobs;
  std::vector<int32_t> placed;
  jobs.reserve(size_t(n));
  placed.reserve(size_t(n));
  ss_status failed = SS_OK;
  std::string err;
  try {
    for (int i = 0; i < n; ++i) {
      int u = units[i];
      if (s.rs->slot(u) >= 0 || plan_.tier[size_t(gidx(layer, u))] == SS_TIER_PAIR) continue;
      int slot = s.rs->place(u, false);
      if (slot < 0) break;  // no free slot left: seeds never evict
      placed.push_back(u);
      JobPtr j = make_job(layer, u, at(s.pool, uint64_t(slot) * d_.unit_stride(layer)), false);
      jobs.push_back(j);
      submit(j);
    }
  } catch (const Error& e) {
    failed = e.status();
    keep_message(err, e.what());
  } catch (const std::bad_alloc&) {
    failed = SS_E_NOMEM;
  } catch (const std::exception& e) {
    failed = SS_E_BACKEND;
    keep_message(err, e.what());
  }
  // A placed seed without a finished read (never submitted, or failed) must not stay resident. Every job in `jobs`
  // was submitted, and a submission that failed published Failed, so these waits always end.
  std::vector<int32_t> landed;
  landed.reserve(placed.size());
  for (int32_t u : placed) {
    bool ok = false;
    for (const JobPtr& j : jobs)
      if (j->unit == u) {
        wait_job_quiet(*j);
        ok = j->state.load() == Job::Done;
      }
    if (!ok) s.rs->drop(u);
    else landed.push_back(u);
  }
  // Report the first failed read (its unit was dropped above: it would be a hit on garbage).
  for (const JobPtr& j : jobs)
    if (j->state.load() != Job::Done && failed == SS_OK) {
      failed = j->err_status;
      err = j->err;
    }
  for (const JobPtr& j : jobs)
    if (j->done_ev) ev_pool_.push_back(j->done_ev);
  bool synced = false;
  try {
    be_->stream_sync(nullptr);
    synced = true;
  } catch (const Error& e) {
    if (failed == SS_OK) {
      failed = e.status();
      keep_message(err, e.what());
    }
  }
  // Exclusive: a seeded unit's copy out of its block has completed (stream sync): the block is free at once.
  if (excl_ && synced)
    for (int32_t u : landed) free_block(layer, u);
  if (failed != SS_OK) fail(failed, err);
}

void Ctx::begin(const uint32_t* rows, int n_layers_in) {
  require(!in_forward_, SS_E_INVALID, "begin: previous forward not ended");
  require(rows == nullptr || n_layers_in == n_layers(), SS_E_INVALID, "begin: rows_per_layer needs one per layer");
  open_forward();
  if (rows)
    for (size_t L = 0; L < L_.size(); ++L) L_[L].skip = rows[L] == 0;
  if (excl_) {
    uint64_t f = 0;
    for (const auto& v : free_blocks_) f += v.size();
    xst_.free_blocks = f;
    xst_.min_free_blocks = std::min(xst_.min_free_blocks, f);
  }
  in_forward_ = true;
}

void Ctx::ingest_routes(int layer, const int32_t* ids, int n) {
  LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  require(s.rs != nullptr, SS_E_INVALID, "routes: layer " + std::to_string(layer) + " has no units");
  require(!s.routes_given, SS_E_INVALID, "routes: given twice for layer " + std::to_string(layer));
  require(!s.skip, SS_E_INVALID, "routes: layer " + std::to_string(layer) + " is skipped in this forward");
  s.routes_given = true;
  for (int i = 0; i < n; ++i) {
    int32_t u = ids[i];
    if (u < 0) continue;
    require(u < l.n_units, SS_E_INVALID, "routes: unit id " + std::to_string(u) + " out of range");
    ++s.count[size_t(u)];
  }
  ss_layer_stats& st = fwd_stats_[size_t(layer)];
  std::vector<int32_t> misses, to_admit;
  std::vector<char> busy(size_t(l.n_units), 0);
  for (int u = 0; u < l.n_units; ++u) {
    if (s.loc[size_t(u)] != None) busy[size_t(u)] = 1;
    if (!s.count[size_t(u)] || plan_.tier[size_t(gidx(layer, u))] == SS_TIER_PAIR) continue;
    busy[size_t(u)] = 1;  // a hit of this route must never be the victim of one of its misses
    s.routed.push_back(u);
    if (s.loc[size_t(u)] != None) {  // prefetched this forward
      if (s.spec_unclaimed[size_t(u)]) {
        s.spec_unclaimed[size_t(u)] = 0;
        ++st.spec_used;
        if (s.job[size_t(u)]) promote(s.job[size_t(u)]);
      } else {
        ++st.prefetched;
      }
      misses.push_back(u);
    } else if (s.rs->slot(u) >= 0) {
      s.hits.push_back(u);
    } else {
      to_admit.push_back(u);
      misses.push_back(u);
    }
  }
  st.routed += s.routed.size();
  st.hits += s.hits.size();
  Admission a = s.rs->admit(to_admit, busy, true, eviction_cap(layer, false));
  xst_.deferred += uint64_t(a.deferred);
  fetch_admitted(layer, a, false);
  for (int32_t u : a.transient)
    if (!fetch_into_ring(layer, u, false)) {
      s.pending.push_back(u);
      s.loc[size_t(u)] = Ring;  // placed when a ring slot frees
      ra_request(layer, u, true);
    }
  st.admitted += a.admitted.size();
  st.transient += a.transient.size();
  st.evicted += a.evicted.size();
  s.rs->observe(s.routed, misses);
}

void Ctx::routes(int layer, const int32_t* ids, int n) {
  check_layer(layer, "routes");
  require(in_forward_, SS_E_INVALID, "routes: outside a forward (call ss_begin)");
  require(n >= 0 && (ids || n == 0), SS_E_INVALID, "routes: bad ids");
  ingest_routes(layer, ids, n);
  ra_pump();
}

void Ctx::routes_dev(int layer, const int32_t* ids_dev, int n, void* stream) {
  check_layer(layer, "routes_dev");
  require(in_forward_, SS_E_INVALID, "routes_dev: outside a forward (call ss_begin)");
  LayerState& s = L_[size_t(layer)];
  require(n >= 0 && s.ids_n < 0 && !s.routes_given, SS_E_INVALID, "routes_dev: given twice or bad count");
  if (size_t(n) > s.ids_cap) {
    if (s.ids_host) be_->free_pinned(s.ids_host);
    s.ids_cap = std::max<size_t>(size_t(n), 1024);
    s.ids_host = static_cast<int32_t*>(be_->alloc_pinned(sizeof(int32_t) * s.ids_cap));
  }
  check_stream(stream, "routes_dev");
  // Copied on the engine stream right behind the router; read at the next call that needs this layer's routes.
  be_->copy_d2h(s.ids_host, ids_dev, sizeof(int32_t) * size_t(n), stream);
  be_->event_record(s.ids_ev, stream);
  s.ids_n = n;
}

void Ctx::ingest_dev_routes(int layer) {
  LayerState& s = L_[size_t(layer)];
  if (s.ids_n < 0) return;
  be_->event_sync(s.ids_ev);
  int n = s.ids_n;
  s.ids_n = -1;
  ingest_routes(layer, s.ids_host, n);
}

void Ctx::routes_all(int layer) {
  check_layer(layer, "layer_all");
  require(in_forward_, SS_E_INVALID, "layer_all: outside a forward (call ss_begin)");
  int n = d_.layers[size_t(layer)].n_units;
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int u = 0; u < n; ++u) ids[size_t(u)] = u;
  ingest_routes(layer, ids.data(), n);
  ra_pump();
}

void Ctx::prefetch(int layer, const int32_t* ids, int n, bool speculative) {
  check_layer(layer, "prefetch");
  open_forward();  // allowed between forwards: the reads belong to the next forward
  LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  require(s.rs != nullptr, SS_E_INVALID, "prefetch: layer has no units");
  if (s.routes_given || s.acquired) return;  // too late to help
  ss_layer_stats& st = fwd_stats_[size_t(layer)];
  uint32_t issued = 0;
  std::vector<char> busy(size_t(l.n_units), 0);
  for (int u = 0; u < l.n_units; ++u) busy[size_t(u)] = s.loc[size_t(u)] != None;
  for (int i = 0; i < n; ++i) {
    if (speculative && issued >= o_.lookahead_budget) break;
    int32_t u = ids[i];
    if (u < 0 || u >= l.n_units) continue;
    if (s.loc[size_t(u)] != None || s.rs->slot(u) >= 0) continue;
    if (plan_.tier[size_t(gidx(layer, u))] == SS_TIER_PAIR) continue;
    // A speculative read uses only the ring unless spec_persistent: it never evicts a resident on a guess.
    Admission a =
        speculative && !o_.spec_persistent ? Admission{} : s.rs->admit({u}, busy, true, eviction_cap(layer, speculative));
    xst_.deferred += uint64_t(a.deferred);  // counted for every admission, also one that admitted nothing
    if (!a.admitted.empty()) {
      fetch_admitted(layer, a, speculative);
      st.evicted += a.evicted.size();
    } else if (!fetch_into_ring(layer, u, speculative)) {
      if (speculative) break;  // ring full: speculation never waits
      s.pending.push_back(u);
      s.loc[size_t(u)] = Ring;
      ra_request(layer, u, true);
    }
    busy[size_t(u)] = 1;
    if (speculative) {
      s.spec_unclaimed[size_t(u)] = 1;
      ++st.spec_issued;
    }
    ++issued;
  }
  ra_pump();
}

void Ctx::host_prefetch(int layer, const int32_t* ids, int n) {
  check_layer(layer, "host_prefetch");
  require(n >= 0 && (ids || n == 0), SS_E_INVALID, "host_prefetch: bad ids");
  if (!ra_on()) return;  // readahead_units = 0 (or nothing can be read from a file): a no-op
  open_forward();        // allowed between forwards: the requests belong to the next forward
  require(L_[size_t(layer)].rs != nullptr, SS_E_INVALID, "host_prefetch: layer has no units");
  for (int i = 0; i < n; ++i) ra_request(layer, ids[i], false);
  ra_pump();
}

void Ctx::prefetch_layer(int layer) {
  check_layer(layer, "prefetch_layer");
  int n = d_.layers[size_t(layer)].n_units;
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int u = 0; u < n; ++u) ids[size_t(u)] = u;
  prefetch(layer, ids.data(), n, false);
}

void Ctx::acquire(int layer, void* stream, bool wait_all, ss_layer_view* out) {
  check_layer(layer, "acquire");
  require(in_forward_, SS_E_INVALID, "acquire: outside a forward");
  LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  require(s.rs != nullptr, SS_E_INVALID, "acquire: layer has no units");
  require(!s.acquired, SS_E_INVALID, "acquire: layer acquired twice in one forward");
  check_stream(stream, "acquire");
  ingest_dev_routes(layer);
  ra_pump();
  require(s.routes_given, SS_E_INVALID, "acquire: no routes for layer " + std::to_string(layer) +
                                            " (ss_layer_routes / ss_layer_all first)");
  dispatch_pending();
  require(s.pending.empty(), SS_E_CAPACITY,
          "acquire: the stream ring cannot hold layer " + std::to_string(layer) + "'s " +
              std::to_string(s.pending.size()) + " remaining transient units (release earlier layers or plan more "
              "ring slots)");

  // Units that became resident in an earlier forward: their copies were issued then, maybe not finished. The engine
  // stream waits once for the last copy issued for this layer (device-side; a no-op when it has landed).
  if (s.ready_wait) {
    be_->stream_wait(stream, s.ready_ev);
    s.ready_wait = false;
  }

  // Groups: 0 = resident before this forward; then the fetched units by route count (desc), id.
  std::vector<int32_t> fetched;
  for (int32_t u : s.routed)
    if (s.job[size_t(u)]) fetched.push_back(u);
  std::stable_sort(fetched.begin(), fetched.end(),
                   [&](int32_t a, int32_t b) { return s.count[size_t(a)] > s.count[size_t(b)]; });
  s.groups.clear();
  s.groups.push_back(s.hits);
  size_t gsz = o_.group_units ? o_.group_units : std::max<size_t>(fetched.size(), 1);
  for (size_t k = 0; k < fetched.size(); k += gsz) {
    std::vector<int32_t> g(fetched.begin() + std::ptrdiff_t(k),
                           fetched.begin() + std::ptrdiff_t(std::min(fetched.size(), k + gsz)));
    std::sort(g.begin(), g.end());
    s.groups.push_back(g);
  }
  if (wait_all) {
    uint64_t t0 = now_ns();
    for (int32_t u : fetched) {
      wait_job(*s.job[size_t(u)]);
      if (s.job[size_t(u)]->done_ev) be_->stream_wait(stream, s.job[size_t(u)]->done_ev);
    }
    fwd_stats_[size_t(layer)].wait_ns += now_ns() - t0;
  }

  // Unit table: routed units + residents not in flight.
  s.version ^= 1;
  int v = s.version;
  be_->event_sync(s.table_ev[v]);
  uint64_t* t = s.table_host[v];
  for (int u = 0; u < l.n_units; ++u) {
    void* a = nullptr;
    if (s.job[size_t(u)] ? s.count[size_t(u)] != 0 : s.rs->slot(u) >= 0) a = unit_addr(layer, u);
    t[u] = uint64_t(reinterpret_cast<uintptr_t>(a));
  }
  if (!be_->unified()) be_->copy_h2d(s.table_dev[v], t, sizeof(uint64_t) * size_t(l.n_units), stream);
  s.group_view.clear();
  for (const auto& g : s.groups) s.group_view.push_back({g.data(), int(g.size())});
  s.acquired = true;
  out->table_dev = static_cast<const uint64_t*>(s.table_dev[v]);
  out->table_host = t;
  out->n_units = l.n_units;
  out->groups = s.group_view.data();
  out->n_groups = int(s.group_view.size());
}

void Ctx::group_wait(int layer, int group, void* stream) {
  check_layer(layer, "group_wait");
  LayerState& s = L_[size_t(layer)];
  require(s.acquired && !s.released, SS_E_INVALID, "group_wait: layer not acquired");
  require(group >= 0 && size_t(group) < s.groups.size(), SS_E_INVALID, "group_wait: no such group");
  check_stream(stream, "group_wait");
  if (group == 0) return;
  uint64_t t0 = now_ns();
  for (int32_t u : s.groups[size_t(group)]) {
    wait_job(*s.job[size_t(u)]);
    if (s.job[size_t(u)]->done_ev) be_->stream_wait(stream, s.job[size_t(u)]->done_ev);
  }
  fwd_stats_[size_t(layer)].wait_ns += now_ns() - t0;
}

void Ctx::retire_ring(int layer) {
  LayerState& s = L_[size_t(layer)];
  for (size_t i = 0; i < ring_.size(); ++i) {
    if (ring_[i].owner != layer) continue;
    Job& j = *ring_[i].job;
    int expect = Job::Queued;
    if (j.ra_child) {
      std::unique_lock<std::mutex> lk(q_mu_);
      if (j.state.compare_exchange_strong(expect, Job::Cancelled)) {
        if (j.ra->child_queued) {
          auto it = std::find(demand_q_.begin(), demand_q_.end(), ring_[i].job);
          if (it != demand_q_.end()) demand_q_.erase(it);
        }
        ra_child_epilogue_locked(j);
      } else {
        lk.unlock();
        wait_job_quiet(j);
      }
    } else if (!j.state.compare_exchange_strong(expect, Job::Cancelled)) {
      wait_job_quiet(j);
    }
    done_cv_.notify_all();
    ring_owner_ev_[i] = s.release_ev;
    ring_[i].owner = -1;
    ring_[i].job.reset();
  }
}

void Ctx::release(int layer, void* stream) {
  check_layer(layer, "release");
  LayerState& s = L_[size_t(layer)];
  require(s.acquired && !s.released, SS_E_INVALID, "release: layer not acquired or already released");
  check_stream(stream, "release");
  be_->event_record(s.release_ev, stream);
  // The table this acquire handed out is read until here: its mirror is rewritten only after this point (unified
  // backends read the mirror itself; CUDA's H2D from it sits before this point on the same stream).
  be_->event_record(s.table_ev[s.version], stream);
  s.released = true;
  for (int u = 0; u < d_.layers[size_t(layer)].n_units; ++u)
    if (s.spec_unclaimed[size_t(u)]) {
      s.spec_unclaimed[size_t(u)] = 0;
      ++fwd_stats_[size_t(layer)].spec_dropped;
    }
  retire_ring(layer);
  dispatch_pending();
  ra_pump();
}

void Ctx::wait_job_quiet(Job& j) {
  std::unique_lock<std::mutex> lk(q_mu_);
  done_cv_.wait(lk, [&] {
    int st = j.state.load();
    return st == Job::Done || st == Job::Cancelled || st == Job::Failed;
  });
}

void Ctx::end() {
  require(in_forward_, SS_E_INVALID, "end: no forward open");
  for (size_t L = 0; L < L_.size(); ++L)
    require(!L_[L].acquired || L_[L].released, SS_E_INVALID,
            "end: layer " + std::to_string(L) + " acquired but not released");
  // Read-ahead: no new request from here on; unused, waiting and unattached work is abandoned; producers are drained
  // (auxiliary: their failures are not this forward's).
  if (ra_on()) {
    {
      std::lock_guard<std::mutex> lk(q_mu_);
      ra_frozen_ = true;
      ra_sweep_locked();
    }
    done_cv_.notify_all();
    for (const JobPtr& p : ra_jobs_) wait_job_quiet(*p);
  }
  // Layers never acquired: cancel what has not started, wait for the rest, free their ring slots.
  ss_status failed = SS_OK;
  std::string err;
  for (const JobPtr& j : fwd_jobs_) {
    int expect = Job::Queued;
    if (j->spec && j->state.compare_exchange_strong(expect, Job::Cancelled)) continue;
    wait_job_quiet(*j);
    if (j->state.load() == Job::Failed && failed == SS_OK) {
      failed = j->err_status;
      err = j->err;
    }
  }
  done_cv_.notify_all();
  if (ra_on()) ra_drain_and_reconcile();
  for (size_t L = 0; L < L_.size(); ++L) {
    LayerState& s = L_[L];
    if (!s.rs) continue;
    for (int u = 0; u < d_.layers[L].n_units; ++u) {
      if (s.spec_unclaimed[size_t(u)]) ++fwd_stats_[L].spec_dropped;
      // A unit admitted to a persistent slot whose read was cancelled (speculative, never started) is not resident.
      if (s.loc[size_t(u)] == Persistent && s.job[size_t(u)] && s.job[size_t(u)]->state.load() != Job::Done)
        s.rs->drop(u);
    }
    for (size_t i = 0; i < ring_.size(); ++i)
      if (ring_[i].owner == int(L)) {
        ring_[i].owner = -1;
        ring_[i].job.reset();
      }
  }
  // Exclusive (every job of the forward has ended above): promotions free their blocks; the copy stream waits for
  // this forward's write-backs (every block they wrote is complete before a later forward reads it), then marks
  // blocks_ev_ after every H2D of this forward (the next forward's first write-back waits for it before reusing a
  // freed block or reading a slot filled in this forward).
  if (excl_) {
    for (size_t L = 0; L < L_.size(); ++L) {
      LayerState& s = L_[L];
      if (!s.rs) continue;
      for (int u = 0; u < d_.layers[L].n_units; ++u)
        if (s.loc[size_t(u)] == Persistent && s.job[size_t(u)] && s.job[size_t(u)]->from_pinned &&
            s.job[size_t(u)]->state.load() == Job::Done)
          free_block(int(L), u);
    }
    if (!be_->unified()) {
      std::lock_guard<std::mutex> lk(copy_mu_);
      if (!wb_evs_.empty()) {
        be_->event_record(wb_mark_ev_, wb_stream_);
        be_->stream_wait(nullptr, wb_mark_ev_);
      }
      be_->event_record(blocks_ev_, nullptr);
      wb_gate_ = true;
    }
    for (int64_t g : wb_units_) wb_pending_[size_t(g)] = nullptr;
    wb_units_.clear();
    for (void* e : wb_evs_) ev_pool_.push_back(e);  // every wait on them is enqueued (CUDA captures the record)
    wb_evs_.clear();
    uint64_t f = 0;
    for (const auto& v : free_blocks_) f += v.size();
    xst_.free_blocks = f;
  }
  // Persistent copies of this forward are issued (Done) but may still be in flight on the copy stream: the next
  // acquire of each such layer waits for this point (they would otherwise be hits with no wait).
  if (!be_->unified()) {
    std::lock_guard<std::mutex> lk(copy_mu_);
    for (size_t L = 0; L < L_.size(); ++L) {
      LayerState& s = L_[L];
      if (!s.rs) continue;
      bool any = false;
      for (int u = 0; u < d_.layers[L].n_units && !any; ++u)
        any = s.loc[size_t(u)] == Persistent && s.job[size_t(u)] && s.job[size_t(u)]->state.load() == Job::Done;
      if (any) {
        be_->event_record(s.ready_ev, nullptr);
        s.ready_wait = true;
      }
    }
  }
  for (const JobPtr& j : fwd_jobs_)
    if (j->done_ev) ev_pool_.push_back(j->done_ev);
  fwd_jobs_.clear();
  fwd_stream_set_ = false;
  fwd_stream_ = nullptr;
  ss_layer_stats tot{};
  for (size_t L = 0; L < L_.size(); ++L) {
    add(stats_[L], fwd_stats_[L]);
    add(tot, fwd_stats_[L]);
  }
  if (hook_) hook_(hook_user_, &tot, fwd_stats_.data(), n_layers());
  in_forward_ = false;
  fwd_open_ = false;
  ra_frozen_ = false;
  ++forward_;
  // A forward whose read failed is still folded above: it is published like any other (design.md §7.1).
  maybe_publish();
  if (failed != SS_OK) fail(failed, err);
}

void Ctx::check_unit(int layer, int unit) {
  check_layer(layer, "check_unit");
  LayerState& s = L_[size_t(layer)];
  const Layer& l = d_.layers[size_t(layer)];
  require(s.acquired && !s.released, SS_E_INVALID, "check_unit: layer not acquired");
  require(unit >= 0 && unit < l.n_units, SS_E_INVALID, "check_unit: unit out of range");
  void* a = reinterpret_cast<void*>(uintptr_t(s.table_host[s.version][unit]));
  require(a != nullptr, SS_E_INVALID, "check_unit: unit not available in this forward");
  if (s.job[size_t(unit)]) {
    wait_job(*s.job[size_t(unit)]);
    if (s.job[size_t(unit)]->done_ev) be_->event_sync(s.job[size_t(unit)]->done_ev);
  }
  std::vector<char> got(size_t(l.unit_bytes)), want(size_t(round_up(l.unit_bytes, kHostAlign)) + kHostAlign);
  be_->copy_d2h(got.data(), a, got.size(), nullptr);
  be_->stream_sync(nullptr);
  char* w = reinterpret_cast<char*>(round_up(uint64_t(reinterpret_cast<uintptr_t>(want.data())), kHostAlign));
  read_unit_host(layer, unit, w, false, nullptr);
  require(std::memcmp(got.data(), w, got.size()) == 0, SS_E_HASH,
          "check_unit: landed bytes of (" + std::to_string(layer) + ", " + std::to_string(unit) +
              ") differ from the source");
  if (!l.hash.empty())
    require(xxh64(got.data(), got.size()) == l.hash[size_t(unit)], SS_E_HASH,
            "check_unit: landed bytes of (" + std::to_string(layer) + ", " + std::to_string(unit) +
                ") differ from the descriptor hash");
}

// ---- stats -------------------------------------------------------------------------------------------------------------

ss_layer_stats Ctx::total() const {
  ss_layer_stats t{};
  for (const auto& s : stats_) add(t, s);
  return t;
}

void Ctx::stats_reset() {
  std::fill(stats_.begin(), stats_.end(), ss_layer_stats{});
  ra_st_ = RaStats{};
  // Only at a boundary (the exclusive counters move inside a forward) and never from a poisoned context: otherwise
  // the next ss_end publishes.
  if (!in_forward_ && !fwd_open_ && !poisoned_) maybe_publish();
}

void Ctx::publish() {
  std::lock_guard<std::mutex> lk(snap_mu_);
  std::copy(stats_.begin(), stats_.end(), snap_.layers.begin());
  snap_.ra = ra_st_;
  snap_.x = xst_;
  snap_.forwards = forward_;
  ++snap_.seq;
}

bool Ctx::snapshot(ss_layer_stats* total, ss_snapshot_info* info) const {
  snap_armed_.store(true, std::memory_order_relaxed);
  ss_layer_stats t{};
  ss_snapshot_info i{};
  {
    std::lock_guard<std::mutex> lk(snap_mu_);
    if (snap_.seq == 0) return false;
    for (const auto& s : snap_.layers) add(t, s);
    i.sequence = snap_.seq;
    i.forwards = snap_.forwards;
    if (ra_on()) {
      i.readahead_requested = snap_.ra.requested;
      i.readahead_read = snap_.ra.read;
      i.readahead_landed = snap_.ra.landed;
      i.readahead_unused = snap_.ra.unused;
      i.readahead_failed = snap_.ra.failed;
      i.readahead_fallback = snap_.ra.fallback;
      i.readahead_bytes = snap_.ra.bytes;
    }
    if (excl_) {
      i.exclusive_writebacks = snap_.x.writebacks;
      i.exclusive_writeback_bytes = snap_.x.writeback_bytes;
      i.exclusive_deferred = snap_.x.deferred;
      i.exclusive_promoted = snap_.x.promoted;
      i.exclusive_free_blocks = snap_.x.free_blocks;
      i.exclusive_min_free_blocks = snap_.x.min_free_blocks;
      i.exclusive_rereads = snap_.x.rereads;
      i.exclusive_discards = snap_.x.discards;
    }
  }
  // Fixed at open (never written after ss_open returned).
  if (ra_on()) {
    i.flags |= SS_SNAPSHOT_READAHEAD;
    i.readahead_buffers = uint64_t(ra_host_.size());
    i.readahead_buffer_bytes = ra_bytes_;
  }
  if (excl_) {
    i.flags |= SS_SNAPSHOT_EXCLUSIVE;
    i.exclusive_blocks = n_blocks_;
  }
  i.struct_size = uint32_t(sizeof i);
  if (total) *total = t;
  if (info) *info = i;
  return true;
}

bool Ctx::snapshot_json(Json* out) const {
  snap_armed_.store(true, std::memory_order_relaxed);
  Published p;
  p.layers.resize(d_.layers.size());  // allocated before the lock
  {
    std::lock_guard<std::mutex> lk(snap_mu_);
    if (snap_.seq == 0) return false;
    std::copy(snap_.layers.begin(), snap_.layers.end(), p.layers.begin());
    p.ra = snap_.ra;
    p.x = snap_.x;
    p.forwards = snap_.forwards;
  }
  *out = format_stats(StatsView{p.forwards, p.layers.data(), &p.ra, &p.x});
  return true;
}

Json Ctx::stats_json() const { return format_stats(StatsView{forward_, stats_.data(), &ra_st_, &xst_}); }

// The stats-v1 document of a set of counters (live or published); the fixed parts come from the context.
Json Ctx::format_stats(const StatsView& v) const {
  auto one = [](const ss_layer_stats& s) {
    Json j = Json::object();
    j["routed"] = Json::number(s.routed);
    j["hits"] = Json::number(s.hits);
    j["admitted"] = Json::number(s.admitted);
    j["transient"] = Json::number(s.transient);
    j["evicted"] = Json::number(s.evicted);
    j["prefetched"] = Json::number(s.prefetched);
    j["spec_issued"] = Json::number(s.spec_issued);
    j["spec_used"] = Json::number(s.spec_used);
    j["spec_dropped"] = Json::number(s.spec_dropped);
    j["bytes_pinned"] = Json::number(s.bytes_pinned);
    j["bytes_disk"] = Json::number(s.bytes_disk);
    j["wait_ns"] = Json::number(s.wait_ns);
    return j;
  };
  Json j = Json::object();
  j["schema"] = Json::string("strix-stream/stats-v1");
  j["backend"] = Json::string(be_->name());
  j["forwards"] = Json::number(v.forwards);
  Json direct = Json::array();
  for (const auto& f : files_) direct.push(Json::boolean(f->direct()));
  j["sources_direct"] = direct;
  const size_t n = d_.layers.size();
  ss_layer_stats t{};
  for (size_t L = 0; L < n; ++L) add(t, v.layers[L]);
  j["total"] = one(t);
  Json layers = Json::array();
  for (size_t L = 0; L < n; ++L) layers.push(one(v.layers[L]));
  j["layers"] = layers;
  if (ra_on()) {  // readahead_units = 0: the JSON is today's
    Json r = Json::object();
    r["buffers"] = Json::number(uint64_t(ra_host_.size()));
    r["buffer_bytes"] = Json::number(ra_bytes_);
    r["requested"] = Json::number(v.ra->requested);
    r["read"] = Json::number(v.ra->read);
    r["landed"] = Json::number(v.ra->landed);
    r["unused"] = Json::number(v.ra->unused);
    r["failed"] = Json::number(v.ra->failed);
    r["fallback"] = Json::number(v.ra->fallback);
    r["bytes"] = Json::number(v.ra->bytes);
    j["readahead"] = r;
  }
  if (excl_) {
    Json x = Json::object();
    x["writebacks"] = Json::number(v.x->writebacks);
    x["writeback_bytes"] = Json::number(v.x->writeback_bytes);
    x["discards"] = Json::number(v.x->discards);
    x["deferred"] = Json::number(v.x->deferred);
    x["rereads"] = Json::number(v.x->rereads);
    x["promoted"] = Json::number(v.x->promoted);
    x["free_blocks"] = Json::number(v.x->free_blocks);
    x["min_free_blocks"] = Json::number(v.x->min_free_blocks);
    x["blocks"] = Json::number(n_blocks_);
    j["exclusive"] = x;
  }
  return j;
}

void Ctx::test_hold_readers(bool hold) {
  {
    std::lock_guard<std::mutex> lk(q_mu_);
    hold_ = hold;
  }
  q_cv_.notify_all();
}

void Ctx::test_fill_free_blocks(int byte) {
  require(excl_ && !in_forward_, SS_E_INVALID, "test_fill_free_blocks: exclusive mode, between forwards");
  for (size_t k = 0; k < free_blocks_.size(); ++k)
    for (void* b : free_blocks_[k]) std::memset(b, byte, size_t(block_bytes_[k]));
}

void Ctx::test_sync_copy() { be_->stream_sync(nullptr); }

void Ctx::test_gate_writeback(void* ev) {
  require(excl_ && !be_->unified(), SS_E_INVALID, "test_gate_writeback: exclusive mode on a discrete device only");
  be_->stream_wait(wb_stream_, ev);
}

ss_exclusive_stats Ctx::exclusive_stats() const {
  ss_exclusive_stats x = xst_;
  x.struct_size = uint32_t(sizeof x);
  return x;
}

void Ctx::check_exclusive() const {
  require(excl_, SS_E_INVALID, "check_exclusive: the context is not in the exclusive tier mode");
  require(!in_forward_ && !fwd_open_, SS_E_INVALID, "check_exclusive: only between forwards");
  std::set<const void*> owned;
  for (size_t L = 0; L < d_.layers.size(); ++L) {
    const Layer& l = d_.layers[L];
    if (!L_[L].rs) continue;
    for (int u = 0; u < l.n_units; ++u) {
      const int64_t g = gidx(int(L), u);
      if (plan_.tier[size_t(g)] == SS_TIER_PAIR) continue;  // another device's unit: never here
      const void* b = pinned_of_[size_t(g)];
      require(!wb_pending_[size_t(g)], SS_E_INVALID, "check_exclusive: a write-back is still pending at a boundary");
      if (!b) continue;
      require(l.streamable, SS_E_INVALID, "check_exclusive: a non-streamable unit owns a block");
      require(L_[L].rs->slot(u) < 0, SS_E_INVALID,
              "check_exclusive: unit (" + std::to_string(L) + ", " + std::to_string(u) + ") is in a slot and a block");
      require(owned.insert(b).second, SS_E_INVALID, "check_exclusive: a block is owned twice");
    }
  }
  for (const auto& v : free_blocks_)
    for (const void* b : v) require(owned.insert(b).second, SS_E_INVALID, "check_exclusive: a free block is owned");
  require(owned.size() == n_blocks_, SS_E_INVALID,
          "check_exclusive: " + std::to_string(owned.size()) + " blocks accounted for, " + std::to_string(n_blocks_) +
              " allocated");
}

}  // namespace ts
