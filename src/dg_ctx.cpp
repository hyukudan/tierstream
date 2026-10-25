// Device-gated decode mode: the context (dg_ctx.hpp). CUDA transport, open (arena, blocks, fill), the token
// boundary with its completion edge, and close (design rev 3.2 §4.5, §5).

#include "dg_ctx.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>
#include <set>

#include "errors.hpp"

#if defined(SS_WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>

#include "dg_core.hpp"
#include "dg_kernels.hpp"
#include "dg_service.hpp"
#include "file_reader.hpp"
#include "xxh64.hpp"
#endif

namespace ts {

#if !defined(SS_WITH_CUDA)

std::unique_ptr<DgCtx> open_dg_ctx(const Desc& d, const Plan& p, const ss_options& o) {
  (void)dg_layout(d, p, o);  // every scope refusal first
  fail(SS_E_UNSUPPORTED, "device-gated: needs the CUDA backend (configure with -DSS_CUDA=ON)");
}

#else

namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    fail(e == cudaErrorMemoryAllocation ? SS_E_NOMEM : SS_E_BACKEND,
         std::string("device-gated: ") + what + ": " + cudaGetErrorString(e));
}
void ck_drv(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) {
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    fail(SS_E_BACKEND, std::string("device-gated: ") + what + ": " + (s ? s : "CUDA driver error"));
  }
}
void ck_launch(int e, const char* what) { ck(cudaError_t(e), what); }

char* at(void* p, uint64_t off) { return static_cast<char*>(p) + off; }
uint64_t round_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// Copies, publications, events: every operation on the core's own streams.
class CudaTransport final : public DgTransport {
 public:
  CudaTransport(const DgLayout& x, const std::vector<char*>& arena, int32_t* slot_of_dev, uint32_t* ready_dev,
                const std::vector<char*>& blocks)
      : x_(x), arena_(arena), slot_of_(slot_of_dev), ready_(ready_dev), blocks_(blocks) {
    try {
      int lo = 0, hi = 0;
      ck(cudaDeviceGetStreamPriorityRange(&lo, &hi), "cudaDeviceGetStreamPriorityRange");
      ck(cudaStreamCreateWithPriority(&s_[kDem], cudaStreamNonBlocking, hi), "cudaStreamCreate (demand)");
      ck(cudaStreamCreateWithPriority(&s_[kPf], cudaStreamNonBlocking, lo), "cudaStreamCreate (prefetch)");
      ck(cudaStreamCreateWithFlags(&s_[kD2h], cudaStreamNonBlocking), "cudaStreamCreate (write-back)");
      ck(cudaHostAlloc(reinterpret_cast<void**>(&table_pin_), size_t(x_.keys ? x_.keys : 1) * 4, cudaHostAllocPortable),
         "cudaHostAlloc (table staging)");
    } catch (...) {  // the destructor does not run for a throwing constructor
      for (cudaStream_t s : s_)
        if (s) cudaStreamDestroy(s);
      if (table_pin_) cudaFreeHost(table_pin_);
      throw;
    }
  }
  ~CudaTransport() override {
    if (table_pin_) cudaFreeHost(table_pin_);
    for (cudaEvent_t e : dma_a_)
      if (e) cudaEventDestroy(e);
    for (cudaEvent_t e : dma_b_)
      if (e) cudaEventDestroy(e);
    for (cudaStream_t s : s_)
      if (s) cudaStreamDestroy(s);
  }
  // dg_diag: two timing events per gated layer on the demand stream (separate from the protocol's untimed events).
  void enable_diag(uint32_t G) {
    dma_a_.assign(G, nullptr);
    dma_b_.assign(G, nullptr);
    dma_used_.assign(G, 0);
    for (uint32_t g = 0; g < G; ++g) {
      ck(cudaEventCreate(&dma_a_[g]), "cudaEventCreate (dma)");
      ck(cudaEventCreate(&dma_b_[g]), "cudaEventCreate (dma)");
    }
  }
  void diag_mark(uint32_t g, int which) override {
    if (g >= dma_a_.size()) return;
    ck(cudaEventRecord(which == 0 ? dma_a_[g] : dma_b_[g], s_[kDem]), "cudaEventRecord (dma)");
    if (which == 0) dma_used_[g] = 1;
  }
  // After the demand stream is synced (the boundary): the layer's marked device time, -1 when it was not marked.
  double take_dma_ms(uint32_t g) {
    if (g >= dma_used_.size() || !dma_used_[g]) return -1;
    dma_used_[g] = 0;
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, dma_a_[g], dma_b_[g]), "cudaEventElapsedTime (dma)");
    return double(ms);
  }
  void h2d(Stream s, int32_t slot, int32_t block) override {
    for (size_t p = 0; p < x_.part_bytes.size(); ++p)
      ck(cudaMemcpyAsync(at(arena_[p], uint64_t(slot) * x_.part_bytes[p]), at(blocks_[size_t(block)], x_.part_offset[p]),
                         size_t(x_.part_bytes[p]), cudaMemcpyHostToDevice, s_[s]),
         "cudaMemcpyAsync (fetch)");
  }
  void d2h(int32_t slot, int32_t block) override {
    for (size_t p = 0; p < x_.part_bytes.size(); ++p)
      ck(cudaMemcpyAsync(at(blocks_[size_t(block)], x_.part_offset[p]), at(arena_[p], uint64_t(slot) * x_.part_bytes[p]),
                         size_t(x_.part_bytes[p]), cudaMemcpyDeviceToHost, s_[kD2h]),
         "cudaMemcpyAsync (write-back)");
  }
  // Default flags: a system-scope fence before the write (the release the kernels' acquire loads pair with).
  void write_entry(Stream s, uint32_t key, int32_t v) override {
    ck_drv(cuStreamWriteValue32(reinterpret_cast<CUstream>(s_[s]), reinterpret_cast<CUdeviceptr>(slot_of_ + key),
                                cuuint32_t(v), CU_STREAM_WRITE_VALUE_DEFAULT),
           "cuStreamWriteValue32 (table)");
  }
  void signal_ready(uint32_t g, uint32_t seq) override {
    ck_drv(cuStreamWriteValue32(reinterpret_cast<CUstream>(s_[kDem]), reinterpret_cast<CUdeviceptr>(ready_ + g),
                                cuuint32_t(seq), CU_STREAM_WRITE_VALUE_DEFAULT),
           "cuStreamWriteValue32 (ready)");
  }
  void* event_create() override {
    cudaEvent_t e = nullptr;
    ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
    return e;
  }
  void event_destroy(void* ev) override {
    if (ev) cudaEventDestroy(static_cast<cudaEvent_t>(ev));
  }
  void record(void* ev, Stream s) override { ck(cudaEventRecord(static_cast<cudaEvent_t>(ev), s_[s]), "cudaEventRecord"); }
  void wait(Stream s, void* ev) override {
    ck(cudaStreamWaitEvent(s_[s], static_cast<cudaEvent_t>(ev), 0), "cudaStreamWaitEvent");
  }
  bool done(void* ev) override {
    const cudaError_t e = cudaEventQuery(static_cast<cudaEvent_t>(ev));
    if (e == cudaErrorNotReady) return false;
    ck(e, "cudaEventQuery");
    return true;
  }
  void event_sync(void* ev) override { ck(cudaEventSynchronize(static_cast<cudaEvent_t>(ev)), "cudaEventSynchronize"); }
  void sync(Stream s) override { ck(cudaStreamSynchronize(s_[s]), "cudaStreamSynchronize"); }
  // From pinned staging (as the reference, cache.cpp:552-556): a pageable source would stage through the driver's
  // own buffer on every boundary.
  void upload_table(const int32_t* t, uint32_t n) override {
    std::memcpy(table_pin_, t, size_t(n) * 4);
    ck(cudaMemcpyAsync(slot_of_, table_pin_, size_t(n) * 4, cudaMemcpyHostToDevice, s_[kDem]), "cudaMemcpyAsync (table)");
    sync(kDem);
  }
  void gate(Stream s, void* ev) { wait(s, ev); }
  std::vector<cudaEvent_t> dma_a_, dma_b_;
  std::vector<uint8_t> dma_used_;
  void drain_all() noexcept {
    for (cudaStream_t s : s_)
      if (s) cudaStreamSynchronize(s);
  }

 private:
  const DgLayout& x_;
  std::vector<char*> arena_;
  int32_t* slot_of_;
  uint32_t* ready_;
  const std::vector<char*>& blocks_;
  cudaStream_t s_[3] = {nullptr, nullptr, nullptr};
  int32_t* table_pin_ = nullptr;  // keys entries
};

class CudaDgCtx final : public DgCtx {
 public:
  CudaDgCtx(const Desc& d, const Plan& p, const ss_options& o) : d_(d), o_(o), x_(dg_layout(d, p, o)) {
    try {
      init();
    } catch (...) {
      release_all(true);
      throw;
    }
  }
  ~CudaDgCtx() override { close(); }

  const DgLayout& layout() const override { return x_; }
  bool poisoned() const override { return poisoned_; }
  void poison() override { poisoned_ = true; }
  int n_sources() const override { return int(files_.size()); }
  bool source_direct(int i) const override { return files_[size_t(i)]->direct(); }

  void begin(const uint32_t* rows, int n_layers) override {
    require(!in_token_, SS_E_INVALID, "ss_begin: a gated token is already open (call ss_end)");
    if (rows) {
      require(n_layers == int(d_.layers.size()), SS_E_INVALID, "ss_begin: rows_per_layer must cover every layer");
      for (int L : x_.gated)
        require(rows[size_t(L)] > 0, SS_E_INVALID,
                "ss_begin: gated layer " + std::to_string(L) + " has 0 rows (every gated layer runs in every token)");
    }
    // design rev 3.4 §11.1: no step ran outside a pair (every word still holds the last token's sequence)
    bool clean = dg_load_acquire(m_.done) == seq_ && dg_load_acquire(m_.began) == seq_;
    for (size_t g = 0; clean && g < x_.gated.size(); ++g)
      clean = dg_load_acquire(m_.flag + g) == seq_ && int32_t(dg_load_acquire(m_.pflag + g) - seq_) <= 0;
    if (!clean) {
      dg_store_release(m_.host_abort, 1);
      svc_->stop_join();
      poisoned_ = true;
      fail(SS_E_BACKEND, "device-gated: sequence desync at ss_begin: a gated step ran outside ss_begin / ss_end (done " +
                             std::to_string(dg_load_acquire(m_.done)) + ", began " +
                             std::to_string(dg_load_acquire(m_.began)) + ", last token " + std::to_string(seq_) +
                             "; the context is poisoned)");
    }
    ++seq_;
    in_token_ = true;
    device_busy_ = true;  // until its end kernel is seen
    dg_store_release(m_.host_seq, seq_);  // the begin kernel authorizes exactly this sequence
    svc_->begin(seq_);
  }

  void end() override {
    require(in_token_, SS_E_INVALID, "ss_end: no gated token is open");
    in_token_ = false;
    // One deadline for both waits.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(uint64_t(o_.dg_wait_timeout_ms) + 5000);
    auto left = [&] {
      const auto now = std::chrono::steady_clock::now();
      return now < deadline ? std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)
                            : std::chrono::milliseconds(0);
    };
    // 1. The completion edge: the end kernel ran after every reader of the token (test mutation 2 skips it). A done
    //    past this token, or a later step's begin, is a step outside the pair: fail at once (design rev 3.4 §11.1).
    if (!skip_edge_) {
      const Seen w = wait_seq(m_.done, seq_, left());
      const uint32_t began = dg_load_acquire(m_.began);
      if (w != kAt || began != seq_) {
        dg_store_release(m_.host_abort, 1);
        svc_->stop_join();
        poisoned_ = true;
        if (w == kBefore)
          fail(SS_E_BACKEND, "device-gated: the device did not finish token " + std::to_string(seq_) +
                                 " (no end kernel: the context is poisoned, the token's outputs must be discarded)");
        fail(SS_E_BACKEND, "device-gated: sequence desync at ss_end: another gated step ran in token " +
                               std::to_string(seq_) + " (done " + std::to_string(dg_load_acquire(m_.done)) +
                               ", began " + std::to_string(began) +
                               "; the context is poisoned, the token's outputs must be discarded)");
      }
    }
    device_busy_ = false;
    // 2. The service's terminal word for this token: the core is the engine thread's again.
    const DgService::State st = svc_->wait_terminal(seq_, left());
    if (st != DgService::kServed) {
      dg_store_release(m_.host_abort, 1);
      svc_->stop_join();
      poisoned_ = true;
      const std::string why = svc_->error();
      fail(SS_E_BACKEND, "device-gated: token " + std::to_string(seq_) + " failed in the service: " +
                             (why.empty() ? std::string("it did not finish") : why) +
                             " (the context is poisoned, the token's outputs must be discarded)");
    }
    // 3. The device's error word (written before done).
    const uint32_t code = dg_load_acquire(m_.err_code);
    if (code != kDgOk) {
      poisoned_ = true;
      const char* what = code == kDgTimeout   ? "a wait timed out"
                         : code == kDgBadId   ? "an id was out of range"
                         : code == kDgDesync  ? "a step ran outside ss_begin / ss_end (sequence desync)"
                         : code == kDgAbsent  ? "a slot lookup met an absent entry"
                                              : "the host aborted the token";
      const uint32_t aux = dg_load_acquire(m_.err_aux);
      const std::string where = code == kDgDesync   ? " (that step's seq " + std::to_string(aux) + ")"
                                : code == kDgAbsent ? std::string()
                                                    : " at gated layer " + std::to_string(aux);
      fail(SS_E_BACKEND, std::string("device-gated: token ") + std::to_string(seq_) + ": " + what + where +
                             " (the context is poisoned, the token's outputs must be discarded)");
    }
    // 4. The boundary.
    std::vector<uint32_t> keep;
    for (const auto& k : keep_) keep.insert(keep.end(), k.begin(), k.end());
    core_->end_token(keep);  // syncs the demand stream: the diagnostics marks have completed
    if (o_.dg_diag)
      for (uint32_t g = 0; g < uint32_t(dma_ms_.size()); ++g) dma_ms_[g] = transport_->take_dma_ms(g);
    fold();
  }

  void seed(int layer, const int32_t* units, int n) override {
    require(!in_token_, SS_E_INVALID, "ss_seed: between tokens only");
    const int g = gated_index(layer);
    require(g >= 0, SS_E_INVALID, "ss_seed: layer " + std::to_string(layer) + " is not gated");
    require(n >= 0 && (n == 0 || units), SS_E_INVALID, "ss_seed: bad arguments");
    std::vector<uint32_t> keys;
    for (int i = 0; i < n; ++i) {
      require(units[i] >= 0 && uint32_t(units[i]) < core_->units(uint32_t(g)), SS_E_INVALID, "ss_seed: unit out of range");
      keys.push_back(core_->key(uint32_t(g), uint32_t(units[i])));
    }
    core_->seed(keys);
  }

  void keep(int g, const int32_t* ids, int n) override {
    require(!in_token_, SS_E_INVALID, "ss_dg_keep: between tokens only");
    require(g >= 0 && uint32_t(g) < x_.gated.size(), SS_E_INVALID, "ss_dg_keep: gated layer out of range");
    require(n >= 0 && (n == 0 || ids), SS_E_INVALID, "ss_dg_keep: bad arguments");
    std::vector<uint32_t> k;
    for (int i = 0; i < n; ++i) {
      require(ids[i] >= 0 && uint32_t(ids[i]) < core_->units(uint32_t(g)), SS_E_INVALID, "ss_dg_keep: id out of range");
      k.push_back(core_->key(uint32_t(g), uint32_t(ids[i])));
    }
    keep_[size_t(g)] = std::move(k);
  }

  void check_unit(int layer, int unit) override {
    require(!in_token_, SS_E_INVALID, "ss_check_unit: between tokens in device-gated mode");
    const int g = gated_index(layer);
    require(g >= 0, SS_E_INVALID, "ss_check_unit: layer " + std::to_string(layer) + " is not gated");
    require(unit >= 0 && uint32_t(unit) < core_->units(uint32_t(g)), SS_E_INVALID, "ss_check_unit: unit out of range");
    core_->drain();
    const uint32_t k = core_->key(uint32_t(g), uint32_t(unit));
    std::vector<char> got(size_t(x_.unit_bytes));
    if (core_->slot_of(k) >= 0 &&
        (core_->home(k) == DgCore::Home::Device || core_->home(k) == DgCore::Home::Evicting)) {
      for (size_t p = 0; p < x_.part_bytes.size(); ++p)
        ck(cudaMemcpy(got.data() + x_.part_offset[p], at(arena_[p], uint64_t(core_->slot_of(k)) * x_.part_bytes[p]),
                      size_t(x_.part_bytes[p]), cudaMemcpyDeviceToHost),
           "cudaMemcpy (check)");
    } else {
      std::memcpy(got.data(), blocks_[size_t(core_->block_of(k))], got.size());
    }
    read_unit(layer, unit, staging_, false, nullptr);
    require(std::memcmp(got.data(), staging_, got.size()) == 0, SS_E_HASH,
            "check_unit: the bytes of (" + std::to_string(layer) + ", " + std::to_string(unit) +
                ") differ from the source");
    core_->check();
  }

  void device_view(ss_dg_device_view* v) const override {
    std::memset(v, 0, sizeof *v);
    v->struct_size = uint32_t(sizeof *v);
    v->protocol = SS_DG_PROTOCOL;
    v->n_gated = uint32_t(x_.gated.size());
    v->n_parts = uint32_t(x_.part_bytes.size());
    v->guard_slot = x_.guard_slot;
    v->slots = x_.slots;
    v->slot_of = slot_of_;
    v->unit_base = dev_.unit_base;
    v->abort = dev_.abort;
    for (size_t p = 0; p < x_.part_bytes.size(); ++p) v->parts[p] = {arena_[p], x_.part_bytes[p], x_.part_bytes[p]};
    v->absent_err = dev_.absent_err;
  }

  void enqueue_begin(void* s) override {
    require(s != nullptr, SS_E_INVALID, "ss_dg_enqueue_begin: null stream");
    ck_launch(dgk::launch_begin(dev_, s), "begin kernel");
  }
  void enqueue_export(int g, const int32_t* ids, int n, int n_pred, void* s) override {
    require(s && ids, SS_E_INVALID, "ss_dg_enqueue_export: null argument");
    check_g(g, "ss_dg_enqueue_export");
    require(n >= 1 && uint32_t(n) <= x_.B, SS_E_INVALID,
            "ss_dg_enqueue_export: n must be 1..dg_max_ids (" + std::to_string(x_.B) + ")");
    require(n_pred >= 0 && n_pred <= int(kDgMaxIds), SS_E_INVALID, "ss_dg_enqueue_export: n_pred must be 0..64");
    require(n_pred == 0 || uint32_t(g) + 1 < x_.gated.size(), SS_E_INVALID,
            "ss_dg_enqueue_export: the last gated layer cannot promise a prediction");
    ck_launch(dgk::launch_export(dev_, uint32_t(g), ids, uint32_t(n), uint32_t(n_pred), s), "export kernel");
  }
  void enqueue_predict(int g, const int32_t* ids, int n, void* s) override {
    require(s && ids, SS_E_INVALID, "ss_dg_enqueue_predict: null argument");
    check_g(g, "ss_dg_enqueue_predict");
    require(uint32_t(g) + 1 < x_.gated.size(), SS_E_INVALID, "ss_dg_enqueue_predict: the last gated layer has no next");
    require(n >= 1 && n <= int(kDgMaxIds), SS_E_INVALID, "ss_dg_enqueue_predict: n must be 1..64");
    ck_launch(dgk::launch_predict(dev_, uint32_t(g), ids, uint32_t(n), s), "predict kernel");
  }
  void enqueue_wait(int g, void* s) override {
    require(s != nullptr, SS_E_INVALID, "ss_dg_enqueue_wait: null stream");
    check_g(g, "ss_dg_enqueue_wait");
    ck_launch(dgk::launch_wait(dev_, uint32_t(g), s), "wait kernel");
  }
  void enqueue_end(void* s) override {
    require(s != nullptr, SS_E_INVALID, "ss_dg_enqueue_end: null stream");
    ck_launch(dgk::launch_end(dev_, s), "end kernel");
  }

  void set_budgets(uint32_t spare, uint32_t per_layer, uint32_t per_token) override {
    require(!in_token_, SS_E_INVALID, "ss_dg_set_budgets: between tokens only");
    require(spare <= x_.D, SS_E_INVALID,
            "ss_dg_set_budgets: spare_target " + std::to_string(spare) + " above the billed spares (" +
                std::to_string(x_.D) + ")");
    core_->set_spare_target(spare);
    core_->set_prefetch_budget(per_layer, per_token);
  }

  void test_mutate(int m) override {
    hooks("test_mutate");
    require(!in_token_, SS_E_INVALID, "test_mutate: between tokens");
    core_->set_mutation(m);
  }
  void test_pressure(uint32_t f, uint32_t r) override {
    hooks("test_pressure");
    require(!in_token_, SS_E_INVALID, "test_pressure: between tokens");
    core_->set_pressure(f, r);
  }
  void test_check() override {
    hooks("test_check");
    require(!in_token_, SS_E_INVALID, "test_check: between tokens");
    core_->check();
  }
  void test_gate(int stream, void* ev) override {
    hooks("test_gate");
    require(!in_token_ && (stream == 0 || stream == 1), SS_E_INVALID, "test_gate: between tokens, stream 0 or 1");
    transport_->gate(stream == 0 ? DgTransport::kDem : DgTransport::kPf, ev);
  }
  void test_flags(uint32_t f) override {
    hooks("test_flags");
    dev_.test_flags = f & dgk::kSkipWait;
    skip_edge_ = (f & 2u) != 0;
  }
  void test_stop_service() override {
    hooks("test_stop_service");
    svc_->stop_join();
  }
  // Test-only hooks act only on a context opened with dg_test_hooks = 1 (design rev 3.4 §11.5).
  void hooks(const char* what) const {
    require(o_.dg_test_hooks != 0, SS_E_INVALID, std::string(what) + ": open the context with dg_test_hooks = 1");
  }

  void stats(ss_dg_stats* out) const override {
    const DgStats a = diff(fold_, base_);
    std::memset(out, 0, sizeof *out);
    out->struct_size = uint32_t(sizeof *out);
    out->tokens = a.tokens;
    out->lookups = a.lookups;
    out->misses = a.misses;
    out->covered = a.covered;
    out->fetched = a.fetched;
    out->prefetched = a.prefetched;
    out->prefetch_used = a.prefetch_used;
    out->pred_invalid = a.pred_invalid;
    out->resurrected = a.resurrected;
    out->bank_serves = a.bank_serves;
    out->collected_mid = a.collected_mid;
    out->evicted = a.evicted;
    out->evict_short = a.evict_short;
    out->writebacks_done = a.writebacks_done;
    out->stalled = a.stalled;
    out->free_slots = fold_free_slots_;
    out->blocks = x_.blocks;
    out->slots = x_.slots;
    out->predicted = a.predicted;
    out->predicted_true = a.predicted_true;
    out->miss_layers = a.miss_layers;
    out->covered_waits = a.covered_waits;
    // the device's wait timeouts since open, live (a timed-out token is never folded: it poisons the context)
    out->device_timeouts = m_.err_count ? dg_load_acquire(m_.err_count) : 0;
  }
  void layer_stats(int layer, ss_layer_stats* out) const override {
    require(layer >= 0 && size_t(layer) < d_.layers.size(), SS_E_INVALID, "ss_stats_layer: layer out of range");
    *out = ss_layer_stats{};
    const int g = gated_index(layer);
    if (g >= 0) *out = to_layer(fold_l_[size_t(g)], base_l_[size_t(g)]);
  }
  ss_layer_stats total() const override {
    ss_layer_stats t{};
    for (size_t g = 0; g < fold_l_.size(); ++g) add(t, to_layer(fold_l_[g], base_l_[g]));
    return t;
  }
  ss_exclusive_stats exclusive_stats() const override {
    const DgStats& a = fold_;  // since open: the exclusive counters are never reset (ss.h)
    ss_exclusive_stats x{};
    x.struct_size = uint32_t(sizeof x);
    x.writebacks = a.evicted;
    x.writeback_bytes = a.evicted * x_.unit_bytes;
    x.promoted = a.fetched - a.bank_serves + a.prefetched;  // copies into persistent slots
    x.free_blocks = fold_free_blocks_;
    x.min_free_blocks = min_free_blocks_;
    return x;
  }
  void stats_reset() override {
    require(!in_token_, SS_E_INVALID, "ss_stats_reset: inside a gated token (nothing changed)");
    base_ = fold_;
    std::copy(fold_l_.begin(), fold_l_.end(), base_l_.begin());
    publish();  // the exclusive counters and the minimum free blocks are not reset
  }
  void set_hook(ss_stats_hook h, void* user) override {
    hook_ = h;
    hook_user_ = user;
  }
  Json stats_json() const override {
    return format(diff(fold_, base_), fold_, layers_now(), fold_free_slots_, fold_free_blocks_, min_free_blocks_);
  }
  bool snapshot(ss_layer_stats* total, ss_snapshot_info* info) const override {
    snap_armed_.store(true, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(snap_mu_);
    if (snap_.seq == 0) return false;
    ss_layer_stats t{};
    for (const ss_layer_stats& l : snap_.layers) add(t, l);
    if (total) *total = t;
    if (info) {
      *info = ss_snapshot_info{};
      info->struct_size = uint32_t(sizeof *info);
      info->flags = SS_SNAPSHOT_EXCLUSIVE;
      info->sequence = snap_.seq;
      info->forwards = snap_.s.tokens;
      info->exclusive_writebacks = snap_.life.evicted;
      info->exclusive_writeback_bytes = snap_.life.evicted * x_.unit_bytes;
      info->exclusive_promoted = snap_.life.fetched - snap_.life.bank_serves + snap_.life.prefetched;
      info->exclusive_free_blocks = snap_.free_blocks;
      info->exclusive_min_free_blocks = snap_.min_free_blocks;
      info->exclusive_blocks = x_.blocks;
    }
    return true;
  }
  bool snapshot_json(Json* out) const override {
    snap_armed_.store(true, std::memory_order_relaxed);
    Published p;
    {
      std::lock_guard<std::mutex> lk(snap_mu_);
      if (snap_.seq == 0) return false;
      p = snap_;
    }
    *out = format(p.s, p.life, p.layers, p.free_slots, p.free_blocks, p.min_free_blocks);
    return true;
  }
  void timeline(int g, ss_dg_layer_timeline* out) const override {
    require(!in_token_, SS_E_INVALID, "ss_dg_timeline: between tokens only");
    require(o_.dg_diag, SS_E_INVALID, "ss_dg_timeline: open with dg_diag = 1");
    check_g(g, "ss_dg_timeline");
    const auto& tl = svc_->timeline();
    require(size_t(g) < tl.size(), SS_E_INVALID, "ss_dg_timeline: no token completed yet");
    const DgLayerTimeline& t = tl[size_t(g)];
    std::memset(out, 0, sizeof *out);
    out->struct_size = uint32_t(sizeof *out);
    out->layer = uint32_t(x_.gated[size_t(g)]);
    out->seen_us = t.seen_us;
    out->demand_us = t.demand_us;
    out->pred_wait_us = t.pred_wait_us;
    out->prefetch_us = t.prefetch_us;
    out->serve_us = t.serve_us;
    out->ids = t.ids;
    out->misses = t.misses;
    out->covered = t.covered;
    out->fetched = t.fetched;
    out->prefetched = t.prefetched;
    out->dma_ms = size_t(g) < dma_ms_.size() ? dma_ms_[size_t(g)] : -1;
    out->predicted_true = t.predicted_true;
    out->covered_waits = t.covered_waits;
  }

 private:

  // ---- stats: the engine-owned folded copy ----
  struct Published {
    uint64_t seq = 0;
    DgStats s;                           // since open or the last ss_stats_reset
    DgStats life;                        // since open (the exclusive counters are never reset)
    std::vector<ss_layer_stats> layers;  // per descriptor layer, since open or the last reset
    uint32_t free_slots = 0, free_blocks = 0, min_free_blocks = 0;
  };
  static DgStats diff(const DgStats& a, const DgStats& b) {
    DgStats d;
    d.tokens = a.tokens - b.tokens;
    d.lookups = a.lookups - b.lookups;
    d.misses = a.misses - b.misses;
    d.covered = a.covered - b.covered;
    d.fetched = a.fetched - b.fetched;
    d.prefetched = a.prefetched - b.prefetched;
    d.prefetch_used = a.prefetch_used - b.prefetch_used;
    d.resurrected = a.resurrected - b.resurrected;
    d.bank_serves = a.bank_serves - b.bank_serves;
    d.collected_mid = a.collected_mid - b.collected_mid;
    d.evicted = a.evicted - b.evicted;
    d.evict_short = a.evict_short - b.evict_short;
    d.writebacks_done = a.writebacks_done - b.writebacks_done;
    d.covered_waits = a.covered_waits - b.covered_waits;
    d.predicted = a.predicted - b.predicted;
    d.predicted_true = a.predicted_true - b.predicted_true;
    d.miss_layers = a.miss_layers - b.miss_layers;
    d.stalled = a.stalled - b.stalled;
    d.pred_invalid = a.pred_invalid - b.pred_invalid;
    return d;
  }
  // A gated layer's counters in the host contract's terms: every routed unit is a lookup, a demand copy into a
  // persistent slot is admitted, a bank serve is transient, a prefetch is a speculative read, every copy comes from
  // a pinned block and none from a file.
  ss_layer_stats to_layer(const DgLayerCounts& a, const DgLayerCounts& b) const {
    ss_layer_stats l{};
    l.routed = a.lookups - b.lookups;
    l.hits = l.routed - (a.misses - b.misses);
    l.transient = a.bank_serves - b.bank_serves;
    l.admitted = (a.fetched - b.fetched) - l.transient;
    l.evicted = a.evicted - b.evicted;
    l.spec_issued = a.prefetched - b.prefetched;
    l.spec_used = a.prefetch_used - b.prefetch_used;
    l.spec_dropped = l.spec_issued >= l.spec_used ? l.spec_issued - l.spec_used : 0;
    l.bytes_pinned = ((a.fetched - b.fetched) + (a.prefetched - b.prefetched)) * x_.unit_bytes;
    return l;
  }
  static void add(ss_layer_stats& t, const ss_layer_stats& l) {
    t.routed += l.routed;
    t.hits += l.hits;
    t.admitted += l.admitted;
    t.transient += l.transient;
    t.evicted += l.evicted;
    t.prefetched += l.prefetched;
    t.spec_issued += l.spec_issued;
    t.spec_used += l.spec_used;
    t.spec_dropped += l.spec_dropped;
    t.bytes_pinned += l.bytes_pinned;
    t.bytes_disk += l.bytes_disk;
    t.wait_ns += l.wait_ns;
  }
  // Into a vector sized at open (no allocation).
  void fill_layers(std::vector<ss_layer_stats>& v) const {
    std::fill(v.begin(), v.end(), ss_layer_stats{});
    for (size_t g = 0; g < fold_l_.size(); ++g) v[size_t(x_.gated[g])] = to_layer(fold_l_[g], base_l_[g]);
  }
  std::vector<ss_layer_stats> layers_now() const {
    std::vector<ss_layer_stats> v(d_.layers.size());
    fill_layers(v);
    return v;
  }
  // strix-stream/stats-v1 with the host contract's total / layers / exclusive objects, plus "device_gated".
  // s: resettable; life: since open (the exclusive object, as in the host contract).
  Json format(const DgStats& s, const DgStats& life, const std::vector<ss_layer_stats>& layers, uint32_t free_slots,
              uint32_t free_blocks, uint32_t min_free) const {
    auto one = [](const ss_layer_stats& l) {
      Json j = Json::object();
      j["routed"] = Json::number(l.routed);
      j["hits"] = Json::number(l.hits);
      j["admitted"] = Json::number(l.admitted);
      j["transient"] = Json::number(l.transient);
      j["evicted"] = Json::number(l.evicted);
      j["prefetched"] = Json::number(l.prefetched);
      j["spec_issued"] = Json::number(l.spec_issued);
      j["spec_used"] = Json::number(l.spec_used);
      j["spec_dropped"] = Json::number(l.spec_dropped);
      j["bytes_pinned"] = Json::number(l.bytes_pinned);
      j["bytes_disk"] = Json::number(l.bytes_disk);
      j["wait_ns"] = Json::number(l.wait_ns);
      return j;
    };
    Json j = Json::object();
    j["schema"] = Json::string("strix-stream/stats-v1");
    j["backend"] = Json::string("cuda");
    j["forwards"] = Json::number(s.tokens);
    Json direct = Json::array();
    for (const auto& f : files_) direct.push(Json::boolean(f->direct()));
    j["sources_direct"] = direct;
    ss_layer_stats t{};
    Json ls = Json::array();
    for (const ss_layer_stats& l : layers) {
      add(t, l);
      ls.push(one(l));
    }
    j["total"] = one(t);
    j["layers"] = ls;
    Json x = Json::object();
    x["writebacks"] = Json::number(life.evicted);
    x["writeback_bytes"] = Json::number(life.evicted * x_.unit_bytes);
    x["deferred"] = Json::number(uint64_t(0));
    x["rereads"] = Json::number(uint64_t(0));
    x["promoted"] = Json::number(life.fetched - life.bank_serves + life.prefetched);
    x["free_blocks"] = Json::number(uint64_t(free_blocks));
    x["min_free_blocks"] = Json::number(uint64_t(min_free));
    x["blocks"] = Json::number(uint64_t(x_.blocks));
    j["exclusive"] = x;
    Json g = Json::object();
    const std::pair<const char*, uint64_t> fields[] = {
        {"tokens", s.tokens},           {"lookups", s.lookups},           {"misses", s.misses},
        {"covered", s.covered},         {"fetched", s.fetched},           {"prefetched", s.prefetched},
        {"prefetch_used", s.prefetch_used}, {"pred_invalid", s.pred_invalid}, {"resurrected", s.resurrected},
        {"bank_serves", s.bank_serves}, {"collected_mid", s.collected_mid}, {"evicted", s.evicted},
        {"evict_short", s.evict_short}, {"writebacks_done", s.writebacks_done}, {"stalled", s.stalled},
        {"free_slots", uint64_t(free_slots)}, {"blocks", uint64_t(x_.blocks)}, {"slots", uint64_t(x_.slots)}};
    for (const auto& f : fields) g[f.first] = Json::number(f.second);
    j["device_gated"] = g;
    return j;
  }
  // After a successful boundary (and at open): copy the core's counters, call the hook with this token's numbers,
  // publish the snapshot when armed. Nothing here allocates (every buffer is sized at open), so a token whose ss_end
  // fails never shows in the stats.
  void fold() {
    std::copy(fold_l_.begin(), fold_l_.end(), prev_l_.begin());
    fold_ = core_->stats();
    const std::vector<DgLayerCounts>& lc = core_->layer_counts();
    std::copy(lc.begin(), lc.end(), fold_l_.begin());
    fold_free_slots_ = core_->free_slots();
    fold_free_blocks_ = core_->free_blocks();
    min_free_blocks_ = std::min(min_free_blocks_, fold_free_blocks_);
    if (hook_) {
      std::fill(hook_fwd_.begin(), hook_fwd_.end(), ss_layer_stats{});
      ss_layer_stats tot{};
      for (size_t g = 0; g < fold_l_.size(); ++g) {
        hook_fwd_[size_t(x_.gated[g])] = to_layer(fold_l_[g], prev_l_[g]);
        add(tot, hook_fwd_[size_t(x_.gated[g])]);
      }
      hook_(hook_user_, &tot, hook_fwd_.data(), int(hook_fwd_.size()));
    }
    publish();
  }
  // A poisoned context publishes no more: the last successful publication stays.
  void publish() {
    if (!snap_armed_.load(std::memory_order_relaxed) || poisoned_) return;
    pending_.s = diff(fold_, base_);
    pending_.life = fold_;
    fill_layers(pending_.layers);
    pending_.free_slots = fold_free_slots_;
    pending_.free_blocks = fold_free_blocks_;
    pending_.min_free_blocks = min_free_blocks_;
    std::lock_guard<std::mutex> lk(snap_mu_);
    pending_.seq = snap_.seq + 1;
    std::swap(snap_, pending_);  // vectors of equal size: swapped, never reallocated
  }

  void check_g(int g, const char* what) const {
    require(g >= 0 && uint32_t(g) < x_.gated.size(), SS_E_INVALID, std::string(what) + ": gated layer out of range");
  }

  // A device-written sequence word against seq (signed): at it, past it (a later step), or still before at the limit.
  enum Seen { kBefore, kAt, kPast };
  static Seen wait_seq(const uint32_t* w, uint32_t seq, std::chrono::milliseconds limit) {
    const auto end = std::chrono::steady_clock::now() + limit;
    for (uint32_t spins = 0;; ++spins) {
      const int32_t d = int32_t(dg_load_acquire(w) - seq);
      if (d == 0) return kAt;
      if (d > 0) return kPast;
      if ((spins & 1023u) == 0 && std::chrono::steady_clock::now() > end) return kBefore;
      if (spins > 20000) std::this_thread::yield();
    }
  }

  // One unit's bytes (its parts back to back) into dst, with the covering aligned range through the staging buffer
  // when the direct-I/O alignment rules are not met.
  void read_unit(int layer, int unit, void* dst, bool hash, uint64_t* digest) {
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
      const bool in_place = !f.direct() || (F % al == 0 && part.bytes % al == 0 && uintptr_t(out) % al == 0);
      if (in_place) {
        require(f.pread(out, part.bytes, F) == part.bytes, SS_E_IO, "short read");
      } else {
        const uint64_t cap = (bounce_bytes_ - 2 * al) / al * al;
        for (uint64_t c = 0; c < part.bytes;) {
          const uint64_t len = std::min(cap, part.bytes - c);
          const uint64_t a0 = (F + c) / al * al, delta = F + c - a0, a1 = round_up(F + c + len, al);
          require(f.pread(bounce_, a1 - a0, a0) >= delta + len, SS_E_IO, "short read");
          std::memcpy(out + c, at(bounce_, delta), size_t(len));
          c += len;
        }
      }
      if (hash) h.update(out, size_t(part.bytes));
      pos += part.bytes;
    }
    if (digest) *digest = h.digest();
  }

  void read_checked(int layer, int unit, void* dst) {
    const Layer& l = d_.layers[size_t(layer)];
    const bool want = o_.verify_hash && !l.hash.empty();
    uint64_t digest = 0;
    read_unit(layer, unit, dst, want, &digest);
    if (want && digest != l.hash[size_t(unit)])
      fail(SS_E_HASH, "unit (" + std::to_string(layer) + ", " + std::to_string(unit) +
                          ") bytes differ from the descriptor hash");
  }

  void init() {
    device_ = o_.device;
    ck(cudaSetDevice(device_), "cudaSetDevice");
    ck_launch(dgk::preload(), "loading the kernels");
    for (const auto& s : d_.sources) files_.push_back(std::make_unique<File>(s.path, o_.direct_io != 0));
    bounce_bytes_ = x_.bounce_bytes;  // the sizes admission billed (dg_layout)
    ck(cudaHostAlloc(&bounce_, size_t(bounce_bytes_), cudaHostAllocPortable), "cudaHostAlloc (bounce)");
    ck(cudaHostAlloc(reinterpret_cast<void**>(&staging_), size_t(x_.staging_bytes), cudaHostAllocPortable),
       "cudaHostAlloc (staging)");

    // Device: the arena (one array per part; the guard slot last, zeroed), the table, the controls.
    for (uint64_t pb : x_.part_bytes) {
      void* a = nullptr;
      ck(cudaMalloc(&a, size_t(uint64_t(x_.slots) * pb)), "cudaMalloc (arena)");
      arena_.push_back(static_cast<char*>(a));
      ck(cudaMemset(at(a, uint64_t(x_.guard_slot) * pb), 0, size_t(pb)), "cudaMemset (guard slot)");
    }
    ck(cudaMalloc(reinterpret_cast<void**>(&slot_of_), size_t(x_.keys) * 4), "cudaMalloc (slot table)");
    const uint32_t G = uint32_t(x_.gated.size());
    const size_t ctl_words = size_t(DgLayout::ctl_words(G));
    ck(cudaMalloc(reinterpret_cast<void**>(&ctl_), ctl_words * 4), "cudaMalloc (controls)");
    ck(cudaMemset(ctl_, 0, ctl_words * 4), "cudaMemset (controls)");
    dev_.seq = ctl_;
    dev_.abort = ctl_ + 1;
    dev_.err_count = ctl_ + 2;
    dev_.err_code = ctl_ + 3;
    dev_.err_aux = ctl_ + 4;
    dev_.any_miss = ctl_ + 5;
    dev_.absent_err = ctl_ + 5 + G + (G + 1);  // the last control word (design rev 3.4 §11.3)
    uint32_t* ub = ctl_ + 5 + G;
    ck(cudaMemcpy(ub, x_.unit_base.data(), (G + 1) * 4, cudaMemcpyHostToDevice), "cudaMemcpy (unit_base)");
    dev_.unit_base = ub;
    dev_.slot_of = slot_of_;
    dev_.timeout_ns = uint64_t(o_.dg_wait_timeout_ms) * 1000000ull;

    // Mapped control area.
    void* mh = nullptr;
    ck(cudaHostAlloc(&mh, size_t(x_.mapped_bytes), cudaHostAllocMapped | cudaHostAllocPortable), "cudaHostAlloc (mapped)");
    mapped_ = mh;
    std::memset(mh, 0, size_t(x_.mapped_bytes));
    void* md = nullptr;
    ck(cudaHostGetDevicePointer(&md, mh, 0), "cudaHostGetDevicePointer");
    auto carve = [&](uint64_t& off, uint64_t bytes, auto** host, auto** devp) {
      using T = std::remove_pointer_t<std::remove_reference_t<decltype(*host)>>;
      *host = reinterpret_cast<T*>(at(mh, off));
      *devp = reinterpret_cast<T*>(at(md, off));
      off += (bytes + 63) / 64 * 64;
    };
    uint64_t off = 0;
    carve(off, uint64_t(G) * sizeof(DgRecord), &m_.rec, &dev_.rec);
    carve(off, uint64_t(G) * sizeof(DgPredRecord), &m_.prec, &dev_.prec);
    carve(off, uint64_t(G) * 4, &m_.flag, &dev_.flag);
    carve(off, uint64_t(G) * 4, &m_.pflag, &dev_.pflag);
    carve(off, uint64_t(G) * 4, &m_.ready, &dev_.ready);
    uint32_t* words_h = nullptr;
    uint32_t* words_d = nullptr;
    carve(off, 7 * 4, &words_h, &words_d);
    require(off <= x_.mapped_bytes, SS_E_BACKEND, "device-gated: mapped layout (internal)");
    m_.done = words_h;
    m_.err_code = words_h + 1;
    m_.err_aux = words_h + 2;
    m_.host_abort = words_h + 3;
    m_.host_seq = words_h + 4;
    m_.began = words_h + 5;
    m_.err_count = words_h + 6;
    dev_.done = words_d;
    dev_.m_err_code = words_d + 1;
    dev_.m_err_aux = words_d + 2;
    dev_.host_abort = words_d + 3;
    dev_.host_seq = words_d + 4;
    dev_.began = words_d + 5;
    dev_.m_err_count = words_d + 6;

    // Pinned blocks: P + Q of the host stride, in chunks of at most 1 GiB.
    const uint64_t hs = x_.host_stride;
    const uint64_t per_chunk = std::max<uint64_t>(1, (uint64_t(1) << 30) / hs);
    for (uint64_t left = x_.blocks; left > 0;) {
      const uint64_t n = std::min(per_chunk, left);
      void* c = nullptr;
      ck(cudaHostAlloc(&c, size_t(n * hs), cudaHostAllocPortable), "cudaHostAlloc (blocks)");
      chunks_.push_back(c);
      for (uint64_t i = 0; i < n; ++i) blocks_.push_back(at(c, i * hs));
      left -= n;
    }

    // The core: residents in slots 0..R-1, every other unit in a block (key order), spares free.
    transport_ = std::make_unique<CudaTransport>(x_, arena_, slot_of_, dev_.ready, blocks_);
    DgConfig c;
    for (int L : x_.gated) c.n_units.push_back(uint32_t(d_.layers[size_t(L)].n_units));
    c.persistent_slots = x_.persistent_slots;
    c.bank_slots = x_.B;
    c.blocks = x_.blocks;
    const uint32_t T = o_.dg_spare_target == 0xFFFFFFFFu ? x_.D : o_.dg_spare_target;
    require(T <= x_.D, SS_E_INVALID,
            "device-gated: dg_spare_target " + std::to_string(T) + " above the billed spares (" + std::to_string(x_.D) +
                ")");
    c.spare_target = T;
    c.prefetch_reserve_auto = o_.dg_prefetch_reserve == 0xFFFFFFFFu;  // follows the spare target (cache.cpp:171)
    c.prefetch_reserve = c.prefetch_reserve_auto ? std::min<uint32_t>(24, T / 4) : o_.dg_prefetch_reserve;
    c.prefetch_per_layer = o_.dg_prefetch_per_layer;
    c.prefetch_per_token = o_.dg_prefetch_per_token;
    core_ = std::make_unique<DgCore>(c, transport_.get());
    // design rev 3.4 §11.7: the seed (validated, duplicates skipped), then the plan's residents, then key order; place()
    // keeps the first persistent_slots - T distinct keys, and that order is the recency order.
    std::vector<uint32_t> order;
    if (o_.dg_seed_n) {
      require(o_.dg_seed != nullptr, SS_E_INVALID, "device-gated: dg_seed_n > 0 with a null dg_seed");
      for (uint32_t i = 0; i < o_.dg_seed_n; ++i) {
        const ss_dg_unit& e = o_.dg_seed[i];
        require(e.gated >= 0 && uint32_t(e.gated) < G && e.unit >= 0 && uint32_t(e.unit) < core_->units(uint32_t(e.gated)),
                SS_E_INVALID,
                "device-gated: dg_seed entry " + std::to_string(i) + " (" + std::to_string(e.gated) + ", " +
                    std::to_string(e.unit) + ") is out of range");
        order.push_back(core_->key(uint32_t(e.gated), uint32_t(e.unit)));
      }
    }
    o_.dg_seed = nullptr;  // read only during ss_open
    o_.dg_seed_n = 0;
    order.insert(order.end(), x_.residents.begin(), x_.residents.end());
    for (uint32_t k = 0; k < x_.keys; ++k) order.push_back(k);
    core_->place(order);
    if (o_.dg_diag) {
      transport_->enable_diag(G);
      core_->set_diag(true);
    }
    dma_ms_.assign(G, -1);

    // Fill: blocks read in place, residents through the staging buffer and one H2D per part.
    for (uint32_t g = 0; g < G; ++g) {
      const int L = x_.gated[g];
      for (uint32_t u = 0; u < core_->units(g); ++u) {
        const uint32_t k = core_->key(g, u);
        if (core_->slot_of(k) >= 0) {
          read_checked(L, int(u), staging_);
          for (size_t p = 0; p < x_.part_bytes.size(); ++p)
            ck(cudaMemcpy(at(arena_[p], uint64_t(core_->slot_of(k)) * x_.part_bytes[p]), staging_ + x_.part_offset[p],
                          size_t(x_.part_bytes[p]), cudaMemcpyHostToDevice),
               "cudaMemcpy (fill)");
        } else {
          read_checked(L, int(u), blocks_[size_t(core_->block_of(k))]);
        }
      }
    }
    transport_->upload_table(core_->table().data(), x_.keys);
    core_->check();
    keep_.resize(G);
    // Every stats buffer sized once: folding and publishing never allocate.
    fold_l_.assign(G, DgLayerCounts{});
    prev_l_.assign(G, DgLayerCounts{});
    base_l_.assign(G, DgLayerCounts{});
    hook_fwd_.assign(d_.layers.size(), ss_layer_stats{});
    pending_.layers.assign(d_.layers.size(), ss_layer_stats{});
    snap_.layers.assign(d_.layers.size(), ss_layer_stats{});
    fold();  // the open's state: zero counters, the free slots and blocks
    base_ = fold_;
    std::copy(fold_l_.begin(), fold_l_.end(), base_l_.begin());

    // The service.
    DgServiceConfig sc;
    sc.max_ids = x_.B;
    sc.timeout = std::chrono::milliseconds(o_.dg_wait_timeout_ms);
    sc.boost = int(o_.dg_service_boost);
    sc.diag = o_.dg_diag != 0;
    const int dev = device_;
    sc.bind = [dev] { ck(cudaSetDevice(dev), "cudaSetDevice (service)"); };
    svc_ = std::make_unique<DgService>(core_.get(), m_, sc);
    svc_->start();
  }

  // Close (design §5): stop and join the service first; free only when the device is known to be done with the
  // last token; otherwise keep every allocation an outstanding operation may reach.
  void close() noexcept {
    if (closed_) return;
    closed_ = true;
    if (m_.host_abort) dg_store_release(m_.host_abort, 1);
    if (svc_) svc_->stop_join();
    bool device_done = !device_busy_;  // also after an ss_end that failed before seeing the end kernel
    if (!device_done && m_.done)
      device_done = wait_seq(m_.done, seq_, std::chrono::milliseconds(uint64_t(o_.dg_wait_timeout_ms) + 5000)) != kBefore;
    // a step that began and has not ended (outside any pair) may still reach the memory
    if (device_done && m_.began && int32_t(dg_load_acquire(m_.began) - dg_load_acquire(m_.done)) > 0) {
      const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(uint64_t(o_.dg_wait_timeout_ms) + 5000);
      while (int32_t(dg_load_acquire(m_.began) - dg_load_acquire(m_.done)) > 0 && std::chrono::steady_clock::now() < end)
        std::this_thread::yield();
      device_done = int32_t(dg_load_acquire(m_.began) - dg_load_acquire(m_.done)) <= 0;
    }
    release_all(device_done);
  }

  void release_all(bool free_memory) noexcept {
    if (!free_memory) {
      std::fprintf(stderr,
                   "strix-stream device-gated: closing while token %u may still run on the device: its arena, tables, "
                   "control words and pinned blocks are left allocated\n",
                   unsigned(seq_));
      svc_.release();     // NOLINT: deliberately leaked with the memory it references
      core_.release();
      transport_.release();
      return;
    }
    if (transport_) transport_->drain_all();
    svc_.reset();
    core_.reset();
    transport_.reset();
    for (char* a : arena_) cudaFree(a);
    arena_.clear();
    if (slot_of_) cudaFree(slot_of_);
    if (ctl_) cudaFree(ctl_);
    if (mapped_) cudaFreeHost(mapped_);
    for (void* c : chunks_) cudaFreeHost(c);
    if (bounce_) cudaFreeHost(bounce_);
    if (staging_) cudaFreeHost(staging_);
    slot_of_ = nullptr;
    ctl_ = nullptr;
    mapped_ = nullptr;
    bounce_ = nullptr;
    staging_ = nullptr;
    chunks_.clear();
  }

  const Desc& d_;
  ss_options o_;
  DgLayout x_;
  int device_ = 0;
  std::vector<std::unique_ptr<File>> files_;
  void* bounce_ = nullptr;
  uint64_t bounce_bytes_ = 0;
  char* staging_ = nullptr;
  std::vector<char*> arena_;
  int32_t* slot_of_ = nullptr;
  uint32_t* ctl_ = nullptr;
  void* mapped_ = nullptr;
  DgMapped m_;
  dgk::Dev dev_;
  std::vector<void*> chunks_;
  std::vector<char*> blocks_;
  std::unique_ptr<CudaTransport> transport_;
  std::unique_ptr<DgCore> core_;
  std::unique_ptr<DgService> svc_;
  std::vector<std::vector<uint32_t>> keep_;
  std::vector<double> dma_ms_;  // dg_diag: the last token's per-layer demand device time (-1 = not measured)
  DgStats fold_, base_;
  std::vector<DgLayerCounts> fold_l_, prev_l_, base_l_;
  std::vector<ss_layer_stats> hook_fwd_;
  Published pending_;
  uint32_t fold_free_slots_ = 0, fold_free_blocks_ = 0, min_free_blocks_ = UINT32_MAX;
  ss_stats_hook hook_ = nullptr;
  void* hook_user_ = nullptr;
  mutable std::atomic<bool> snap_armed_{false};
  mutable std::mutex snap_mu_;
  Published snap_;  // guarded by snap_mu_
  uint32_t seq_ = 0;
  bool in_token_ = false, device_busy_ = false, poisoned_ = false, closed_ = false, skip_edge_ = false;
};

}  // namespace

std::unique_ptr<DgCtx> open_dg_ctx(const Desc& d, const Plan& p, const ss_options& o) {
  if (o.backend != SS_BACKEND_CUDA) {
    (void)dg_layout(d, p, o);
    fail(SS_E_UNSUPPORTED, "device-gated: needs the CUDA backend");
  }
  return std::make_unique<CudaDgCtx>(d, p, o);
}

#endif

}  // namespace ts
