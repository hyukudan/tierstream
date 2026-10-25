// Device-gated decode mode: one context (design rev 3.2). Owns the arena, the slot table, the device and mapped
// control words, the pinned blocks, the core, the transport and the service thread. CUDA only; other builds refuse
// at open after the layout checks.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "desc.hpp"
#include "dg_core.hpp"
#include "dg_layout.hpp"
#include "json.hpp"
#include "plan.hpp"
#include "strix_stream/ss.h"

namespace ts {

class DgCtx {
 public:
  virtual ~DgCtx() = default;
  virtual const DgLayout& layout() const = 0;
  virtual bool poisoned() const = 0;
  virtual void poison() = 0;
  virtual int n_sources() const = 0;
  virtual bool source_direct(int i) const = 0;

  // Host side, the engine thread.
  virtual void begin(const uint32_t* rows, int n_layers) = 0;
  virtual void end() = 0;
  virtual void reset_open() = 0;
  virtual void seed(int layer, const int32_t* units, int n) = 0;
  virtual void keep(int g, const int32_t* ids, int n) = 0;
  virtual void check_unit(int layer, int unit) = 0;

  // Device side: kernel launchers on the engine's stream (capturable).
  virtual void device_view(ss_dg_device_view* out) const = 0;
  virtual void enqueue_begin(void* stream) = 0;
  virtual void enqueue_export(int g, const int32_t* ids_dev, int n, int n_pred, void* stream) = 0;
  virtual void enqueue_predict(int g, const int32_t* ids_dev, int n, void* stream) = 0;
  virtual void enqueue_wait(int g, void* stream) = 0;
  virtual void enqueue_end(void* stream) = 0;

  // Stats: an engine-owned copy folded at each successful ss_end (never the service's live counters), so every call
  // is safe on the engine thread at any time; the snapshot calls are safe on any thread (design rev 3.2 §5).
  virtual void stats(ss_dg_stats* out) const = 0;
  virtual void layer_stats(int layer, ss_layer_stats* out) const = 0;
  virtual ss_layer_stats total() const = 0;
  virtual ss_exclusive_stats exclusive_stats() const = 0;
  virtual void stats_reset() = 0;
  virtual void set_hook(ss_stats_hook h, void* user) = 0;
  virtual Json stats_json() const = 0;
  virtual bool snapshot(ss_layer_stats* total, ss_snapshot_info* info) const = 0;
  virtual bool snapshot_json(Json* out) const = 0;
  virtual void timeline(int g, ss_dg_layer_timeline* out) const = 0;
  virtual void boundary_timeline(ss_dg_boundary_timeline* out) const = 0;

  // Test hooks (tests/test_dg_cuda.cu), between tokens: core mutation and pressure, the invariant check, a stream
  // gate (the demand / prefetch stream waits `event` before its next operations), kernel-level mutations (flags:
  // 1 = skip the ready wait, 2 = ss_end skips the completion edge), and stopping the service now.
  virtual void set_budgets(uint32_t spare_target, uint32_t per_layer, uint32_t per_token) = 0;
  virtual void set_sched(uint32_t wb_defer, uint32_t flush_ready) = 0;
  virtual uint32_t residency(ss_dg_unit* out, uint32_t cap) = 0;
  // Window prefetch (ss.h: Window prefetch), between tokens.
  virtual void winpf_set(const DgWinpf& w) = 0;
  virtual void window_hint(const int32_t* symbols, const float* weights, uint32_t rows) = 0;
  virtual void test_mutate(int m) = 0;
  virtual void test_pressure(uint32_t free_per_token, uint32_t evict_recent) = 0;
  virtual void test_check() = 0;
  virtual uint32_t test_coming() = 0;
  virtual void test_gate(int stream, void* event) = 0;
  virtual void test_flags(uint32_t flags) = 0;
  virtual void test_stop_service() = 0;

  int gated_index(int layer) const {
    const DgLayout& x = layout();
    return layer >= 0 && size_t(layer) < x.ordinal.size() ? x.ordinal[size_t(layer)] : -1;
  }
};

// tierstream/hint-prior-v1 (ss.h: Window prefetch) for the gated layers of d: every check, then the parsed prior.
DgPrior parse_hint_prior(const void* data, uint64_t len, const Desc& d, const std::vector<int>& gated);
// The identity a prior binds to (32 bytes).
std::string hint_prior_identity(const Desc& d, const std::vector<int>& gated);

// Checks the layout (every refusal), then opens on the options' backend: CUDA only.
std::unique_ptr<DgCtx> open_dg_ctx(const Desc& d, const Plan& p, const ss_options& o);

}  // namespace ts
