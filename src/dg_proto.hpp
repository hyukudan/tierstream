// Device-gated decode mode: the words the device and the library share (docs/design.md §3.7). The same layout is
// read by the service thread (dg_service.cpp) and written by the kernels (dg_kernels.cu).
#pragma once

#include <atomic>
#include <cstdint>

namespace ts {

constexpr uint32_t kDgMaxIds = 64;  // ids per gated layer per token (SS_DG_MAX_IDS)
// Layer tokens (ss_dg_begin_layer): exports ("parts") of ONE gated layer per token (SS_DG_MAX_LAYER_PARTS).
constexpr uint32_t kDgMaxParts = 32;

// One gated layer's export: written by the export kernel before flag[g], read by the service after it.
struct DgRecord {
  uint32_t n_ids, n_pred_promised, mask_lo, mask_hi;
  uint32_t seq, pad[3];
  uint32_t ids[kDgMaxIds];
};
// The prediction for gated layer g + 1, exported at layer g: written by the predict kernel before pflag[g].
struct DgPredRecord {
  uint32_t n_pred, pad[3];
  uint32_t pred[kDgMaxIds];
};
static_assert(sizeof(DgRecord) == 288 && sizeof(DgPredRecord) == 272, "the device layout");

// Device error codes (err_code; err_aux = the gated layer).
// kDgDesync: a step the host did not open (its begin kernel's seq != host_seq; err_aux = that seq); kDgAbsent: a
// slot lookup met an entry < 0 without abort (design rev 3.4 §11.1, §11.3).
enum DgErrCode : uint32_t { kDgOk = 0, kDgTimeout = 1, kDgBadId = 2, kDgHostAbort = 3, kDgDesync = 4, kDgAbsent = 5 };

#if defined(__CUDACC__)
#define TS_DG_HD __host__ __device__
#else
#define TS_DG_HD
#endif

// A layer token's end (end_layer_kernel; the CPU tests' device model runs the same function): a layer token exports
// through its part words only, so no gated layer's export flag may hold the token's seq yet. One that does was fully
// exported inside the layer token (an engine bug: its readers may read slots the token's mid-token evictions reuse):
// stray(g') reports it (the kernel: kDgDesync, err_aux = g'), before the stamp would hide it from the next ss_begin's
// flag check. Then every flag takes seq (that check's expectation).
template <class Load, class Store, class Stray>
TS_DG_HD inline void dg_end_layer_flags(uint32_t n_gated, uint32_t seq, Load load, Store store, Stray stray) {
  for (uint32_t g = 0; g < n_gated; ++g) {
    if (load(g) == seq) stray(g);
    store(g, seq);
  }
}

// Host addresses of the mapped control area (cudaHostAlloc mapped, or plain memory in the CPU tests).
struct DgMapped {
  DgRecord* rec = nullptr;        // [G]
  DgPredRecord* prec = nullptr;   // [G]
  uint32_t* flag = nullptr;       // [G], device-written
  uint32_t* pflag = nullptr;      // [G], device-written
  uint32_t* ready = nullptr;      // [G], library-written (stream writes on the demand stream)
  uint32_t* done = nullptr;       // device-written by the end kernel
  uint32_t* err_code = nullptr;   // device-written before done
  uint32_t* err_aux = nullptr;
  uint32_t* host_abort = nullptr;  // host-written, read by the wait kernel
  uint32_t* host_seq = nullptr;    // host-written by ss_begin (release), read by the begin kernel
  uint32_t* began = nullptr;       // device-written by the begin kernel (its seq), after a system fence
  uint32_t* err_count = nullptr;   // device-written by the end kernel: wait timeouts since open
  // Layer tokens: part k's record and flag (device-written by the part export kernel) and ready word (library-written
  // on the demand stream), [kDgMaxParts] each.
  DgRecord* lm_rec = nullptr;
  uint32_t* lm_flag = nullptr;
  uint32_t* lm_ready = nullptr;
};

// Acquire / release on a shared 32-bit word. GCC / Clang builtins (TSan sees them); MSVC volatile accesses are
// acquire / release on x86 (/volatile:ms) and the fences keep the compiler from moving accesses across them.
inline uint32_t dg_load_acquire(const uint32_t* p) {
#if defined(__GNUC__) || defined(__clang__)
  return __atomic_load_n(p, __ATOMIC_ACQUIRE);
#else
  const uint32_t v = *static_cast<const volatile uint32_t*>(p);
  std::atomic_thread_fence(std::memory_order_acquire);
  return v;
#endif
}
inline void dg_store_release(uint32_t* p, uint32_t v) {
#if defined(__GNUC__) || defined(__clang__)
  __atomic_store_n(p, v, __ATOMIC_RELEASE);
#else
  std::atomic_thread_fence(std::memory_order_release);
  *static_cast<volatile uint32_t*>(p) = v;
#endif
}

}  // namespace ts
