// Device-gated decode mode: the words the device and the library share (docs/design.md §3.7). The same layout is
// read by the service thread (dg_service.cpp) and written by the kernels (dg_kernels.cu).
#pragma once

#include <atomic>
#include <cstdint>

namespace ts {

constexpr uint32_t kDgMaxIds = 64;  // ids per gated layer per token (SS_DG_MAX_IDS)

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
enum DgErrCode : uint32_t { kDgOk = 0, kDgTimeout = 1, kDgBadId = 2, kDgHostAbort = 3 };

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
