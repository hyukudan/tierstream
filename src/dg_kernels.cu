// Device-gated decode mode: kernels (dg_kernels.hpp). Port of wdsrtx's StreamSite kernels (ds41/rtx-replica a8cccdc7,
// src/cuda/ds41_stream_site.cu) with the library's sticky abort: begin, export, predict, wait, end.

#include <cuda_runtime.h>

#include "dg_kernels.hpp"

namespace ts {
namespace dgk {

namespace {

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Acquire loads at system scope: they pair with the library's stream writes (cuStreamWriteValue32 with default flags
// fences the preceding copies and table writes) and with the host's release stores.
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int32_t ld_acquire_sys(const int32_t* p) {
  int32_t v;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

// The first error of the token wins the code; abort is sticky (a failed token poisons the context). err_count counts
// the wait timeouts (ss_dg_stats.device_timeouts).
__device__ void set_error(const Dev& d, uint32_t code, uint32_t g) {
  if (atomicCAS(d.err_code, 0u, code) == 0u) *d.err_aux = g;
  if (code == kDgTimeout) atomicAdd(d.err_count, 1u);
  atomicExch(d.abort, 1u);
}

// design rev 3.4 §11.1: the step is authorized only if its sequence is the one ss_begin wrote; any other step is
// aborted here, so every slot lookup it makes returns the guard and every wait returns at once.
__global__ void begin_kernel(Dev d) {
  const uint32_t seq = *d.seq + 1u;
  *d.seq = seq;
  if (ld_acquire_sys(d.host_seq) != seq) set_error(d, kDgDesync, seq);
  __threadfence_system();
  *reinterpret_cast<volatile uint32_t*>(d.began) = seq;
}

__global__ __launch_bounds__(64) void export_kernel(Dev d, uint32_t g, const int32_t* ids, uint32_t n, uint32_t n_pred) {
  __shared__ unsigned long long mask;
  __shared__ uint32_t bad;
  const uint32_t t = threadIdx.x;
  if (t == 0) {
    mask = 0ull;
    bad = 0u;
  }
  __syncthreads();
  const uint32_t seq = *d.seq;
  const uint32_t base = d.unit_base[g], units = d.unit_base[g + 1] - base;
  if (t < n) {
    const uint32_t id = uint32_t(ids[t]);
    d.rec[g].ids[t] = id;
    if (id >= units)
      atomicExch(&bad, 1u);  // never looked up; the service fails the token too
    else if (ld_acquire_sys(d.slot_of + base + id) < 0)
      atomicOr(&mask, 1ull << t);
  }
  __syncthreads();
  if (t == 0) {
    if (bad) set_error(d, kDgBadId, g);
    DgRecord& r = d.rec[g];
    r.n_ids = n;
    r.n_pred_promised = n_pred;
    r.mask_lo = uint32_t(mask);
    r.mask_hi = uint32_t(mask >> 32);
    r.seq = seq;
    d.any_miss[g] = mask != 0ull ? 1u : 0u;
    if (d.gts) {
      volatile unsigned long long* s = d.gts + 4u * g;
      s[1] = 0ull;
      s[2] = 0ull;
      s[0] = gtimer();
    }
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(d.flag + g) = seq;
  }
}

__global__ __launch_bounds__(64) void predict_kernel(Dev d, uint32_t g, const int32_t* ids, uint32_t n) {
  const uint32_t t = threadIdx.x;
  if (t < n) d.prec[g].pred[t] = uint32_t(ids[t]);
  __syncthreads();
  if (t == 0) {
    d.prec[g].n_pred = n;
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(d.pflag + g) = *d.seq;
  }
}

__global__ void wait_kernel(Dev d, uint32_t g) {
  if (d.test_flags & kSkipWait) return;
  if (*reinterpret_cast<volatile uint32_t*>(d.abort)) return;
  if (!d.any_miss[g]) return;
  const uint32_t seq = *d.seq;
  const unsigned long long t0 = gtimer();
  if (d.gts) reinterpret_cast<volatile unsigned long long*>(d.gts)[4u * g + 1u] = t0;
  for (;;) {
    if (int32_t(ld_acquire_sys(d.ready + g) - seq) >= 0) {
      if (d.gts) {
        reinterpret_cast<volatile unsigned long long*>(d.gts)[4u * g + 2u] = gtimer();
        __threadfence_system();
      }
      return;
    }
    if (ld_acquire_sys(d.host_abort)) {
      set_error(d, kDgHostAbort, g);
      return;
    }
    if (gtimer() - t0 > d.timeout_ns) {
      set_error(d, kDgTimeout, g);
      return;
    }
  }
}

__global__ void end_kernel(Dev d) {
  // an absent entry met without abort (design rev 3.4 §11.3): first code wins, never over an earlier one
  if (*reinterpret_cast<volatile uint32_t*>(d.absent_err) && !*reinterpret_cast<volatile uint32_t*>(d.abort))
    set_error(d, kDgAbsent, 0);
  *reinterpret_cast<volatile uint32_t*>(d.m_err_count) = *d.err_count;
  *reinterpret_cast<volatile uint32_t*>(d.m_err_code) = *d.err_code;
  *reinterpret_cast<volatile uint32_t*>(d.m_err_aux) = *d.err_aux;
  __threadfence_system();
  *reinterpret_cast<volatile uint32_t*>(d.done) = *d.seq;
}

// Layer tokens: as export_kernel, into part `part`'s record (no prediction, no diag stamps). The service serves the
// parts in order; a part's record is never rewritten inside its token.
__global__ __launch_bounds__(64) void export_part_kernel(Dev d, uint32_t g, uint32_t part, const int32_t* ids, uint32_t n) {
  __shared__ unsigned long long mask;
  __shared__ uint32_t bad;
  const uint32_t t = threadIdx.x;
  if (t == 0) {
    mask = 0ull;
    bad = 0u;
  }
  __syncthreads();
  const uint32_t seq = *d.seq;
  const uint32_t base = d.unit_base[g], units = d.unit_base[g + 1] - base;
  if (t < n) {
    const uint32_t id = uint32_t(ids[t]);
    d.lm_rec[part].ids[t] = id;
    if (id >= units)
      atomicExch(&bad, 1u);
    else if (ld_acquire_sys(d.slot_of + base + id) < 0)
      atomicOr(&mask, 1ull << t);
  }
  __syncthreads();
  if (t == 0) {
    if (bad) set_error(d, kDgBadId, g);
    DgRecord& r = d.lm_rec[part];
    r.n_ids = n;
    r.n_pred_promised = 0;
    r.mask_lo = uint32_t(mask);
    r.mask_hi = uint32_t(mask >> 32);
    r.seq = seq;
    r.pad[0] = g;
    r.pad[1] = part;
    d.lm_any_miss[part] = mask != 0ull ? 1u : 0u;
    __threadfence_system();
    *reinterpret_cast<volatile uint32_t*>(d.lm_flag + part) = seq;
  }
}

__global__ void wait_part_kernel(Dev d, uint32_t part) {
  if (d.test_flags & kSkipWait) return;
  if (*reinterpret_cast<volatile uint32_t*>(d.abort)) return;
  if (!d.lm_any_miss[part]) return;
  const uint32_t seq = *d.seq;
  const unsigned long long t0 = gtimer();
  for (;;) {
    if (int32_t(ld_acquire_sys(d.lm_ready + part) - seq) >= 0) return;
    if (ld_acquire_sys(d.host_abort)) {
      set_error(d, kDgHostAbort, part);
      return;
    }
    if (gtimer() - t0 > d.timeout_ns) {
      set_error(d, kDgTimeout, part);
      return;
    }
  }
}

// The layer token's end: every gated layer's export flag takes the token's seq (a layer token exports through its
// part words only), then the end kernel's work. A flag already at seq is a stray full export inside the layer token
// (dg_end_layer_flags): kDgDesync with err_aux = that gated layer, before the stamp hides it.
__global__ void end_layer_kernel(Dev d) {
  const uint32_t seq = *d.seq;
  dg_end_layer_flags(
      d.n_gated, seq, [&](uint32_t g) { return *reinterpret_cast<volatile uint32_t*>(d.flag + g); },
      [&](uint32_t g, uint32_t v) { *reinterpret_cast<volatile uint32_t*>(d.flag + g) = v; },
      [&](uint32_t g) { set_error(d, kDgDesync, g); });
  if (*reinterpret_cast<volatile uint32_t*>(d.absent_err) && !*reinterpret_cast<volatile uint32_t*>(d.abort))
    set_error(d, kDgAbsent, 0);
  *reinterpret_cast<volatile uint32_t*>(d.m_err_count) = *d.err_count;
  *reinterpret_cast<volatile uint32_t*>(d.m_err_code) = *d.err_code;
  *reinterpret_cast<volatile uint32_t*>(d.m_err_aux) = *d.err_aux;
  __threadfence_system();
  *reinterpret_cast<volatile uint32_t*>(d.done) = seq;
}

__global__ void spin_kernel(unsigned long long ns) {
  const unsigned long long t0 = gtimer();
  while (gtimer() - t0 < ns) {
  }
}

template <class T>
struct same {
  using type = T;
};
// One block of `threads` on stream s: the status of this launch alone. A triple-chevron launch reports through
// cudaGetLastError, which also returns an error some earlier, unrelated call left pending (the engine's, say) and
// would poison the context for it; cudaLaunchKernel returns its own status and leaves that error to its owner.
template <class... A>
int launch(void (*k)(A...), unsigned threads, void* s, typename same<A>::type... a) {
  void* args[] = {static_cast<void*>(&a)...};
  return int(cudaLaunchKernel(reinterpret_cast<const void*>(k), dim3(1), dim3(threads), args, 0,
                              static_cast<cudaStream_t>(s)));
}

}  // namespace

int launch_begin(const Dev& d, void* s) {
  return launch(begin_kernel, 1, s, d);
}
int launch_export(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, uint32_t n_pred, void* s) {
  return launch(export_kernel, 64, s, d, g, ids, n, n_pred);
}
int launch_predict(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, void* s) {
  return launch(predict_kernel, 64, s, d, g, ids, n);
}
int launch_wait(const Dev& d, uint32_t g, void* s) {
  return launch(wait_kernel, 1, s, d, g);
}
int launch_end(const Dev& d, void* s) {
  return launch(end_kernel, 1, s, d);
}
int launch_export_part(const Dev& d, uint32_t g, uint32_t part, const int32_t* ids, uint32_t n, void* s) {
  return launch(export_part_kernel, 64, s, d, g, part, ids, n);
}
int launch_wait_part(const Dev& d, uint32_t part, void* s) {
  return launch(wait_part_kernel, 1, s, d, part);
}
int launch_end_layer(const Dev& d, void* s) {
  return launch(end_layer_kernel, 1, s, d);
}
int launch_spin(unsigned long long ns, void* s) {
  return launch(spin_kernel, 1, s, ns);
}

int preload() {
  cudaFuncAttributes a;
  for (const void* f : {(const void*)begin_kernel, (const void*)export_kernel, (const void*)predict_kernel,
                        (const void*)wait_kernel, (const void*)end_kernel, (const void*)spin_kernel,
                        (const void*)export_part_kernel, (const void*)wait_part_kernel, (const void*)end_layer_kernel}) {
    const cudaError_t e = cudaFuncGetAttributes(&a, f);
    if (e != cudaSuccess) return int(e);
  }
  return 0;
}

}  // namespace dgk
}  // namespace ts
