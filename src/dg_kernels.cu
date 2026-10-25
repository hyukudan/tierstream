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

__global__ void spin_kernel(unsigned long long ns) {
  const unsigned long long t0 = gtimer();
  while (gtimer() - t0 < ns) {
  }
}

int last() { return int(cudaGetLastError()); }

}  // namespace

int launch_begin(const Dev& d, void* s) {
  begin_kernel<<<1, 1, 0, static_cast<cudaStream_t>(s)>>>(d);
  return last();
}
int launch_export(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, uint32_t n_pred, void* s) {
  export_kernel<<<1, 64, 0, static_cast<cudaStream_t>(s)>>>(d, g, ids, n, n_pred);
  return last();
}
int launch_predict(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, void* s) {
  predict_kernel<<<1, 64, 0, static_cast<cudaStream_t>(s)>>>(d, g, ids, n);
  return last();
}
int launch_wait(const Dev& d, uint32_t g, void* s) {
  wait_kernel<<<1, 1, 0, static_cast<cudaStream_t>(s)>>>(d, g);
  return last();
}
int launch_end(const Dev& d, void* s) {
  end_kernel<<<1, 1, 0, static_cast<cudaStream_t>(s)>>>(d);
  return last();
}
int launch_spin(unsigned long long ns, void* s) {
  spin_kernel<<<1, 1, 0, static_cast<cudaStream_t>(s)>>>(ns);
  return last();
}

int preload() {
  cudaFuncAttributes a;
  for (const void* f : {(const void*)begin_kernel, (const void*)export_kernel, (const void*)predict_kernel,
                        (const void*)wait_kernel, (const void*)end_kernel, (const void*)spin_kernel}) {
    const cudaError_t e = cudaFuncGetAttributes(&a, f);
    if (e != cudaSuccess) return int(e);
  }
  return 0;
}

}  // namespace dgk
}  // namespace ts
