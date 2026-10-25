// The rename's device-side gate (design notes §4): tierstream/ts.h's ts_dg_slot, called from a
// kernel, gives what ss_dg_slot gives for a resident id, an absent id, an out-of-range id and an aborted view.
// The view is synthetic (no context): 2 gated layers of 3 and 2 units, guard slot 7.
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "tierstream/ts.h"

#define CK(x)                                                                                    \
  do {                                                                                           \
    cudaError_t e_ = (x);                                                                        \
    if (e_ != cudaSuccess) {                                                                     \
      std::printf("  %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));       \
      return 1;                                                                                  \
    }                                                                                            \
  } while (0)

struct Case {
  uint32_t g;
  int32_t id;
};

__global__ void slots(ts_dg_device_view v, const Case* cs, int n, int32_t* out_ts, int32_t* out_ss) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out_ts[i] = ts_dg_slot(&v, cs[i].g, cs[i].id);
  out_ss[i] = ss_dg_slot(&v, cs[i].g, cs[i].id);
}

int main() {
  int nd = 0;
  if (cudaGetDeviceCount(&nd) != cudaSuccess || nd == 0) {
    std::printf("no CUDA device\n");
    return 1;
  }
  const int32_t slot_of[5] = {4, -1, 2, 5, -1};  // layer 0: ids 0 (resident), 1 (absent), 2; layer 1: 0, 1 (absent)
  const uint32_t unit_base[3] = {0, 3, 5};
  const Case cs[] = {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, -1}, {1, 0}, {1, 1}, {1, 2}};
  const int32_t want_live[] = {4, 7, 2, 7, 7, 5, 7, 7};
  const int n = int(sizeof cs / sizeof cs[0]);
  int32_t *d_slot, *d_ts, *d_ss;
  uint32_t *d_base, *d_abort;
  Case* d_cs;
  CK(cudaMalloc(&d_slot, sizeof slot_of));
  CK(cudaMalloc(&d_base, sizeof unit_base));
  CK(cudaMalloc(&d_abort, sizeof(uint32_t)));
  CK(cudaMalloc(&d_cs, sizeof cs));
  CK(cudaMalloc(&d_ts, n * sizeof(int32_t)));
  CK(cudaMalloc(&d_ss, n * sizeof(int32_t)));
  CK(cudaMemcpy(d_slot, slot_of, sizeof slot_of, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_base, unit_base, sizeof unit_base, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cs, cs, sizeof cs, cudaMemcpyHostToDevice));
  ts_dg_device_view v;
  std::memset(&v, 0, sizeof v);
  v.struct_size = uint32_t(sizeof v);
  v.protocol = TS_DG_PROTOCOL;
  v.n_gated = 2;
  v.guard_slot = 7;
  v.slots = 8;
  v.slot_of = d_slot;
  v.unit_base = d_base;
  v.abort = d_abort;
  int failures = 0;
  for (uint32_t aborted = 0; aborted < 2; ++aborted) {
    CK(cudaMemcpy(d_abort, &aborted, sizeof aborted, cudaMemcpyHostToDevice));
    slots<<<1, 32>>>(v, d_cs, n, d_ts, d_ss);
    CK(cudaGetLastError());
    int32_t got_ts[n], got_ss[n];
    CK(cudaMemcpy(got_ts, d_ts, sizeof got_ts, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(got_ss, d_ss, sizeof got_ss, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      const int32_t want = aborted ? 7 : want_live[i];
      if (got_ts[i] != want || got_ss[i] != want) {
        std::printf("  abort %u layer %u id %d: ts_dg_slot %d, ss_dg_slot %d, want %d\n", aborted, cs[i].g, cs[i].id,
                    got_ts[i], got_ss[i], want);
        ++failures;
      }
    }
  }
  cudaFree(d_slot);
  cudaFree(d_base);
  cudaFree(d_abort);
  cudaFree(d_cs);
  cudaFree(d_ts);
  cudaFree(d_ss);
  std::printf(failures ? "FAIL ts_dg_slot\n" : "ok   ts_dg_slot == ss_dg_slot (16 cases)\n");
  return failures ? 1 : 0;
}
