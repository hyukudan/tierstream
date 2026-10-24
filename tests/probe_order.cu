// Stream-ordering probe (one short PC-BIG; Sol sol-10 readahead/exclusive): does this GPU/driver let an H2D on one
// non-blocking stream finish while a D2H queued earlier on another stream is held behind an event wait?
// Four conditions, each 3 times, everything allocated and warmed up before the first gate:
//   control   : gate spin only (on its own stream), then the H2D        -> can H2D finish while the gate is pending?
//   disjoint  : wb waits the gate, then D2H into a separate pinned C; then the H2D from A
//   shared    : wb waits the gate, then D2H dev X -> pinned B (B poisoned); then the H2D B -> R with NO wait
//   explicit  : as shared, plus the copy stream waits the D2H's event (positive ordering control)
// For each: "h2d done while gate pending" (event polled against the gate) and, for shared/explicit, which bytes R got
// (X = ordered, poison = the H2D ran before the D2H). API call durations are printed.
//   ss-probe-order [--gate-ms N]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_fail = 0;
#define CU(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("  %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      ++g_fail;                                                                                    \
    }                                                                                              \
  } while (0)

__global__ void spin(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

double ms_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

constexpr size_t kBytes = 64 * 1024;

struct Probe {
  cudaStream_t gate_s = nullptr, wb = nullptr, cp = nullptr;
  cudaEvent_t gate = nullptr, wb_done = nullptr, h2d_done = nullptr;
  unsigned char *A = nullptr, *B = nullptr, *C = nullptr, *host_r = nullptr;  // pinned
  unsigned char *X = nullptr, *R = nullptr;                                     // device
  long long gate_cycles = 0;

  Probe(int gate_ms) {
    int khz = 0;
    CU(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    gate_cycles = (long long)khz * gate_ms;  // kHz x ms = cycles
    CU(cudaStreamCreateWithFlags(&gate_s, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&wb, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
    for (cudaEvent_t* e : {&gate, &wb_done, &h2d_done}) CU(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    for (unsigned char** p : {&A, &B, &C, &host_r}) CU(cudaHostAlloc(reinterpret_cast<void**>(p), kBytes, cudaHostAllocPortable));
    CU(cudaMalloc(&X, kBytes));
    CU(cudaMalloc(&R, kBytes));
    std::memset(A, 0x11, kBytes);
    CU(cudaMemset(X, 0x5A, kBytes));
    // Warm-up: one copy each way on every stream, one spin.
    for (cudaStream_t s : {gate_s, wb, cp}) {
      CU(cudaMemcpyAsync(R, A, kBytes, cudaMemcpyHostToDevice, s));
      CU(cudaMemcpyAsync(C, X, kBytes, cudaMemcpyDeviceToHost, s));
    }
    spin<<<1, 1, 0, gate_s>>>(1000);
    CU(cudaDeviceSynchronize());
  }
  ~Probe() {
    cudaDeviceSynchronize();
    for (cudaEvent_t e : {gate, wb_done, h2d_done}) cudaEventDestroy(e);
    for (unsigned char* p : {A, B, C, host_r}) cudaFreeHost(p);
    cudaFree(X);
    cudaFree(R);
    for (cudaStream_t s : {gate_s, wb, cp}) cudaStreamDestroy(s);
  }

  // Returns 1 if the H2D finished while the gate was still pending, 0 if not; *r_byte = first byte of R after all.
  int run(const char* name, int cond, int* r_byte) {
    CU(cudaDeviceSynchronize());
    std::memset(B, 0xEE, kBytes);  // poison: an H2D that runs before the D2H copies this
    CU(cudaMemset(R, 0x00, kBytes));
    CU(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    spin<<<1, 1, 0, gate_s>>>(gate_cycles);
    CU(cudaEventRecord(gate, gate_s));
    double t_enq_wb = 0, t_enq_cp = 0;
    if (cond >= 1) {
      auto t = std::chrono::steady_clock::now();
      CU(cudaStreamWaitEvent(wb, gate, 0));
      CU(cudaMemcpyAsync(cond == 1 ? C : B, X, kBytes, cudaMemcpyDeviceToHost, wb));
      CU(cudaEventRecord(wb_done, wb));
      t_enq_wb = ms_since(t);
    }
    {
      auto t = std::chrono::steady_clock::now();
      if (cond == 3) CU(cudaStreamWaitEvent(cp, wb_done, 0));
      CU(cudaMemcpyAsync(R, cond >= 2 ? B : A, kBytes, cudaMemcpyHostToDevice, cp));
      CU(cudaEventRecord(h2d_done, cp));
      t_enq_cp = ms_since(t);
    }
    // Poll the H2D's event against the gate.
    int while_pending = -1;
    for (int i = 0; i < 200000 && while_pending < 0; ++i) {
      const bool h = cudaEventQuery(h2d_done) == cudaSuccess;
      const bool g = cudaEventQuery(gate) == cudaSuccess;
      if (h) while_pending = g ? 0 : 1;
      else if (g) while_pending = 0;  // the gate ended first
    }
    const double t_h2d = ms_since(t0);
    CU(cudaDeviceSynchronize());
    CU(cudaMemcpy(host_r, R, kBytes, cudaMemcpyDeviceToHost));
    *r_byte = host_r[0];
    std::printf("  %-8s h2d done while gate pending: %s; R = 0x%02x; enqueue wb %.3f ms, cp %.3f ms; seen at %.1f ms\n",
                name, while_pending == 1 ? "yes" : "no", *r_byte, t_enq_wb, t_enq_cp, t_h2d);
    return while_pending;
  }
};

}  // namespace

int main(int argc, char** argv) {
  int gate_ms = 200;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--gate-ms") && i + 1 < argc) gate_ms = std::atoi(argv[++i]);
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 2;
  }
  const char* clb = std::getenv("CUDA_LAUNCH_BLOCKING");
  const char* cdmc = std::getenv("CUDA_DEVICE_MAX_CONNECTIONS");
  std::printf("device %d: %s; tcc %d; async engines %d; concurrent kernels %d; CUDA_LAUNCH_BLOCKING=%s "
              "CUDA_DEVICE_MAX_CONNECTIONS=%s; gate %d ms\n",
              dev, p.name, p.tccDriver, p.asyncEngineCount, p.concurrentKernels, clb ? clb : "unset",
              cdmc ? cdmc : "unset", gate_ms);
  Probe pr(gate_ms);
  const char* names[4] = {"control", "disjoint", "shared", "explicit"};
  int yes[4] = {0, 0, 0, 0}, poison[4] = {0, 0, 0, 0};
  for (int rep = 0; rep < 3; ++rep)
    for (int c = 0; c < 4; ++c) {
      int r = 0;
      if (pr.run(names[c], c, &r) == 1) ++yes[c];
      if (c >= 2 && r == 0xEE) ++poison[c];
    }
  std::printf("summary: control %d/3, disjoint %d/3, shared %d/3 (poisoned %d/3), explicit %d/3 (poisoned %d/3) "
              "H2D done while gate pending\n",
              yes[0], yes[1], yes[2], poison[2], yes[3], poison[3]);
  // Interpretation (Sol sol-10): control yes + disjoint no = library-independent serialization; disjoint yes + shared
  // no = shared-memory tracking; shared yes (poisoned) = the hazard is observable here; explicit must be ordered.
  const bool explicit_ok = yes[3] == 0 && poison[3] == 0;
  std::printf("explicit ordering control: %s\n", explicit_ok ? "OK" : "BROKEN");
  return g_fail || !explicit_ok ? 1 : 0;
}
