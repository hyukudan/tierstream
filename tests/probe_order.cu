// Stream-ordering probe (one short PC-BIG; Sol sol-10 / sol-14): does this GPU/driver let an H2D on one non-blocking
// stream finish while a D2H queued earlier on another stream is held behind an event wait?
// Four conditions, each 3 times, with the integrated hazard's transfer size (8 KiB); everything is allocated and
// warmed up before the first gate:
//   control   : gate spin only (its own stream), then the H2D A -> R        -> can H2D finish while the gate is pending?
//   disjoint  : wb waits the gate, D2H X -> separate pinned C; then the H2D A -> R
//   shared    : wb waits the gate, D2H X -> pinned B (B poisoned 0xEE); then the H2D B -> R with NO wait
//   explicit  : as shared, plus the copy stream waits the D2H's event (positive ordering control)
// Each observation is YES (H2D done while the gate was pending), NO (the gate ended first) or UNKNOWN (the gate had
// already ended when the H2D was submitted, the poll ran out, or an event query failed): UNKNOWN is never evidence.
// R is checked in full: control / disjoint expect 0x11 everywhere, explicit 0x5A, shared reports ordered (0x5A),
// poison (0xEE) or mixed.
//   ss-probe-order [--gate-ms N] [--bytes N]
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
extern "C" __declspec(dllimport) int __stdcall IsDebuggerPresent(void);
#endif

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

enum Obs { kNo = 0, kYes = 1, kUnknown = 2 };
const char* obs_name(int o) { return o == kYes ? "YES" : o == kNo ? "NO" : "UNKNOWN"; }

// 1 = complete, 0 = not ready, -1 = error.
int query(cudaEvent_t e) {
  const cudaError_t r = cudaEventQuery(e);
  if (r == cudaSuccess) return 1;
  if (r == cudaErrorNotReady) return 0;
  std::printf("  cudaEventQuery: %s\n", cudaGetErrorString(r));
  return -1;
}

const char* fill_name(const std::vector<unsigned char>& v, unsigned char* only) {
  bool all = true;
  for (unsigned char c : v) all = all && c == v[0];
  *only = v[0];
  if (!all) return "mixed";
  return v[0] == 0x5A ? "ordered(0x5A)" : v[0] == 0xEE ? "poison(0xEE)" : v[0] == 0x11 ? "source(0x11)" : "other";
}

struct Probe {
  size_t bytes;
  cudaStream_t gate_s = nullptr, wb = nullptr, cp = nullptr;
  cudaEvent_t gate = nullptr, wb_done = nullptr, h2d_done = nullptr;
  unsigned char *A = nullptr, *B = nullptr, *C = nullptr, *host_r = nullptr;  // pinned
  unsigned char *X = nullptr, *R = nullptr;                                     // device
  long long gate_cycles = 0;

  Probe(int gate_ms, size_t n) : bytes(n) {
    int khz = 0;
    CU(cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, 0));
    gate_cycles = (long long)khz * gate_ms;  // kHz x ms = cycles
    CU(cudaStreamCreateWithFlags(&gate_s, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&wb, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&cp, cudaStreamNonBlocking));
    for (cudaEvent_t* e : {&gate, &wb_done, &h2d_done}) CU(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    for (unsigned char** p : {&A, &B, &C, &host_r})
      CU(cudaHostAlloc(reinterpret_cast<void**>(p), bytes, cudaHostAllocPortable));
    CU(cudaMalloc(&X, bytes));
    CU(cudaMalloc(&R, bytes));
    std::memset(A, 0x11, bytes);
    CU(cudaMemset(X, 0x5A, bytes));
    for (cudaStream_t s : {gate_s, wb, cp}) {  // warm-up: one copy each way on every stream, one spin, one wait
      CU(cudaMemcpyAsync(R, A, bytes, cudaMemcpyHostToDevice, s));
      CU(cudaMemcpyAsync(C, X, bytes, cudaMemcpyDeviceToHost, s));
    }
    spin<<<1, 1, 0, gate_s>>>(1000);
    CU(cudaEventRecord(gate, gate_s));
    CU(cudaStreamWaitEvent(wb, gate, 0));
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

  // Observation for one condition; *content = R's full-buffer classification, *ok_content whether it is the expected
  // one (control/disjoint: source; explicit: ordered; shared: any, reported).
  int run(const char* name, int cond, const char** content, bool* ok_content) {
    CU(cudaDeviceSynchronize());
    std::memset(B, 0xEE, bytes);
    CU(cudaMemset(R, 0x00, bytes));
    CU(cudaDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    spin<<<1, 1, 0, gate_s>>>(gate_cycles);
    CU(cudaEventRecord(gate, gate_s));
    const double t_gate = ms_since(t0);
    double t_enq_wb = 0, t_enq_cp = 0;
    if (cond >= 1) {
      auto t = std::chrono::steady_clock::now();
      CU(cudaStreamWaitEvent(wb, gate, 0));
      CU(cudaMemcpyAsync(cond == 1 ? C : B, X, bytes, cudaMemcpyDeviceToHost, wb));
      CU(cudaEventRecord(wb_done, wb));
      t_enq_wb = ms_since(t);
    }
    double t_wait = 0;
    if (cond == 3) {  // the explicit control's wait goes in before the gate check: the check is the last call before H2D
      auto t = std::chrono::steady_clock::now();
      CU(cudaStreamWaitEvent(cp, wb_done, 0));
      t_wait = ms_since(t);
    }
    const int gate_at_submit = query(gate);  // must still be pending, or the observation is UNKNOWN
    {
      auto t = std::chrono::steady_clock::now();
      CU(cudaMemcpyAsync(R, cond >= 2 ? B : A, bytes, cudaMemcpyHostToDevice, cp));
      CU(cudaEventRecord(h2d_done, cp));
      t_enq_cp = ms_since(t);
    }
    int obs = kUnknown;
    double t_seen = -1;
    if (gate_at_submit == 0) {
      for (long i = 0; i < 50000000L && obs == kUnknown; ++i) {
        const int h = query(h2d_done);
        const int g = query(gate);
        if (h < 0 || g < 0) break;  // query error: UNKNOWN
        if (h == 1) {
          obs = g == 0 ? kYes : kNo;
          t_seen = ms_since(t0);
        } else if (g == 1) {
          obs = kNo;  // the gate ended before the H2D completed
          t_seen = ms_since(t0);
        }
      }
    }
    CU(cudaDeviceSynchronize());
    std::vector<unsigned char> r(bytes);
    CU(cudaMemcpy(r.data(), R, bytes, cudaMemcpyDeviceToHost));
    unsigned char b0 = 0;
    *content = fill_name(r, &b0);
    const char* expect = cond <= 1 ? "source(0x11)" : cond == 3 ? "ordered(0x5A)" : nullptr;
    *ok_content = !expect || std::strcmp(*content, expect) == 0;
    std::printf("  %-8s %-7s gate at submit %s; R %s%s; enqueue gate %.3f ms, wb %.3f ms, wait %.3f ms, cp %.3f ms; decided at %.1f "
                "ms\n",
                name, obs_name(obs), gate_at_submit == 0 ? "pending" : gate_at_submit == 1 ? "ENDED" : "ERROR",
                *content, *ok_content ? "" : " (UNEXPECTED)", t_gate, t_enq_wb, t_wait, t_enq_cp, t_seen);
    return obs;
  }
};

}  // namespace

int main(int argc, char** argv) {
  int gate_ms = 200;
  size_t bytes = 8192;  // the integrated hazard's unit (layer 2 records, tests/fixture.hpp)
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--gate-ms") && i + 1 < argc) gate_ms = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--bytes") && i + 1 < argc) bytes = size_t(std::atoll(argv[++i]));
  }
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 2;
  }
  auto env = [](const char* k) {
    const char* v = std::getenv(k);
    return v ? v : "unset";
  };
  int debugger = 0;
#if defined(_WIN32)
  debugger = IsDebuggerPresent();
#endif
  std::printf("device %d: %s; tcc %d; async engines %d; CUDA_LAUNCH_BLOCKING=%s CUDA_DEVICE_MAX_CONNECTIONS=%s "
              "CUDA_INJECTION64_PATH=%s NSYS_PROFILING_SESSION_ID=%s debugger %d; gate %d ms; %zu bytes\n",
              dev, p.name, p.tccDriver, p.asyncEngineCount, env("CUDA_LAUNCH_BLOCKING"),
              env("CUDA_DEVICE_MAX_CONNECTIONS"), env("CUDA_INJECTION64_PATH"), env("NSYS_PROFILING_SESSION_ID"),
              debugger, gate_ms, bytes);
  Probe pr(gate_ms, bytes);
  const char* names[4] = {"control", "disjoint", "shared", "explicit"};
  int cnt[4][3] = {};
  int poison = 0, bad_content = 0;
  for (int rep = 0; rep < 3; ++rep)
    for (int c = 0; c < 4; ++c) {
      const char* content = nullptr;
      bool ok = true;
      ++cnt[c][pr.run(names[c], c, &content, &ok)];
      if (!ok) ++bad_content;
      if (c == 2 && std::strcmp(content, "poison(0xEE)") == 0) ++poison;
    }
  std::printf("summary: control %d/%d/%d, disjoint %d/%d/%d, shared %d/%d/%d (poisoned %d), explicit %d/%d/%d "
              "(YES/NO/UNKNOWN of 3); unexpected contents %d\n",
              cnt[0][kYes], cnt[0][kNo], cnt[0][kUnknown], cnt[1][kYes], cnt[1][kNo], cnt[1][kUnknown], cnt[2][kYes],
              cnt[2][kNo], cnt[2][kUnknown], poison, cnt[3][kYes], cnt[3][kNo], cnt[3][kUnknown], bad_content);
  const bool explicit_ok = cnt[3][kYes] == 0 && cnt[3][kUnknown] == 0;
  std::printf("explicit ordering control: %s\n", explicit_ok && bad_content == 0 ? "OK" : "BROKEN");
  return g_fail || !explicit_ok || bad_content ? 1 : 0;
}
