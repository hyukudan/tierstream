// Device-gated decode mode on a GPU (design rev 3.2 §8.3; one short GPU run). A synthetic MoE step per token: for
// every gated layer, export its routed ids, wait, and checksum every part of every routed unit through ss_dg_slot and
// the part strides, in a CUDA graph captured once (and eagerly). Every checksum is compared with the file's bytes.
// Phases: eager, graph, prediction on, 8 -> 0 -> 8 graph switching, pressure (banks, resurrections, mid-token
// collects, MRU evictions), the failure paths (a stopped service, an out-of-range id) and three mutations that must
// produce wrong checksums or a failed token: no ready wait (copies delayed by a gate), no completion edge (a delayed
// last reader under MRU evictions), and the ids check.
//   ss-test-dg-cuda [--only NAME]
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <functional>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "desc.hpp"
#include "dg_fixture.hpp"
#include "strix_stream/ss.h"
#include "strix_stream/ss_test.h"
#include "desc.hpp"
#include "dg_ctx.hpp"
#include "dg_score.hpp"
#include "json.hpp"
#include "sha256.hpp"


using sstest::DgModel;

namespace {

#define CU(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("  %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      ++sstest::failures();                                                                        \
    }                                                                                              \
  } while (0)

__device__ __forceinline__ unsigned long long gtimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__global__ void spin_ns(unsigned long long ns) {
  const unsigned long long t0 = gtimer();
  while (gtimer() - t0 < ns) {
  }
}

// A gate the host releases (pw1/u2: a timed spin on the gate stream had ended before the step reached the point under
// test). The kernel waits on a mapped host flag, bounded by max_ns; arm() pushes it past WDDM batching and proves it
// holds before returning.
__global__ void wait_flag(const volatile uint32_t* f, unsigned long long max_ns) {
  const unsigned long long t0 = gtimer();
  while (*f == 0u && gtimer() - t0 < max_ns) {
  }
}
struct HostGate {
  uint32_t* h = nullptr;
  uint32_t* d = nullptr;
  cudaStream_t s = nullptr;
  cudaEvent_t ev = nullptr;
  HostGate() {
    cudaFuncAttributes fa;  // loaded now, never lazily inside a gated step
    CU(cudaFuncGetAttributes(&fa, (const void*)wait_flag));
    CU(cudaHostAlloc(reinterpret_cast<void**>(&h), 4, cudaHostAllocMapped));
    CU(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0));
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));
  }
  // returns true when the gate is armed and still holding 5 ms later
  bool arm(unsigned long long max_ns = 10000000000ull) {
    *reinterpret_cast<volatile uint32_t*>(h) = 0u;
    wait_flag<<<1, 1, 0, s>>>(d, max_ns);
    CU(cudaGetLastError());
    CU(cudaEventRecord(ev, s));
    (void)cudaStreamQuery(s);  // WDDM: submit now
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(5)) {
    }
    return cudaEventQuery(ev) == cudaErrorNotReady;
  }
  void release() { *reinterpret_cast<volatile uint32_t*>(h) = 1u; }
  ~HostGate() {
    release();
    if (s) (void)cudaStreamSynchronize(s);
    if (ev) (void)cudaEventDestroy(ev);
    if (s) (void)cudaStreamDestroy(s);
    if (h) (void)cudaFreeHost(h);
  }
};

// One block per routed id: the unit's bytes (its parts back to back) summed as byte x (position % 251 + 1).
__global__ void checksum(ss_dg_device_view v, uint32_t g, const int32_t* ids, unsigned long long* out) {
  const int32_t slot = ss_dg_slot(&v, g, ids[blockIdx.x]);
  unsigned long long s = 0;
  uint64_t pos0 = 0;
  for (uint32_t p = 0; p < v.n_parts; ++p) {
    const unsigned char* base = static_cast<const unsigned char*>(v.parts[p].base) + uint64_t(slot) * v.parts[p].stride;
    for (uint64_t i = threadIdx.x; i < v.parts[p].bytes; i += blockDim.x)
      s += (unsigned long long)base[i] * ((pos0 + i) % 251 + 1);
    pos0 += v.parts[p].bytes;
  }
  atomicAdd(out + blockIdx.x, s);
}

// Resident-first reads (ss_dg_slot_if_ready), before layer g's wait: the checksum of each routed id whose bytes are
// already complete (ready[i] = 1), nothing for the others (ready[i] = 0).
__global__ void checksum_early(ss_dg_device_view v, uint32_t g, const int32_t* ids, unsigned long long* out,
                               uint32_t* ready) {
  const int32_t slot = ss_dg_slot_if_ready(&v, g, ids[blockIdx.x]);
  if (slot < 0) {
    if (threadIdx.x == 0) ready[blockIdx.x] = 0;
    return;
  }
  unsigned long long s = 0;
  uint64_t pos0 = 0;
  for (uint32_t p = 0; p < v.n_parts; ++p) {
    const unsigned char* base = static_cast<const unsigned char*>(v.parts[p].base) + uint64_t(slot) * v.parts[p].stride;
    for (uint64_t i = threadIdx.x; i < v.parts[p].bytes; i += blockDim.x)
      s += (unsigned long long)base[i] * ((pos0 + i) % 251 + 1);
    pos0 += v.parts[p].bytes;
  }
  atomicAdd(out + blockIdx.x, s);
  if (threadIdx.x == 0) ready[blockIdx.x] = 1;
}

// Test-only host bug (design rev 3.4 §11.3): after layer g's wait, unpublish the slot of its first routed id.
__global__ void unpublish_first(ss_dg_device_view v, uint32_t g, const int32_t* ids) {
  int32_t* table = const_cast<int32_t*>(v.slot_of);
  table[v.unit_base[g] + uint32_t(ids[0])] = -1;
  __threadfence_system();
}

// the write-back modes a test runs in (this branch has only the copy engine)
#define WB_MODES {0u}
#define WB_SET(o, m) ((void)(o), (void)(m))

uint64_t host_sum(const std::vector<uint8_t>& v) {
  uint64_t s = 0;
  for (size_t i = 0; i < v.size(); ++i) s += uint64_t(v[i]) * (i % 251 + 1);
  return s;
}

struct Rig {
  DgModel m;
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  uint32_t n;  // ids per layer per token (== dg_max_ids)
  cudaStream_t s = nullptr, side = nullptr;
  cudaEvent_t fork = nullptr, join = nullptr;
  int32_t *ids_dev = nullptr, *pred_dev = nullptr, *ids_host = nullptr, *pred_host = nullptr;
  unsigned long long *sums_dev = nullptr, *sums_host = nullptr;
  ss_dg_device_view v{};
  std::vector<std::vector<uint64_t>> want;  // [layer][unit]
  cudaGraphExec_t graph[2] = {nullptr, nullptr};  // [predict]
  std::mt19937 rng{11};
  unsigned long long last_spin_ns = 0;  // a delay before the last layer's checksum (in the captured graph)
  uint64_t distinct = 0;  // distinct routed ids summed over layers and tokens (the service dedups each layer's ids)
  // resident-first reads (enable_early): checksums taken before each layer's wait, and their tallies
  unsigned long long *esums_dev = nullptr, *esums_host = nullptr;
  uint32_t *eready_dev = nullptr, *eready_host = nullptr;
  uint64_t early_total = 0, early_ready = 0, early_wrong = 0;
  void enable_early() {
    const size_t cells = size_t(m.G) * n;
    CU(cudaMalloc(&esums_dev, cells * 8));
    CU(cudaMalloc(&eready_dev, cells * 4));
    CU(cudaHostAlloc(&esums_host, cells * 8, cudaHostAllocDefault));
    CU(cudaHostAlloc(&eready_host, cells * 4, cudaHostAllocDefault));
    cudaFuncAttributes fa;
    CU(cudaFuncGetAttributes(&fa, (const void*)checksum_early));
  }

  // tweak: changes the options before ss_open; open_st: receives ss_open's status instead of requiring SS_OK.
  Rig(const std::string& tag, int resident, uint32_t dyn, uint32_t max_ids, uint32_t timeout_ms = 10000,
      bool hooks = true, const std::function<void(ss_options&)>& tweak = {}, ss_status* open_st = nullptr, bool hashes = false)
      : m(tag), n(max_ids) {
    // load the test's own kernels now: a first launch inside a gated step (lazy module loading) would wait for the
    // step's spinning wait kernel, which waits for the service: a stall until the wait deadline (dgts-t3)
    cudaFuncAttributes fa;
    for (const void* f : {(const void*)checksum, (const void*)spin_ns, (const void*)unpublish_first})
      CU(cudaFuncGetAttributes(&fa, f));
    if (hashes) {
      auto desc = ss::desc_from_json(ss::json_parse(m.json));
      for (int l=0; l<m.G; ++l) for (int u=0; u<m.U; ++u) {
        const auto expected = m.bytes(l,u);
        desc.layers[size_t(l)].hash.push_back(ss::xxh64(expected.data(),expected.size()));
      }
      m.json = ss::json_write(ss::desc_to_json(desc));
    }
    CHECK_OK(ss_desc_from_json(m.json.data(), m.json.size(), &d));
    sstest::dg_budget(b, m, dyn, max_ids);
    std::vector<uint8_t> t = m.tiers(resident);
    CHECK_OK(ss_plan_from_tiers(d, &b, t.data(), t.size(), &p));
    sstest::dg_options(o, max_ids, SS_BACKEND_CUDA);
    o.dg_wait_timeout_ms = timeout_ms;
    o.dg_diag = 1;
    o.dg_wb_defer = 0;  // the library default is 1: each test that defers says so
    o.dg_test_hooks = hooks ? 1u : 0u;  // the mutation / pressure / gate hooks (ss_test.h)
    if (tweak) tweak(o);
    if (open_st) *open_st = ss_open(d, &b, p, &o, &c);
    else CHECK_OK(ss_open(d, &b, p, &o, &c));
    if (!c) return;
    CHECK_OK(ss_dg_get_device_view(c, &v));
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CU(cudaStreamCreateWithFlags(&side, cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    CU(cudaEventCreateWithFlags(&join, cudaEventDisableTiming));
    const size_t cells = size_t(m.G) * n;
    CU(cudaMalloc(&ids_dev, cells * 4));
    CU(cudaMalloc(&pred_dev, cells * 4));
    CU(cudaMalloc(&sums_dev, cells * 8));
    CU(cudaHostAlloc(&ids_host, cells * 4, cudaHostAllocDefault));
    CU(cudaHostAlloc(&pred_host, cells * 4, cudaHostAllocDefault));
    CU(cudaHostAlloc(&sums_host, cells * 8, cudaHostAllocDefault));
    want.resize(size_t(m.G));
    for (int L = 0; L < m.G; ++L)
      for (int u = 0; u < m.U; ++u) want[size_t(L)].push_back(host_sum(m.bytes(L, u)));
  }
  ~Rig() {
    if (s) cudaStreamSynchronize(s);
    for (auto& g : graph)
      if (g) (void)cudaGraphExecDestroy(g);
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
    (void)cudaFree(ids_dev);
    (void)cudaFree(pred_dev);
    (void)cudaFree(sums_dev);
    (void)cudaFreeHost(ids_host);
    (void)cudaFreeHost(pred_host);
    (void)cudaFreeHost(sums_host);
    (void)cudaFree(esums_dev);
    (void)cudaFree(eready_dev);
    (void)cudaFreeHost(esums_host);
    (void)cudaFreeHost(eready_host);
    if (fork) (void)cudaEventDestroy(fork);
    if (join) (void)cudaEventDestroy(join);
    if (side) (void)cudaStreamDestroy(side);
    if (s) (void)cudaStreamDestroy(s);
  }

  // The step: begin; per layer export, (predict on the side stream, joined), wait, checksum; end.
  void enqueue_step(bool predict) {
    CHECK_OK(ss_dg_enqueue_begin(c, s));
    for (int g = 0; g < m.G; ++g) {
      const bool pr = predict && g + 1 < m.G;
      CHECK_OK(ss_dg_enqueue_export(c, g, ids_dev + size_t(g) * n, int(n), pr ? int(n) : 0, s));
      if (pr) {
        CU(cudaEventRecord(fork, s));
        CU(cudaStreamWaitEvent(side, fork, 0));
        CHECK_OK(ss_dg_enqueue_predict(c, g, pred_dev + size_t(g) * n, int(n), side));
        CU(cudaEventRecord(join, side));
        CU(cudaStreamWaitEvent(s, join, 0));
      }
      if (esums_dev)
        checksum_early<<<n, 128, 0, s>>>(v, uint32_t(g), ids_dev + size_t(g) * n, esums_dev + size_t(g) * n,
                                         eready_dev + size_t(g) * n);
      CHECK_OK(ss_dg_enqueue_wait(c, g, s));
      if (g + 1 == m.G && last_spin_ns) spin_ns<<<1, 1, 0, s>>>(last_spin_ns);
      checksum<<<n, 128, 0, s>>>(v, uint32_t(g), ids_dev + size_t(g) * n, sums_dev + size_t(g) * n);
    }
    CHECK_OK(ss_dg_enqueue_end(c, s));
  }
  void capture(bool predict) {
    cudaGraph_t gr = nullptr;
    CU(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    enqueue_step(predict);
    CU(cudaStreamEndCapture(s, &gr));
    CU(cudaGraphInstantiate(&graph[predict ? 1 : 0], gr, 0));
    (void)cudaGraphDestroy(gr);
  }

  uint32_t pick(int L) {
    const double x = std::uniform_real_distribution<double>(0, 1)(rng);
    return uint32_t(double(m.U) * x * x * x) % uint32_t(m.U);
  }
  // One token; returns the number of wrong checksums, or -1 when ss_end failed (status in *st).
  // fixed: when non-empty, every layer routes these n ids (instead of random ones).
  int token(bool use_graph, bool predict, ss_status* st = nullptr, int bad_id_layer = -1,
            const std::vector<int32_t>& fixed = {}) {
    for (int g = 0; g < m.G; ++g)
      for (uint32_t i = 0; i < n; ++i) {
        ids_host[size_t(g) * n + i] = fixed.empty() ? int32_t(pick(g)) : fixed[i % fixed.size()];
        pred_host[size_t(g) * n + i] = int32_t(pick(g < m.G - 1 ? g + 1 : g));
      }
    if (bad_id_layer >= 0) ids_host[size_t(bad_id_layer) * n] = m.U + 3;
    for (int g = 0; g < m.G; ++g) {
      std::vector<int32_t> u(ids_host + size_t(g) * n, ids_host + size_t(g) * n + n);
      std::sort(u.begin(), u.end());
      distinct += uint64_t(std::unique(u.begin(), u.end()) - u.begin());
    }
    const size_t cells = size_t(m.G) * n;
    CU(cudaMemcpyAsync(ids_dev, ids_host, cells * 4, cudaMemcpyHostToDevice, s));
    CU(cudaMemcpyAsync(pred_dev, pred_host, cells * 4, cudaMemcpyHostToDevice, s));
    CU(cudaMemsetAsync(sums_dev, 0, cells * 8, s));
    if (esums_dev) {
      CU(cudaMemsetAsync(esums_dev, 0, cells * 8, s));
      CU(cudaMemsetAsync(eready_dev, 0, cells * 4, s));
    }
    const ss_status bs = ss_begin(c, nullptr, 0);
    if (bs != SS_OK) {  // a mutation run: the begin precondition caught the previous token (rev 3.4 §11.1)
      if (st) {
        *st = bs;
        CU(cudaStreamSynchronize(s));
        return -1;
      }
      CHECK_OK(bs);
    }
    if (use_graph) CU(cudaGraphLaunch(graph[predict ? 1 : 0], s));
    else enqueue_step(predict);
    CU(cudaMemcpyAsync(sums_host, sums_dev, cells * 8, cudaMemcpyDeviceToHost, s));
    if (esums_dev) {
      CU(cudaMemcpyAsync(esums_host, esums_dev, cells * 8, cudaMemcpyDeviceToHost, s));
      CU(cudaMemcpyAsync(eready_host, eready_dev, cells * 4, cudaMemcpyDeviceToHost, s));
    }
    const ss_status e = ss_end(c);
    if (st) *st = e;
    CU(cudaStreamSynchronize(s));
    if (e != SS_OK) return -1;
    int wrong = 0;
    for (int g = 0; g < m.G; ++g)
      for (uint32_t i = 0; i < n; ++i) {
        const size_t k = size_t(g) * n + i;
        const int32_t id = ids_host[k];
        if (sums_host[k] != want[size_t(g)][size_t(id)]) ++wrong;
        if (esums_dev) {
          ++early_total;
          if (eready_host[k]) {
            ++early_ready;
            if (esums_host[k] != want[size_t(g)][size_t(id)]) ++early_wrong;
          }
        }
      }
    return wrong;
  }
  // Fills the ids (no step), returns nothing; the caller enqueues.
  void fill_ids() {
    for (int g = 0; g < m.G; ++g)
      for (uint32_t i = 0; i < n; ++i) {
        ids_host[size_t(g) * n + i] = int32_t(pick(g));
        pred_host[size_t(g) * n + i] = int32_t(pick(g < m.G - 1 ? g + 1 : g));
      }
    const size_t cells = size_t(m.G) * n;
    CU(cudaMemcpyAsync(ids_dev, ids_host, cells * 4, cudaMemcpyHostToDevice, s));
    CU(cudaMemcpyAsync(pred_dev, pred_host, cells * 4, cudaMemcpyHostToDevice, s));
    CU(cudaMemsetAsync(sums_dev, 0, cells * 8, s));
  }
  bool sums_all_zero() {
    const size_t cells = size_t(m.G) * n;
    CU(cudaMemcpy(sums_host, sums_dev, cells * 8, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < cells; ++i)
      if (sums_host[i] != 0) return false;
    return true;
  }
  ss_dg_stats stats() {
    ss_dg_stats x{};
    CHECK_OK(ss_stats_dg(c, &x));
    return x;
  }
};

// Wrong checksums and failed tokens, counted apart (a failed token never looks clean).
struct Tally {
  int wrong = 0, failed = 0;
  void add(int w) {
    if (w < 0) ++failed;
    else wrong += w;
  }
};

// After a failed ss_end: the context refuses everything but close and the stats calls.
void expect_poisoned(Rig& r) {
  CHECK_EQ(int(ss_dg_reset_open(r.c)), int(SS_E_BACKEND));
  CHECK_EQ(int(ss_begin(r.c, nullptr, 0)), int(SS_E_BACKEND));
  ss_dg_device_view v{};
  CHECK_EQ(int(ss_dg_get_device_view(r.c, &v)), int(SS_E_BACKEND));
  CHECK_EQ(ss_dg_gated_index(r.c, 0), -1);
  CHECK_EQ(ss_source_direct(r.c, 0), -1);
  CHECK_EQ(int(ss_dg_enqueue_begin(r.c, r.s)), int(SS_E_BACKEND));
  CHECK_EQ(int(ss_dg_set_budgets(r.c, 0, 0, 0)), int(SS_E_BACKEND));
  ss_dg_stats x{};
  CHECK_OK(ss_stats_dg(r.c, &x));
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(r.c, &js, nullptr));
  ss_free(js);
}

void print(const char* name, const ss_dg_stats& x) {
  std::printf("  %s: tokens %llu, misses %llu (covered %llu), prefetched %llu (used %llu), banks %llu, resurrected "
              "%llu, mid collects %llu, evicted %llu, free slots %llu\n",
              name, (unsigned long long)x.tokens, (unsigned long long)x.misses, (unsigned long long)x.covered,
              (unsigned long long)x.prefetched, (unsigned long long)x.prefetch_used,
              (unsigned long long)x.bank_serves, (unsigned long long)x.resurrected,
              (unsigned long long)x.collected_mid, (unsigned long long)x.evicted, (unsigned long long)x.free_slots);
}

}  // namespace

// Only this short startup fixture is registered with ctest (< 4 MiB VRAM).
TEST(dg_cuda_load_serial_pipeline) {
  struct Env {
    std::string name, old; bool had;
    Env(const char* n, const char* value) : name(n) {
      const char* v = std::getenv(n); had = v != nullptr; if (v) old = v;
      set(value);
    }
    void set(const char* value) {
#if defined(_WIN32)
      _putenv_s(name.c_str(), value ? value : "");
#else
      if (value) setenv(name.c_str(), value, 1); else unsetenv(name.c_str());
#endif
    }
    ~Env() { set(had ? old.c_str() : nullptr); }
  } serial("DG_LOAD_SERIAL", "1"), readers("DG_LOAD_READERS", "3"),
    buffers("DG_LOAD_BUFFERS", "2"), bytes("DG_LOAD_BUFFER_MIB", "1");
  std::vector<int32_t> old_table;
  std::vector<std::vector<char>> old_parts;
  for (int mode = 0; mode < 2; ++mode) {
    serial.set(mode ? "0" : "1");
    Rig r(mode ? "load-pipeline" : "load-serial", 16, 4, 8, 10000, true, {}, nullptr, true);
    if (!r.c) return;
    std::vector<int32_t> table(size_t(r.m.G*r.m.U));
    CU(cudaMemcpy(table.data(), r.v.slot_of, table.size()*sizeof(int32_t), cudaMemcpyDeviceToHost));
    std::vector<std::vector<char>> parts;
    for (uint32_t p = 0; p < r.v.n_parts; ++p) {
      const auto& part = r.v.parts[p];
      // Only initialized resident slots, followed by the explicitly zeroed guard.
      std::vector<char> got(size_t(16*r.m.G+1)*size_t(part.bytes));
      CU(cudaMemcpy(got.data(), part.base, size_t(16*r.m.G)*size_t(part.bytes), cudaMemcpyDeviceToHost));
      CU(cudaMemcpy(got.data()+size_t(16*r.m.G)*size_t(part.bytes),
                    static_cast<const char*>(part.base)+uint64_t(r.v.guard_slot)*part.stride,
                    size_t(part.bytes), cudaMemcpyDeviceToHost));
      parts.push_back(std::move(got));
    }
    // check_unit compares every host home and resident byte against independent
    // serial source reads and also checks the core's placement invariants.
    for (int l=0; l<r.m.G; ++l)
      for (int u=0; u<r.m.U; ++u) CHECK_OK(ss_check_unit(r.c,l,u));
    CHECK_OK(ss_test_dg_check(r.c));
    if (!mode) {old_table=std::move(table); old_parts=std::move(parts);}
    else {CHECK(table==old_table); CHECK(parts==old_parts);}
  }
}

TEST(dg_cuda_eager_graph_prediction) {
  Rig r("main", 16, 8, 8);
  if (!r.c) return;
  Tally k;
  for (int t = 0; t < 20; ++t) k.add(r.token(false, false));
  r.capture(false);
  r.capture(true);
  for (int t = 0; t < 200; ++t) k.add(r.token(true, false));
  for (int t = 0; t < 200; ++t) k.add(r.token(true, true));
  for (int t = 0; t < 120; ++t) k.add(r.token(true, (t / 10) % 2 == 0));  // 8 -> 0 -> 8 switching
  CHECK_EQ(k.failed, 0);
  CHECK_EQ(k.wrong, 0);
  CHECK_OK(ss_test_dg_check(r.c));
  for (int L = 0; L < r.m.G; ++L)
    for (int u = 0; u < r.m.U; u += 9) CHECK_OK(ss_check_unit(r.c, L, u));
  const ss_dg_stats x = r.stats();
  print("main", x);
  CHECK_EQ(x.tokens, 540ull);
  CHECK(x.misses > 0 && x.prefetched > 0 && x.prefetch_used > 0 && x.evicted > 0);
  ss_dg_layer_timeline tl{};
  CHECK_OK(ss_dg_timeline_sized(r.c, 2, &tl, sizeof tl));
  CHECK_EQ(tl.ids, r.n);
}

// The stats contract (review items R2a-R2c): the hook once per successful token, the snapshot published at each one, a reset
// that clears the resettable counters but not the exclusive (lifetime) ones.
int g_hook_calls = 0;
uint64_t g_hook_routed = 0;
void count_hook(void*, const ss_layer_stats* fwd_total, const ss_layer_stats*, int) {
  ++g_hook_calls;
  g_hook_routed += fwd_total->routed;
}

TEST(dg_cuda_stats) {
  Rig r("stats", 8, 4, 8);
  if (!r.c) return;
  ss_set_stats_hook(r.c, count_hook, nullptr);
  ss_layer_stats tot{};
  ss_snapshot_info info{};
  CHECK_EQ(int(ss_stats_snapshot(r.c, &tot, &info)), int(SS_E_INVALID));  // arms it
  r.capture(false);
  Tally k;
  for (int t = 0; t < 10; ++t) k.add(r.token(true, false));
  CHECK_EQ(k.failed, 0);
  CHECK_EQ(k.wrong, 0);
  CHECK_EQ(g_hook_calls, 10);
  CHECK_EQ(g_hook_routed, r.distinct);
  CHECK_OK(ss_stats_snapshot(r.c, &tot, &info));
  CHECK_EQ(info.sequence, 10ull);
  CHECK_EQ(info.forwards, 10ull);
  CHECK_EQ(tot.routed, r.distinct);
  ss_exclusive_stats x0{};
  CHECK_OK(ss_stats_exclusive(r.c, &x0));
  CHECK(x0.writebacks > 0);
  ss_stats_reset(r.c);
  ss_dg_stats d{};
  CHECK_OK(ss_stats_dg(r.c, &d));
  CHECK_EQ(d.tokens, 0ull);
  ss_layer_stats t2{};
  CHECK_OK(ss_stats_total(r.c, &t2));
  CHECK_EQ(t2.routed, 0ull);
  ss_exclusive_stats x1{};
  CHECK_OK(ss_stats_exclusive(r.c, &x1));
  CHECK_EQ(x1.writebacks, x0.writebacks);  // never reset
  CHECK_EQ(x1.min_free_blocks, x0.min_free_blocks);
  CHECK_OK(ss_stats_snapshot(r.c, &tot, &info));
  CHECK_EQ(info.sequence, 11ull);  // the reset published
  CHECK_EQ(tot.routed, 0ull);
  CHECK_EQ(info.exclusive_writebacks, x0.writebacks);
  char* js = nullptr;
  CHECK_OK(ss_stats_snapshot_json(r.c, &js, nullptr));
  CHECK(js && std::strstr(js, "\"device_gated\"") && std::strstr(js, "\"exclusive\""));
  ss_free(js);
}

TEST(dg_cuda_pressure) {
  Rig r("pressure", 8, 2, 8);
  if (!r.c) return;
  CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
  r.capture(true);
  Tally k;
  for (int t = 0; t < 300; ++t) {
    k.add(r.token(true, true));
    CHECK_OK(ss_test_dg_check(r.c));  // the ownership invariants at every boundary
  }
  CHECK_EQ(k.failed, 0);
  CHECK_EQ(k.wrong, 0);
  const ss_dg_stats x = r.stats();
  print("pressure", x);
  CHECK(x.bank_serves > 0);
  CHECK(x.resurrected > 0);
  CHECK(x.collected_mid > 0);
  CHECK(x.covered > 0);
}

// Spares as the engine caches have them (VRAM plan 2.7): a global count of 20 spare slots (not a multiple of the
// 6 layers) on the plan's 12 spare blocks (dyn 2 x 6): Q < D. Exact under pressure, invariants at every boundary. (Every
// promotion frees its block, so with misses the blocks rarely run out here; evict_short is covered by the core test
// dg_spare_blocks_below_slots.)
TEST(dg_cuda_global_spares) {
  Rig r("gspares", 8, 2, 8, 10000, true, [](ss_options& o) { o.dg_spare_slots = 20; });
  if (!r.c) return;
  CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
  r.capture(true);
  Tally k;
  for (int t = 0; t < 200; ++t) {
    k.add(r.token(true, true));
    CHECK_OK(ss_test_dg_check(r.c));
  }
  const ss_dg_stats x = r.stats();
  print("global spares", x);
  CHECK_EQ(k.failed, 0);
  CHECK_EQ(k.wrong, 0);
  CHECK_EQ(x.blocks, uint64_t(6 * (64 - 8) + 12));  // P + Q
  CHECK(x.evicted > 0);
}

// dg_flush_ready (wdsrtx's WDDM flush proposal): exact with the flush on; the mean demand time (flag seen -> copies and
// ready enqueued) and the per-token wall time are printed for both settings (measured, not asserted).
TEST(dg_cuda_flush_ready) {
  for (uint32_t flush : {0u, 1u}) {
    Rig r(flush ? "flush1" : "flush0", 16, 8, 8, 10000, true, [flush](ss_options& o) { o.dg_flush_ready = flush; });
    if (!r.c) return;
    r.capture(true);
    Tally k;
    double demand = 0;
    int layers = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < 100; ++t) {
      k.add(r.token(true, true));
      for (int g = 0; g < r.m.G; ++g) {
        ss_dg_layer_timeline tl{};
        CHECK_OK(ss_dg_timeline_sized(r.c, g, &tl, uint32_t(sizeof tl)));
        if (tl.misses > 0) {
          demand += tl.demand_us;
          ++layers;
        }
      }
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 100;
    std::printf("  flush_ready %u: wrong %d, failed %d, %.3f ms/token, mean demand_us %.1f over %d layers with misses\n",
                flush, k.wrong, k.failed, ms, layers ? demand / layers : 0.0, layers);
    CHECK_EQ(k.wrong + k.failed, 0);
  }
}

// dg_wb_defer: the boundary's write-backs submitted by the service between layers. The eager / graph / prediction,
// pressure and global-spares workloads stay bitwise exact with ownership invariants and ss_check_unit; on the demand
// stream (default budget), with a larger budget, on the write-back stream, with a 1-byte budget and a small cap
// (partial write-backs), with tight spares (forced waits).
void print_wb(const char* name, const ss_dg_stats& x) {
  std::printf("  %s: evicted %llu, deferred %llu, cancelled %llu, forced %llu (wait %.3f ms), boundary-flushed %llu "
              "(%.0f B/token); bytes issued %llu, completed %llu, never copied %llu, flushed %llu, outstanding %llu\n",
              name, (unsigned long long)x.evicted, (unsigned long long)x.wb_deferred,
              (unsigned long long)x.wb_cancelled, (unsigned long long)x.wb_forced, double(x.wb_forced_wait_ns) / 1e6,
              (unsigned long long)x.wb_boundary_flushed,
              x.tokens ? double(x.wb_boundary_flushed_bytes) / double(x.tokens) : 0.0,
              (unsigned long long)x.wb_issued_bytes, (unsigned long long)x.wb_completed_bytes,
              (unsigned long long)x.wb_cancelled_bytes, (unsigned long long)x.wb_boundary_flushed_bytes,
              (unsigned long long)x.wb_outstanding_bytes);
}

void add_wb(ss_dg_stats& sum, const ss_dg_stats& x) {
  sum.evicted += x.evicted;
  sum.wb_deferred += x.wb_deferred;
  sum.wb_cancelled += x.wb_cancelled;
  sum.wb_forced += x.wb_forced;
  sum.wb_boundary_flushed += x.wb_boundary_flushed;
}

// The three workloads under one setting; returns their write-back counters summed.
ss_dg_stats wb_defer_cases(const char* tag, const std::function<void(ss_options&)>& wb, int dyn_main = 8) {
  ss_dg_stats sum{};
  auto tweak = [&](ss_options& o) {
    o.dg_wb_defer = 1;
    wb(o);
  };
  const std::string t = tag;
  {  // eager, graph, prediction, 8 -> 0 -> 8 switching
    Rig r(t + "-main", 16, uint32_t(dyn_main), 8, 10000, true, tweak);
    if (!r.c) return sum;
    Tally k;
    for (int i = 0; i < 20; ++i) k.add(r.token(false, false));
    r.capture(false);
    r.capture(true);
    for (int i = 0; i < 100; ++i) k.add(r.token(true, false));
    for (int i = 0; i < 100; ++i) k.add(r.token(true, true));
    for (int i = 0; i < 60; ++i) k.add(r.token(true, (i / 10) % 2 == 0));
    CHECK_EQ(k.failed, 0);
    CHECK_EQ(k.wrong, 0);
    CHECK_OK(ss_test_dg_check(r.c));
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 9) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print_wb((t + " main").c_str(), x);
    add_wb(sum, x);
    CHECK(x.evicted > 0);
    CHECK_EQ(x.wb_deferred, x.evicted);
    CHECK(x.wb_issued_bytes > 0 && x.wb_completed_bytes <= x.wb_issued_bytes);
    // a few more tokens after ss_check_unit's drain: still exact
    for (int i = 0; i < 10; ++i) k.add(r.token(true, true));
    CHECK_EQ(k.wrong + k.failed, 0);
  }
  {  // pressure: banks, resurrections (cancelled or not), mid-token collects, MRU evictions
    Rig r(t + "-pressure", 8, 2, 8, 10000, true, tweak);
    if (!r.c) return sum;
    CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
    r.capture(true);
    Tally k;
    for (int i = 0; i < 200; ++i) {
      k.add(r.token(true, true));
      CHECK_OK(ss_test_dg_check(r.c));
    }
    CHECK_EQ(k.failed, 0);
    CHECK_EQ(k.wrong, 0);
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 5) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print("pressure", x);
    print_wb((t + " pressure").c_str(), x);
    add_wb(sum, x);
    CHECK(x.bank_serves > 0);
    CHECK(x.resurrected > 0);
    CHECK(x.wb_cancelled <= x.resurrected);
    CHECK_EQ(x.wb_deferred, x.evicted);
  }
  {  // global spares, fewer spare blocks than spare slots
    Rig r(t + "-gspares", 8, 2, 8, 10000, true, [&](ss_options& o) {
      tweak(o);
      o.dg_spare_slots = 20;
    });
    if (!r.c) return sum;
    CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
    r.capture(true);
    Tally k;
    for (int i = 0; i < 150; ++i) {
      k.add(r.token(true, true));
      CHECK_OK(ss_test_dg_check(r.c));
    }
    CHECK_EQ(k.wrong + k.failed, 0);
    const ss_dg_stats x = r.stats();
    print_wb((t + " global spares").c_str(), x);
    add_wb(sum, x);
    CHECK(x.wb_deferred > 0);
  }
  return sum;
}

TEST(dg_cuda_wb_defer) {
  const ss_dg_stats s = wb_defer_cases("wbd", [](ss_options&) {});
  CHECK(s.wb_cancelled > 0);  // MRU victims come back before their first part
}

// A credit of 12288 bytes per served layer: one and a half units of this model (2048 + 2048 + 4096 = 8192).
TEST(dg_cuda_wb_defer_layer_bytes) {
  const ss_dg_stats s = wb_defer_cases("wbd3", [](ss_options& o) { o.dg_wb_layer_bytes = 3 * 4096; });
  CHECK(s.wb_cancelled > 0);
}

// The write-back stream behind a post-ready event instead of the demand stream.
TEST(dg_cuda_wb_defer_d2h_stream) {
  const ss_dg_stats s = wb_defer_cases("wbds", [](ss_options& o) { o.dg_wb_stream = 1; });
  CHECK(s.wb_cancelled > 0);
}

// A 1-byte credit (one part per layer at most, then a debt) under a 4096-byte cap: write-backs stay partial across
// layers and are resurrected or flushed part-way.
TEST(dg_cuda_wb_defer_partials) {
  const ss_dg_stats s = wb_defer_cases("wbdp", [](ss_options& o) {
    o.dg_wb_layer_bytes = 1;
    o.dg_wb_max_outstanding = 4096;
  });
  CHECK(s.wb_cancelled > 0);
  CHECK(s.wb_boundary_flushed > 0);
}

// Tight spares (2 per layer in the main workload too): demands find no free slot and wait for deferred write-backs.
TEST(dg_cuda_wb_defer_tight_spares) {
  const ss_dg_stats s = wb_defer_cases("wbdt", [](ss_options&) {}, 2);
  CHECK(s.wb_forced > 0);
}

// An older, smaller ss_dg_stats: nothing is written past its size.
TEST(dg_cuda_wb_defer_stats_abi) {
  Rig r("wbd-abi", 8, 2, 8, 10000, true, [](ss_options& o) { o.dg_wb_defer = 1; });
  if (!r.c) return;
  Tally k;
  for (int i = 0; i < 5; ++i) k.add(r.token(false, false));
  CHECK_EQ(k.wrong + k.failed, 0);
  ss_dg_stats old;
  std::memset(&old, 0x5a, sizeof old);
  const uint32_t old_size = uint32_t(offsetof(ss_dg_stats, wb_deferred));
  CHECK_OK(ss_stats_dg_sized(r.c, &old, old_size));
  CHECK_EQ(old.struct_size, old_size);
  CHECK_EQ(old.tokens, 5ull);
  const unsigned char* tail = reinterpret_cast<const unsigned char*>(&old) + old_size;
  bool untouched = true;
  for (size_t i = 0; i < sizeof old - old_size; ++i) untouched = untouched && tail[i] == 0x5a;
  CHECK(untouched);
}

// dg_wb_defer with victims still deferred when the service stops: the token fails, the context is poisoned, close
// drops the unsubmitted write-backs and frees after draining the submitted ones.
TEST(dg_cuda_wb_defer_stopped) {
  Rig r("wbd-stop", 8, 2, 8, 10000, true, [](ss_options& o) { o.dg_wb_defer = 1; });
  if (!r.c) return;
  CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
  r.capture(true);
  Tally k;
  for (int i = 0; i < 10; ++i) k.add(r.token(true, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK(r.stats().wb_deferred > 0);
  CHECK_OK(ss_test_dg_stop_service(r.c));
  ss_status st = SS_OK;
  CHECK_EQ(r.token(true, true, &st), -1);
  CHECK_EQ(int(st), int(SS_E_BACKEND));
  expect_poisoned(r);
}

// With dg_wb_defer = 0 nothing is deferred: the new counters stay 0 (the issued bytes count the boundary's copies).
TEST(dg_cuda_wb_defer_off_counters) {
  Rig r("wbd-off", 8, 2, 8);
  if (!r.c) return;
  CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
  r.capture(true);
  Tally k;
  for (int i = 0; i < 50; ++i) k.add(r.token(true, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  const ss_dg_stats x = r.stats();
  print_wb("defer off", x);
  CHECK_EQ(x.wb_deferred + x.wb_cancelled + x.wb_forced + x.wb_boundary_flushed, 0ull);
  CHECK_EQ(x.wb_issued_bytes, x.evicted * r.m.unit());
}

// A stopped service: the device's waits end on host_abort within a round trip, ss_end fails, close frees.
TEST(dg_cuda_stopped_service) {
  Rig r("stop", 4, 4, 8, 10000);
  if (!r.c) return;
  ss_snapshot_info info{};
  CHECK_EQ(int(ss_stats_snapshot(r.c, nullptr, &info)), int(SS_E_INVALID));  // arms it
  r.capture(false);
  CHECK_EQ(r.token(true, false), 0);
  CHECK_OK(ss_stats_snapshot(r.c, nullptr, &info));
  const uint64_t published = info.sequence;
  CHECK_OK(ss_test_dg_stop_service(r.c));
  const auto t0 = std::chrono::steady_clock::now();
  ss_status st = SS_OK;
  CHECK_EQ(r.token(true, false, &st), -1);
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  stopped service: ss_end %d in %.3f s: %s\n", int(st), s, ss_last_error());
  CHECK_EQ(int(st), int(SS_E_BACKEND));
  CHECK(s < 3.0);  // not the 10 s device deadline
  expect_poisoned(r);
  ss_stats_reset(r.c);  // a poisoned context publishes no more
  CHECK_OK(ss_stats_snapshot(r.c, nullptr, &info));
  CHECK_EQ(info.sequence, published);
}

TEST(dg_cuda_bad_id) {
  Rig r("badid", 16, 4, 8);
  if (!r.c) return;
  r.capture(false);
  ss_status st = SS_OK;
  CHECK_EQ(r.token(true, false, &st, 2), -1);
  std::printf("  bad id: ss_end %d: %s\n", int(st), ss_last_error());
  CHECK_EQ(int(st), int(SS_E_BACKEND));
  expect_poisoned(r);
}

// Mutation: no ready wait. The demand copies of one token are held back by a gate (a ~150 ms spin on another stream);
// with the wait the readers see the units, without it they read the guard slot (zeros).
TEST(dg_cuda_mutation_no_ready_wait) {
  for (int mut = 0; mut < 2; ++mut) {
    Rig r(mut ? "nowait-m" : "nowait", 4, 8, 8);
    if (!r.c) return;
    if (mut) CHECK_OK(ss_test_dg_flags(r.c, 1));  // before the capture: the flag is a kernel argument
    r.capture(false);
    cudaStream_t gs = nullptr;
    cudaEvent_t gate = nullptr;
    CU(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
    Tally k;
    for (int t = 0; t < 5; ++t) {
      spin_ns<<<1, 1, 0, gs>>>(150000000ull);
      CU(cudaEventRecord(gate, gs));
      CHECK_OK(ss_test_dg_gate(r.c, 0, gate));
      ss_status st = SS_OK;
      k.add(r.token(true, false, mut ? &st : nullptr));
      if (k.failed) {
        std::printf("  mutation failed the token: %s\n", ss_last_error());
        break;  // poisoned from here
      }
    }
    std::printf("  %s: wrong %d, failed tokens %d\n", mut ? "mutation (no ready wait)" : "normal (gated copies)",
                k.wrong, k.failed);
    if (mut) CHECK(k.wrong > 0 || k.failed > 0);  // caught: wrong bytes, or an absent entry reported (rev 3.4)
    else CHECK_EQ(k.wrong + k.failed, 0);
    CU(cudaStreamSynchronize(gs));
    (void)cudaEventDestroy(gate);
    (void)cudaStreamDestroy(gs);
  }
}

// Mutation: no completion edge. The last layer's reader is delayed ~100 ms and every boundary evicts the most
// recently used units: without the edge the boundary unpublishes them while the reader still resolves its slots.
TEST(dg_cuda_mutation_no_completion_edge) {
  for (int mut = 0; mut < 2; ++mut) {
    Rig r(mut ? "edge-m" : "edge", 8, 4, 8);
    if (!r.c) return;
    CHECK_OK(ss_test_dg_pressure(r.c, 0, 24));
    r.last_spin_ns = 100000000ull;
    r.capture(false);
    if (mut) CHECK_OK(ss_test_dg_flags(r.c, 2));
    Tally k;
    for (int t = 0; t < 6; ++t) {
      ss_status st = SS_OK;
      k.add(r.token(true, false, mut ? &st : nullptr));
      if (k.failed) {
        std::printf("  mutation failed the token: %s\n", ss_last_error());
        break;  // poisoned from here
      }
    }
    std::printf("  %s: wrong %d, failed tokens %d\n",
                mut ? "mutation (no completion edge)" : "normal (completion edge)", k.wrong, k.failed);
    if (mut) CHECK(k.wrong > 0 || k.failed > 0);  // caught: wrong bytes, or a desync / absent report (rev 3.4)
    else CHECK_EQ(k.wrong + k.failed, 0);
  }
}

int main(int argc, char** argv) {
  const char* only = nullptr;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 2;
  }
  std::printf("device %d: %s, %.1f GiB, async engines %d\n", dev, p.name, double(p.totalGlobalMem) / double(1ull << 30),
              p.asyncEngineCount);
  return sstest::run_all(only);
}

// design rev 3.4 §11.1: an eager step outside any pair (stream-synchronized before ss_begin) is not authorized: its
// reads all return the zeroed guard, and the next ss_begin fails at once.
TEST(dg_cuda_desync_warmup) {
  Rig r("desync1", 16, 8, 8);
  if (!r.c) return;
  Tally k;
  for (int t = 0; t < 5; ++t) k.add(r.token(false, false));
  CHECK_EQ(k.wrong + k.failed, 0);
  r.fill_ids();
  r.enqueue_step(false);  // no ss_begin
  CU(cudaStreamSynchronize(r.s));
  CHECK(r.sums_all_zero());
  CHECK_EQ(int(ss_begin(r.c, nullptr, 0)), int(SS_E_BACKEND));
  std::printf("  warm-up outside the pair: %s\n", ss_last_error());
  CHECK(std::string(ss_last_error()).find("sequence desync at ss_begin") != std::string::npos);
  expect_poisoned(r);
}

// Two steps inside one pair: the second is not authorized (guard reads); ss_end fails at once, not after a timeout.
TEST(dg_cuda_desync_two_steps) {
  Rig r("desync2", 16, 8, 8);
  if (!r.c) return;
  Tally k;
  for (int t = 0; t < 5; ++t) k.add(r.token(false, false));
  CHECK_EQ(k.wrong + k.failed, 0);
  r.fill_ids();
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  r.enqueue_step(false);
  r.enqueue_step(false);
  CU(cudaStreamSynchronize(r.s));
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));
  const double s_end = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  two steps in one pair: ss_end failed in %.3f s: %s\n", s_end, ss_last_error());
  CHECK(s_end < 2.0);
  CHECK(std::string(ss_last_error()).find("sequence desync at ss_end") != std::string::npos);
  expect_poisoned(r);
}

// design rev 3.4 §11.3: a reader that meets an entry < 0 without abort (a host bug, injected) reports it; ss_end fails.
TEST(dg_cuda_absent_entry) {
  for (uint32_t wbm : WB_MODES) {
  Rig r("absent", 16, 8, 8, 10000, true, [wbm](ss_options& o) { WB_SET(o, wbm); });
  if (!r.c) return;
  Tally k;
  for (int t = 0; t < 5; ++t) k.add(r.token(false, false));
  CHECK_EQ(k.wrong + k.failed, 0);
  r.fill_ids();
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_dg_enqueue_begin(r.c, r.s));
  for (int g = 0; g < r.m.G; ++g) {
    CHECK_OK(ss_dg_enqueue_export(r.c, g, r.ids_dev + size_t(g) * r.n, int(r.n), 0, r.s));
    CHECK_OK(ss_dg_enqueue_wait(r.c, g, r.s));
    if (g == 1) unpublish_first<<<1, 1, 0, r.s>>>(r.v, uint32_t(g), r.ids_dev + size_t(g) * r.n);
    checksum<<<r.n, 128, 0, r.s>>>(r.v, uint32_t(g), r.ids_dev + size_t(g) * r.n, r.sums_dev + size_t(g) * r.n);
  }
  CHECK_OK(ss_dg_enqueue_end(r.c, r.s));
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  absent entry (write-back mode %u): ss_end after %.3f s: %s\n", wbm, secs, ss_last_error());
  CHECK(std::string(ss_last_error()).find("absent entry") != std::string::npos);
  CU(cudaStreamSynchronize(r.s));
  print("absent", r.stats());
  expect_poisoned(r);
  }
}

// design rev 3.4 §11.6: dma_ms on the layers with misses (-1 on the others), the new counters, device_timeouts 0.
// The device stamps (gpu_wait_us <= gpu_export_ready_us on a miss layer, -1 on the others), wb_left_ms (-1 or >= 0)
// and the boundary timeline (its write-back count = the token's evictions).
TEST(dg_cuda_dma_and_counters) {
  Rig r("dma", 16, 8, 8);
  if (!r.c) return;
  r.capture(true);
  Tally k;
  int measured = 0, unmeasured = 0, bad = 0, wb_seen = 0;
  double wait_us = 0, path_us = 0;
  ss_dg_boundary_timeline bt{};
  CHECK_EQ(int(ss_dg_boundary_timeline_sized(r.c, &bt, uint32_t(sizeof bt))), int(SS_E_INVALID));  // no token yet
  for (int t = 0; t < 60; ++t) {
    const uint64_t ev0 = r.stats().evicted;
    k.add(r.token(true, true));
    for (int g = 0; g < r.m.G; ++g) {
      ss_dg_layer_timeline tl{};
      CHECK_OK(ss_dg_timeline_sized(r.c, g, &tl, uint32_t(sizeof tl)));
      if (tl.misses > 0) {
        ++measured;
        if (!(tl.dma_ms >= 0)) ++bad;
        if (!(tl.gpu_wait_us >= 0 && tl.gpu_export_ready_us >= tl.gpu_wait_us)) ++bad;
        if (!(tl.wb_left_ms == -1 || tl.wb_left_ms == -2 || tl.wb_left_ms >= 0)) ++bad;
        if (tl.fetched > 0 ? !(tl.dma_first_ms >= 0 && tl.dma_first_ms <= tl.dma_ms) : tl.dma_first_ms != -1) ++bad;
        wb_seen += tl.wb_left_ms >= 0;
        wait_us += tl.gpu_wait_us;
        path_us += tl.gpu_export_ready_us;
      } else {
        ++unmeasured;
        if (tl.dma_ms != -1 || tl.gpu_wait_us != -1 || tl.gpu_export_ready_us != -1 || tl.wb_left_ms != -1 ||
            tl.dma_first_ms != -1)
          ++bad;
      }
    }
    CHECK_OK(ss_dg_boundary_timeline_sized(r.c, &bt, uint32_t(sizeof bt)));
    if (uint64_t(bt.writebacks) != r.stats().evicted - ev0) ++bad;
    if (!(bt.sync_us >= 0 && bt.select_us >= 0 && bt.upload_us >= 0 && bt.issue_us >= 0)) ++bad;
  }
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK_EQ(bad, 0);
  CHECK(measured > 0);
  std::printf("  device stamps: mean wait %.1f us, export -> ready %.1f us over %d miss layers (%d with the previous "
              "boundary's write-backs measured)\n",
              wait_us / measured, path_us / measured, measured, wb_seen);
  const ss_dg_stats x = r.stats();
  std::printf("  dma: %d layers measured, %d without a miss; predicted %llu (true %llu), miss_layers %llu, covered "
              "%llu (waits %llu), device_timeouts %llu\n",
              measured, unmeasured, (unsigned long long)x.predicted, (unsigned long long)x.predicted_true,
              (unsigned long long)x.miss_layers, (unsigned long long)x.covered, (unsigned long long)x.covered_waits,
              (unsigned long long)x.device_timeouts);
  CHECK(x.predicted > 0 && x.miss_layers > 0);
  CHECK(x.predicted_true <= x.predicted);
  CHECK_EQ(x.device_timeouts, 0ull);
  std::printf("  dma: open the diag-off rig\n");
  // without dg_diag there is no timeline at all (no DMA events are recorded): the getter refuses, the token runs
  Rig q("dma-off", 16, 8, 8, 10000, true, [](ss_options& o) { o.dg_diag = 0; });
  if (!q.c) return;
  Tally kq;
  for (int t = 0; t < 4; ++t) kq.add(q.token(false, false));
  CHECK_EQ(kq.wrong + kq.failed, 0);
  ss_dg_layer_timeline tl{};
  CHECK_EQ(int(ss_dg_timeline_sized(q.c, 0, &tl, uint32_t(sizeof tl))), int(SS_E_INVALID));
}

// The diagnostics getters' ABI: refused without dg_diag; an older, smaller struct gets only its own fields.
TEST(dg_cuda_diag_abi) {
  std::printf("  diag abi: open (diag off)\n");
  {
    Rig q("diag-off", 16, 8, 8, 10000, true, [](ss_options& o) { o.dg_diag = 0; });
    if (!q.c) return;
    Tally kq;
    for (int t = 0; t < 2; ++t) kq.add(q.token(false, false));
    CHECK_EQ(kq.wrong + kq.failed, 0);
    std::printf("  diag abi: boundary getter refused without diag\n");
    ss_dg_boundary_timeline bq{};
    CHECK_EQ(int(ss_dg_boundary_timeline_sized(q.c, &bq, uint32_t(sizeof bq))), int(SS_E_INVALID));
    std::printf("  diag abi: close (diag off)\n");
  }
  std::printf("  diag abi: open (diag on)\n");
  Rig r("diag-on", 16, 8, 8);
  if (!r.c) return;
  Tally k;
  for (int t = 0; t < 4; ++t) k.add(r.token(false, false));
  CHECK_EQ(k.wrong + k.failed, 0);
  std::printf("  diag abi: old-size timeline\n");
  ss_dg_layer_timeline old;
  std::memset(&old, 0x5a, sizeof old);
  const uint32_t old_size = uint32_t(offsetof(ss_dg_layer_timeline, gpu_wait_us));
  CHECK_OK(ss_dg_timeline_sized(r.c, 0, &old, old_size));
  CHECK_EQ(old.struct_size, old_size);
  const unsigned char* tail = reinterpret_cast<const unsigned char*>(&old) + old_size;
  bool untouched = true;
  for (size_t i = 0; i < sizeof old - old_size; ++i) untouched = untouched && tail[i] == 0x5a;
  CHECK(untouched);
  ss_dg_boundary_timeline bt{};
  CHECK_OK(ss_dg_boundary_timeline_sized(r.c, &bt, uint32_t(sizeof bt)));
  std::printf("  diag abi: boundary sync %.1f select %.1f upload %.1f issue %.1f us, %u write-backs; close\n",
              bt.sync_us, bt.select_us, bt.upload_us, bt.issue_us, bt.writebacks);
}

// Resident-first reads: ss_dg_slot_if_ready before each layer's wait returns only slots whose bytes are complete
// (every early checksum exact), under pressure, eager and captured, with and without prediction and deferred
// write-backs; some ids are ready early and some are not.
TEST(dg_cuda_slot_if_ready) {
  for (uint32_t defer : {0u, 1u}) {
    Rig r(defer ? "early-defer" : "early", 16, 8, 8, 10000, true, [defer](ss_options& o) { o.dg_wb_defer = defer; });
    if (!r.c) return;
    r.enable_early();
    Tally k;
    for (int t = 0; t < 20; ++t) k.add(r.token(false, t % 2 == 1));
    r.capture(false);
    r.capture(true);
    for (int t = 0; t < 60; ++t) k.add(r.token(true, t % 3 != 0));
    CHECK_EQ(k.wrong + k.failed, 0);
    std::printf("  slot_if_ready (defer %u): %llu of %llu routed ids ready before the wait, %llu early checksums wrong\n",
                defer, (unsigned long long)r.early_ready, (unsigned long long)r.early_total,
                (unsigned long long)r.early_wrong);
    CHECK_EQ(r.early_wrong, 0ull);
    CHECK(r.early_ready > 0 && r.early_ready < r.early_total);
    CHECK_OK(ss_test_dg_check(r.c));
  }
}

// design rev 3.4 §11.5 / §11.8: hooks refused without dg_test_hooks; ss_dg_set_budgets bounded and between tokens.
TEST(dg_cuda_hooks_and_budgets) {
  Rig r("hooks", 16, 8, 8, 10000, false);
  if (!r.c) return;
  CHECK_EQ(int(ss_test_dg_check(r.c)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_test_dg_pressure(r.c, 1, 1)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_dg_set_budgets(r.c, 1u << 30, 4, 48)), int(SS_E_INVALID));  // above the billed spares
  CHECK_OK(ss_dg_set_budgets(r.c, 2, 0, 0));
  Tally k;
  for (int t = 0; t < 30; ++t) k.add(r.token(false, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK_EQ(r.stats().prefetched, 0ull);  // budget 0: no prefetch copy
  CHECK(r.stats().predicted > 0);        // but the predictions are counted
  // inside a token: refused, and the token still completes
  r.fill_ids();
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_EQ(int(ss_dg_set_budgets(r.c, 2, 4, 48)), int(SS_E_INVALID));
  r.enqueue_step(false);
  CHECK_OK(ss_end(r.c));
  CU(cudaStreamSynchronize(r.s));
}

// design rev 3.4 §11.7 / §11.8: option refusals at open.
TEST(dg_cuda_open_refusals) {
  struct Case {
    const char* what;
    std::function<void(ss_options&)> tweak;
  };
  static const ss_dg_unit bad[1] = {{0, 1 << 20}};
  const Case cases[] = {
      {"above the billed spares", [](ss_options& o) { o.dg_spare_target = 1u << 30; }},
      {"is out of range", [](ss_options& o) { o.dg_seed = bad; o.dg_seed_n = 1; }},
      {"null dg_seed", [](ss_options& o) { o.dg_seed = nullptr; o.dg_seed_n = 2; }},
  };
  for (const Case& c : cases) {
    ss_status st = SS_OK;
    Rig r("refuse", 16, 8, 8, 10000, true, c.tweak, &st);
    CHECK_EQ(int(st), int(SS_E_INVALID));
    CHECK(r.c == nullptr);
    if (std::string(ss_last_error()).find(c.what) == std::string::npos)
      std::printf("  refusal '%s' missing in: %s\n", c.what, ss_last_error());
    CHECK(std::string(ss_last_error()).find(c.what) != std::string::npos);
  }
}

// design rev 3.4 §11.7: a seed of units the plan did not make resident: every unit's bytes check out in its swapped
// home, the first token routing only seeded units misses nothing, and the slot / block counts match an unseeded open.
TEST(dg_cuda_seed_open) {
  std::vector<ss_dg_unit> seed;
  Rig plain("seed0", 4, 8, 8);
  if (!plain.c) return;
  for (int g = 0; g < plain.m.G; ++g)
    for (int u = plain.m.U - 1; u >= plain.m.U - 4; --u) seed.push_back({g, u});  // the last 4 units of each layer
  Rig r("seed1", 4, 8, 8, 10000, true, [&](ss_options& o) {
    o.dg_seed = seed.data();
    o.dg_seed_n = uint32_t(seed.size());
  });
  if (!r.c) return;
  for (int L = 0; L < r.m.G; ++L)
    for (int u = 0; u < r.m.U; ++u) CHECK_OK(ss_check_unit(r.c, L, u));
  const ss_dg_stats a = r.stats(), b0 = plain.stats();
  CHECK_EQ(a.free_slots, b0.free_slots);
  CHECK_EQ(a.blocks, b0.blocks);
  const std::vector<int32_t> ids = {int32_t(r.m.U - 1), int32_t(r.m.U - 2), int32_t(r.m.U - 3), int32_t(r.m.U - 4)};
  Tally k;
  k.add(r.token(false, false, nullptr, -1, ids));
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK_EQ(r.stats().misses, 0ull);
  k.add(plain.token(false, false, nullptr, -1, ids));
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK(plain.stats().misses > 0);  // the same routes miss without the seed
}

// design rev 3.4 §11.3: an old-size view (no absent_err) still reads the guard for an absent entry but cannot report
// it; ABSENT never overrides an earlier code (a bad id here).
TEST(dg_cuda_absent_old_view_and_order) {
  {
    Rig r("absent-old", 16, 8, 8);
    if (!r.c) return;
    ss_dg_device_view old{};
    CHECK_OK(ss_dg_get_device_view_sized(r.c, &old, uint32_t(offsetof(ss_dg_device_view, absent_err))));
    CHECK_EQ(old.struct_size, uint32_t(offsetof(ss_dg_device_view, absent_err)));
    r.fill_ids();
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_dg_enqueue_begin(r.c, r.s));
    for (int g = 0; g < r.m.G; ++g) {
      CHECK_OK(ss_dg_enqueue_export(r.c, g, r.ids_dev + size_t(g) * r.n, int(r.n), 0, r.s));
      CHECK_OK(ss_dg_enqueue_wait(r.c, g, r.s));
      if (g == 1) unpublish_first<<<1, 1, 0, r.s>>>(r.v, uint32_t(g), r.ids_dev + size_t(g) * r.n);
      checksum<<<r.n, 128, 0, r.s>>>(old, uint32_t(g), r.ids_dev + size_t(g) * r.n, r.sums_dev + size_t(g) * r.n);
    }
    CHECK_OK(ss_dg_enqueue_end(r.c, r.s));
    CHECK_OK(ss_end(r.c));  // nothing to report through an old view: the documented limit
    CU(cudaStreamSynchronize(r.s));
  }
  {
    // layer 0 routes residents only (no wait) and meets an unpublished entry: absent_err set, no abort; the LAST layer
    // routes non-residents whose demand copies are held past the wait deadline: a timeout (abort) there (the last, so
    // no service flag wait follows it). The end kernel folds ABSENT only without abort: the timeout is reported.
    Rig r("absent-order", 16, 8, 8, 300);
    if (!r.c) return;
    std::vector<int32_t> ids(size_t(r.m.G) * r.n);
    for (int g = 0; g < r.m.G; ++g)
      for (uint32_t i = 0; i < r.n; ++i) ids[size_t(g) * r.n + i] = int32_t(g == r.m.G - 1 ? 40 + i : i);  // residents: 0..15
    CU(cudaMemcpy(r.ids_dev, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
    CU(cudaMemset(r.sums_dev, 0, ids.size() * 8));
    cudaStream_t gs = nullptr;
    cudaEvent_t gate = nullptr;
    CU(cudaStreamCreateWithFlags(&gs, cudaStreamNonBlocking));
    CU(cudaEventCreateWithFlags(&gate, cudaEventDisableTiming));
    spin_ns<<<1, 1, 0, gs>>>(1000000000ull);  // 1 s > the 300 ms wait deadline, < ss_end's deadline + 5 s
    CU(cudaEventRecord(gate, gs));
    CHECK_OK(ss_test_dg_gate(r.c, 0, gate));  // the demand stream's next copies wait for it
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_dg_enqueue_begin(r.c, r.s));
    for (int g = 0; g < r.m.G; ++g) {
      CHECK_OK(ss_dg_enqueue_export(r.c, g, r.ids_dev + size_t(g) * r.n, int(r.n), 0, r.s));
      CHECK_OK(ss_dg_enqueue_wait(r.c, g, r.s));
      if (g == 0) unpublish_first<<<1, 1, 0, r.s>>>(r.v, uint32_t(g), r.ids_dev);
      checksum<<<r.n, 128, 0, r.s>>>(r.v, uint32_t(g), r.ids_dev + size_t(g) * r.n, r.sums_dev + size_t(g) * r.n);
    }
    CHECK_OK(ss_dg_enqueue_end(r.c, r.s));
    CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));
    std::printf("  absent then timeout: %s\n", ss_last_error());
    CHECK(std::string(ss_last_error()).find("a wait timed out at gated layer " + std::to_string(r.m.G - 1)) != std::string::npos);
    CHECK(std::string(ss_last_error()).find("absent") == std::string::npos);
    CU(cudaStreamSynchronize(r.s));
    CU(cudaStreamSynchronize(gs));
    (void)cudaEventDestroy(gate);
    (void)cudaStreamDestroy(gs);
  }
}

// design rev 3.4 §11.2: a step that never launches is caught by ss_end's deadline (the service waits for began with
// no deadline of its own), and close returns.
TEST(dg_cuda_never_launched) {
  Rig r("nolaunch", 16, 8, 8, 500);
  if (!r.c) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  const auto t0 = std::chrono::steady_clock::now();
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("  never launched: ss_end failed after %.2f s: %s\n", secs, ss_last_error());
  CHECK(std::string(ss_last_error()).find("did not finish") != std::string::npos);
  CHECK(secs < 10.0);
  ss_close(r.c);
  r.c = nullptr;
}

// design rev 3.4 §11.5: every hook refused without dg_test_hooks.
TEST(dg_cuda_hooks_refused) {
  Rig r("hooks2", 16, 8, 8, 10000, false);
  if (!r.c) return;
  CHECK_EQ(int(ss_test_dg_mutate(r.c, 1)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_test_dg_gate(r.c, 0, nullptr)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_test_dg_flags(r.c, 1)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_test_dg_stop_service(r.c)), int(SS_E_INVALID));
  Tally k;
  for (int t = 0; t < 3; ++t) k.add(r.token(false, false));  // the context is unharmed
  CHECK_EQ(k.wrong + k.failed, 0);
}

// ---- winpf (window prefetch): window hints between tokens (prior prefetches of gated layers 0
// and 1), the l+1 union ranked by row votes inside the token, unused prefetches reverted at the boundary; eager, graph,
// pressure, with and without deferred write-backs and with a tiny prior budget. Every checksum must be exact. ----
namespace {

std::vector<uint8_t> winpf_prior(const DgModel& m, bool bad_identity = false) {
  const ts::Desc d = ts::desc_from_json(ts::json_parse(m.json));
  std::vector<int> gated;
  for (int L = 0; L < m.G; ++L) gated.push_back(L);
  std::vector<uint8_t> b;
  auto u32 = [&](uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(v >> (8 * i)));
  };
  auto u16 = [&](uint32_t v) {
    b.push_back(uint8_t(v));
    b.push_back(uint8_t(v >> 8));
  };
  for (char ch : std::string("TSHPRIO1")) b.push_back(uint8_t(ch));
  u32(1);
  u32(uint32_t(m.G));
  for (int L = 0; L < m.G; ++L) u32(uint32_t(m.U));
  std::string id = ts::hint_prior_identity(d, gated);
  if (bad_identity) id[0] ^= 1;
  b.insert(b.end(), id.begin(), id.end());
  const std::string prov = "dg_cuda_winpf";
  u32(uint32_t(prov.size()));
  b.insert(b.end(), prov.begin(), prov.end());
  const uint32_t P = 2, k = 4, n = 40;
  u32(P);
  u32(k);
  u32(n);
  for (uint32_t s = 0; s < n; ++s) {
    u32(s);
    for (uint32_t j = 0; j < P * k; ++j) u16((s * 5 + j * 3) % 24);  // low ids: the hot ones of Rig::pick
    for (uint32_t j = 0; j < P * k; ++j) u16(65535u - 7000u * (j % k));
  }
  const std::string hex = ts::sha256_hex(b.data(), b.size());
  for (size_t i = 0; i < 64; i += 2) b.push_back(uint8_t(std::stoi(hex.substr(i, 2), nullptr, 16)));
  return b;
}

void winpf_on(Rig& r, uint32_t window_us) {
  ss_dg_winpf_params w;
  ss_dg_winpf_defaults(&w);
  CHECK_EQ(w.enable, 0u);
  w.enable = 1;
  w.top_k = 4;  // the fixture's top_k: n = 8 ids = 2 rows; the prediction 8 = 2 rows x 4
  w.l1_cap = 2;
  w.l1_pmin = 0.3f;
  w.prior_cap = 2;
  w.prior_pmin = 0.2f;
  w.prior_window_us = window_us;
  w.copy_gbps = 1.0f;  // 1 us x 1 GB/s = 1000 bytes: window_us / 8 units of this model's 8 KiB
  CHECK_OK(ss_dg_winpf_set(r.c, &w));
}

int hinted_token(Rig& r, bool graph) {
  int32_t sym[2] = {int32_t(r.rng() % 48), int32_t(r.rng() % 48)};  // 40..47: no prior entry
  CHECK_OK(ss_dg_window_hint(r.c, sym, nullptr, 1));                 // the bonus row, then the draft
  const float w = 0.5f;
  CHECK_OK(ss_dg_window_hint(r.c, sym + 1, &w, 1));
  return r.token(graph, true);
}

void winpf_cases(const std::string& t, bool defer, uint32_t window_us, bool must_bind = false) {
  const std::vector<uint8_t> prior = winpf_prior(DgModel(t));
  const auto tweak = [&](ss_options& o) {
    o.dg_prior = prior.data();
    o.dg_prior_len = prior.size();
    o.dg_wb_defer = defer ? 1u : 0u;
  };
  ss_dg_stats sum{};
  {
    Rig r(t, 16, 8, 8, 10000, true, tweak);
    if (!r.c) return;
    winpf_on(r, window_us);
    Tally k;
    for (int i = 0; i < 20; ++i) k.add(hinted_token(r, false));
    r.capture(true);
    for (int i = 0; i < 200; ++i) {
      k.add(hinted_token(r, true));
      if (i % 20 == 0) CHECK_OK(ss_test_dg_check(r.c));
    }
    for (int i = 0; i < 40; ++i) k.add(r.token(true, true));  // no hint: today's prefetch for those tokens
    CHECK_EQ(k.failed, 0);
    CHECK_EQ(k.wrong, 0);
    // a hint inside a token is refused (no poison); the token itself completes exact
    r.fill_ids();
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    int32_t sym = 3;
    CHECK_EQ(int(ss_dg_window_hint(r.c, &sym, nullptr, 1)), int(SS_E_INVALID));
    CU(cudaGraphLaunch(r.graph[1], r.s));
    CHECK_OK(ss_end(r.c));
    CU(cudaStreamSynchronize(r.s));
    CHECK_OK(ss_test_dg_check(r.c));
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 7) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print((t + " main").c_str(), x);
    sum = x;
  }
  {  // pressure: banks, resurrections, mid-token collects, MRU evictions with window prefetch on
    Rig r(t + "-pressure", 8, 2, 8, 10000, true, tweak);
    if (!r.c) return;
    winpf_on(r, window_us);
    CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
    r.capture(true);
    Tally k;
    for (int i = 0; i < 200; ++i) {
      k.add(hinted_token(r, true));
      CHECK_OK(ss_test_dg_check(r.c));
    }
    CHECK_EQ(k.failed, 0);
    CHECK_EQ(k.wrong, 0);
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 5) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print((t + " pressure").c_str(), x);
    // Without deferral a resurrection needs a re-demand before an 8 KiB write-back has completed (a timing window of
    // microseconds: 0-11 per run on the RTX PRO); deferred victims stay evicting into the next token, so there it
    // must happen.
    CHECK(x.bank_serves > 0);
    if (defer) CHECK(x.resurrected > 0);
    for (auto f : {&ss_dg_stats::winpf_hints, &ss_dg_stats::prior_issued, &ss_dg_stats::prior_used,
                   &ss_dg_stats::prior_reverted, &ss_dg_stats::l1_issued, &ss_dg_stats::l1_used,
                   &ss_dg_stats::l1_reverted, &ss_dg_stats::prior_budget_stops, &ss_dg_stats::l1_priced_out})
      sum.*f += x.*f;
  }
  std::printf("  %s winpf: hints %llu, prior issued %llu used %llu late %llu reverted %llu; l1 candidates %llu issued %llu "
              "used %llu late %llu reverted %llu priced out %llu ahead %llu\n",
              t.c_str(), (unsigned long long)sum.winpf_hints, (unsigned long long)sum.prior_issued,
              (unsigned long long)sum.prior_used, (unsigned long long)sum.prior_late,
              (unsigned long long)sum.prior_reverted, (unsigned long long)sum.l1_candidates,
              (unsigned long long)sum.l1_issued, (unsigned long long)sum.l1_used, (unsigned long long)sum.l1_late,
              (unsigned long long)sum.l1_reverted, (unsigned long long)sum.l1_priced_out,
              (unsigned long long)sum.l1_skipped_ahead);
  CHECK(sum.winpf_hints > 0);
  CHECK(sum.prior_issued > 0 && sum.prior_used > 0);
  CHECK(sum.l1_issued > 0 && sum.l1_used > 0);
  CHECK(sum.prior_reverted + sum.l1_reverted > 0);
  if (must_bind) {  // the byte budget (queued write-back bytes included) and the precision floor both bound admission
    std::printf("  %s: prior budget stops %llu, l1 priced out %llu\n", t.c_str(), (unsigned long long)sum.prior_budget_stops,
                (unsigned long long)sum.l1_priced_out);
    CHECK(sum.prior_budget_stops > 0);
    CHECK(sum.l1_priced_out > 0);
  }
}

}  // namespace

TEST(dg_cuda_winpf) { winpf_cases("wpf", false, 1500); }

// with deferred write-backs and a prior budget of ~6 units (write-back bytes still queued count against it)
TEST(dg_cuda_winpf_defer_budget) { winpf_cases("wpfd", true, 20, true); }

// a prior for another layout (identity), or with a broken checksum, fails ss_open
TEST(dg_cuda_winpf_prior_refused) {
  for (int bad = 0; bad < 2; ++bad) {
    std::vector<uint8_t> prior = winpf_prior(DgModel("wpf-bad"), bad == 0);
    if (bad == 1) prior[prior.size() - 40] ^= 1;  // a record byte: the checksum
    ss_status st = SS_OK;
    Rig r("wpf-bad", 16, 8, 8, 10000, true,
          [&](ss_options& o) {
            o.dg_prior = prior.data();
            o.dg_prior_len = prior.size();
          },
          &st);
    CHECK_EQ(int(st), int(SS_E_INVALID));
    CHECK(r.c == nullptr);
    std::printf("  refused: %s\n", ss_last_error());
  }
}

// winpf ABI: the old-size ss_dg_stats (ending at wbsched's last field) is written only up to its size; the params
// struct: NULL = off, invalid values refused (no poison), an old-size params struct without top_k cannot enable.
TEST(dg_cuda_winpf_abi) {
  const std::vector<uint8_t> prior = winpf_prior(DgModel("wpf-abi"));
  Rig r("wpf-abi", 16, 8, 8, 10000, true, [&](ss_options& o) {
    o.dg_prior = prior.data();
    o.dg_prior_len = prior.size();
  });
  if (!r.c) return;
  winpf_on(r, 1500);
  r.capture(true);
  Tally k;
  for (int i = 0; i < 30; ++i) k.add(hinted_token(r, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  // old-size stats: up to the field before winpf_hints, canaries after
  const uint32_t old = uint32_t(offsetof(ss_dg_stats, winpf_hints));
  std::vector<uint8_t> buf(sizeof(ss_dg_stats) + 64, 0xA5);
  ss_dg_stats* s = reinterpret_cast<ss_dg_stats*>(buf.data());
  CHECK_OK(ss_stats_dg_sized(r.c, s, old));
  CHECK_EQ(s->struct_size, old);
  bool canary = true;
  for (size_t i = old; i < buf.size(); ++i) canary = canary && buf[i] == 0xA5;
  CHECK(canary);
  const ss_dg_stats full = r.stats();
  CHECK_EQ(s->wb_forced_wait_ns, full.wb_forced_wait_ns);
  CHECK(full.winpf_hints == 60 && full.l1_candidates > 0);
  // params
  ss_dg_winpf_params p;
  ss_dg_winpf_defaults(&p);
  p.enable = 1;  // top_k missing
  CHECK_EQ(int(ss_dg_winpf_set(r.c, &p)), int(SS_E_INVALID));
  p.top_k = 4;
  p.l1_pmin = 2.0f;
  CHECK_EQ(int(ss_dg_winpf_set(r.c, &p)), int(SS_E_INVALID));
  p.l1_pmin = std::numeric_limits<float>::quiet_NaN();
  CHECK_EQ(int(ss_dg_winpf_set(r.c, &p)), int(SS_E_INVALID));
  p.l1_pmin = 0.4f;
  p.copy_gbps = 0;
  CHECK_EQ(int(ss_dg_winpf_set(r.c, &p)), int(SS_E_INVALID));
  ss_dg_winpf_params o2;
  ss_dg_winpf_defaults(&o2);
  o2.enable = 1;
  o2.struct_size = uint32_t(offsetof(ss_dg_winpf_params, top_k));  // an older caller: top_k reads as 0
  CHECK_EQ(int(ss_dg_winpf_set(r.c, &o2)), int(SS_E_INVALID));
  std::vector<int32_t> rows(65, 1);
  CHECK_EQ(int(ss_dg_window_hint(r.c, rows.data(), nullptr, 65)), int(SS_E_INVALID));
  const float badw = 1.5f;
  CHECK_EQ(int(ss_dg_window_hint(r.c, rows.data(), &badw, 1)), int(SS_E_INVALID));
  CHECK_OK(ss_dg_winpf_set(r.c, nullptr));  // off
  for (int i = 0; i < 10; ++i) k.add(r.token(true, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK_EQ(r.stats().winpf_hints, full.winpf_hints);
  // sized defaults: an old size writes only its bytes
  std::vector<uint8_t> pb(sizeof(ss_dg_winpf_params) + 16, 0x5A);
  ss_dg_winpf_defaults_sized(reinterpret_cast<ss_dg_winpf_params*>(pb.data()), uint32_t(offsetof(ss_dg_winpf_params, top_k)));
  canary = true;
  for (size_t i = offsetof(ss_dg_winpf_params, top_k); i < pb.size(); ++i) canary = canary && pb[i] == 0x5A;
  CHECK(canary);
}

// Prior files refused through ss_open: bad magic, truncated, one byte too long, wrong identity, bad checksum.
TEST(dg_cuda_winpf_prior_malformed) {
  const DgModel m("wpf-mal");
  const std::vector<uint8_t> good = winpf_prior(m);
  for (int bad = 0; bad < 5; ++bad) {
    std::vector<uint8_t> p = bad == 3 ? winpf_prior(m, true) : good;
    if (bad == 0) p[0] = 'X';
    if (bad == 1) p.resize(p.size() / 2);
    if (bad == 2) p.push_back(0);
    if (bad == 4) p[p.size() - 40] ^= 1;
    ss_status st = SS_OK;
    Rig r("wpf-mal", 16, 8, 8, 10000, true,
          [&](ss_options& o) {
            o.dg_prior = p.data();
            o.dg_prior_len = p.size();
          },
          &st);
    CHECK_EQ(int(st), int(SS_E_INVALID));
    CHECK(r.c == nullptr);
    std::printf("  refused %d: %s\n", bad, ss_last_error());
  }
}

// A prior copy held back on the prefetch stream (a gate) is still in flight at its layer's export: the export reads
// -1, the demand covers it and waits for its event before ready (late), and every checksum is exact.
TEST(dg_cuda_winpf_delayed_prior) {
  const std::vector<uint8_t> prior = winpf_prior(DgModel("wpf-late"));
  Rig r("wpf-late", 4, 8, 8, 10000, true, [&](ss_options& o) {
    o.dg_prior = prior.data();
    o.dg_prior_len = prior.size();
  });
  if (!r.c) return;
  winpf_on(r, 1500);
  r.capture(true);
  HostGate gate;
  CHECK_OK(ss_test_dg_pressure(r.c, 2, 8));  // the most recent units are evicted at every boundary: misses again
  Tally k;
  int held = 0;
  double token_ms_max = 0;
  for (int i = 0; i < 20; ++i) {
    held += gate.arm();
    CHECK_OK(ss_test_dg_gate(r.c, 1, gate.ev));  // the prefetch stream's next copies wait for it
    // every layer routes the prior's layer-0 units of symbol s (winpf_prior: (5 s + 3 j) % 24, j < 4), so the
    // prior names units the export wants
    const int32_t s = int32_t(i % 40);
    std::vector<int32_t> fixed;
    for (int j = 0; j < 4; ++j) fixed.push_back((5 * s + 3 * j) % 24);
    int32_t sym[2] = {s, s};
    CHECK_OK(ss_dg_window_hint(r.c, sym, nullptr, 2));
    // the host releases the gate 200 ms into the token: the step's first exports meet the prior copies in flight
    std::thread rel([&] {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      gate.release();
    });
    const auto t0 = std::chrono::steady_clock::now();
    k.add(r.token(true, true, nullptr, -1, fixed));
    token_ms_max = std::max(token_ms_max, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    rel.join();
    CU(cudaStreamSynchronize(gate.s));
  }
  CHECK_EQ(k.wrong + k.failed, 0);
  const ss_dg_stats x = r.stats();
  print("delayed prior", x);
  std::printf("  gate held at arm %d / 20, longest token %.0f ms; prior issued %llu used %llu late %llu\n", held,
              token_ms_max, (unsigned long long)x.prior_issued, (unsigned long long)x.prior_used,
              (unsigned long long)x.prior_late);
  CHECK_EQ(held, 20);
  CHECK(x.prior_issued > 0);
  CHECK(x.prior_used > 0);
  // late is observable only where the step reaches its first export while the gated copy still holds. On the RTX PRO
  // 6000 (one copy engine, WDDM; u3) the step itself starts only when the gate releases (every token ~200 ms = the
  // release), so the prior copies have landed by the export: exactness under the held copies is what this proves
  // there; the cover-wait accounting is proven by the CPU oracle (test_dg prior_late_and_revert).
  if (token_ms_max < 150) CHECK(x.prior_late > 0);  // the gate held the copies past the export
  else std::printf("  late not observable: the step started after the gate released (longest token %.0f ms)\n", token_ms_max);
  CHECK_OK(ss_test_dg_check(r.c));
}

// Prior copies still queued (gated) when the service stops: the token fails, the context is poisoned, close drains
// the queued copies before freeing (no crash, no hang).
TEST(dg_cuda_winpf_poison_with_prior_in_flight) {
  const std::vector<uint8_t> prior = winpf_prior(DgModel("wpf-stop"));
  Rig r("wpf-stop", 4, 8, 8, 2000, true, [&](ss_options& o) {
    o.dg_prior = prior.data();
    o.dg_prior_len = prior.size();
  });
  if (!r.c) return;
  winpf_on(r, 1500);
  r.capture(true);
  Tally k;
  for (int i = 0; i < 5; ++i) k.add(hinted_token(r, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  HostGate gate;
  CHECK(gate.arm(60000000000ull));  // held until the host releases it (bounded at 60 s; the failed token takes ~10 s)
  CHECK_OK(ss_test_dg_gate(r.c, 1, gate.ev));
  // a symbol whose prior units are not on the device: copies queue behind the gate
  uint32_t coming = 0;
  for (int32_t sym = 0; sym < 40 && coming == 0; ++sym) {
    int32_t rows[2] = {sym, sym};
    CHECK_OK(ss_dg_window_hint(r.c, rows, nullptr, 2));
    CHECK_OK(ss_test_dg_coming(r.c, &coming));
  }
  std::printf("  prior copies queued behind the gate: %u\n", coming);
  CHECK(coming > 0);
  CHECK_OK(ss_test_dg_stop_service(r.c));
  ss_status st = SS_OK;
  const auto tt = std::chrono::steady_clock::now();
  CHECK_EQ(r.token(true, true, &st), -1);
  std::printf("  failed token: %.0f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tt).count());
  CHECK_EQ(int(st), int(SS_E_BACKEND));
  expect_poisoned(r);
  CHECK_EQ(int(cudaEventQuery(gate.ev)), int(cudaErrorNotReady));  // the gate still holds the prior copies
  // close with the copies still queued: it drains them (released 300 ms into close) before freeing anything
  std::thread rel([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    gate.release();
  });
  const auto t0 = std::chrono::steady_clock::now();
  ss_close(r.c);
  r.c = nullptr;
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  rel.join();
  std::printf("  close with prior copies in flight: %.0f ms\n", ms);
  CHECK(ms >= 250);  // close waited for the queued copies (they were released 300 ms in)
  CHECK(ms < 5000);
}

// An options struct of an older caller (ending before dg_prior_len) with a hostile suffix: the suffix is never read,
// the context opens without a prior and window prefetch still runs (l+1 only).
TEST(dg_cuda_winpf_old_options) {
  std::vector<uint8_t> junk(64, 0xEE);
  Rig r("wpf-old", 16, 8, 8, 10000, true, [&](ss_options& o) {
    o.dg_prior_len = 0xEEEEEEEEEEEEull;  // would be refused or crash if read
    o.dg_prior = junk.data();
    o.struct_size = uint32_t(offsetof(ss_options, dg_prior_len));
  });
  if (!r.c) return;
  winpf_on(r, 1500);
  r.capture(true);
  Tally k;
  for (int i = 0; i < 30; ++i) k.add(hinted_token(r, true));
  CHECK_EQ(k.wrong + k.failed, 0);
  const ss_dg_stats x = r.stats();
  CHECK_EQ(x.prior_issued, 0ull);  // no prior was loaded
  CHECK(x.l1_issued > 0);
}

// Compile here, execute only in a booked GPU window. ~3 MiB expert fixture, comfortably below 300 MiB total.
// Keep captured arena/table pointers live across reset; verify canonical placement, all unit bytes, diagnostics,
// statistics rebasing, unconsumed prior Coming entries, minimum spares and both write-back modes.
TEST(dg_cuda_reset_open) {
  for (bool defer : {false, true}) {
    const std::string tag = defer ? "reset-open-defer" : "reset-open";
    const auto prior = winpf_prior(DgModel(tag));
    Rig r(tag, 16, 1, 4, 10000, true, [&](ss_options& o) {
      o.dg_wb_defer = defer ? 1u : 0u;
      o.dg_prior = prior.data();
      o.dg_prior_len = prior.size();
    });
    if (!r.c) continue;
    const size_t keys = size_t(r.m.G) * r.m.U;
    std::vector<int32_t> open_table(keys), got(keys);
    CU(cudaMemcpy(open_table.data(), r.v.slot_of, keys * 4, cudaMemcpyDeviceToHost));
    CHECK_OK(ss_dg_reset_open(r.c));  // legal before token 1; no service handoff or sequence rewrite
    r.capture(true);
    CHECK_OK(ss_dg_set_budgets(r.c, 0, 3, 12));
    CHECK_OK(ss_test_dg_pressure(r.c, 1, 3));
    for (int i = 0; i < 12; ++i) CHECK_EQ(r.token(true, true), 0);
    winpf_on(r, 1500);
    const int32_t symbol = 5;
    CHECK_OK(ss_dg_window_hint(r.c, &symbol, nullptr, 1));  // left unconsumed, potentially Coming
    const auto lifetime = r.stats();
    CHECK_OK(ss_dg_reset_open(r.c));
    CHECK_OK(ss_test_dg_check(r.c));
    CU(cudaMemcpy(got.data(), r.v.slot_of, keys * 4, cudaMemcpyDeviceToHost));
    CHECK(got == open_table);
    ss_dg_device_view after{};
    CHECK_OK(ss_dg_get_device_view(r.c, &after));
    CHECK(after.slot_of == r.v.slot_of);
    for (uint32_t p = 0; p < after.n_parts; ++p) CHECK(after.parts[p].base == r.v.parts[p].base);
    for (int g = 0; g < r.m.G; ++g)
      for (int u = 0; u < r.m.U; ++u) CHECK_OK(ss_check_unit(r.c, g, u));
    ss_dg_layer_timeline tl{};
    ss_dg_boundary_timeline bt{};
    CHECK_EQ(int(ss_dg_timeline_sized(r.c, 0, &tl, sizeof tl)), int(SS_E_INVALID));
    CHECK_EQ(int(ss_dg_boundary_timeline_sized(r.c, &bt, sizeof bt)), int(SS_E_INVALID));
    auto st = r.stats();
    CHECK_EQ(st.tokens, 0ull);
    CHECK_EQ(st.fetched, 0ull);
    CHECK_EQ(st.wb_completed_bytes, 0ull);
    CHECK_EQ(st.wb_outstanding_bytes, 0ull);
    CHECK(lifetime.tokens > 0);
    CHECK_OK(ss_dg_reset_open(r.c));
    // Refusal inside a token must not poison: finish that token through the previously captured graph.
    r.fill_ids();
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_EQ(int(ss_dg_reset_open(r.c)), int(SS_E_INVALID));
    CU(cudaGraphLaunch(r.graph[1], r.s));
    CHECK_OK(ss_end(r.c));
    CU(cudaStreamSynchronize(r.s));
    CHECK_EQ(r.token(true, true), 0);
    st = r.stats();
    CHECK_EQ(st.tokens, 2ull);
    CHECK(st.fetched < 10000 && st.wb_completed_bytes < (1ull << 30));  // no unsigned baseline underflow
    CHECK_OK(ss_dg_timeline_sized(r.c, 0, &tl, sizeof tl));
    CHECK_OK(ss_dg_boundary_timeline_sized(r.c, &bt, sizeof bt));
  }
}

// ---- scored eviction (docs/design.md §3.8): only the boundary's victims change; every read stays exact ----
namespace {

// eager, graph, prediction, 8 -> 0 -> 8 switching; pressure (banks, resurrections, MRU evictions); global spares with
// fewer spare blocks than slots; deferral off or on
void scored_cases(const std::string& t, bool defer, const std::function<void(ss_options&)>& extra = {}) {
  auto tweak = [&](ss_options& o) {
    o.policy = SS_POLICY_SCORED;
    o.dg_wb_defer = defer ? 1 : 0;
    if (extra) extra(o);
  };
  auto sane = [](const ss_dg_stats& x, uint32_t G) {
    CHECK_EQ(x.score_fallbacks, 0ull);
    CHECK_EQ(x.score_layers, x.tokens * G);
    CHECK(x.score_candidates > 0);
    CHECK(x.scored_victims <= x.evicted);
  };
  {
    Rig r(t + "-main", 16, 8, 8, 10000, true, tweak);
    if (!r.c) return;
    Tally k;
    for (int i = 0; i < 20; ++i) k.add(r.token(false, false));
    r.capture(false);
    r.capture(true);
    for (int i = 0; i < 100; ++i) k.add(r.token(true, false));
    for (int i = 0; i < 100; ++i) k.add(r.token(true, true));
    for (int i = 0; i < 60; ++i) k.add(r.token(true, (i / 10) % 2 == 0));
    CHECK_EQ(k.failed, 0);
    CHECK_EQ(k.wrong, 0);
    CHECK_OK(ss_test_dg_check(r.c));
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 9) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print((t + " main").c_str(), x);
    sane(x, uint32_t(r.m.G));
    CHECK(x.evicted > 0 && x.scored_victims > 0);
    if (defer) CHECK_EQ(x.wb_deferred, x.evicted);
    char* js = nullptr;
    size_t len = 0;
    CHECK_OK(ss_stats_to_json(r.c, &js, &len));
    CHECK(js && std::string(js, len).find("\"policy\":\"scored\"") != std::string::npos);
    ss_free(js);
    for (int i = 0; i < 10; ++i) k.add(r.token(true, true));  // after ss_check_unit's drain: still exact
    CHECK_EQ(k.wrong + k.failed, 0);
  }
  {
    Rig r(t + "-pressure", 8, 2, 8, 10000, true, tweak);
    if (!r.c) return;
    CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
    r.capture(true);
    Tally k;
    for (int i = 0; i < 200; ++i) {
      k.add(r.token(true, true));
      CHECK_OK(ss_test_dg_check(r.c));
    }
    CHECK_EQ(k.failed + k.wrong, 0);
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 5) CHECK_OK(ss_check_unit(r.c, L, u));
    const ss_dg_stats x = r.stats();
    print((t + " pressure").c_str(), x);
    sane(x, uint32_t(r.m.G));
    CHECK(x.bank_serves > 0);
    CHECK(x.resurrected > 0);
  }
  {
    Rig r(t + "-gspares", 8, 2, 8, 10000, true, [&](ss_options& o) {
      tweak(o);
      o.dg_spare_slots = 20;
    });
    if (!r.c) return;
    CHECK_OK(ss_test_dg_pressure(r.c, 2, 6));
    r.capture(true);
    Tally k;
    for (int i = 0; i < 150; ++i) {
      k.add(r.token(true, true));
      CHECK_OK(ss_test_dg_check(r.c));
    }
    CHECK_EQ(k.wrong + k.failed, 0);
    const ss_dg_stats x = r.stats();
    print((t + " global spares").c_str(), x);
    sane(x, uint32_t(r.m.G));
    CHECK(x.evict_short > 0 || x.evicted > 0);
  }
}

// A profile of the fixture model: identity from its layers and unit hashes, counts favouring "u -> u + 1".
std::string scored_profile(const DgModel& m, const char* prov, uint64_t salt = 0) {
  std::vector<uint32_t> units(size_t(m.G), uint32_t(m.U));
  std::vector<std::vector<uint64_t>> h(size_t(m.G));
  std::vector<ts::DgScoreIdLayer> idl;
  for (int L = 0; L < m.G; ++L) {
    h[size_t(L)] = m.hashes(L);
    if (salt) h[size_t(L)][0] ^= salt;
    idl.push_back({uint32_t(L), uint32_t(m.U), m.unit(), &h[size_t(L)]});
  }
  std::vector<std::vector<uint32_t>> cells(size_t(m.G), std::vector<uint32_t>(size_t(m.U) * size_t(m.U), 1));
  for (auto& c : cells)
    for (int u = 0; u < m.U; ++u) c[size_t(u) * size_t(m.U) + size_t((u + 1) % m.U)] = 1000;
  return ts::dg_profile_write(units, ts::dg_score_identity(idl), prov, cells);
}

}  // namespace

TEST(dg_cuda_scored) {
  scored_cases("sc", false);
  scored_cases("scd", true);
}

// A profile-seeded table: loaded at open (provenance in the stats JSON), exact reads in every mode, restored by
// ss_dg_reset_open; a profile of another identity refuses the open.
TEST(dg_cuda_scored_profile) {
  const DgModel mm("scp");
  const std::string prof = scored_profile(mm, "gpu test profile");
  {
    Rig r("scp", 16, 8, 8, 10000, true, [&](ss_options& o) {
      o.policy = SS_POLICY_SCORED;
      o.dg_score_profile = prof.data();
      o.dg_score_profile_len = prof.size();
    }, nullptr, true);
    if (!r.c) return;
    Tally k;
    for (int i = 0; i < 20; ++i) k.add(r.token(false, false));
    r.capture(false);
    r.capture(true);
    for (int i = 0; i < 80; ++i) k.add(r.token(true, i % 2 == 0));
    CHECK_EQ(k.wrong + k.failed, 0);
    CHECK_OK(ss_test_dg_check(r.c));
    char* js = nullptr;
    size_t len = 0;
    CHECK_OK(ss_stats_to_json(r.c, &js, &len));
    CHECK(js && std::string(js, len).find("gpu test profile") != std::string::npos);
    ss_free(js);
    const ss_dg_stats x = r.stats();
    print("scored profile", x);
    CHECK_EQ(x.score_fallbacks, 0ull);
    CHECK(x.scored_victims > 0);
    CHECK_OK(ss_dg_reset_open(r.c));
    for (int i = 0; i < 40; ++i) k.add(r.token(true, true));
    CHECK_EQ(k.wrong + k.failed, 0);
    for (int L = 0; L < r.m.G; ++L)
      for (int u = 0; u < r.m.U; u += 7) CHECK_OK(ss_check_unit(r.c, L, u));
  }
  {
    const std::string other = scored_profile(mm, "", 1);
    ss_status st = SS_OK;
    Rig r("scp", 16, 8, 8, 10000, true, [&](ss_options& o) {
      o.policy = SS_POLICY_SCORED;
      o.dg_score_profile = other.data();
      o.dg_score_profile_len = other.size();
    }, &st, true);
    CHECK_EQ(int(st), int(SS_E_INVALID));
    CHECK(std::string(ss_last_error()).find("identity mismatch") != std::string::npos);
  }
}
