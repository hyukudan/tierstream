// Device-gated decode mode on a GPU (design rev 3.2 §8.3; one short PC-BIG). A synthetic MoE step per token: for
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
#include <random>
#include <string>
#include <vector>

#include "check.hpp"
#include "dg_fixture.hpp"
#include "strix_stream/ss.h"
#include "strix_stream/ss_test.h"


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

  // tweak: changes the options before ss_open; open_st: receives ss_open's status instead of requiring SS_OK.
  Rig(const std::string& tag, int resident, uint32_t dyn, uint32_t max_ids, uint32_t timeout_ms = 10000,
      bool hooks = true, const std::function<void(ss_options&)>& tweak = {}, ss_status* open_st = nullptr)
      : m(tag), n(max_ids) {
    // load the test's own kernels now: a first launch inside a gated step (lazy module loading) would wait for the
    // step's spinning wait kernel, which waits for the service: a stall until the wait deadline (dgts-t3)
    cudaFuncAttributes fa;
    for (const void* f : {(const void*)checksum, (const void*)spin_ns, (const void*)unpublish_first})
      CU(cudaFuncGetAttributes(&fa, f));
    CHECK_OK(ss_desc_from_json(m.json.data(), m.json.size(), &d));
    sstest::dg_budget(b, m, dyn, max_ids);
    std::vector<uint8_t> t = m.tiers(resident);
    CHECK_OK(ss_plan_from_tiers(d, &b, t.data(), t.size(), &p));
    sstest::dg_options(o, max_ids, SS_BACKEND_CUDA);
    o.dg_wait_timeout_ms = timeout_ms;
    o.dg_diag = 1;
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
    const ss_status e = ss_end(c);
    if (st) *st = e;
    CU(cudaStreamSynchronize(s));
    if (e != SS_OK) return -1;
    int wrong = 0;
    for (int g = 0; g < m.G; ++g)
      for (uint32_t i = 0; i < n; ++i) {
        const int32_t id = ids_host[size_t(g) * n + i];
        if (sums_host[size_t(g) * n + i] != want[size_t(g)][size_t(id)]) ++wrong;
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

// The stats contract (sol-6 R2a-R2c): the hook once per successful token, the snapshot published at each one, a reset
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

// Spares as the engine caches have them (Sol rtx-vram-plan 2.7): a global count of 20 spare slots (not a multiple of the
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
TEST(dg_cuda_dma_and_counters) {
  Rig r("dma", 16, 8, 8);
  if (!r.c) return;
  r.capture(true);
  Tally k;
  int measured = 0, unmeasured = 0, bad = 0;
  for (int t = 0; t < 60; ++t) {
    k.add(r.token(true, true));
    for (int g = 0; g < r.m.G; ++g) {
      ss_dg_layer_timeline tl{};
      CHECK_OK(ss_dg_timeline_sized(r.c, g, &tl, uint32_t(sizeof tl)));
      if (tl.misses > 0) {
        ++measured;
        if (!(tl.dma_ms >= 0)) ++bad;
      } else {
        ++unmeasured;
        if (tl.dma_ms != -1) ++bad;
      }
    }
  }
  CHECK_EQ(k.wrong + k.failed, 0);
  CHECK_EQ(bad, 0);
  CHECK(measured > 0);
  const ss_dg_stats x = r.stats();
  std::printf("  dma: %d layers measured, %d without a miss; predicted %llu (true %llu), miss_layers %llu, covered "
              "%llu (waits %llu), device_timeouts %llu\n",
              measured, unmeasured, (unsigned long long)x.predicted, (unsigned long long)x.predicted_true,
              (unsigned long long)x.miss_layers, (unsigned long long)x.covered, (unsigned long long)x.covered_waits,
              (unsigned long long)x.device_timeouts);
  CHECK(x.predicted > 0 && x.miss_layers > 0);
  CHECK(x.predicted_true <= x.predicted);
  CHECK_EQ(x.device_timeouts, 0ull);
  // without dg_diag there is no timeline at all (no DMA events are recorded): the getter refuses, the token runs
  Rig q("dma-off", 16, 8, 8, 10000, true, [](ss_options& o) { o.dg_diag = 0; });
  if (!q.c) return;
  Tally kq;
  for (int t = 0; t < 4; ++t) kq.add(q.token(false, false));
  CHECK_EQ(kq.wrong + kq.failed, 0);
  ss_dg_layer_timeline tl{};
  CHECK_EQ(int(ss_dg_timeline_sized(q.c, 0, &tl, uint32_t(sizeof tl))), int(SS_E_INVALID));
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
