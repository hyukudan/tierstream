// CUDA backend test (one short PC-BIG on the RTX PRO).
// Exactness and hazards: a kernel reads every byte of every routed unit through the device unit table on the engine
// stream, AFTER a spin kernel that keeps the stream busy, and the host releases the layer and moves on without any
// synchronization: slots are reused while earlier readers are still queued, and forwards run back to back. All
// checksums are compared once at the end. A missing wait (copy before release, read before landing, table rewritten
// while in use, a hit from the previous forward's copy in flight) shows as a wrong checksum.
// Then a corrupt-hash fixture (the read failure path) and a pinned-tier / unbuffered-file bandwidth probe.
//   ss-test-cuda [--bw-dir DIR] [--no-bw]
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <random>
#include <set>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

using sstest::Fixture;

namespace {

#define CU(x)                                                                                      \
  do {                                                                                             \
    cudaError_t e_ = (x);                                                                          \
    if (e_ != cudaSuccess) {                                                                       \
      std::printf("  %s:%d: %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      ++sstest::failures();                                                                        \
    }                                                                                              \
  } while (0)

__global__ void spin(long long cycles) {
  long long t0 = clock64();
  while (clock64() - t0 < cycles) {
  }
}

// One block per listed unit: 64-bit sum of (byte x (index % 251 + 1)), order-free so the reduction is exact.
__global__ void checksum(const uint64_t* table, const int32_t* units, int n, uint64_t bytes,
                         unsigned long long* out) {
  int k = blockIdx.x;
  if (k >= n) return;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(table[units[k]]);
  unsigned long long s = 0;
  if (p)
    for (uint64_t i = threadIdx.x; i < bytes; i += blockDim.x) s += (unsigned long long)p[i] * (i % 251 + 1);
  else if (threadIdx.x == 0)
    s = 0xDEADDEADDEADDEADull;  // a null entry can never match
  atomicAdd(&out[k], s);
}

uint64_t host_sum(const std::vector<uint8_t>& v) {
  uint64_t s = 0;
  for (size_t i = 0; i < v.size(); ++i) s += uint64_t(v[i]) * (i % 251 + 1);
  return s;
}

struct Run {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  ~Run() {
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
};

void plan(Run& r, std::initializer_list<const char*> tiers) {
  std::string js = "{\"tiers\":[";
  bool first = true;
  for (const char* t : tiers) {
    js += first ? "\"" : ",\"";
    js += t;
    js += "\"";
    first = false;
  }
  js += "]}";
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
}

// Asynchronous checker: every check appends unit ids and expected sums; the device writes into its own result
// slots; nothing synchronizes until verify().
struct Gpu {
  static constexpr int kMax = 1 << 15;
  cudaStream_t s = nullptr;
  int32_t* units = nullptr;               // device, kMax
  unsigned long long* sums = nullptr;     // device, kMax
  int32_t* units_host = nullptr;          // pinned staging for the unit lists (never reused before verify)
  std::vector<uint64_t> want;
  std::vector<std::pair<int, int>> who;   // (layer, unit) per slot
  int used = 0;
  long long spin_cycles = 2000000;        // ~1 ms at 2 GHz
  Gpu() {
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CU(cudaMalloc(&units, kMax * sizeof(int32_t)));
    CU(cudaMalloc(&sums, kMax * sizeof(unsigned long long)));
    CU(cudaMemset(sums, 0, kMax * sizeof(unsigned long long)));
    CU(cudaHostAlloc(&units_host, kMax * sizeof(int32_t), cudaHostAllocDefault));
  }
  ~Gpu() {
    cudaStreamSynchronize(s);
    cudaFree(units);
    cudaFree(sums);
    cudaFreeHost(units_host);
    cudaStreamDestroy(s);
  }
  void check(const ss_layer_view& v, const Fixture& fx, int L, const std::vector<int32_t>& list, bool delay = true) {
    int n = int(list.size());
    if (!n || used + n > kMax) {
      if (n) std::printf("  checker full\n");
      return;
    }
    std::memcpy(units_host + used, list.data(), n * sizeof(int32_t));
    CU(cudaMemcpyAsync(units + used, units_host + used, n * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    if (delay) spin<<<1, 1, 0, s>>>(spin_cycles);
    checksum<<<n, 256, 0, s>>>(v.table_dev, units + used, n, fx.expected[size_t(L)][0].size(), sums + used);
    for (int32_t u : list) {
      want.push_back(host_sum(fx.expected[size_t(L)][size_t(u)]));
      who.push_back({L, u});
    }
    used += n;
  }
  void verify() {
    CU(cudaStreamSynchronize(s));
    std::vector<unsigned long long> got(static_cast<size_t>(used));
    CU(cudaMemcpy(got.data(), sums, size_t(used) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int k = 0; k < used; ++k)
      if (got[size_t(k)] != want[size_t(k)]) {
        if (bad++ < 10) std::printf("  check %d: layer %d unit %d: device checksum differs\n", k, who[size_t(k)].first,
                                    who[size_t(k)].second);
      }
    std::printf("  %d device checksums, %d wrong\n", used, bad);
    CHECK_EQ(bad, 0);
    CU(cudaMemset(sums, 0, kMax * sizeof(unsigned long long)));
    want.clear();
    who.clear();
    used = 0;
  }
};

std::vector<int32_t> route(std::mt19937& g, int E, int k, int rows) {
  std::vector<int32_t> ids;
  std::uniform_int_distribution<int> hot(0, 3), any(0, E - 1), coin(0, 2);
  for (int r = 0; r < rows; ++r) {
    std::set<int> row;
    while (int(row.size()) < k) row.insert(coin(g) ? hot(g) : any(g));
    ids.insert(ids.end(), row.begin(), row.end());
  }
  return ids;
}

void decode_variant(const char* name, ss_policy policy, int spec_persistent, bool backing) {
  Fixture fx(std::string("cuda-") + name);
  Gpu gpu;
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  ss_budget_defaults(&r.b);
  r.b.device_bytes = uint64_t(1) << 30;
  r.b.margin_fraction = 0;
  r.b.margin_gib = 1.0 / 1024;
  r.b.ring_slots = 6;
  r.b.host_pinned_bytes = 2 << 20;
  r.b.dynamic_slots_per_layer = 2;
  if (backing) plan(r, {"", "BBBBPPPPPPDDDDDD", "BBBBPPDD", "D", "BBBBDDDDDDDDDDDD"});
  else plan(r, {"", "RRRRPPPPPPDDDDDD", "RRRRPPDD", "D", "RRRRDDDDDDDDDDDD"});
  ss_options_defaults(&r.o);
  r.o.backend = SS_BACKEND_CUDA;
  r.o.policy = policy;
  r.o.pin_planned = 0;
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  r.o.group_units = 2;
  r.o.spec_persistent = spec_persistent;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  std::mt19937 g(77 + int(policy) * 3 + spec_persistent);
  std::vector<int32_t*> ids_dev_list;
  for (int step = 0; step < 40; ++step) {
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    for (int L : {1, 2, 4}) {
      int E = L == 2 ? Fixture::kE2 : Fixture::kE1, k = L == 2 ? 2 : 4;
      auto ids = route(g, E, k, 1 + step % 3);
      if (L == 1) {
        int32_t pred[2] = {int32_t(step % 8), int32_t((step * 5) % 8)};
        CHECK_OK(ss_prefetch(r.c, 2, pred, 2, SS_PREFETCH_SPECULATIVE));
      }
      ss_layer_view v{};
      std::set<int32_t> distinct(ids.begin(), ids.end());
      std::vector<int32_t> list(distinct.begin(), distinct.end());
      if (step % 3 == 0) {
        // Device routes, copied back on the engine stream (freed after the final sync).
        int32_t* ids_dev = nullptr;
        CU(cudaMalloc(&ids_dev, ids.size() * sizeof(int32_t)));
        ids_dev_list.push_back(ids_dev);
        CU(cudaMemcpyAsync(ids_dev, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, gpu.s));
        CU(cudaStreamSynchronize(gpu.s));  // ids is a pageable temporary
        CHECK_OK(ss_layer_routes_dev(r.c, L, ids_dev, int(ids.size()), gpu.s));
        CHECK_OK(ss_acquire(r.c, L, gpu.s, &v));
        gpu.check(v, fx, L, list);
      } else if (step % 3 == 1) {
        CHECK_OK(ss_layer_routes(r.c, L, ids.data(), int(ids.size())));
        CHECK_OK(ss_acquire_groups(r.c, L, gpu.s, &v));
        for (int gi = 0; gi < v.n_groups; ++gi) {
          CHECK_OK(ss_group_wait(r.c, L, gi, gpu.s));
          gpu.check(v, fx, L, std::vector<int32_t>(v.groups[gi].units, v.groups[gi].units + v.groups[gi].n_units));
        }
      } else {
        CHECK_OK(ss_layer_routes(r.c, L, ids.data(), int(ids.size())));
        CHECK_OK(ss_acquire(r.c, L, gpu.s, &v));
        gpu.check(v, fx, L, list);
      }
      if (step == 5) CHECK_OK(ss_check_unit(r.c, L, ids[0]));
      CHECK_OK(ss_release(r.c, L, gpu.s));  // no sync: the spin + checksum are still queued
    }
    CHECK_OK(ss_layer_all(r.c, 3));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 3, gpu.s, &v));
    gpu.check(v, fx, 3, {0});
    CHECK_OK(ss_release(r.c, 3, gpu.s));
    CHECK_OK(ss_end(r.c));
    if (step % 10 == 0) CHECK_EQ(int(ss_acquire(r.c, 1, (void*)gpu.s, &v)), int(SS_E_INVALID));  // outside a forward
  }
  gpu.verify();
  for (int32_t* p : ids_dev_list) cudaFree(p);
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(r.c, &t));
  CHECK(t.hits > 0 && t.bytes_disk > 0 && t.bytes_pinned > 0);
  CHECK_EQ(t.hits + t.admitted + t.transient + t.prefetched + t.spec_used, t.routed);
  std::printf("  %s: routed %llu hits %llu admitted %llu transient %llu evicted %llu spec %llu used %llu dropped %llu "
              "pinned %.1f MB disk %.1f MB\n",
              name, (unsigned long long)t.routed, (unsigned long long)t.hits, (unsigned long long)t.admitted,
              (unsigned long long)t.transient, (unsigned long long)t.evicted, (unsigned long long)t.spec_issued,
              (unsigned long long)t.spec_used, (unsigned long long)t.spec_dropped, double(t.bytes_pinned) / 1e6,
              double(t.bytes_disk) / 1e6);
}

}  // namespace

TEST(cuda_decode_lru_async) { decode_variant("lru", SS_POLICY_LRU, 0, false); }
TEST(cuda_decode_scored_async) { decode_variant("scored", SS_POLICY_SCORED, 0, false); }
TEST(cuda_decode_lru_spec_persistent_backed) { decode_variant("lru-specp-backed", SS_POLICY_LRU, 1, true); }

TEST(cuda_prefill_layer_major_async) {
  Fixture fx("cuda-prefill");
  Gpu gpu;
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  ss_budget_defaults(&r.b);
  r.b.device_bytes = uint64_t(1) << 30;
  r.b.margin_fraction = 0;
  r.b.margin_gib = 1.0 / 1024;
  r.b.ring_slots = 12;
  r.b.host_pinned_bytes = 2 << 20;
  plan(r, {"", "RRRRRRRRPPPPDDDD", "RRRRRRRR", "P", "RRRRRRRRDDDDPPPP"});
  ss_options_defaults(&r.o);
  r.o.backend = SS_BACKEND_CUDA;
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  std::vector<int32_t> all(16);
  for (int u = 0; u < 16; ++u) all[size_t(u)] = u;
  uint32_t rows[5] = {512, 512, 0, 512, 512};
  for (int pass = 0; pass < 6; ++pass) {
    CHECK_OK(ss_begin(r.c, rows, 5));
    CHECK_OK(ss_prefetch_layer(r.c, 1));
    CHECK_OK(ss_prefetch_layer(r.c, 3));
    CHECK_OK(ss_prefetch_layer(r.c, 4));  // half waits for ring slots held by layer 1
    ss_layer_view v{};
    CHECK_OK(ss_layer_all(r.c, 1));
    CHECK_OK(ss_acquire(r.c, 1, gpu.s, &v));
    gpu.check(v, fx, 1, all);
    CHECK_OK(ss_release(r.c, 1, gpu.s));  // layer 4's copies into these ring slots wait for this point
    CHECK_OK(ss_layer_all(r.c, 3));
    CHECK_OK(ss_acquire(r.c, 3, gpu.s, &v));
    gpu.check(v, fx, 3, {0});
    CHECK_OK(ss_release(r.c, 3, gpu.s));
    CHECK_OK(ss_layer_all(r.c, 4));
    CHECK_OK(ss_acquire(r.c, 4, gpu.s, &v));
    gpu.check(v, fx, 4, all);
    CHECK_OK(ss_release(r.c, 4, gpu.s));
    CHECK_OK(ss_end(r.c));
  }
  gpu.verify();
}

TEST(cuda_read_failure_reported_and_recovered) {
  Fixture fx("cuda-hash", /*corrupt_hash=*/true);
  Gpu gpu;
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  ss_budget_defaults(&r.b);
  r.b.device_bytes = uint64_t(1) << 30;
  r.b.margin_fraction = 0;
  r.b.margin_gib = 1.0 / 1024;
  r.b.ring_slots = 8;
  r.b.dynamic_slots_per_layer = 4;
  plan(r, {"", "DDDDDDDDDDDDDDDD", "DDDDDDDD", "D", "DDDDDDDDDDDDDDDD"});
  ss_options_defaults(&r.o);
  r.o.backend = SS_BACKEND_CUDA;
  r.o.verify_hash = 1;
  r.o.pin_planned = 0;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  // A seed whose read fails is rolled back.
  int32_t bad_seed[2] = {2, 3};
  CHECK_EQ(int(ss_seed(r.c, 1, bad_seed, 2)), int(SS_E_HASH));
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t ids[2] = {2, 3};
  CHECK_OK(ss_layer_routes(r.c, 1, ids, 2));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, gpu.s, &v)), int(SS_E_HASH));  // unit 3 is read again and fails again
  CHECK(ss_end(r.c) != SS_OK);
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_layer_routes(r.c, 1, ids, 1));
  CHECK_OK(ss_acquire(r.c, 1, gpu.s, &v));
  gpu.check(v, fx, 1, {2});
  CHECK_OK(ss_release(r.c, 1, gpu.s));
  CHECK_OK(ss_end(r.c));
  gpu.verify();
  ss_layer_stats s;
  CHECK_OK(ss_stats_layer(r.c, 1, &s));
  CHECK_EQ(s.hits, uint64_t(2));  // unit 2 (seeded fine) in both forwards; unit 3 never became a hit
}

namespace {
std::string g_bw_dir;
bool g_bw = true;
}  // namespace

TEST(cuda_bandwidth_pinned_and_nvme) {
  if (!g_bw) return;
  // Two layers of 32 x 8 MiB records in one file: layer 0 from the pinned tier, layer 1 read from the file, each
  // landing in a 32-slot ring. 256 MiB each way, 3 repetitions (a small end-to-end sample, not sustained bandwidth).
  const uint64_t unit = uint64_t(8) << 20, n = 32;
  std::filesystem::path dir = g_bw_dir.empty() ? Fixture::temp_root() / "ss-test-bw" : std::filesystem::path(g_bw_dir);
  std::filesystem::create_directories(dir);
  std::filesystem::path file = dir / "bw.bin";
  if (!std::filesystem::exists(file) || std::filesystem::file_size(file) != 2 * n * unit) {
    std::ofstream f(file, std::ios::binary);
    std::vector<char> buf(static_cast<size_t>(unit));
    for (uint64_t k = 0; k < 2 * n; ++k) {
      for (size_t i = 0; i < buf.size(); i += 4096) buf[i] = char(k + i);
      f.write(buf.data(), std::streamsize(unit));
    }
  }
  std::string path = file.generic_string();
  std::string js = R"({"schema":"strix-stream/desc-v1","engine":"bw","sources":[{"path":")" + path +
                   R"(","align":4096}],"layers":[)";
  for (int L = 0; L < 2; ++L)
    js += std::string(L ? "," : "") + R"({"index":)" + std::to_string(L) +
          R"(,"kind":"moe","n_units":32,"top_k":8,"parts":[{"format":"BF16","offset":)" +
          std::to_string(uint64_t(L) * n * unit) + R"(,"stride":8388608,"bytes":8388608}]})";
  js += "]}";
  Run r;
  CHECK_OK(ss_desc_from_json(js.data(), js.size(), &r.d));
  if (!r.d) return;
  ss_budget_defaults(&r.b);
  r.b.device_bytes = uint64_t(4) << 30;
  r.b.ring_slots = 32;
  r.b.host_pinned_bytes = n * unit;
  std::string t0(32, 'P'), t1(32, 'D');
  plan(r, {t0.c_str(), t1.c_str()});
  ss_options_defaults(&r.o);
  r.o.backend = SS_BACKEND_CUDA;
  r.o.reader_threads = 4;
  r.o.staging_buffers = 4;
  r.o.staging_bytes = uint64_t(16) << 20;  // pinned total: 256 MiB tier + 64 MiB staging
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  int direct = ss_source_direct(r.c, 0);
  std::printf("  source opened %s\n", direct == 1 ? "unbuffered (FILE_FLAG_NO_BUFFERING / O_DIRECT)" : "BUFFERED");
  Gpu gpu;
  for (int rep = 0; rep < 3; ++rep) {
    for (int L = 0; L < 2; ++L) {
      CHECK_OK(ss_begin(r.c, nullptr, 0));
      auto t = std::chrono::steady_clock::now();
      CHECK_OK(ss_layer_all(r.c, L));
      ss_layer_view v{};
      CHECK_OK(ss_acquire(r.c, L, gpu.s, &v));
      CU(cudaStreamSynchronize(gpu.s));
      double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
      CHECK_OK(ss_release(r.c, L, gpu.s));
      CHECK_OK(ss_end(r.c));
      std::printf("  %s -> device: %.0f MiB in %.3f s = %.2f GB/s\n",
                  L ? (direct == 1 ? "file (unbuffered)" : "file (buffered)") : "pinned tier",
                  double(n * unit) / double(1 << 20), s, double(n * unit) / s / 1e9);
    }
  }
}

int main(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--no-bw")) g_bw = false;
    else if (!std::strcmp(argv[i], "--bw-dir") && i + 1 < argc) g_bw_dir = argv[++i];
  }
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 2;
  }
  std::printf("device %d: %s, %.1f GiB\n", dev, p.name, double(p.totalGlobalMem) / double(1ull << 30));
  return sstest::run_all();
}
