// CUDA backend test (one short PC-BIG on the RTX PRO): a kernel reads every byte of every routed unit through the
// device unit table on the engine stream, right after ss_acquire / ss_group_wait. A missing copy-stream wait would
// show as a wrong checksum. Then a pinned-tier and an unbuffered-NVMe bandwidth probe (256 MiB each).
//   ss-test-cuda [--bw-dir D:\tmp\ss-test] [--no-bw]
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

// One block per listed unit: 64-bit sum of (byte x (index % 251 + 1)), order-free so the reduction is exact.
__global__ void checksum(const uint64_t* table, const int32_t* units, int n, uint64_t bytes,
                         unsigned long long* out) {
  int k = blockIdx.x;
  if (k >= n) return;
  const unsigned char* p = reinterpret_cast<const unsigned char*>(table[units[k]]);
  unsigned long long s = 0;
  if (p)
    for (uint64_t i = threadIdx.x; i < bytes; i += blockDim.x) s += (unsigned long long)p[i] * (i % 251 + 1);
  else
    s = 0xFFFFFFFFFFFFFFFFull / blockDim.x;  // a null entry can never match
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

struct Gpu {
  cudaStream_t s = nullptr;
  int32_t* units = nullptr;
  unsigned long long* sums = nullptr;
  Gpu() {
    CU(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    CU(cudaMalloc(&units, 64 * sizeof(int32_t)));
    CU(cudaMalloc(&sums, 64 * sizeof(unsigned long long)));
  }
  ~Gpu() {
    cudaStreamSynchronize(s);
    cudaFree(units);
    cudaFree(sums);
    cudaStreamDestroy(s);
  }
  // Checks units[0..n) of layer L through the device table (async on s; compared after a sync).
  void check(const ss_layer_view& v, const Fixture& fx, int L, const std::vector<int32_t>& list) {
    int n = int(list.size());
    if (!n) return;
    CU(cudaMemcpyAsync(units, list.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice, s));
    CU(cudaMemsetAsync(sums, 0, n * sizeof(unsigned long long), s));
    uint64_t bytes = fx.expected[size_t(L)][0].size();
    checksum<<<n, 256, 0, s>>>(v.table_dev, units, n, bytes, sums);
    std::vector<unsigned long long> got(static_cast<size_t>(n));
    CU(cudaMemcpyAsync(got.data(), sums, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
    CU(cudaStreamSynchronize(s));
    for (int k = 0; k < n; ++k) {
      bool ok = got[size_t(k)] == host_sum(fx.expected[size_t(L)][size_t(list[size_t(k)])]);
      if (!ok) std::printf("  layer %d unit %d: device checksum differs\n", L, list[size_t(k)]);
      CHECK(ok);
    }
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

}  // namespace

TEST(cuda_decode_through_device_table) {
  Fixture fx("cuda-decode");
  Gpu gpu;
  for (int pol = 0; pol < 2; ++pol) {
    Run r;
    CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
    ss_budget_defaults(&r.b);
    r.b.device_bytes = uint64_t(1) << 30;
    r.b.margin_fraction = 0;
    r.b.margin_gib = 1.0 / 1024;
    r.b.ring_slots = 6;
    r.b.host_pinned_bytes = 1 << 20;
    r.b.dynamic_slots_per_layer = 2;
    plan(r, {"", "RRRRPPPPPPDDDDDD", "RRRRPPDD", "D", "RRRRDDDDDDDDDDDD"});
    ss_options_defaults(&r.o);
    r.o.backend = SS_BACKEND_CUDA;
    r.o.policy = pol ? SS_POLICY_SCORED : SS_POLICY_LRU;
    r.o.pin_planned = 0;
    r.o.verify_hash = 1;
    r.o.staging_bytes = 64 * 1024;
    r.o.group_units = 2;
    CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
    if (!r.c) return;
    std::mt19937 g(77 + pol);
    for (int step = 0; step < 30; ++step) {
      CHECK_OK(ss_begin(r.c, nullptr, 0));
      for (int L : {1, 2, 4}) {
        int E = L == 2 ? Fixture::kE2 : Fixture::kE1, k = L == 2 ? 2 : 4;
        auto ids = route(g, E, k, 1 + step % 3);
        if (L == 1) {
          int32_t pred[2] = {int32_t(step % 8), int32_t((step * 5) % 8)};
          CHECK_OK(ss_prefetch(r.c, 2, pred, 2, SS_PREFETCH_SPECULATIVE));
        }
        ss_layer_view v{};
        if (step % 3 == 0) {
          // Device routes: the ids are copied back on the engine stream.
          int32_t* ids_dev = nullptr;
          CU(cudaMalloc(&ids_dev, ids.size() * sizeof(int32_t)));
          CU(cudaMemcpyAsync(ids_dev, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice, gpu.s));
          CHECK_OK(ss_layer_routes_dev(r.c, L, ids_dev, int(ids.size()), gpu.s));
          CHECK_OK(ss_acquire(r.c, L, gpu.s, &v));
          CU(cudaFreeAsync(ids_dev, gpu.s));
        } else {
          CHECK_OK(ss_layer_routes(r.c, L, ids.data(), int(ids.size())));
        }
        if (step % 3 == 1) {
          CHECK_OK(ss_acquire_groups(r.c, L, gpu.s, &v));
          for (int gi = 0; gi < v.n_groups; ++gi) {
            CHECK_OK(ss_group_wait(r.c, L, gi, gpu.s));
            gpu.check(v, fx, L, std::vector<int32_t>(v.groups[gi].units, v.groups[gi].units + v.groups[gi].n_units));
          }
        } else {
          if (step % 3 == 2) CHECK_OK(ss_acquire(r.c, L, gpu.s, &v));
          std::set<int32_t> distinct(ids.begin(), ids.end());
          gpu.check(v, fx, L, std::vector<int32_t>(distinct.begin(), distinct.end()));
        }
        if (step == 5) CHECK_OK(ss_check_unit(r.c, L, ids[0]));
        CHECK_OK(ss_release(r.c, L, gpu.s));
      }
      CHECK_OK(ss_layer_all(r.c, 3));
      ss_layer_view v{};
      CHECK_OK(ss_acquire(r.c, 3, gpu.s, &v));
      gpu.check(v, fx, 3, {0});
      CHECK_OK(ss_release(r.c, 3, gpu.s));
      CHECK_OK(ss_end(r.c));
    }
    ss_layer_stats t;
    CHECK_OK(ss_stats_total(r.c, &t));
    CHECK(t.bytes_pinned > 0 && t.bytes_disk > 0 && t.hits > 0);
    std::printf("  policy %s: routed %llu hits %llu admitted %llu transient %llu spec %llu/%llu\n",
                pol ? "scored" : "lru", (unsigned long long)t.routed, (unsigned long long)t.hits,
                (unsigned long long)t.admitted, (unsigned long long)t.transient, (unsigned long long)t.spec_used,
                (unsigned long long)t.spec_issued);
  }
}

TEST(cuda_prefill_layer_major) {
  Fixture fx("cuda-prefill");
  Gpu gpu;
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  ss_budget_defaults(&r.b);
  r.b.device_bytes = uint64_t(1) << 30;
  r.b.margin_fraction = 0;
  r.b.margin_gib = 1.0 / 1024;
  r.b.ring_slots = 12;
  r.b.host_pinned_bytes = 1 << 20;
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
  for (int pass = 0; pass < 3; ++pass) {
    CHECK_OK(ss_begin(r.c, rows, 5));
    CHECK_OK(ss_prefetch_layer(r.c, 1));
    CHECK_OK(ss_prefetch_layer(r.c, 3));
    CHECK_OK(ss_prefetch_layer(r.c, 4));
    ss_layer_view v{};
    CHECK_OK(ss_layer_all(r.c, 1));
    CHECK_OK(ss_acquire(r.c, 1, gpu.s, &v));
    gpu.check(v, fx, 1, all);
    CHECK_OK(ss_release(r.c, 1, gpu.s));
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
}

namespace {
std::string g_bw_dir;
bool g_bw = true;
}  // namespace

TEST(cuda_bandwidth_pinned_and_nvme) {
  if (!g_bw) return;
  // Two layers of 32 x 8 MiB records in one file: layer 0 from the pinned tier, layer 1 read unbuffered from the
  // file (never cached by the OS), each landing in a 32-slot ring. 256 MiB each way.
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
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
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
      std::printf("  %s -> device: %.0f MiB in %.3f s = %.2f GB/s\n", L ? "file (unbuffered)" : "pinned tier",
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
