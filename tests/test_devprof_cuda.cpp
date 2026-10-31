// Device profiles, NVIDIA side (devprof.hpp detect_cuda / detect_windows_host): a FAKE driver API + NVML (an RTX PRO
// 6000 Blackwell and an RTX 4090), the measured H2D bandwidth from the engine's per-GPU cache (ds41_gpu_profile's
// strix-gpu-<uuid>.json), a corrupt cache, an absent runtime, and the Windows host facts. The real driver query runs
// only with TS_DEVPROF_REAL_CUDA=1 (no context, nothing allocated; not part of the default run).
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "check.hpp"
#include "devprof.hpp"

namespace fs = std::filesystem;
using ts::devprof::CudaApi;
using ts::devprof::Profile;

namespace {

constexpr const char* kProUuid = "GPU-3f2a9c41-0b7e-4d12-8a6f-1c2d3e4f5a6b";
constexpr const char* kProKey = "3f2a9c410b7e4d128a6f1c2d3e4f5a6b";
constexpr const char* k4090Uuid = "GPU-aa11bb22-cc33-dd44-ee55-ff6677889900";

struct FakeGpu {
  CudaApi::Props props;
  int gen, width, max_gen, max_width;
};

std::vector<FakeGpu> two_gpus() {
  FakeGpu pro;
  pro.props.name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
  pro.props.uuid = kProUuid;
  pro.props.pci = "0000:01:00.0";
  pro.props.cc_major = 12;
  pro.props.cc_minor = 0;
  pro.props.sms = 188;
  pro.props.warp = 32;
  pro.props.smem_per_block = 49152;
  pro.props.total = 97887272960ull;
  pro.gen = 5, pro.width = 16, pro.max_gen = 5, pro.max_width = 16;
  FakeGpu g4;
  g4.props.name = "NVIDIA GeForce RTX 4090";
  g4.props.uuid = k4090Uuid;
  g4.props.pci = "0000:02:00.0";
  g4.props.cc_major = 8;
  g4.props.cc_minor = 9;
  g4.props.sms = 128;
  g4.props.warp = 32;
  g4.props.smem_per_block = 49152;
  g4.props.total = 25757220864ull;
  g4.gen = 4, g4.width = 16, g4.max_gen = 4, g4.max_width = 16;
  return {pro, g4};
}

CudaApi fake(const std::vector<FakeGpu>& gpus, bool nvml = true) {
  CudaApi a;
  a.device_count = [gpus](int* n) {
    *n = int(gpus.size());
    return 0;
  };
  a.props = [gpus](int i, CudaApi::Props* p) {
    if (i < 0 || i >= int(gpus.size())) return false;
    *p = gpus[size_t(i)].props;
    return true;
  };
  if (nvml) {
    a.pcie = [gpus](const std::string& pci, int* gen, int* width, int* mgen, int* mwidth) {
      for (const FakeGpu& g : gpus)
        if (g.props.pci == pci) {
          *gen = g.gen, *width = g.width, *mgen = g.max_gen, *mwidth = g.max_width;
          return true;
        }
      return false;
    };
    a.driver = [] { return std::string("580.88"); };
  }
  return a;
}

fs::path temp_dir(const char* tag) {
  fs::path root = fs::temp_directory_path();
  if (const char* e = std::getenv("TS_TEST_DIR")) root = e;
  const fs::path d = root / (std::string("ts-devprof-cuda-") + tag);
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d);
  return d;
}

void write(const fs::path& f, const std::string& text) {
  std::ofstream o(f, std::ios::binary);
  o << text;
}

bool has_notice(const Profile& p, const std::string& part) {
  for (const std::string& n : p.notices)
    if (n.find(part) != std::string::npos) return true;
  return false;
}

bool has_source(const Profile& p, const char* s) {
  for (const std::string& n : p.sources)
    if (n == s) return true;
  return false;
}

}  // namespace

TEST(two_gpus_from_fake_driver) {
  Profile p;
  const fs::path dir = temp_dir("empty");  // no cache files
  ts::devprof::detect_cuda(p, fake(two_gpus()), dir.string());
  CHECK_EQ(p.devices.size(), size_t(2));
  if (p.devices.size() != 2) return;
  const auto& a = p.devices[0];
  CHECK_EQ(a.api, std::string("cuda"));
  CHECK_EQ(a.id, std::string(kProUuid));
  CHECK_EQ(a.arch, std::string("sm_120"));
  CHECK_EQ(a.compute_units, 188u);
  CHECK_EQ(a.wave, 32u);
  CHECK_EQ(a.lds_bytes, uint64_t(49152));
  CHECK_EQ(a.memory_bytes, uint64_t(97887272960ull));
  CHECK_EQ(a.cc_major, 12);
  CHECK_EQ(a.cc_minor, 0);
  CHECK_EQ(a.pcie_gen, 5);
  CHECK_EQ(a.pcie_width, 16);
  CHECK_EQ(a.driver, std::string("580.88"));
  CHECK(a.h2d_gbps == 0);
  CHECK(a.gtt_bytes == 0 && a.visible_bytes == 0);
  const auto& b = p.devices[1];
  CHECK_EQ(b.id, std::string(k4090Uuid));
  CHECK_EQ(b.name, std::string("NVIDIA GeForce RTX 4090"));
  CHECK_EQ(b.arch, std::string("sm_89"));
  CHECK_EQ(b.compute_units, 128u);
  CHECK_EQ(b.memory_bytes, uint64_t(25757220864ull));
  CHECK_EQ(b.pcie_gen, 4);
  CHECK_EQ(b.pcie_width, 16);
  CHECK(!p.unified);
  CHECK(has_source(p, "cuda"));
  CHECK(has_source(p, "nvml"));
  CHECK(p.evidence.find("devices[0].h2d_gbps") == nullptr);
}

TEST(no_nvml_leaves_pcie_unknown) {
  Profile p;
  ts::devprof::detect_cuda(p, fake(two_gpus(), false), temp_dir("nonvml").string());
  CHECK_EQ(p.devices.size(), size_t(2));
  if (p.devices.empty()) return;
  CHECK_EQ(p.devices[0].pcie_gen, -1);
  CHECK_EQ(p.devices[0].pcie_width, -1);
  CHECK(p.devices[0].driver.empty());
  CHECK(!has_source(p, "nvml"));
}

TEST(h2d_from_engine_cache) {
  const fs::path dir = temp_dir("cache");
  // ds41_gpu_profile's writer: {"name":...,"uuid":<32 hex>,"h2d_gbps":...,"measured":...}
  const fs::path f = dir / (std::string("strix-gpu-") + kProKey + ".json");
  write(f, std::string("{\"name\":\"NVIDIA RTX PRO 6000 Blackwell Workstation Edition\",\"uuid\":\"") + kProKey +
               "\",\"h2d_gbps\":25.1,\"measured\":\"2000-01-01T00:00:00\"}\n");
  Profile p;
  ts::devprof::detect_cuda(p, fake(two_gpus()), dir.string());
  CHECK_EQ(p.devices.size(), size_t(2));
  if (p.devices.size() != 2) return;
  CHECK(p.devices[0].h2d_gbps == 25.1);
  CHECK(p.devices[1].h2d_gbps == 0);  // no cache file for the 4090
  const ts::Json* ev = p.evidence.find("devices[0].h2d_gbps");
  CHECK(ev != nullptr && ev->is_string());
  if (ev) CHECK_EQ(ev->text, "cache:" + (dir.string() + "/strix-gpu-" + kProKey + ".json"));
  CHECK(p.evidence.find("devices[1].h2d_gbps") == nullptr);
  CHECK(p.notices.empty());
}

TEST(empty_cache_dir_reads_no_cache) {
  // SS_DEVPROF_NO_CACHE reaches detect_cuda as "": no default directory is read, even with one that holds a file
  const fs::path dir = temp_dir("nocache");
  write(dir / (std::string("strix-gpu-") + kProKey + ".json"), "{\"h2d_gbps\":25.1}\n");
#if defined(_WIN32)
  _putenv_s("STRIX_GPU_PROFILE_DIR", dir.string().c_str());
#else
  setenv("STRIX_GPU_PROFILE_DIR", dir.string().c_str(), 1);
#endif
  CHECK_EQ(ts::devprof::default_gpu_cache_dir(), dir.string());
  Profile p;
  ts::devprof::detect_cuda(p, fake(two_gpus()), "");
  CHECK_EQ(p.devices.size(), size_t(2));
  if (!p.devices.empty()) CHECK(p.devices[0].h2d_gbps == 0);
  CHECK(p.evidence.find("devices[0].h2d_gbps") == nullptr);
  // the resolved default reads it
  Profile q;
  ts::devprof::detect_cuda(q, fake(two_gpus()), ts::devprof::default_gpu_cache_dir());
  if (!q.devices.empty()) CHECK(q.devices[0].h2d_gbps == 25.1);
#if defined(_WIN32)
  _putenv_s("STRIX_GPU_PROFILE_DIR", "");
#else
  unsetenv("STRIX_GPU_PROFILE_DIR");
#endif
}

TEST(evidence_index_follows_existing_devices) {
  const fs::path dir = temp_dir("index");
  write(dir / (std::string("strix-gpu-") + kProKey + ".json"), "{\"h2d_gbps\":24.5}");
  Profile p;
  p.devices.emplace_back();  // a device another probe found first
  ts::devprof::detect_cuda(p, fake(two_gpus()), dir.string());
  CHECK_EQ(p.devices.size(), size_t(3));
  CHECK(p.evidence.find("devices[1].h2d_gbps") != nullptr);
  CHECK(p.evidence.find("devices[0].h2d_gbps") == nullptr);
}

TEST(corrupt_cache_is_a_notice) {
  const fs::path dir = temp_dir("corrupt");
  write(dir / (std::string("strix-gpu-") + kProKey + ".json"), "{\"h2d_gbps\":25.1,");
  Profile p;
  ts::devprof::detect_cuda(p, fake(two_gpus()), dir.string());
  CHECK_EQ(p.devices.size(), size_t(2));
  if (p.devices.empty()) return;
  CHECK(p.devices[0].h2d_gbps == 0);
  CHECK(has_notice(p, "not readable"));
  CHECK(p.evidence.find("devices[0].h2d_gbps") == nullptr);

  const fs::path dir2 = temp_dir("nofield");
  write(dir2 / (std::string("strix-gpu-") + kProKey + ".json"), "{\"name\":\"x\"}");
  Profile q;
  ts::devprof::detect_cuda(q, fake(two_gpus()), dir2.string());
  CHECK(!q.devices.empty() && q.devices[0].h2d_gbps == 0);
  CHECK(has_notice(q, "no h2d_gbps"));
}

TEST(empty_api_changes_nothing) {
  Profile p;
  ts::devprof::detect_cuda(p, CudaApi{}, "");
  CHECK(p.devices.empty());
  CHECK(p.notices.empty());
  CHECK(p.sources.empty());
  CHECK(!p.unified);
}

TEST(failing_driver_is_a_notice) {
  CudaApi a;
  a.device_count = [](int*) { return 100; };  // CUDA_ERROR_NO_DEVICE
  Profile p;
  ts::devprof::detect_cuda(p, a, "");
  CHECK(p.devices.empty());
  CHECK(has_notice(p, "cannot count"));

  auto gpus = two_gpus();
  CudaApi b = fake(gpus);
  b.props = [](int i, CudaApi::Props* pr) {
    if (i == 0) return false;
    pr->name = "x";
    pr->uuid = "GPU-00000000-0000-0000-0000-000000000001";
    pr->pci = "0000:09:00.0";  // unknown to the fake NVML
    pr->cc_major = 9;
    return true;
  };
  Profile q;
  ts::devprof::detect_cuda(q, b, temp_dir("partial").string());
  CHECK_EQ(q.devices.size(), size_t(1));
  CHECK(has_notice(q, "device 0: properties not readable"));
  CHECK(has_notice(q, "no PCIe link reading"));
}

TEST(integrated_device_is_unified) {
  auto gpus = two_gpus();
  gpus.resize(1);
  gpus[0].props.integrated = true;
  Profile p;
  ts::devprof::detect_cuda(p, fake(gpus), temp_dir("integrated").string());
  CHECK(p.unified);
}

TEST(windows_host) {
  Profile p;
  ts::devprof::detect_windows_host(p);
#ifdef _WIN32
  CHECK(p.ram_bytes > 0);
  CHECK(p.ram_available_bytes > 0 && p.ram_available_bytes <= p.ram_bytes);
  CHECK(p.cpu_cores > 0);
  CHECK(has_source(p, "windows"));
#else
  CHECK(p.ram_bytes == 0);
  CHECK(p.sources.empty());
#endif
}

// Opt-in only: queries the real driver without a context (names, attributes, total memory, UUID, NVML link).
TEST(real_query_opt_in) {
  const char* e = std::getenv("TS_DEVPROF_REAL_CUDA");
  if (!e || std::string(e) != "1") {
    std::printf("  skipped (TS_DEVPROF_REAL_CUDA=1 runs it)\n");
    return;
  }
  Profile p;
  ts::devprof::detect_cuda(p, ts::devprof::cuda_runtime(), "");
  for (const auto& d : p.devices)
    std::printf("  %s %s %s sms=%u mem=%llu pcie=%d x%d h2d=%.2f\n", d.id.c_str(), d.name.c_str(), d.arch.c_str(),
                d.compute_units, static_cast<unsigned long long>(d.memory_bytes), d.pcie_gen, d.pcie_width,
                d.h2d_gbps);
  for (const auto& n : p.notices) std::printf("  notice: %s\n", n.c_str());
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
