// Device profiles, NVIDIA side (devprof.hpp): the CUDA driver API and NVML loaded at run time (no toolkit needed to
// build), queried WITHOUT a context - attributes, name, total memory and UUID only, nothing allocated on the GPU and no
// bandwidth probe - plus the measured H2D bandwidth from the engine's per-GPU cache (ds41_gpu_profile:
// strix-gpu-<uuid>.json), and the Windows host facts.
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>

#include "devprof.hpp"
#include "errors.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ts::devprof {
namespace {

// cuda.h values (checked against CUDA 12.8 and 13.0)
constexpr int kAttrSmemPerBlock = 8, kAttrWarp = 10, kAttrSms = 16, kAttrIntegrated = 18, kAttrPciBus = 33,
              kAttrPciDevice = 34, kAttrPciDomain = 50, kAttrCcMajor = 75, kAttrCcMinor = 76;

class Lib {
 public:
  explicit Lib(std::initializer_list<const char*> names) {
    for (const char* n : names) {
#ifdef _WIN32
      h_ = LoadLibraryA(n);
#else
      h_ = dlopen(n, RTLD_LAZY | RTLD_LOCAL);
#endif
      if (h_) break;
    }
  }
  Lib(const Lib&) = delete;
  Lib& operator=(const Lib&) = delete;
  ~Lib() {
    if (!h_) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(h_));
#else
    dlclose(h_);
#endif
  }
  bool ok() const { return h_ != nullptr; }
  template <class F>
  F sym(const char* n) const {
    if (!h_) return nullptr;
#ifdef _WIN32
    return reinterpret_cast<F>(reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(h_), n)));
#else
    return reinterpret_cast<F>(dlsym(h_, n));
#endif
  }

 private:
  void* h_ = nullptr;
};

// The driver API (CUresult = int, CUdevice = int) and NVML (nvmlReturn_t = int, nvmlDevice_t = pointer).
struct Driver {
  Lib lib{
#ifdef _WIN32
      "nvcuda.dll"
#else
      "libcuda.so.1", "libcuda.so"
#endif
  };
  int (*init)(unsigned) = nullptr;
  int (*count)(int*) = nullptr;
  int (*get)(int*, int) = nullptr;
  int (*name)(char*, int, int) = nullptr;
  int (*attr)(int*, int, int) = nullptr;
  int (*total)(size_t*, int) = nullptr;
  int (*uuid)(unsigned char*, int) = nullptr;  // CUuuid = 16 bytes
};

struct Nvml {
  Lib lib{
#ifdef _WIN32
      "nvml.dll", "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll"
#else
      "libnvidia-ml.so.1", "libnvidia-ml.so"
#endif
  };
  bool up = false;
  int (*shutdown)() = nullptr;
  int (*by_bus)(const char*, void**) = nullptr;
  int (*cur_gen)(void*, unsigned*) = nullptr;
  int (*cur_width)(void*, unsigned*) = nullptr;
  int (*max_gen)(void*, unsigned*) = nullptr;
  int (*max_width)(void*, unsigned*) = nullptr;
  int (*driver)(char*, unsigned) = nullptr;
  Nvml() {
    const auto init = lib.sym<int (*)()>("nvmlInit_v2");
    if (!init || init() != 0) return;
    up = true;
    shutdown = lib.sym<int (*)()>("nvmlShutdown");
    by_bus = lib.sym<int (*)(const char*, void**)>("nvmlDeviceGetHandleByPciBusId_v2");
    cur_gen = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetCurrPcieLinkGeneration");
    cur_width = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetCurrPcieLinkWidth");
    max_gen = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetMaxPcieLinkGeneration");
    max_width = lib.sym<int (*)(void*, unsigned*)>("nvmlDeviceGetMaxPcieLinkWidth");
    driver = lib.sym<int (*)(char*, unsigned)>("nvmlSystemGetDriverVersion");
  }
  Nvml(const Nvml&) = delete;
  Nvml& operator=(const Nvml&) = delete;
  ~Nvml() {
    if (up && shutdown) shutdown();
  }
};

std::string uuid_text(const unsigned char* b) {
  // nvidia-smi's form: GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  std::string s = "GPU-";
  char h[3];
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) s += '-';
    std::snprintf(h, sizeof h, "%02x", unsigned(b[i]));
    s += h;
  }
  return s;
}

// The engine's cache key: the 16 UUID bytes as 32 lowercase hex digits (no "GPU-", no dashes).
std::string cache_key(const std::string& uuid) {
  std::string s = uuid.rfind("GPU-", 0) == 0 ? uuid.substr(4) : uuid;
  std::string k;
  for (char c : s)
    if (c != '-') k += char(std::tolower(static_cast<unsigned char>(c)));
  return k;
}

std::string join(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  const char last = dir.back();
  return last == '/' || last == '\\' ? dir + file : dir + "/" + file;
}

}  // namespace

CudaApi cuda_runtime() {
  auto d = std::make_shared<Driver>();
  if (!d->lib.ok()) return {};
  d->init = d->lib.sym<int (*)(unsigned)>("cuInit");
  d->count = d->lib.sym<int (*)(int*)>("cuDeviceGetCount");
  d->get = d->lib.sym<int (*)(int*, int)>("cuDeviceGet");
  d->name = d->lib.sym<int (*)(char*, int, int)>("cuDeviceGetName");
  d->attr = d->lib.sym<int (*)(int*, int, int)>("cuDeviceGetAttribute");
  d->total = d->lib.sym<int (*)(size_t*, int)>("cuDeviceTotalMem_v2");
  d->uuid = d->lib.sym<int (*)(unsigned char*, int)>("cuDeviceGetUuid_v2");
  if (!d->uuid) d->uuid = d->lib.sym<int (*)(unsigned char*, int)>("cuDeviceGetUuid");
  if (!d->init || !d->count || !d->get || !d->attr || d->init(0) != 0) return {};

  CudaApi a;
  a.device_count = [d](int* n) { return d->count(n); };
  a.props = [d](int i, CudaApi::Props* p) {
    int dev = 0;
    if (d->get(&dev, i) != 0) return false;
    auto at = [&](int which) {
      int v = 0;
      return d->attr(&v, which, dev) == 0 ? v : 0;
    };
    char name[256] = {0};
    if (d->name && d->name(name, int(sizeof name) - 1, dev) == 0) p->name = name;
    p->cc_major = at(kAttrCcMajor);
    p->cc_minor = at(kAttrCcMinor);
    p->sms = uint32_t(at(kAttrSms));
    p->warp = uint32_t(at(kAttrWarp));
    p->smem_per_block = uint64_t(at(kAttrSmemPerBlock));
    p->integrated = at(kAttrIntegrated) != 0;
    char bus[32];
    std::snprintf(bus, sizeof bus, "%04x:%02x:%02x.0", unsigned(at(kAttrPciDomain)), unsigned(at(kAttrPciBus)),
                  unsigned(at(kAttrPciDevice)));
    p->pci = bus;
    unsigned char u[16] = {0};
    if (d->uuid && d->uuid(u, dev) == 0) p->uuid = uuid_text(u);
    size_t total = 0;
    if (d->total && d->total(&total, dev) == 0) p->total = total;
    return true;
  };
  // free memory needs a context (cuMemGetInfo): not queried, the profile does not use it
  a.mem_info = nullptr;

  auto n = std::make_shared<Nvml>();
  if (n->up) {
    a.pcie = [n](const std::string& pci, int* gen, int* width, int* mgen, int* mwidth) {
      void* h = nullptr;
      if (!n->by_bus || n->by_bus(pci.c_str(), &h) != 0) return false;
      unsigned v = 0;
      if (n->cur_gen && n->cur_gen(h, &v) == 0) *gen = int(v);
      if (n->cur_width && n->cur_width(h, &v) == 0) *width = int(v);
      if (n->max_gen && n->max_gen(h, &v) == 0) *mgen = int(v);
      if (n->max_width && n->max_width(h, &v) == 0) *mwidth = int(v);
      return true;
    };
    a.driver = [n]() -> std::string {
      char v[96] = {0};
      return n->driver && n->driver(v, unsigned(sizeof v)) == 0 ? std::string(v) : std::string();
    };
  }
  return a;
}

// The engine's measurement cache directory (ds41_gpu_profile's default_cache_dir order): STRIX_GPU_PROFILE_DIR, then
// %LOCALAPPDATA%\strix (Windows), $XDG_CACHE_HOME/strix, ~/.cache/strix; "" when none is set.
std::string default_gpu_cache_dir() {
  if (const char* v = std::getenv("STRIX_GPU_PROFILE_DIR")) return v;
#ifdef _WIN32
  if (const char* v = std::getenv("LOCALAPPDATA")) return std::string(v) + "\\strix";
#endif
  if (const char* v = std::getenv("XDG_CACHE_HOME")) return std::string(v) + "/strix";
  if (const char* v = std::getenv("HOME")) return std::string(v) + "/.cache/strix";
  return {};
}

void detect_cuda(Profile& p, const CudaApi& api, const std::string& cache_dir) {
  if (!api.present()) return;
  try {
    int n = 0;
    if (api.device_count(&n) != 0 || n < 0) {
      p.notices.push_back("cuda: cannot count the devices");
      return;
    }
    p.sources.push_back("cuda");
    const std::string driver = api.driver ? api.driver() : std::string();
    if (api.pcie) p.sources.push_back("nvml");
    const std::string& dir = cache_dir;  // "" = no cache (SS_DEVPROF_NO_CACHE); detect() resolves the default
    for (int i = 0; i < n; ++i) {
      CudaApi::Props pr;
      if (!api.props || !api.props(i, &pr)) {
        p.notices.push_back("cuda: device " + std::to_string(i) + ": properties not readable");
        continue;
      }
      Device d;
      d.api = "cuda";
      d.id = !pr.uuid.empty() ? pr.uuid : pr.pci;
      d.name = pr.name;
      d.arch = "sm_" + std::to_string(pr.cc_major) + std::to_string(pr.cc_minor);
      d.compute_units = pr.sms;
      d.wave = pr.warp;
      d.lds_bytes = pr.smem_per_block;
      d.memory_bytes = pr.total;
      d.cc_major = pr.cc_major;
      d.cc_minor = pr.cc_minor;
      d.driver = driver;
      const std::string at = "devices[" + std::to_string(p.devices.size()) + "]";
      if (api.pcie && !pr.pci.empty()) {
        int gen = -1, width = -1, mgen = -1, mwidth = -1;
        if (api.pcie(pr.pci, &gen, &width, &mgen, &mwidth)) {
          d.pcie_gen = gen;
          d.pcie_width = width;
        } else {
          p.notices.push_back("nvml: " + at + " (" + pr.pci + "): no PCIe link reading");
        }
      }
      if (!pr.uuid.empty() && !dir.empty()) {
        const std::string file = join(dir, "strix-gpu-" + cache_key(pr.uuid) + ".json");
        std::ifstream in(file, std::ios::binary);
        if (in) {
          std::stringstream ss;
          ss << in.rdbuf();
          try {
            const Json j = json_parse(ss.str());
            const Json* v = j.is_object() ? j.find("h2d_gbps") : nullptr;
            const double g = v ? json_f64(*v, file + ": h2d_gbps") : 0;
            if (g > 0) {
              d.h2d_gbps = g;
              p.evidence[at + ".h2d_gbps"] = Json::string("cache:" + file);
            } else {
              p.notices.push_back("cuda: " + at + ": cache " + file + " has no h2d_gbps");
            }
          } catch (const std::exception& e) {
            p.notices.push_back("cuda: " + at + ": cache " + file + " not readable: " + e.what());
          }
        }
      }
      if (pr.integrated) p.unified = true;
      p.devices.push_back(std::move(d));
    }
  } catch (const std::exception& e) {
    p.notices.push_back(std::string("cuda: ") + e.what());
  }
}

void detect_windows_host(Profile& p) {
#ifdef _WIN32
  MEMORYSTATUSEX m{};
  m.dwLength = sizeof m;
  if (GlobalMemoryStatusEx(&m)) {
    p.ram_bytes = m.ullTotalPhys;
    p.ram_available_bytes = m.ullAvailPhys;
  } else {
    p.notices.push_back("windows: GlobalMemoryStatusEx failed");
  }
  const DWORD cores = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  if (cores) p.cpu_cores = uint32_t(cores);
  p.sources.push_back("windows");
#else
  (void)p;
#endif
}

}  // namespace ts::devprof
