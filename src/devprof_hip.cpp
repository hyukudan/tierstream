// Device profiles: the HIP runtime, loaded at run time (devprof.hpp hip_runtime / detect_hip). Only flat C entry points
// are used (hipGetDeviceCount, hipDeviceGetAttribute, hipDeviceGetName, hipDeviceGetPCIBusId, hipGetDevice,
// hipSetDevice, hipMemGetInfo): no hipDeviceProp_t, whose layout changes between ROCm releases, so the arch is the KFD
// one of the device with the same PCI address. The caller's current device is restored after every query.
#include <cctype>
#include <cstddef>
#include <exception>
#include <string>
#include <utility>

#include <cstdio>
#include <functional>

#include "devprof.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ts::devprof {
namespace {

// hipDeviceAttribute_t values (hip/hip_runtime_api.h, ROCm 7.2.4; the CUDA-compatible block has kept its order since
// ROCm 5 - an unused slot replaced hipDeviceAttributeName instead of renumbering).
constexpr int kAttrIntegrated = 16;
constexpr int kAttrMaxSharedMemoryPerBlock = 74;
constexpr int kAttrMultiprocessorCount = 63;  // CU mode: CUs; WGP mode: half of them (the KFD count wins)
constexpr int kAttrWarpSize = 87;

using CountFn = int (*)(int*);
using AttrFn = int (*)(int*, int, int);
using NameFn = int (*)(char*, int, int);
using SetFn = int (*)(int);
using GetFn = int (*)(int*);
using MemFn = int (*)(size_t*, size_t*);

void* open_library() {
#if defined(_WIN32)
  for (const char* n : {"amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"})
    if (HMODULE h = LoadLibraryA(n)) return reinterpret_cast<void*>(h);
#else
  for (const char* n : {"libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"})
    if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
#endif
  return nullptr;
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

template <class F>
F fn(void* lib, const char* name) {
  return reinterpret_cast<F>(symbol(lib, name));
}

// "dddd:bb:dd.f" lowercase with a 4-digit domain ("C5:00.0" and "0000:c5:00.0" are the same device); "" if malformed.
std::string normal_pci(const std::string& v) {
  std::string t;
  for (char c : v) t += char(std::tolower(static_cast<unsigned char>(c)));
  unsigned dom = 0, bus = 0, dev = 0, fnc = 0;
  char tail = 0;
  if (std::sscanf(t.c_str(), "%x:%x:%x.%x%c", &dom, &bus, &dev, &fnc, &tail) != 4) {
    dom = 0;
    if (std::sscanf(t.c_str(), "%x:%x.%x%c", &bus, &dev, &fnc, &tail) != 3) return {};
  }
  char b[32];
  std::snprintf(b, sizeof b, "%04x:%02x:%02x.%x", dom & 0xffffu, bus & 0xffu, dev & 0x1fu, fnc & 7u);
  return b;
}

}  // namespace

// hipSetDevice(d) + hipMemGetInfo, then the caller's device back (also when the query fails).
bool hip_mem_info_restoring(const std::function<int(int*)>& get, const std::function<int(int)>& set,
                            const std::function<int(size_t*, size_t*)>& mem, int d, uint64_t* free_b,
                            uint64_t* total_b) {
  int prev = -1;
  if (get(&prev) != 0) return false;  // cannot restore: do not move the caller's device
  if (set(d) != 0) {
    set(prev);
    return false;
  }
  size_t f = 0, t = 0;
  const bool ok = mem(&f, &t) == 0;
  set(prev);
  if (!ok) return false;
  *free_b = f;
  *total_b = t;
  return true;
}

HipApi hip_runtime() {
  HipApi api;
  // loaded once and never unloaded (the runtime keeps threads of its own)
  static void* const lib = open_library();
  if (!lib) return api;
  const auto count = fn<CountFn>(lib, "hipGetDeviceCount");
  const auto attr = fn<AttrFn>(lib, "hipDeviceGetAttribute");
  const auto name = fn<NameFn>(lib, "hipDeviceGetName");
  const auto pci = fn<NameFn>(lib, "hipDeviceGetPCIBusId");
  const auto set = fn<SetFn>(lib, "hipSetDevice");
  const auto mem = fn<MemFn>(lib, "hipMemGetInfo");
  if (!count || !attr) return api;
  api.device_count = [count](int* n) { return count(n); };
  api.props = [attr, name, pci](int d, HipApi::Props* p) {
    int v = 0;
    if (attr(&v, kAttrMultiprocessorCount, d) != 0) return false;
    p->cus = uint32_t(v);
    if (attr(&v, kAttrWarpSize, d) == 0) p->wave = uint32_t(v);
    if (attr(&v, kAttrMaxSharedMemoryPerBlock, d) == 0) p->lds = uint64_t(v);
    if (attr(&v, kAttrIntegrated, d) == 0) p->integrated = v != 0;
    char buf[256] = {};
    if (name && name(buf, int(sizeof buf - 1), d) == 0) p->name = buf;
    char bus[64] = {};
    if (pci && pci(bus, int(sizeof bus - 1), d) == 0) p->pci = bus;
    return true;
  };
  const auto get = fn<GetFn>(lib, "hipGetDevice");
  if (set && mem && get)
    api.mem_info = [set, mem, get](int d, uint64_t* free_b, uint64_t* total_b) {
      return hip_mem_info_restoring(get, set, [mem](size_t* f, size_t* t) { return mem(f, t); }, d, free_b, total_b);
    };
  return api;
}

namespace {

void detect_hip_impl(Profile& p, const HipApi& api) {
  if (!api.present()) {
    p.notices.push_back("hip: no HIP runtime");
    return;
  }
  int n = 0;
  if (api.device_count(&n) != 0 || n <= 0) {
    p.notices.push_back("hip: no HIP device");
    return;
  }
  struct Rt {
    HipApi::Props pr;
    bool ok = false;
    std::string pci;
    uint64_t total = 0;
  };
  std::vector<Rt> rt(static_cast<size_t>(n));
  bool all_answered = true;
  for (int d = 0; d < n; ++d) {
    Rt& r = rt[size_t(d)];
    r.ok = api.props && api.props(d, &r.pr);
    if (!r.ok) {
      p.notices.push_back("hip: device " + std::to_string(d) + ": properties unavailable");
      all_answered = false;
    }
    r.pci = normal_pci(r.pr.pci);
    uint64_t free_b = 0;
    if (!(api.mem_info && api.mem_info(d, &free_b, &r.total) && r.total)) {
      r.total = 0;
      p.notices.push_back("hip: device " + std::to_string(d) + ": hipMemGetInfo unavailable");
    }
  }
  // the KFD (sysfs) devices, joined by PCI address - never by position: HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES
  // renumber and hide devices
  std::vector<size_t> kfd;
  for (size_t i = 0; i < p.devices.size(); ++i)
    if (p.devices[i].api == "hip") kfd.push_back(i);
  bool kfd_pci = false;
  for (size_t i : kfd) kfd_pci = kfd_pci || !normal_pci(p.devices[i].id).empty();
  std::vector<long> match(size_t(n), -1);  // runtime device -> index into p.devices
  std::vector<bool> taken(p.devices.size(), false);
  for (int d = 0; d < n; ++d) {
    const Rt& r = rt[size_t(d)];
    if (r.pci.empty()) continue;
    for (size_t i : kfd)
      if (!taken[i] && normal_pci(p.devices[i].id) == r.pci) {
        match[size_t(d)] = long(i);
        taken[i] = true;
        break;
      }
  }
  if (!kfd_pci && kfd.size() == 1 && n == 1) {  // no PCI address in sysfs: only the one-to-one case is safe
    match[0] = long(kfd[0]);
    taken[kfd[0]] = true;
  } else if (!kfd_pci && !kfd.empty()) {
    p.notices.push_back("hip: sysfs gives no PCI address and " + std::to_string(kfd.size()) + " KFD / " +
                        std::to_string(n) + " runtime devices: not joined (runtime values only)");
  }
  std::vector<Device> out;
  for (size_t i = 0; i < p.devices.size(); ++i) {
    if (p.devices[i].api != "hip") {
      out.push_back(p.devices[i]);
      continue;
    }
    if (taken[i]) continue;  // placed below with its runtime device
    if (all_answered) {
      p.notices.push_back("hip: " + p.devices[i].id + " (" + p.devices[i].arch +
                          ") is not visible to the HIP runtime (HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES?): dropped");
      continue;
    }
    out.push_back(p.devices[i]);  // the runtime did not answer for every device: keep what sysfs found
  }
  bool any_integrated = false;
  for (int d = 0; d < n; ++d) {
    const Rt& r = rt[size_t(d)];
    Device dev;
    if (match[size_t(d)] >= 0) {
      dev = p.devices[size_t(match[size_t(d)])];  // KFD arch / CUs kept only for the same PCI device
    } else {
      dev.api = "hip";
      dev.id = !r.pci.empty() ? r.pci : "hip-" + std::to_string(d);
      dev.arch = r.pr.arch;  // the runtime's arch when it has one; empty -> a generic class
      dev.compute_units = r.pr.cus;
    }
    if (r.ok) {
      if (!r.pr.name.empty()) dev.name = r.pr.name;
      if (dev.arch.empty()) dev.arch = r.pr.arch;
      if (!dev.compute_units) dev.compute_units = r.pr.cus;  // KFD's simd_count / simd_per_cu wins (WGP mode)
      if (r.pr.wave) dev.wave = r.pr.wave;
      if (r.pr.lds) dev.lds_bytes = r.pr.lds;
      if (!r.pci.empty()) dev.id = r.pci;
      any_integrated = any_integrated || r.pr.integrated;
    }
    if (r.total) dev.memory_bytes = r.total;  // what hipMalloc can get (the GTT pool on an APU): wins over sysfs
    out.push_back(std::move(dev));
  }
  p.devices = std::move(out);
  // the runtime knows whether the GPU is integrated: it decides when every device answered
  if (any_integrated) p.unified = true;
  else if (all_answered) p.unified = false;
  p.sources.push_back("hip");
}

}  // namespace

void detect_hip(Profile& p, const HipApi& api) {
  try {
    detect_hip_impl(p, api);
  } catch (const std::exception& e) {
    p.notices.push_back(std::string("hip: detection stopped: ") + e.what());
  }
}

}  // namespace ts::devprof
