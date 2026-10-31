// Device profiles (ss.h "Device profiles", docs/design.md §5.2): strix/device-profile-v1, its detection, the class
// table, the override merge and the planner hooks (caps, single box vs pair, prefill piece).
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"
#include "strix_stream/ss.h"

namespace ts::devprof {

struct Nvme {
  std::string path;      // mount point / drive ("D:", "/mnt/models") or block device ("nvme0n1")
  std::string model;
  double read_gbps = 0;  // measured sequential read, GB/s (0 = not measured)
};

struct Device {
  std::string api = "cpu";  // "hip" | "cuda" | "cpu"
  std::string id;           // GPU UUID / PCI bus id / "kfd-node-N"
  std::string name, arch;   // arch: "gfx1151", "sm_120"
  uint32_t compute_units = 0, wave = 0;
  uint64_t lds_bytes = 0, memory_bytes = 0, gtt_bytes = 0, visible_bytes = 0;
  double mem_gbps = 0, h2d_gbps = 0;  // 0 = unknown (JSON null)
  int32_t cc_major = 0, cc_minor = 0;
  int32_t pcie_gen = -1, pcie_width = -1;
  std::string driver;
};

struct Link {
  std::string kind = "other";  // "usb4" | "ethernet" | "pcie" | "other"
  uint32_t count = 0;          // cables / ports up
  double gbps_measured = 0;    // summed over the cables, 0 = not measured
};

struct Profile {
  std::string cls = "generic";
  bool detected = false;
  bool generic = true;
  // host
  uint64_t ram_bytes = 0, ram_available_bytes = 0;
  bool unified = false;
  uint32_t cpu_cores = 0;
  std::vector<Nvme> nvme;
  std::vector<Device> devices;
  std::vector<Link> links;
  // margins
  uint64_t host_floor_bytes = 0, device_reserve_bytes = 0;
  // class limits the engines read instead of literals (max_piece_rows, uma_ceiling_bytes, persist_blocks, ...):
  // today's values on the class they were qualified on, absent elsewhere (the engine keeps its own rule)
  std::map<std::string, uint64_t> limits;
  // provenance: evidence ids per field ("devices[0].mem_gbps" -> "measured-membw"), merged override
  // paths, notices (what detection could not read, generic class), sources (which probes answered)
  Json evidence = Json::object();
  std::vector<std::string> overrides, notices, sources;
};

// JSON (strix/device-profile-v1). from_json refuses unknown keys and a wrong schema (SS_E_RECIPE).
Json to_json(const Profile& p);
std::string to_text(const Profile& p);  // canonical
std::string hash(const Profile& p);     // xxh64 of to_text, 16 hex digits
Profile from_json(const Json& j);
// Partial profile over p (objects by key, arrays by index); every merged leaf path is appended to p.overrides as
// "origin:path". Throws SS_E_RECIPE (p unchanged) on a bad key / type.
void merge(Profile& p, const Json& partial, const std::string& origin);

// Classes (devprof_class.cpp): the shipped table; profiles/<class>.json in the repo are checked against it.
struct ClassRow {
  const char* name;
  const char* api;         // "hip" | "cuda"
  const char* arch;        // exact arch ("gfx1151", "sm_120"); "" = any
  const char* name_part;   // case-insensitive substring of the device name; "" = any
  uint64_t min_memory_bytes, max_memory_bytes;  // device memory_bytes range (0 = no bound)
  bool unified;
  uint64_t host_floor_bytes, device_reserve_bytes;
  double mem_gbps;         // measured (evidence below), 0 = none
  const char* evidence;    // receipt id of mem_gbps / the margins
  std::vector<std::pair<std::string, uint64_t>> limits;
};
const std::vector<ClassRow>& classes();
// Picks the class of p's first GPU (or "generic-cpu" without one) and fills the class fields detection left at 0
// (margins, mem_gbps, unified) - detected values win. No match: "generic-hip" / "generic-cuda" + a notice.
// forced: a class name (an override's "class"), else picked; SS_E_RECIPE for an unknown name.
void apply_class(Profile& p, const std::string& forced = "");
// Takes back what a class filled (evidence "class:..."), except the paths in keep (set explicitly).
void unclass(Profile& p, const std::vector<std::string>& keep);
Json class_json(const ClassRow& c);  // profiles/<class>.json content

// Detection (devprof_linux.cpp / devprof_hip.cpp / devprof_cuda.cpp). Each fills what it can and adds a notice for
// what it could not read; none throws.
// Linux: <root>/proc/meminfo (MemTotal / MemAvailable), <root>/proc/cpuinfo or sysfs cpu list (cores), KFD topology
// <root>/sys/class/kfd/kfd/topology/nodes/*/properties (gfx_target_version, simd_count, simd_per_cu, wave front size,
// lds_size_in_kb, name), amdgpu DRM <root>/sys/class/drm/card*/device/mem_info_{gtt,vram,vis_vram}_total,
// Thunderbolt / USB4 <root>/sys/bus/thunderbolt/devices (domains, xdomain peers = links), NVMe <root>/sys/class/nvme.
void detect_linux(Profile& p, const std::string& root);

// Runtime APIs as function tables so tests can fake them; the real tables load the shared libraries at run time
// (libamdhip64 / amdhip64_6.dll, cudart / nvcuda + nvml) and are empty when absent.
struct HipApi {
  std::function<int(int*)> device_count;
  // name, gcnArchName, multiProcessorCount, warpSize, sharedMemPerBlock, totalGlobalMem, integrated, pciBusID
  struct Props {
    std::string name, arch, pci;
    uint32_t cus = 0, wave = 0;
    uint64_t lds = 0, total = 0;
    bool integrated = false;
  };
  std::function<bool(int, Props*)> props;
  std::function<bool(int, uint64_t*, uint64_t*)> mem_info;  // free, total (hipSetDevice + hipMemGetInfo)
  bool present() const { return bool(device_count); }
};
HipApi hip_runtime();  // loaded at run time; empty when no HIP runtime
void detect_hip(Profile& p, const HipApi& api);

struct CudaApi {
  std::function<int(int*)> device_count;
  struct Props {
    std::string name, uuid, pci;
    int cc_major = 0, cc_minor = 0;
    uint32_t sms = 0, warp = 0;
    uint64_t smem_per_block = 0, total = 0;
    bool integrated = false;
  };
  std::function<bool(int, Props*)> props;
  std::function<bool(int, uint64_t*, uint64_t*)> mem_info;  // free, total
  // NVML (optional): PCIe current / max gen and width by PCI bus id, the driver version
  std::function<bool(const std::string&, int*, int*, int*, int*)> pcie;
  std::function<std::string()> driver;
  bool present() const { return bool(device_count); }
};
CudaApi cuda_runtime();  // loaded at run time; empty when no CUDA driver / runtime
// Measured H2D GB/s comes from the cache (ds41_gpu_profile's strix-gpu-<uuid>.json h2d_gbps, or ours) when present.
void detect_cuda(Profile& p, const CudaApi& api, const std::string& cache_dir);
std::string default_gpu_cache_dir();

// Host facts on Windows (GlobalMemoryStatusEx, logical processors); no-op elsewhere.
void detect_windows_host(Profile& p);

// The whole detection (ss_devprof_detect): host -> runtimes -> class -> override.
struct DetectOptions {
  uint32_t flags = 0;
  std::string root = "/";
  std::string override_path;  // empty = none
  std::string cache_dir;
};
Profile detect(const DetectOptions& o, const HipApi* hip = nullptr, const CudaApi* cuda = nullptr);

// Planner hooks (devprof_plan.cpp).
void budget(const Profile& p, uint32_t device, ss_budget_in& in);
// The one cap ss_devprof_budget would give (unified: the pool; discrete: the device's VRAM cap).
uint64_t device_budget(const Profile& p, uint32_t device);
bool topology(const std::vector<const Profile*>& boxes, const ss_devprof_topology_in& in, ss_devprof_topology_out& out);
uint32_t piece_rows(const Profile& p, const ss_devprof_piece_in& in);  // throws SS_E_CAPACITY

}  // namespace ts::devprof
