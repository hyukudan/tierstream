// Device profiles, Linux detection (devprof.hpp detect_linux / detect_hip) on fake procfs / sysfs trees: a Strix Halo
// node as its sysfs really reads (gfx1151, 40 CUs, the GPU node has no CPU cores, PCI c5:00.0, GTT + carve-out, two
// USB4 peers, NVMe), a successor with a newer gfx and more RAM, a discrete card, DRM matching (render minor, PCI slot,
// the one card left), an empty root, the HIP runtime through a fake function table (joined by PCI address, hidden and
// runtime-only devices, the caller's device restored), and the whole detect() on a fake root.
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.hpp"
#include "devprof.hpp"

#if defined(_WIN32)
#include <process.h>
#define TS_GETPID _getpid
#else
#include <unistd.h>
#define TS_GETPID getpid
#endif

namespace fs = std::filesystem;
using namespace ts::devprof;

namespace ts::devprof {  // devprof_hip.cpp, for the tests
bool hip_mem_info_restoring(const std::function<int(int*)>& get, const std::function<int(int)>& set,
                            const std::function<int(size_t*, size_t*)>& mem, int d, uint64_t* free_b,
                            uint64_t* total_b);
}

namespace {

constexpr uint64_t kGiB = uint64_t(1) << 30;

// One fake tree under the temp directory, removed when the test ends.
struct Tree {
  fs::path root;
  explicit Tree(const char* tag) {
    root = fs::temp_directory_path() / ("ts-devprof-linux-" + std::to_string(TS_GETPID()) + "-" + tag);
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);
  }
  ~Tree() {
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  void put(const std::string& rel, const std::string& text) const {
    const fs::path p = root / rel;
    fs::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
  }
  void dir(const std::string& rel) const { fs::create_directories(root / rel); }
  std::string str() const { return root.string(); }
};

const char* kNodes = "sys/class/kfd/kfd/topology/nodes/";

void cpu_node(const Tree& t, int node, int cores) {
  t.put(kNodes + std::to_string(node) + "/properties",
        "cpu_cores_count " + std::to_string(cores) +
            "\nsimd_count 0\nmem_banks_count 1\ngfx_target_version 0\nlocation_id 0\ndomain 0\n");
}

// location < 0: no location_id / domain lines (older kernels)
void gpu_node(const Tree& t, int node, uint64_t target, int simds, int render_minor, int cpu_cores = 0,
              long location = -1) {
  std::string s = "cpu_cores_count " + std::to_string(cpu_cores) + "\nsimd_count " + std::to_string(simds) +
                  "\nsimd_per_cu 2\nwave_front_size 32\nlds_size_in_kb 64\ngfx_target_version " +
                  std::to_string(target) + "\ndrm_render_minor " + std::to_string(render_minor) + "\n";
  if (location >= 0) s += "location_id " + std::to_string(location) + "\ndomain 0\n";
  t.put(kNodes + std::to_string(node) + "/properties", s);
  t.put(kNodes + std::to_string(node) + "/name", "gfx" + std::to_string(target) + "\n");
}

void drm(const Tree& t, const std::string& dev, uint64_t gtt, uint64_t vram, uint64_t vis, const char* vendor,
         const char* slot = nullptr) {
  t.put("sys/class/drm/" + dev + "/device/mem_info_gtt_total", std::to_string(gtt) + "\n");
  t.put("sys/class/drm/" + dev + "/device/mem_info_vram_total", std::to_string(vram) + "\n");
  t.put("sys/class/drm/" + dev + "/device/mem_info_vis_vram_total", std::to_string(vis) + "\n");
  if (vendor) t.put("sys/class/drm/" + dev + "/device/mem_info_vram_vendor", std::string(vendor) + "\n");
  if (slot)
    t.put("sys/class/drm/" + dev + "/device/uevent", "DRIVER=amdgpu\nPCI_SLOT_NAME=" + std::string(slot) + "\n");
}

void meminfo(const Tree& t, uint64_t total_kb, uint64_t avail_kb) {
  t.put("proc/meminfo", "MemTotal:       " + std::to_string(total_kb) + " kB\nMemFree:        1000 kB\n" +
                            "MemAvailable:   " + std::to_string(avail_kb) + " kB\nBuffers:           0 kB\n");
}

// 16 cores, 2 threads each (Ryzen AI Max+ 395)
void cpuinfo(const Tree& t, int cores, int threads) {
  std::string s;
  int n = 0;
  for (int c = 0; c < cores; ++c)
    for (int h = 0; h < threads; ++h)
      s += "processor\t: " + std::to_string(n++) + "\nphysical id\t: 0\ncore id\t\t: " + std::to_string(c) + "\n\n";
  t.put("proc/cpuinfo", s);
}

// The Strix Halo node as measured (kho=off: GTT 117.16 GiB; 512 MiB carve-out). Its sysfs as read on a Strix Halo
// node: the GPU node has cpu_cores_count 0, location_id 50432 (c5:00.0), domain 0, drm_render_minor 128.
constexpr uint64_t kHaloGtt = 125798105088ull;  // 117.16 GiB
constexpr uint64_t kCarve = 512ull << 20;
constexpr long kHaloLoc = 50432;  // 0xc500

void halo(const Tree& t) {
  meminfo(t, 133120000, 126500000);
  cpuinfo(t, 16, 2);
  cpu_node(t, 0, 32);
  gpu_node(t, 1, 110501, 80, 128, 0, kHaloLoc);
  drm(t, "renderD128", kHaloGtt, kCarve, kCarve, nullptr, "0000:c5:00.0");
  drm(t, "card0", kHaloGtt, kCarve, kCarve, nullptr, "0000:c5:00.0");
  t.dir("sys/class/drm/card0-DP-1");
  // Thunderbolt: two host routers, two XDomain peers (one per cable), a dock router (has 'authorized'), a domain
  t.dir("sys/bus/thunderbolt/devices/domain0");
  t.dir("sys/bus/thunderbolt/devices/domain1");
  t.put("sys/bus/thunderbolt/devices/0-0/authorized", "1\n");
  t.put("sys/bus/thunderbolt/devices/1-0/authorized", "1\n");
  t.put("sys/bus/thunderbolt/devices/0-1/device_name", "node2\n");
  t.put("sys/bus/thunderbolt/devices/1-1/device_name", "node2\n");
  t.put("sys/bus/thunderbolt/devices/0-301/authorized", "1\n");
  t.dir("sys/bus/thunderbolt/devices/0-1.0");  // a service of the peer, not a peer
  t.put("sys/class/nvme/nvme0/model", "WD_BLACK SN850X 4000GB   \n");
}

// The Halo's HIP runtime: integrated, the GTT pool from hipMemGetInfo, 20 'multiprocessors' in WGP mode.
HipApi halo_hip(const std::string& pci = "0000:C5:00.0", uint64_t total = kHaloGtt) {
  HipApi api;
  api.device_count = [](int* n) {
    *n = 1;
    return 0;
  };
  api.props = [pci](int d, HipApi::Props* pr) {
    if (d != 0) return false;
    pr->name = "AMD Radeon 8060S Graphics";
    pr->pci = pci;
    pr->cus = 20;
    pr->wave = 32;
    pr->lds = 65536;
    pr->integrated = true;
    return true;
  };
  api.mem_info = [total](int, uint64_t* f, uint64_t* tot) {
    *f = 100 * kGiB;
    *tot = total;
    return true;
  };
  return api;
}

// A runtime listing these PCI addresses (03:00.0 = a successor with 200 GiB, anything else = a Halo).
HipApi runtime_of(const std::vector<std::string>& pcis) {
  HipApi api;
  const int n = int(pcis.size());
  api.device_count = [n](int* c) {
    *c = n;
    return 0;
  };
  api.props = [pcis](int d, HipApi::Props* pr) {
    pr->pci = pcis[size_t(d)];
    pr->name = pcis[size_t(d)] == "0000:03:00.0" ? "successor" : "halo";
    pr->integrated = true;
    return true;
  };
  api.mem_info = [pcis](int d, uint64_t* f, uint64_t* tot) {
    *f = 1;
    *tot = pcis[size_t(d)] == "0000:03:00.0" ? 200 * kGiB : kHaloGtt;
    return true;
  };
  return api;
}

bool has_notice(const Profile& p, const std::string& part) {
  for (const auto& n : p.notices)
    if (n.find(part) != std::string::npos) return true;
  return false;
}

}  // namespace

TEST(halo_node) {
  Tree t("halo");
  halo(t);
  Profile p;
  detect_linux(p, t.str());
  CHECK_EQ(p.ram_bytes, uint64_t(133120000) * 1024);
  CHECK_EQ(p.ram_available_bytes, uint64_t(126500000) * 1024);
  CHECK_EQ(p.cpu_cores, 16u);
  CHECK_EQ(p.devices.size(), size_t(1));
  if (p.devices.size() == 1) {
    const Device& d = p.devices[0];
    CHECK_EQ(d.api, std::string("hip"));
    CHECK_EQ(d.arch, std::string("gfx1151"));
    CHECK_EQ(d.id, std::string("0000:c5:00.0"));  // from KFD location_id
    CHECK_EQ(d.compute_units, 40u);
    CHECK_EQ(d.wave, 32u);
    CHECK_EQ(d.lds_bytes, uint64_t(65536));
    CHECK_EQ(d.gtt_bytes, kHaloGtt);
    CHECK_EQ(d.visible_bytes, kCarve);
    // sysfs alone has no affirmative APU evidence on this kernel: memory is the VRAM reading, not unified
    CHECK_EQ(d.memory_bytes, kCarve);
    CHECK_EQ(d.name, std::string("gfx110501"));
  }
  CHECK(!p.unified);
  CHECK(has_notice(p, "integrated or discrete not known"));
  CHECK_EQ(p.links.size(), size_t(1));
  if (!p.links.empty()) {
    CHECK_EQ(p.links[0].kind, std::string("usb4"));
    CHECK_EQ(p.links[0].count, 2u);
    CHECK_EQ(p.links[0].gbps_measured, 0.0);
  }
  CHECK_EQ(p.nvme.size(), size_t(1));
  if (!p.nvme.empty()) {
    CHECK_EQ(p.nvme[0].path, std::string("nvme0"));
    CHECK_EQ(p.nvme[0].model, std::string("WD_BLACK SN850X 4000GB"));
  }
  CHECK_EQ(p.sources.size(), size_t(1));
  // + the runtime: integrated, hipMemGetInfo's pool
  detect_hip(p, halo_hip());
  CHECK(p.unified);
  CHECK_EQ(p.devices.size(), size_t(1));
  if (!p.devices.empty()) CHECK_EQ(p.devices[0].memory_bytes, kHaloGtt);
}

TEST(successor_node) {
  // a newer gfx and 256 GiB: read as it is, nothing Halo-specific assumed; the one card found without a render minor
  Tree t("host-a");
  meminfo(t, 266000000, 250000000);
  cpu_node(t, 0, 32);
  gpu_node(t, 1, 110600, 96, 0);
  drm(t, "card0", 240 * kGiB, kCarve, kCarve, nullptr);
  Profile p;
  detect_linux(p, t.str());
  CHECK_EQ(p.ram_bytes, uint64_t(266000000) * 1024);
  CHECK_EQ(p.devices.size(), size_t(1));
  if (!p.devices.empty()) {
    CHECK_EQ(p.devices[0].arch, std::string("gfx1160"));
    CHECK_EQ(p.devices[0].compute_units, 48u);
    CHECK_EQ(p.devices[0].gtt_bytes, 240 * kGiB);  // the one card left, for the one node left
  }
  CHECK(!p.unified);
  CHECK(p.links.empty());
  // no cpuinfo: falls back to cpu/online
  Tree u("online");
  u.put("sys/devices/system/cpu/online", "0-15,32-47\n");
  Profile q;
  detect_linux(q, u.str());
  CHECK_EQ(q.cpu_cores, 32u);
}

TEST(arch_names) {
  // minor / stepping as hex digits (LLVM names): gfx90a, gfx942, gfx1100
  Tree t("arch");
  gpu_node(t, 1, 90010, 8, 0);
  gpu_node(t, 2, 90402, 8, 0);
  gpu_node(t, 10, 110000, 8, 0);
  Profile p;
  detect_linux(p, t.str());
  CHECK_EQ(p.devices.size(), size_t(3));
  if (p.devices.size() == 3) {
    CHECK_EQ(p.devices[0].arch, std::string("gfx90a"));
    CHECK_EQ(p.devices[1].arch, std::string("gfx942"));
    CHECK_EQ(p.devices[2].arch, std::string("gfx1100"));    // node 10 after node 2
    CHECK_EQ(p.devices[2].id, std::string("kfd-node-10"));  // no location_id: the node id
  }
}

TEST(discrete_card) {
  Tree t("dgpu");
  meminfo(t, 65536000, 60000000);
  cpu_node(t, 0, 24);
  gpu_node(t, 1, 110000, 192, 128);
  drm(t, "renderD128", 31 * kGiB, 24 * kGiB, 256ull << 20, "samsung");
  Profile p;
  detect_linux(p, t.str());
  CHECK(!p.unified);
  CHECK_EQ(p.devices.size(), size_t(1));
  if (!p.devices.empty()) {
    CHECK_EQ(p.devices[0].memory_bytes, 24 * kGiB);
    CHECK_EQ(p.devices[0].gtt_bytes, 31 * kGiB);
    CHECK_EQ(p.devices[0].visible_bytes, uint64_t(256) << 20);
  }
  // an absent VRAM vendor is not evidence of an APU
  Tree n("novendor");
  gpu_node(n, 1, 110000, 192, 128);
  drm(n, "renderD128", 31 * kGiB, 24 * kGiB, 24 * kGiB, nullptr);
  Profile r;
  detect_linux(r, n.str());
  CHECK(!r.unified);
  CHECK(!r.devices.empty() && r.devices[0].memory_bytes == 24 * kGiB);
  // a fused CPU + GPU node is an APU whatever the DRM files say
  Tree f("fused");
  gpu_node(f, 0, 90002, 32, 0, 4);
  Profile q;
  detect_linux(q, f.str());
  CHECK(q.unified);
  CHECK(!q.notices.empty());  // no meminfo, no DRM
}

TEST(drm_matching) {
  // A has its render minor (renderD128 = card0, same PCI slot); B has none: B must not get A's card, and with two
  // cards unclaimed (card1, card2) no guess is made
  Tree t("drm");
  gpu_node(t, 1, 110000, 192, 128, 0, 0x0300);
  gpu_node(t, 2, 110000, 96, 0);
  drm(t, "renderD128", 31 * kGiB, 24 * kGiB, 24 * kGiB, "samsung", "0000:03:00.0");
  drm(t, "card0", 31 * kGiB, 24 * kGiB, 24 * kGiB, "samsung", "0000:03:00.0");
  drm(t, "card1", 15 * kGiB, 8 * kGiB, 8 * kGiB, "hynix", "0000:04:00.0");
  drm(t, "card2", 15 * kGiB, 12 * kGiB, 12 * kGiB, "hynix", "0000:05:00.0");
  Profile p;
  detect_linux(p, t.str());
  CHECK_EQ(p.devices.size(), size_t(2));
  if (p.devices.size() == 2) {
    CHECK_EQ(p.devices[0].memory_bytes, 24 * kGiB);
    CHECK_EQ(p.devices[1].memory_bytes, uint64_t(0));  // ambiguous: unknown, not A's 24 GiB
  }
  CHECK(has_notice(p, "no amdgpu DRM mem_info matched"));
  // only card1 left: B takes it (never card0, consumed by A through renderD128)
  Tree u("drm1");
  gpu_node(u, 1, 110000, 192, 128, 0, 0x0300);
  gpu_node(u, 2, 110000, 96, 0);
  drm(u, "renderD128", 31 * kGiB, 24 * kGiB, 24 * kGiB, "samsung", "0000:03:00.0");
  drm(u, "card0", 31 * kGiB, 24 * kGiB, 24 * kGiB, "samsung", "0000:03:00.0");
  drm(u, "card1", 15 * kGiB, 8 * kGiB, 8 * kGiB, "hynix", "0000:04:00.0");
  Profile q;
  detect_linux(q, u.str());
  CHECK(q.devices.size() == 2 && q.devices[1].memory_bytes == 8 * kGiB);
  // B with a PCI location: matched by slot even with two cards left
  Tree w("drmpci");
  gpu_node(w, 1, 110000, 192, 128, 0, 0x0300);
  gpu_node(w, 2, 110000, 96, 0, 0, 0x0500);
  drm(w, "renderD128", 31 * kGiB, 24 * kGiB, 24 * kGiB, "samsung", "0000:03:00.0");
  drm(w, "card1", 15 * kGiB, 8 * kGiB, 8 * kGiB, "hynix", "0000:04:00.0");
  drm(w, "card2", 15 * kGiB, 12 * kGiB, 12 * kGiB, "hynix", "0000:05:00.0");
  Profile r;
  detect_linux(r, w.str());
  CHECK(r.devices.size() == 2 && r.devices[1].memory_bytes == 12 * kGiB);
}

TEST(empty_root) {
  Tree t("empty");
  Profile p;
  detect_linux(p, t.str());
  CHECK_EQ(p.ram_bytes, uint64_t(0));
  CHECK_EQ(p.cpu_cores, 0u);
  CHECK(p.devices.empty());
  CHECK(p.links.empty());
  CHECK(p.nvme.empty());
  CHECK(!p.unified);
  CHECK(p.sources.empty());
  CHECK(p.notices.size() >= 3);  // meminfo, cpu, KFD
  Profile q;
  detect_linux(q, (t.root / "does-not-exist").string());
  CHECK(q.devices.empty());
  // directories where files are expected: notices, never a throw
  Tree b("broken");
  b.dir("proc/meminfo");
  b.dir("proc/cpuinfo");
  b.dir(std::string(kNodes) + "1/properties");
  b.dir("sys/class/drm/renderD128/device/mem_info_gtt_total");
  Profile r;
  bool threw = false;
  try {
    detect_linux(r, b.str());
  } catch (...) {
    threw = true;
  }
  CHECK(!threw);
  CHECK(has_notice(r, "meminfo"));
  CHECK(r.devices.empty());
}

TEST(hip_runtime_fake) {
  Tree t("hip");
  halo(t);
  Profile p;
  detect_linux(p, t.str());
  detect_hip(p, halo_hip());  // "0000:C5:00.0": joined to KFD's 0000:c5:00.0 whatever the case
  CHECK_EQ(p.devices.size(), size_t(1));
  if (!p.devices.empty()) {
    CHECK_EQ(p.devices[0].name, std::string("AMD Radeon 8060S Graphics"));
    CHECK_EQ(p.devices[0].id, std::string("0000:c5:00.0"));
    CHECK_EQ(p.devices[0].compute_units, 40u);      // KFD's count wins over the WGP-mode 20
    CHECK_EQ(p.devices[0].memory_bytes, kHaloGtt);  // hipMemGetInfo wins
    CHECK_EQ(p.devices[0].arch, std::string("gfx1151"));
  }
  CHECK(p.unified);
  CHECK_EQ(p.sources.size(), size_t(2));

  // no sysfs device: the runtime's devices are appended; not integrated -> not unified
  Profile q;
  HipApi two = halo_hip();
  two.device_count = [](int* n) {
    *n = 2;
    return 0;
  };
  two.props = [](int, HipApi::Props* pr) {
    pr->name = "dGPU";
    pr->cus = 60;
    return true;
  };
  detect_hip(q, two);
  CHECK_EQ(q.devices.size(), size_t(2));
  if (q.devices.size() == 2) {
    CHECK_EQ(q.devices[1].id, std::string("hip-1"));
    CHECK_EQ(q.devices[1].compute_units, 60u);
    CHECK_EQ(q.devices[1].api, std::string("hip"));
    CHECK(q.devices[1].arch.empty());
  }
  CHECK(!q.unified);

  // failures are notices, never throws
  Profile r;
  HipApi none;
  detect_hip(r, none);
  HipApi zero;
  zero.device_count = [](int* n) {
    *n = 0;
    return 100;  // hipErrorNoDevice
  };
  detect_hip(r, zero);
  HipApi noprops = zero;
  noprops.device_count = [](int* n) {
    *n = 1;
    return 0;
  };
  detect_hip(r, noprops);
  CHECK_EQ(r.devices.size(), size_t(1));
  CHECK(r.notices.size() >= 4);
  CHECK(!r.unified);
  HipApi throws = halo_hip();
  throws.props = [](int, HipApi::Props*) -> bool { throw std::runtime_error("boom"); };
  Profile x;
  bool threw = false;
  try {
    detect_hip(x, throws);
  } catch (...) {
    threw = true;
  }
  CHECK(!threw);
  CHECK(has_notice(x, "boom"));
  // the real loader: either a runtime (a node) or none (here); never throws
  const HipApi real = hip_runtime();
  (void)real.present();
}

TEST(hip_joins_by_pci_address) {
  // two KFD GPUs (a Halo at c5:00.0, a successor at 03:00.0); the runtime enumerates them in the other order
  Tree t("join");
  gpu_node(t, 1, 110501, 80, 128, 0, kHaloLoc);
  gpu_node(t, 2, 110600, 96, 129, 0, 0x0300);
  drm(t, "renderD128", kHaloGtt, kCarve, kCarve, nullptr, "0000:c5:00.0");
  drm(t, "renderD129", 200 * kGiB, kCarve, kCarve, nullptr, "0000:03:00.0");
  Profile p;
  detect_linux(p, t.str());
  detect_hip(p, runtime_of({"0000:03:00.0", "0000:c5:00.0"}));
  CHECK_EQ(p.devices.size(), size_t(2));
  for (const auto& d : p.devices) {
    if (d.name == "halo") CHECK(d.arch == "gfx1151" && d.memory_bytes == kHaloGtt);
    else CHECK(d.arch == "gfx1160" && d.memory_bytes == 200 * kGiB && d.compute_units == 48u);
  }
  // HIP_VISIBLE_DEVICES=1: only the successor is visible -> the Halo is dropped, the successor never gets gfx1151
  Profile q;
  detect_linux(q, t.str());
  detect_hip(q, runtime_of({"0000:03:00.0"}));
  CHECK_EQ(q.devices.size(), size_t(1));
  if (!q.devices.empty()) {
    CHECK_EQ(q.devices[0].arch, std::string("gfx1160"));
    CHECK_EQ(q.devices[0].name, std::string("successor"));
  }
  CHECK(has_notice(q, "not visible to the HIP runtime"));
  // a runtime device sysfs does not know: appended with no arch (a generic class), never a KFD neighbour's
  Profile r;
  detect_linux(r, t.str());
  detect_hip(r, runtime_of({"0000:07:00.0"}));
  CHECK_EQ(r.devices.size(), size_t(1));
  if (!r.devices.empty()) {
    CHECK(r.devices[0].arch.empty());
    CHECK_EQ(r.devices[0].id, std::string("0000:07:00.0"));
  }
  // KFD without PCI addresses: only one-to-one is joined
  Tree u("join-nopci");
  gpu_node(u, 1, 110501, 80, 0);
  gpu_node(u, 2, 110600, 96, 0);
  Profile s;
  detect_linux(s, u.str());
  detect_hip(s, runtime_of({"0000:c5:00.0", "0000:03:00.0"}));
  CHECK(has_notice(s, "not joined"));
  for (const auto& d : s.devices) CHECK(d.arch.empty());  // runtime values only: no KFD arch guessed
}

TEST(hip_restores_the_callers_device) {
  int current = 3;
  int sets = 0;
  bool fail = false;
  const std::function<int(int*)> get = [&](int* d) {
    *d = current;
    return 0;
  };
  const std::function<int(int)> set = [&](int d) {
    ++sets;
    if (d > 7) return 101;  // hipErrorInvalidDevice
    current = d;
    return 0;
  };
  const std::function<int(size_t*, size_t*)> mem = [&](size_t* f, size_t* t) {
    if (fail) return 1;
    *f = 1;
    *t = 2;
    return 0;
  };
  uint64_t f = 0, tot = 0;
  CHECK(hip_mem_info_restoring(get, set, mem, 0, &f, &tot));
  CHECK_EQ(tot, uint64_t(2));
  CHECK_EQ(current, 3);
  fail = true;
  CHECK(!hip_mem_info_restoring(get, set, mem, 1, &f, &tot));
  CHECK_EQ(current, 3);
  CHECK(!hip_mem_info_restoring(get, set, mem, 9, &f, &tot));
  CHECK_EQ(current, 3);
  const std::function<int(int*)> noget = [](int*) { return 1; };
  sets = 0;
  CHECK(!hip_mem_info_restoring(noget, set, mem, 0, &f, &tot));
  CHECK_EQ(sets, 0);  // cannot restore: the device is never changed
}

TEST(detect_fake_root) {
  Tree t("detect");
  halo(t);
  DetectOptions o;
  o.flags = SS_DEVPROF_NO_RUNTIME | SS_DEVPROF_NO_ENV | SS_DEVPROF_NO_CACHE;
  o.root = t.str();
  const HipApi hip = halo_hip();
  const Profile p = detect(o, &hip);
  CHECK(p.detected);
  CHECK(p.unified);
  CHECK_EQ(p.devices.size(), size_t(1));
  if (!p.devices.empty()) CHECK_EQ(p.devices[0].arch, std::string("gfx1151"));
  CHECK_EQ(p.cls, std::string("strix-halo-395"));
  CHECK(!p.generic);
  // the successor: a generic class and a notice, detected values kept, no Halo limits
  Tree g("detect-host-a");
  meminfo(g, 266000000, 250000000);
  gpu_node(g, 1, 110600, 96, 128, 0, kHaloLoc);
  drm(g, "renderD128", 240 * kGiB, kCarve, kCarve, nullptr, "0000:c5:00.0");
  o.root = g.str();
  const HipApi big = halo_hip("0000:c5:00.0", 240 * kGiB);
  const Profile q = detect(o, &big);
  CHECK(q.generic);
  CHECK(q.unified);
  CHECK(!q.devices.empty() && q.devices[0].arch == "gfx1160");
  CHECK(q.limits.empty());
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
