// Device profiles: Linux detection from procfs / sysfs (devprof.hpp detect_linux). The same files and fields as the
// engine's platform_probe (KFD topology, Thunderbolt domains, MemTotal) and glm_mem (amdgpu DRM mem_info_*,
// MemAvailable), read again here because tierstream does not depend on the engine. Every path is under root, so the
// tests run on a fake tree on any OS.
#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "devprof.hpp"

namespace ts::devprof {
namespace {

namespace fs = std::filesystem;

std::string trim(std::string v) {
  while (!v.empty() && std::isspace(static_cast<unsigned char>(v.back()))) v.pop_back();
  size_t i = 0;
  while (i < v.size() && std::isspace(static_cast<unsigned char>(v[i]))) ++i;
  return v.substr(i);
}

std::optional<std::string> first_line(const fs::path& p) {
  std::ifstream f(p);
  std::string line;
  if (!f || !std::getline(f, line)) return std::nullopt;
  line = trim(std::move(line));
  if (line.empty()) return std::nullopt;
  return line;
}

std::optional<uint64_t> parse_u64(std::string_view t) {
  uint64_t v = 0;
  const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
  if (r.ec != std::errc{} || r.ptr != t.data() + t.size()) return std::nullopt;
  return v;
}

std::optional<uint64_t> read_u64(const fs::path& p) {
  const auto line = first_line(p);
  return line ? parse_u64(*line) : std::nullopt;
}

bool all_digits(std::string_view v) {
  return !v.empty() && std::all_of(v.begin(), v.end(), [](char c) { return c >= '0' && c <= '9'; });
}

// Sorted directory entries (none when the directory is missing).
std::vector<fs::path> entries(const fs::path& dir) {
  std::vector<fs::path> r;
  std::error_code ec;
  for (fs::directory_iterator it(dir, ec), end; it != end && !ec; it.increment(ec)) r.push_back(it->path());
  std::sort(r.begin(), r.end());
  return r;
}

// KFD node properties: "key value" lines, unsigned values only.
std::map<std::string, uint64_t> properties(const fs::path& p) {
  std::ifstream f(p);
  std::map<std::string, uint64_t> r;
  std::string k, v;
  while (f >> k >> v)
    if (const auto x = parse_u64(v)) r[k] = *x;
  return r;
}

uint64_t prop(const std::map<std::string, uint64_t>& m, const char* k) {
  const auto it = m.find(k);
  return it == m.end() ? 0 : it->second;
}

// gfx_target_version = major * 10000 + minor * 100 + stepping; minor and stepping print as one hex digit each
// (110501 -> gfx1151, 90010 -> gfx90a), as the LLVM target names do.
std::string gfx_arch(uint64_t t) {
  static const char hex[] = "0123456789abcdef";
  const uint64_t major = t / 10000, minor = (t / 100) % 100, step = t % 100;
  std::string s = "gfx" + std::to_string(major);
  s += minor < 16 ? std::string(1, hex[minor]) : std::to_string(minor);
  s += step < 16 ? std::string(1, hex[step]) : std::to_string(step);
  return s;
}

// MemTotal / MemAvailable (kB lines) of /proc/meminfo; false when MemTotal is missing.
bool meminfo(const fs::path& root, Profile& p) {
  std::ifstream f(root / "proc/meminfo");
  if (!f) return false;
  std::string line;
  bool total = false;
  while (std::getline(f, line)) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    if (key != "MemTotal" && key != "MemAvailable") continue;
    std::string rest = trim(line.substr(colon + 1));
    const auto sp = rest.find(' ');
    const std::string unit = sp == std::string::npos ? "" : trim(rest.substr(sp + 1));
    const auto v = parse_u64(rest.substr(0, sp));
    if (!v || unit != "kB" || *v > UINT64_MAX / 1024) continue;
    if (key == "MemTotal") {
      p.ram_bytes = *v * 1024;
      total = true;
    } else {
      p.ram_available_bytes = *v * 1024;
    }
  }
  return total;
}

// Physical cores: distinct (physical id, core id) of /proc/cpuinfo; logical processors when the file carries no
// core ids; else the count of /sys/devices/system/cpu/online ("0-31,40").
uint32_t cpu_cores(const fs::path& root) {
  std::ifstream f(root / "proc/cpuinfo");
  if (f) {
    std::set<std::pair<std::string, std::string>> cores;
    uint32_t logical = 0;
    std::string line, phys, core;
    auto flush = [&] {
      if (!core.empty()) cores.insert({phys, core});
      phys.clear();
      core.clear();
    };
    while (std::getline(f, line)) {
      const auto colon = line.find(':');
      if (colon == std::string::npos) {
        if (trim(line).empty()) flush();
        continue;
      }
      const std::string k = trim(line.substr(0, colon)), v = trim(line.substr(colon + 1));
      if (k == "processor") {
        flush();
        ++logical;
      } else if (k == "physical id") {
        phys = v;
      } else if (k == "core id") {
        core = v;
      }
    }
    flush();
    if (!cores.empty()) return uint32_t(cores.size());
    if (logical) return logical;
  }
  const auto online = first_line(root / "sys/devices/system/cpu/online");
  if (!online) return 0;
  uint32_t n = 0;
  std::string_view s = *online;
  while (!s.empty()) {
    const auto comma = s.find(',');
    const std::string_view r = s.substr(0, comma);
    const auto dash = r.find('-');
    const auto a = parse_u64(r.substr(0, dash));
    const auto b = dash == std::string_view::npos ? a : parse_u64(r.substr(dash + 1));
    if (!a || !b || *b < *a) return 0;
    n += uint32_t(*b - *a + 1);
    s = comma == std::string_view::npos ? std::string_view{} : s.substr(comma + 1);
  }
  return n;
}

// The amdgpu device directory of a KFD node: /sys/class/drm/renderD<drm_render_minor>/device when that exists.
fs::path drm_device_of(const fs::path& root, uint64_t render_minor) {
  if (!render_minor) return {};
  const fs::path d = root / "sys/class/drm" / ("renderD" + std::to_string(render_minor)) / "device";
  std::error_code ec;
  return fs::exists(d / "mem_info_gtt_total", ec) ? d : fs::path{};
}

// card* (not connectors "card1-DP-1") with mem_info_gtt_total, in name order (glm_mem's find_drm_device).
std::vector<fs::path> drm_cards(const fs::path& root) {
  std::vector<fs::path> r;
  std::error_code ec;
  for (const auto& e : entries(root / "sys/class/drm")) {
    const std::string n = e.filename().string();
    if (n.rfind("card", 0) != 0 || n.find('-') != std::string::npos) continue;
    if (fs::exists(e / "device/mem_info_gtt_total", ec)) r.push_back(e / "device");
  }
  return r;
}

// "dddd:bb:dd.f" (lowercase hex, as hipDeviceGetPCIBusId prints after normalisation) from KFD's domain and
// location_id (bus << 8 | device << 3 | function).
std::string pci_name(uint64_t domain, uint64_t loc) {
  char b[32];
  std::snprintf(b, sizeof b, "%04x:%02x:%02x.%x", unsigned(domain & 0xffff), unsigned((loc >> 8) & 0xff),
                unsigned((loc >> 3) & 0x1f), unsigned(loc & 7));
  return b;
}

// The PCI slot of an amdgpu device directory (uevent PCI_SLOT_NAME), lowercase; "" when absent.
std::string pci_slot(const fs::path& dev) {
  std::ifstream f(dev / "uevent");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("PCI_SLOT_NAME=", 0) == 0) {
      std::string v = trim(line.substr(14));
      for (auto& c : v) c = char(std::tolower(static_cast<unsigned char>(c)));
      return v;
    }
  return {};
}

// One physical device behind several DRM names (card0 and renderD128 both point at the PCI device): its PCI slot,
// else its canonical path.
std::string drm_identity(const fs::path& dev) {
  const std::string slot = pci_slot(dev);
  if (!slot.empty()) return "pci:" + slot;
  std::error_code ec;
  const fs::path c = fs::canonical(dev, ec);
  return "path:" + (ec ? dev : c).generic_string();
}

// Thunderbolt / USB4 XDomain peers: routes "<domain>-<route>" that are not host routers ("N-0") and carry no
// router attribute ('authorized'), i.e. another host on the cable. Each cable to a peer is its own entry.
uint32_t usb4_peers(const fs::path& root) {
  uint32_t n = 0;
  std::error_code ec;
  for (const auto& e : entries(root / "sys/bus/thunderbolt/devices")) {
    const std::string name = e.filename().string();
    const auto dash = name.find('-');
    if (dash == std::string::npos || !all_digits(std::string_view(name).substr(0, dash))) continue;
    const std::string route = name.substr(dash + 1);
    if (route.empty() || route == "0" || route.find_first_not_of("0123456789abcdef") != std::string::npos) continue;
    if (fs::exists(e / "authorized", ec)) continue;  // a device router (dock, enclosure), not a host
    ++n;
  }
  return n;
}

}  // namespace

namespace {

void detect_linux_impl(Profile& p, const fs::path& root) {
  bool any = false;

  if (meminfo(root, p)) {
    any = true;
    if (!p.ram_available_bytes) p.notices.push_back("linux: no MemAvailable in /proc/meminfo");
  } else {
    p.notices.push_back("linux: no /proc/meminfo (host RAM unknown)");
  }
  if (const uint32_t c = cpu_cores(root)) {
    p.cpu_cores = c;
    any = true;
  } else {
    p.notices.push_back("linux: no /proc/cpuinfo or cpu/online (CPU cores unknown)");
  }

  // KFD topology: one device per node with SIMDs (the CPU nodes have none).
  const auto nodes = entries(root / "sys/class/kfd/kfd/topology/nodes");
  if (nodes.empty()) p.notices.push_back("linux: no KFD topology (no AMD GPU, or amdgpu not loaded)");
  // node directories are numbers: order them numerically ("10" after "9")
  std::vector<std::pair<uint64_t, fs::path>> numbered;
  for (const auto& n : nodes)
    if (const auto id = parse_u64(n.filename().string())) numbered.push_back({*id, n});
  std::sort(numbered.begin(), numbered.end());
  struct Gpu {
    Device d;
    std::string pci;   // "" = KFD gave no location
    fs::path drm;      // the amdgpu device directory, once matched
    bool apu = false;
  };
  std::vector<Gpu> gpus;
  std::set<std::string> consumed;  // DRM devices already given to a node (by drm_identity)
  for (const auto& [id, dir] : numbered) {
    const auto props = properties(dir / "properties");
    const uint64_t simds = prop(props, "simd_count"), target = prop(props, "gfx_target_version");
    if (!simds || !target) continue;
    any = true;
    Gpu g;
    Device& d = g.d;
    d.api = "hip";
    d.arch = gfx_arch(target);
    if (props.count("location_id")) g.pci = pci_name(prop(props, "domain"), prop(props, "location_id"));
    d.id = g.pci.empty() ? "kfd-node-" + std::to_string(id) : g.pci;
    if (const auto name = first_line(dir / "name")) d.name = *name;
    const uint64_t per_cu = prop(props, "simd_per_cu");
    if (per_cu && simds % per_cu == 0) d.compute_units = uint32_t(simds / per_cu);
    d.wave = uint32_t(prop(props, "wave_front_size"));
    d.lds_bytes = prop(props, "lds_size_in_kb") * 1024;
    // APU only on affirmative evidence: a fused CPU + GPU node. (A missing VRAM vendor is not evidence; the HIP
    // runtime's integrated attribute decides when it answers.)
    g.apu = prop(props, "cpu_cores_count") > 0;
    // the node's DRM device: by render minor first (consumed: no other node can fall back to the same device)
    g.drm = drm_device_of(root, prop(props, "drm_render_minor"));
    if (!g.drm.empty()) consumed.insert(drm_identity(g.drm));
    gpus.push_back(std::move(g));
  }
  // then by PCI slot, then the one card left for the one node left; anything else stays unknown
  const auto cards = drm_cards(root);
  for (auto& g : gpus) {
    if (!g.drm.empty() || g.pci.empty()) continue;
    for (const auto& c : cards)
      if (pci_slot(c) == g.pci && !consumed.count(drm_identity(c))) {
        g.drm = c;
        consumed.insert(drm_identity(c));
        break;
      }
  }
  std::vector<const fs::path*> left;
  for (const auto& c : cards)
    if (!consumed.count(drm_identity(c))) left.push_back(&c);
  size_t unmatched = 0;
  for (const auto& g : gpus) unmatched += g.drm.empty() ? 1 : 0;
  for (auto& g : gpus) {
    if (g.drm.empty() && unmatched == 1 && left.size() == 1) g.drm = *left[0];
    Device& d = g.d;
    if (!g.drm.empty()) {
      const uint64_t gtt = read_u64(g.drm / "mem_info_gtt_total").value_or(0);
      const auto vram = read_u64(g.drm / "mem_info_vram_total");
      d.gtt_bytes = gtt;
      d.visible_bytes = read_u64(g.drm / "mem_info_vis_vram_total").value_or(vram.value_or(0));
      d.memory_bytes = g.apu ? gtt + vram.value_or(0) : vram.value_or(0);
      if (!vram) p.notices.push_back("linux: " + d.id + ": no mem_info_vram_total");
      if (!g.apu)
        p.notices.push_back("linux: " + d.id + " (" + d.arch +
                            "): integrated or discrete not known from sysfs; memory = VRAM until the HIP runtime "
                            "answers");
    } else {
      p.notices.push_back("linux: " + d.id + " (" + d.arch + "): no amdgpu DRM mem_info matched (device memory unknown)");
    }
    if (g.apu) p.unified = true;
    p.devices.push_back(std::move(d));
  }

  const uint32_t peers = usb4_peers(root);
  if (peers) {
    Link l;
    l.kind = "usb4";
    l.count = peers;
    p.links.push_back(l);
  }

  for (const auto& e : entries(root / "sys/class/nvme")) {
    const std::string n = e.filename().string();
    if (n.rfind("nvme", 0) != 0 || !all_digits(std::string_view(n).substr(4))) continue;
    Nvme d;
    d.path = n;
    d.model = first_line(e / "model").value_or("");
    p.nvme.push_back(std::move(d));
    any = true;
  }

  if (any) p.sources.push_back("linux-sysfs");
}

}  // namespace

void detect_linux(Profile& p, const std::string& root_text) {
  try {
    detect_linux_impl(p, fs::path(root_text.empty() ? std::string("/") : root_text));
  } catch (const std::exception& e) {
    p.notices.push_back(std::string("linux: detection stopped: ") + e.what());
  }
}

}  // namespace ts::devprof
