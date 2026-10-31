// Device classes: what detection cannot know or should not re-measure every start (class name, conservative margins,
// measured bandwidths with their evidence). profiles/<class>.json in the repo are this table written out (CTest
// devprof checks them byte for byte).
#include <algorithm>
#include <cctype>
#include <string>

#include "errors.hpp"

#include "devprof.hpp"

namespace ts::devprof {

namespace {

constexpr uint64_t kGiB = uint64_t(1) << 30;

// Today's tierstream defaults, so a classed profile reproduces every current plan: the recipes' host free-RAM floor
// (recipe.hpp host_free_ram_floor_gib = 8) and ss_budget's device margin (margin_gib = 2).
constexpr uint64_t kHostFloor = 8 * kGiB;
constexpr uint64_t kDeviceReserve = 2 * kGiB;

bool icontains(const std::string& hay, const char* needle) {
  std::string h = hay, n = needle;
  for (auto& c : h) c = char(std::tolower(static_cast<unsigned char>(c)));
  for (auto& c : n) c = char(std::tolower(static_cast<unsigned char>(c)));
  return h.find(n) != std::string::npos;
}

}  // namespace

const std::vector<ClassRow>& classes() {
  static const std::vector<ClassRow> t = {
      // Strix Halo (Ryzen AI Max+ 395, Radeon 8060S): 40 CUs, unified LPDDR5X; device memory = the GTT pool. mem_gbps:
      // rocm-bandwidth-test on both nodes, 244 of 246.8 GB/s peak with pp_dpm smoothed.
      // Limits = today's engine values on this class: DS pair auto piece cap
      // (deepseek41_serve_sched.hpp), Qwen's UMA ceiling (qwen_effective_config.hpp), gfx1151 persistent blocks
      // (hip_hyper_mix_launcher.hip kPersistBlocks).
      // Qualified on the 128 GB Halo: GTT + carve-out between 96 and 128 GiB (117.66 GiB with kho=off). Any other
      // gfx1151 size is generic-hip until qualified.
      {"strix-halo-395", "hip", "gfx1151", "", 96 * kGiB, 128 * kGiB, true, kHostFloor, kDeviceReserve, 244.0,
       "measured memory read bandwidth, both nodes (244 / 246.8 GB/s)",
       {{"max_piece_rows", 16384}, {"uma_ceiling_bytes", 120 * kGiB}, {"persist_blocks", 40}}},
      // The PC's DS RTX defaults (cuda_ds41_prefill.hpp long-route VRAM cap and reserve, cuda_ds41_replica.cu host
      // margin and pinned ceiling, prefill_service_main.cu commit floor).
      {"rtx-pro-6000-blackwell", "cuda", "sm_120", "RTX PRO 6000", 90 * kGiB, 0, false, kHostFloor, kDeviceReserve, 0,
       "",
       {{"long_vram_cap_bytes", 38 * kGiB},
        {"vram_reserve_bytes", 1 * kGiB},
        {"replica_host_margin_bytes", 6 * kGiB},
        {"max_pinned_bytes", 68 * kGiB},
        {"service_host_commit_floor_bytes", 20 * kGiB}}},
      {"rtx-4090", "cuda", "sm_89", "4090", 20 * kGiB, 26 * kGiB, false, kHostFloor, kDeviceReserve, 0, "", {}},
      // No known class: the detected values only, today's margins.
      {"generic-hip", "hip", "", "", 0, 0, false, kHostFloor, kDeviceReserve, 0, "", {}},
      {"generic-cuda", "cuda", "", "", 0, 0, false, kHostFloor, kDeviceReserve, 0, "", {}},
      {"generic-cpu", "cpu", "", "", 0, 0, false, kHostFloor, 0, 0, "", {}},
  };
  return t;
}

Json class_json(const ClassRow& c) {
  Json j = Json::object();
  j["schema"] = Json::string("strix/device-class-v1");
  j["class"] = Json::string(c.name);
  Json m = Json::object();
  m["api"] = Json::string(c.api);
  m["arch"] = Json::string(c.arch);
  m["name_contains"] = Json::string(c.name_part);
  m["memory_min_bytes"] = Json::number(c.min_memory_bytes);
  m["memory_max_bytes"] = Json::number(c.max_memory_bytes);
  j["match"] = m;
  j["unified"] = Json::boolean(c.unified);
  Json g = Json::object();
  g["host_floor_bytes"] = Json::number(c.host_floor_bytes);
  g["device_reserve_bytes"] = Json::number(c.device_reserve_bytes);
  j["margins"] = g;
  j["mem_gbps"] = c.mem_gbps > 0 ? Json::number(c.mem_gbps) : Json{};
  j["evidence"] = Json::string(c.evidence);
  Json lim = Json::object();
  for (const auto& kv : c.limits) lim[kv.first] = Json::number(kv.second);
  j["limits"] = lim;
  return j;
}

namespace {

bool is_generic(const ClassRow& c) { return std::string(c.name).rfind("generic-", 0) == 0; }

// gfx1151 exists only as the Strix Halo APU: its device memory is the GTT plus the carve-out even when sysfs could not
// say so (the runtime's hipMemGetInfo total is the same number).
uint64_t class_memory(const ClassRow& c, const Device& d) {
  return c.unified ? std::max(d.memory_bytes, d.gtt_bytes + d.visible_bytes) : d.memory_bytes;
}

bool row_matches(const ClassRow& c, const Device* d, const std::string& api) {
  if (api != c.api) return false;
  if (is_generic(c)) return true;
  if (!d) return false;
  if (*c.arch && d->arch != c.arch) return false;
  if (*c.name_part && !icontains(d->name, c.name_part)) return false;
  const uint64_t m = class_memory(c, *d);
  if (c.min_memory_bytes && m < c.min_memory_bytes) return false;
  if (c.max_memory_bytes && m > c.max_memory_bytes) return false;
  return true;
}

std::string owner(const ClassRow& c) { return std::string("class:") + c.name + (*c.evidence ? std::string(" ") + c.evidence : ""); }

}  // namespace

void apply_class(Profile& p, const std::string& forced) {
  const Device* d = p.devices.empty() ? nullptr : &p.devices[0];
  const std::string api = d ? d->api : "cpu";
  const ClassRow* hit = nullptr;
  for (const auto& c : classes()) {
    if (forced.empty() ? row_matches(c, d, api) : forced == c.name) {
      hit = &c;
      break;
    }
  }
  require(hit != nullptr, SS_E_RECIPE, "device profile: no device class named '" + forced + "'");
  p.cls = hit->name;
  p.generic = is_generic(*hit);
  if (p.generic && d)
    p.notices.push_back("no known device class for " + d->api + " '" + d->name + "' (" +
                        (d->arch.empty() ? std::string("arch unknown") : d->arch) +
                        "): detected values and today's margins only; kernel sets qualified on other archs are not "
                        "assumed");
  // What the class fills is marked in "evidence" as class-owned, so a later identity change can take it back.
  const std::string who = owner(*hit);
  if (!p.host_floor_bytes && hit->host_floor_bytes) {
    p.host_floor_bytes = hit->host_floor_bytes;
    p.evidence["margins.host_floor_bytes"] = Json::string(who);
  }
  if (!p.device_reserve_bytes && hit->device_reserve_bytes) {
    p.device_reserve_bytes = hit->device_reserve_bytes;
    p.evidence["margins.device_reserve_bytes"] = Json::string(who);
  }
  if (hit->unified && !p.unified) {
    p.unified = true;
    p.evidence["host.unified"] = Json::string(who);
  }
  for (const auto& kv : hit->limits)
    if (p.limits.emplace(kv.first, kv.second).second) p.evidence["limits." + kv.first] = Json::string(who);
  if (!d) return;
  for (size_t i = 0; i < p.devices.size(); ++i) {
    Device& x = p.devices[i];
    if (!row_matches(*hit, &x, x.api) || p.generic) continue;  // only devices of the class itself
    const std::string at = "devices[" + std::to_string(i) + "].";
    if (hit->unified && x.gtt_bytes + x.visible_bytes > x.memory_bytes) {
      x.memory_bytes = x.gtt_bytes + x.visible_bytes;
      p.evidence[at + "memory_bytes"] = Json::string(who);
    }
    if (hit->mem_gbps > 0 && x.mem_gbps == 0) {
      x.mem_gbps = hit->mem_gbps;
      p.evidence[at + "mem_gbps"] = Json::string(who);
    }
  }
}

void unclass(Profile& p, const std::vector<std::string>& keep) {
  if (!p.evidence.is_object()) return;
  std::vector<std::string> drop;
  for (const auto& kv : p.evidence.obj)
    if (kv.second.is_string() && kv.second.text.rfind("class:", 0) == 0 &&
        std::find(keep.begin(), keep.end(), kv.first) == keep.end())
      drop.push_back(kv.first);
  for (const auto& k : drop) {
    p.evidence.obj.erase(k);
    if (k == "margins.host_floor_bytes") p.host_floor_bytes = 0;
    else if (k == "margins.device_reserve_bytes") p.device_reserve_bytes = 0;
    else if (k == "host.unified") p.unified = false;
    else if (k.rfind("limits.", 0) == 0) p.limits.erase(k.substr(7));
    else if (k.rfind("devices[", 0) == 0) {
      const size_t close = k.find(']');
      const size_t i = size_t(std::stoul(k.substr(8, close - 8)));
      const std::string f = k.substr(close + 2);
      if (i < p.devices.size()) {
        if (f == "mem_gbps") p.devices[i].mem_gbps = 0;
        // memory_bytes: the class only completed GTT + carve-out; leave the larger detected value
      }
    }
  }
}

}  // namespace ts::devprof
