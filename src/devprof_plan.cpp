// Planner hooks of a device profile: the whole-device caps (ss_budget_in), single box vs pair with the layer split,
// and the prefill piece from the memory the plan leaves.
#include <algorithm>
#include <cstdio>
#include <string>

#include "devprof.hpp"
#include "errors.hpp"

namespace ts::devprof {

namespace {

uint64_t sub_sat(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }
std::string gib(uint64_t b) {
  char t[32];
  std::snprintf(t, sizeof t, "%.2f", double(b) / double(uint64_t(1) << 30));
  return t;
}

const Device& device_at(const Profile& p, uint32_t device, const char* what) {
  require(device < p.devices.size(), SS_E_INVALID,
          std::string(what) + ": device " + std::to_string(device) + " is past the profile's " +
              std::to_string(p.devices.size()) + " devices");
  return p.devices[device];
}

// A size the profile knows (known) and what the margins leave of it (bytes; 0 = exhausted). Unknown = no cap.
struct Cap {
  bool known = false;
  uint64_t bytes = 0;
};
Cap host_cap(const Profile& p) {
  return p.ram_available_bytes ? Cap{true, sub_sat(p.ram_available_bytes, p.host_floor_bytes)} : Cap{};
}
Cap vram_cap(const Profile& p, const Device& d) {
  return d.memory_bytes ? Cap{true, sub_sat(d.memory_bytes, p.device_reserve_bytes)} : Cap{};
}
Cap budget_cap(const Profile& p, const Device& d) {
  if (!p.unified) return vram_cap(p, d);
  const Cap h = host_cap(p), v = vram_cap(p, d);
  if (!h.known) return v;
  if (!v.known) return h;
  return Cap{true, std::min(h.bytes, v.bytes)};
}
[[noreturn]] void exhausted(const char* what, const char* side) {
  fail(SS_E_CAPACITY, std::string(what) + ": the profile's margins leave no " + side);
}

// a x b / c rounded half up for a <= c and b < 2^32 (no 128-bit type needed): a and c are halved together until c <
// 2^31, which keeps the ratio within 2^-30 relative - far below half a layer for any layer count < 2^20.
uint64_t share(uint64_t a, uint64_t b, uint64_t c) {
  while (c >= (uint64_t(1) << 31)) {
    a >>= 1;
    c >>= 1;
  }
  return c ? (2 * a * b + c) / (2 * c) : 0;
}

}  // namespace

uint64_t device_budget(const Profile& p, uint32_t device) {
  return budget_cap(p, device_at(p, device, "device profile budget")).bytes;
}

void budget(const Profile& p, uint32_t device, ss_budget_in& in) {
  const Device& d = device_at(p, device, "ss_devprof_budget");
  if (p.unified) {
    if (!in.ram_total_bytes) {
      const Cap c = budget_cap(p, d);
      if (c.known && !c.bytes) exhausted("ss_devprof_budget", "unified memory");
      in.ram_total_bytes = c.bytes;
    }
    in.unified = 1;
    return;
  }
  if (!in.vram_total_bytes) {
    const Cap c = vram_cap(p, d);
    if (c.known && !c.bytes) exhausted("ss_devprof_budget", "device memory");
    in.vram_total_bytes = c.bytes;
  }
  if (!in.ram_total_bytes) {
    const Cap c = host_cap(p);
    if (c.known && !c.bytes) exhausted("ss_devprof_budget", "host memory");
    in.ram_total_bytes = c.bytes;
  }
}

bool topology(const std::vector<const Profile*>& boxes, const ss_devprof_topology_in& in, ss_devprof_topology_out& out) {
  out = ss_devprof_topology_out{};
  out.struct_size = uint32_t(sizeof out);
  const uint32_t n = uint32_t(boxes.size());
  require(n >= 1 && n <= SS_DEVPROF_MAX_WORLD, SS_E_INVALID,
          "ss_devprof_topology: 1 to " + std::to_string(SS_DEVPROF_MAX_WORLD) + " profiles");
  require(in.layers > 0 && in.layers < (1u << 20), SS_E_INVALID, "ss_devprof_topology: layers must be 1 .. 2^20 - 1");
  require(in.pinned_world <= n, SS_E_INVALID,
          "ss_devprof_topology: pinned_world " + std::to_string(in.pinned_world) + " is above the " +
              std::to_string(n) + " profiles given");
  require(in.fixed_bytes <= UINT64_MAX - in.min_expert_bytes, SS_E_INVALID,
          "ss_devprof_topology: fixed_bytes + min_expert_bytes overflows");
  for (uint32_t b = 0; b < n; ++b) {
    require(boxes[b] != nullptr, SS_E_INVALID, "ss_devprof_topology: a NULL profile");
    out.budget_bytes[b] = boxes[b]->devices.empty() ? 0 : device_budget(*boxes[b], 0);
  }
  const uint64_t need = in.fixed_bytes + in.min_expert_bytes;
  // a pair needs a link whose rate was measured (discovery alone does not qualify one)
  bool linked = false;
  for (const auto& l : boxes[0]->links) linked = linked || (l.count > 0 && l.gbps_measured > 0);
  const auto holds = [&](uint32_t w) {  // exact sum: one past 2^64 holds any need
    uint64_t sum = 0;
    for (uint32_t b = 0; b < w; ++b) {
      if (out.budget_bytes[b] > UINT64_MAX - sum) return true;
      sum += out.budget_bytes[b];
    }
    return sum >= need;
  };
  uint32_t world = in.pinned_world;
  bool fits = false;
  if (world) {
    fits = holds(world) && (world == 1 || linked);
  } else {
    for (uint32_t w = 1; w <= (linked ? n : 1u); ++w)
      if (holds(w)) {
        world = w;
        fits = true;
        break;
      }
    if (!world) world = linked ? n : 1;
  }
  out.world = world;
  out.fits = fits ? 1 : 0;
  // Layer split by budget share: box b starts at the round-half-up of layers x (budgets before b) / (budgets of the
  // world); equal budgets give ceil(layers / 2) to box 0. The budgets are halved together until their sum fits.
  uint64_t bud[SS_DEVPROF_MAX_WORLD] = {};
  for (uint32_t b = 0; b < world; ++b) bud[b] = out.budget_bytes[b];
  for (;;) {
    uint64_t t = 0;
    bool over = false;
    for (uint32_t b = 0; b < world && !over; ++b) {
      over = bud[b] > UINT64_MAX - t;
      t += over ? 0 : bud[b];
    }
    if (!over) break;
    for (uint32_t b = 0; b < world; ++b) bud[b] >>= 1;
  }
  uint64_t total = 0, cum = 0;
  for (uint32_t b = 0; b < world; ++b) total += bud[b];
  out.first_layer[0] = 0;
  for (uint32_t b = 1; b < world; ++b) {
    cum += bud[b - 1];
    out.first_layer[b] = total ? uint32_t(share(cum, in.layers, total))
                               : uint32_t((uint64_t(in.layers) * b + world - 1) / world);
  }
  out.first_layer[world] = in.layers;
  std::string why = "need " + gib(need) + " GiB (fixed " + gib(in.fixed_bytes) + " + minimal experts " +
                    gib(in.min_expert_bytes) + "); budgets";
  for (uint32_t b = 0; b < n; ++b) why += (b ? ", " : " ") + boxes[b]->cls + " " + gib(out.budget_bytes[b]);
  why += in.pinned_world ? "; world pinned to " + std::to_string(world) : "; world " + std::to_string(world);
  if (!linked && n > 1) why += "; no measured link";
  if (!fits) why += ": does not fit";
  std::snprintf(out.why, sizeof out.why, "%s", why.c_str());
  return fits;
}

uint32_t piece_rows(const Profile& p, const ss_devprof_piece_in& in) {
  require(in.bytes_per_row > 0, SS_E_INVALID, "ss_devprof_piece_rows: bytes_per_row is 0");
  require(in.row_align > 0, SS_E_INVALID, "ss_devprof_piece_rows: row_align is 0");
  require(!in.max_rows || in.min_rows <= in.max_rows, SS_E_INVALID, "ss_devprof_piece_rows: min_rows > max_rows");
  // the largest aligned row count the caller accepts (and a uint32 holds)
  const uint64_t top = in.max_rows ? in.max_rows : UINT32_MAX;
  const uint64_t top_aligned = top - top % in.row_align;
  require(top_aligned > 0 && top_aligned >= in.min_rows, SS_E_INVALID,
          "ss_devprof_piece_rows: no multiple of row_align " + std::to_string(in.row_align) + " lies in [" +
              std::to_string(in.min_rows) + ", " + std::to_string(top) + "]");
  const Cap cap = budget_cap(p, device_at(p, in.device, "ss_devprof_piece_rows"));
  if (!cap.known) {  // nothing known about the device's memory: the caller's own bound
    require(in.max_rows > 0, SS_E_CAPACITY, "ss_devprof_piece_rows: the profile has no memory size for the device");
    return uint32_t(top_aligned);
  }
  const uint64_t room = sub_sat(sub_sat(cap.bytes, in.planned_bytes), in.piece_reserve_bytes);
  uint64_t rows = std::min(room / in.bytes_per_row, top_aligned);
  rows -= rows % in.row_align;
  if (rows == 0 || rows < in.min_rows)
    fail(SS_E_CAPACITY, "ss_devprof_piece_rows: " + gib(room) + " GiB left after the plan (" + gib(cap.bytes) +
                            " GiB budget - " + gib(in.planned_bytes) + " planned - " + gib(in.piece_reserve_bytes) +
                            " reserve) hold " + std::to_string(rows) + " aligned rows, below the minimum " +
                            std::to_string(std::max<uint32_t>(in.min_rows, 1)));
  return uint32_t(rows);
}

}  // namespace ts::devprof
