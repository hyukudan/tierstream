// A synthetic gated model for the device-gated tests: G MoE layers of U experts, each expert three parts
// (gate / up / down, multiples of 256 bytes) stored like GGUF stacked tensors (one tensor per part per layer, at an
// unaligned file offset, so the reads go through the bounce buffer), and the descriptor / tiers / budget around it.
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "fixture.hpp"
#include "strix_stream/ss.h"
#include "xxh64.hpp"

namespace sstest {

struct DgModel {
  int G = 6, U = 64;
  uint64_t gate = 2048, up = 2048, down = 4096;  // unit = 8 KiB
  std::filesystem::path dir, file;
  std::string json;
  uint64_t base = 1000;  // unaligned start of the first tensor

  uint64_t unit() const { return gate + up + down; }
  uint64_t tensor_off(int L, int part) const {
    const uint64_t per_layer = uint64_t(U) * unit();
    const uint64_t part_off = part == 0 ? 0 : part == 1 ? uint64_t(U) * gate : uint64_t(U) * (gate + up);
    return base + uint64_t(L) * per_layer + part_off;
  }
  uint64_t part_bytes(int part) const { return part == 0 ? gate : part == 1 ? up : down; }
  // The engine bytes of unit (L, u): its three parts back to back.
  std::vector<uint8_t> bytes(int L, int u) const {
    std::vector<uint8_t> v;
    for (int p = 0; p < 3; ++p) {
      const uint64_t off = tensor_off(L, p) + uint64_t(u) * part_bytes(p);
      for (uint64_t i = 0; i < part_bytes(p); ++i) v.push_back(byte_at(3, off + i));
    }
    return v;
  }

  DgModel(const std::string& tag, int g = 6, int u = 64) : G(g), U(u) {
    dir = Fixture::temp_root() / ("ss-test-dg-" + tag);
    std::filesystem::create_directories(dir);
    file = dir / "experts.bin";
    const uint64_t size = base + uint64_t(G) * uint64_t(U) * unit() + 512;
    if (!std::filesystem::exists(file) || std::filesystem::file_size(file) != size) {
      std::vector<char> buf(static_cast<size_t>(size));
      for (uint64_t i = 0; i < size; ++i) buf[size_t(i)] = char(byte_at(3, i));
      std::ofstream f(file, std::ios::binary);
      f.write(buf.data(), std::streamsize(size));
    }
    json = R"({"schema":"strix-stream/desc-v1","engine":"dg-test","unit_align":256,"sources":[{"path":")" +
           file.generic_string() + R"(","align":4096}],"layers":[)";
    const char* names[3] = {"gate", "up", "down"};
    for (int L = 0; L < G; ++L) {
      json += std::string(L ? "," : "") + R"({"index":)" + std::to_string(L) + R"(,"kind":"moe","n_units":)" +
              std::to_string(U) + R"(,"top_k":4,"parts":[)";
      for (int p = 0; p < 3; ++p)
        json += std::string(p ? "," : "") + R"({"name":")" + names[p] + R"(","format":"Q4_K","offset":)" +
                std::to_string(tensor_off(L, p)) + R"(,"stride":)" + std::to_string(part_bytes(p)) +
                R"(,"bytes":)" + std::to_string(part_bytes(p)) + "}";
      json += "]}";
    }
    json += "]}";
  }

  // The xxh64 of each unit's engine bytes of layer L, and the descriptor with them (a score profile is bound to them).
  std::vector<uint64_t> hashes(int L) const {
    std::vector<uint64_t> h;
    for (int u = 0; u < U; ++u) {
      const std::vector<uint8_t> b = bytes(L, u);
      h.push_back(ts::xxh64(b.data(), b.size()));
    }
    return h;
  }
  std::string json_hashed() const {
    std::string js = json;
    for (int L = 0; L < G; ++L) {
      const std::string key = R"({"index":)" + std::to_string(L) + R"(,"kind":"moe")";
      std::string h = R"(,"hash":[)";
      char buf[20];
      const std::vector<uint64_t> v = hashes(L);
      for (size_t i = 0; i < v.size(); ++i) {
        std::snprintf(buf, sizeof buf, "%016llx", (unsigned long long)v[i]);
        h += std::string(i ? "," : "") + "\"" + buf + "\"";
      }
      h += "]";
      const size_t at = js.find(key);
      js.insert(at + key.size(), h);
    }
    return js;
  }

  // Tiers: the first `resident` units of each layer resident, the rest pinned.
  std::vector<uint8_t> tiers(int resident_per_layer) const {
    std::vector<uint8_t> t;
    for (int L = 0; L < G; ++L)
      for (int u = 0; u < U; ++u) t.push_back(uint8_t(u < resident_per_layer ? SS_TIER_RESIDENT : SS_TIER_PINNED));
    return t;
  }
};

// A budget the gated open accepts for this model: exclusive, `dyn` spare slots per layer, a ring of 2 x max_ids + 1,
// spare blocks asked for every dynamic slot.
inline void dg_budget(ss_budget& b, const DgModel& m, uint32_t dyn, uint32_t max_ids) {
  ss_budget_defaults(&b);
  b.device_bytes = uint64_t(1) << 30;
  b.margin_gib = 1.0 / 1024;
  b.margin_fraction = 0;
  b.host_pinned_bytes = uint64_t(m.G) * uint64_t(m.U) * 12288 + (uint64_t(64) << 20);
  b.tier_mode = SS_TIER_MODE_EXCLUSIVE;
  b.dynamic_slots_per_layer = dyn;
  b.ring_slots = 2 * max_ids + 1;
  b.host_spare_units = dyn * uint32_t(m.G);
}

inline void dg_options(ss_options& o, uint32_t max_ids, ss_backend_kind be) {
  ss_options_defaults(&o);
  o.backend = be;
  o.forward_mode = SS_FORWARD_DEVICE_GATED;
  o.pin_planned = 0;
  o.verify_hash = 1;
  o.dg_max_ids = max_ids;
  o.staging_bytes = 64 * 1024;
}

}  // namespace tstest
