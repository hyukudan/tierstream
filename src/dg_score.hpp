// Device-gated decode mode, scored eviction (docs/design.md §3.8): the score profile format and the model identity it
// is bound to. Backend-free, so the CPU tests check it; the core (dg_core.cpp) owns the tables it fills.
//
// Profile (little endian, frozen): "TSSCPRF1", u32 version 1, u32 n_gated, u32 units[n_gated], identity[32],
// u32 prov_len (<= 4096), prov[prov_len] (UTF-8 provenance, reported, never interpreted), then per gated layer in
// order units^2 u32 transition counts row-major [r][e], then sha256[32] of every byte before it. Row sums are not
// stored (rebuilt at load in uint64). identity = sha256 of u32 n_gated, then per gated layer: u32 model layer index,
// u32 units, u64 unit_bytes, the descriptor's per-unit xxh64 list (a descriptor without unit hashes cannot load one).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "desc.hpp"

namespace ts {

constexpr uint32_t kDgScoreWindow = 16;      // unions in the frequency ring
constexpr uint32_t kDgScoreMaxUnits = 65536;  // units per gated layer (the integer bounds follow from it)

struct DgScoreIdLayer {
  uint32_t index = 0, units = 0;
  uint64_t unit_bytes = 0;
  const std::vector<uint64_t>* hash = nullptr;  // one xxh64 per unit
};
// The 32 raw identity bytes. SS_E_INVALID when a layer has no (or a wrong number of) unit hashes.
std::string dg_score_identity(const std::vector<DgScoreIdLayer>& layers);
std::string dg_score_identity(const Desc& d, const std::vector<int>& gated);

// sum_g units(g)^2, checked (SS_E_INVALID on overflow or a layer above kDgScoreMaxUnits units).
uint64_t dg_score_cells(const std::vector<uint32_t>& units);

// A checked profile: every rule of the format (exact size, magic, version, n_gated, units, identity, checksum).
// Throws SS_E_INVALID naming the first mismatch. cells points into data (the first gated layer's first cell).
struct DgProfileView {
  const unsigned char* cells = nullptr;
  std::string provenance;
};
DgProfileView dg_profile_check(const void* data, uint64_t len, const std::vector<uint32_t>& units,
                               const std::string& identity);
// The bytes of a profile (the fitting side, tests). cells[g] = units(g)^2 counts row-major.
std::string dg_profile_write(const std::vector<uint32_t>& units, const std::string& identity, const std::string& prov,
                             const std::vector<std::vector<uint32_t>>& cells);

}  // namespace ts
