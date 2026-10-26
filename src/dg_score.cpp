// Device-gated decode mode, scored eviction: the profile format and its identity (dg_score.hpp).

#include "dg_score.hpp"

#include <cstring>

#include "errors.hpp"
#include "sha256.hpp"

namespace ts {

namespace {

constexpr char kMagic[8] = {'T', 'S', 'S', 'C', 'P', 'R', 'F', '1'};

void put32(std::string& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(char(v >> (8 * i) & 0xff));
}
void put64(std::string& b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(char(v >> (8 * i) & 0xff));
}
uint32_t get32(const unsigned char* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}

std::string raw(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) out.push_back(char(std::stoi(hex.substr(i, 2), nullptr, 16)));
  return out;
}

[[noreturn]] void bad(const std::string& why) { fail(SS_E_INVALID, "dg_score_profile (TSSCPRF1): " + why); }

uint64_t add_ck(uint64_t a, uint64_t b) {
  if (a > UINT64_MAX - b) bad("size overflows");
  return a + b;
}
uint64_t mul_ck(uint64_t a, uint64_t b) {
  if (b && a > UINT64_MAX / b) bad("size overflows");
  return a * b;
}

}  // namespace

std::string dg_score_identity(const std::vector<DgScoreIdLayer>& layers) {
  std::string b;
  put32(b, uint32_t(layers.size()));
  for (size_t g = 0; g < layers.size(); ++g) {
    const DgScoreIdLayer& l = layers[g];
    if (!l.hash || l.hash->size() != l.units)
      fail(SS_E_INVALID, "dg_score_profile: gated layer " + std::to_string(g) +
                             " has no per-unit hashes in the descriptor (a profile is bound to them)");
    put32(b, l.index);
    put32(b, l.units);
    put64(b, l.unit_bytes);
    for (uint64_t h : *l.hash) put64(b, h);
  }
  return raw(sha256_hex(b.data(), b.size()));
}

std::string dg_score_identity(const Desc& d, const std::vector<int>& gated) {
  std::vector<DgScoreIdLayer> v;
  for (int L : gated) {
    const Layer& l = d.layers[size_t(L)];
    v.push_back({uint32_t(l.index), uint32_t(l.n_units), l.unit_bytes, l.hash.empty() ? nullptr : &l.hash});
  }
  return dg_score_identity(v);
}

uint64_t dg_score_cells(const std::vector<uint32_t>& units) {
  uint64_t n = 0;
  for (size_t g = 0; g < units.size(); ++g) {
    require(units[g] <= kDgScoreMaxUnits, SS_E_INVALID,
            "device-gated: scored eviction needs at most 65536 units per gated layer (gated layer " +
                std::to_string(g) + " has " + std::to_string(units[g]) + ")");
    n = add_ck(n, mul_ck(units[g], units[g]));
  }
  return n;
}

DgProfileView dg_profile_check(const void* data, uint64_t len, const std::vector<uint32_t>& units,
                               const std::string& identity) {
  if (!data) bad("null data");
  const auto* p = static_cast<const unsigned char*>(data);
  const uint64_t G = units.size();
  const uint64_t fixed = add_ck(add_ck(8 + 4 + 4, mul_ck(4, G)), 32 + 4);  // through prov_len
  if (len < fixed) bad("truncated header (" + std::to_string(len) + " bytes)");
  if (std::memcmp(p, kMagic, 8) != 0) bad("bad magic");
  if (get32(p + 8) != 1) bad("unsupported version " + std::to_string(get32(p + 8)));
  if (get32(p + 12) != G)
    bad("n_gated " + std::to_string(get32(p + 12)) + " != the context's " + std::to_string(G));
  for (uint64_t g = 0; g < G; ++g)
    if (get32(p + 16 + 4 * g) != units[size_t(g)])
      bad("units of gated layer " + std::to_string(g) + " are " + std::to_string(get32(p + 16 + 4 * g)) +
          ", the context has " + std::to_string(units[size_t(g)]));
  const uint64_t at_id = 16 + 4 * G;
  const uint32_t prov = get32(p + at_id + 32);
  if (prov > 4096) bad("provenance longer than 4096 bytes");
  const uint64_t header = fixed + prov;
  const uint64_t want = add_ck(add_ck(header, mul_ck(4, dg_score_cells(units))), 32);
  if (len != want) bad("size " + std::to_string(len) + " != " + std::to_string(want) + " (header + cells + checksum)");
  if (identity.size() != 32 || std::memcmp(p + at_id, identity.data(), 32) != 0)
    bad("identity mismatch (another model, layer set or descriptor)");
  if (std::memcmp(raw(sha256_hex(p, size_t(len - 32))).data(), p + len - 32, 32) != 0) bad("checksum mismatch");
  DgProfileView v;
  v.provenance.assign(reinterpret_cast<const char*>(p + fixed), prov);
  v.cells = p + header;
  return v;
}

std::string dg_profile_write(const std::vector<uint32_t>& units, const std::string& identity, const std::string& prov,
                             const std::vector<std::vector<uint32_t>>& cells) {
  require(identity.size() == 32 && prov.size() <= 4096 && cells.size() == units.size(), SS_E_INVALID,
          "dg_profile_write: bad arguments");
  std::string b(kMagic, 8);
  put32(b, 1);
  put32(b, uint32_t(units.size()));
  for (uint32_t u : units) put32(b, u);
  b += identity;
  put32(b, uint32_t(prov.size()));
  b += prov;
  for (size_t g = 0; g < units.size(); ++g) {
    require(cells[g].size() == uint64_t(units[g]) * units[g], SS_E_INVALID, "dg_profile_write: cells != units^2");
    for (uint32_t c : cells[g]) put32(b, c);
  }
  b += raw(sha256_hex(b.data(), b.size()));
  return b;
}

}  // namespace ts
