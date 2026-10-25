// Device-gated window prefetch: the tierstream/hint-prior-v1 file (ss.h: Window prefetch), parsed and checked.

#include <cstring>
#include <string>
#include <vector>

#include "dg_ctx.hpp"
#include "errors.hpp"
#include "sha256.hpp"

namespace ts {

namespace {

constexpr char kMagic[8] = {'T', 'S', 'H', 'P', 'R', 'I', 'O', '1'};

void put32(std::string& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back(char(v >> (8 * i) & 0xff));
}
void put64(std::string& b, uint64_t v) {
  for (int i = 0; i < 8; ++i) b.push_back(char(v >> (8 * i) & 0xff));
}

// the 32 raw bytes of a hex digest
std::string raw(const std::string& hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) out.push_back(char(std::stoi(hex.substr(i, 2), nullptr, 16)));
  return out;
}

[[noreturn]] void bad(const std::string& why) { fail(SS_E_INVALID, "dg_prior (tierstream/hint-prior-v1): " + why); }

}  // namespace

std::string hint_prior_identity(const Desc& d, const std::vector<int>& gated) {
  std::string b;
  put32(b, uint32_t(gated.size()));
  for (int L : gated) {
    const Layer& l = d.layers[size_t(L)];
    put32(b, uint32_t(l.index));
    put32(b, uint32_t(l.n_units));
    put64(b, l.unit_bytes);
    for (const Part& p : l.parts) {
      put64(b, p.offset);
      put64(b, p.stride);
      put64(b, p.bytes);
    }
    for (uint64_t h : l.hash) put64(b, h);
  }
  return raw(sha256_hex(b.data(), b.size()));
}

DgPrior parse_hint_prior(const void* data, uint64_t len, const Desc& d, const std::vector<int>& gated) {
  const auto* p = static_cast<const unsigned char*>(data);
  uint64_t at = 0;
  auto need = [&](uint64_t n) {
    if (len < at || len - at < n) bad("truncated");
  };
  auto u32 = [&] {
    need(4);
    const uint32_t v = uint32_t(p[at]) | uint32_t(p[at + 1]) << 8 | uint32_t(p[at + 2]) << 16 | uint32_t(p[at + 3]) << 24;
    at += 4;
    return v;
  };
  need(8);
  if (std::memcmp(p, kMagic, 8) != 0) bad("bad magic");
  at = 8;
  if (u32() != 1) bad("unsupported version");
  const uint32_t G = u32();
  if (G != gated.size()) bad("n_gated " + std::to_string(G) + " != the context's " + std::to_string(gated.size()));
  for (uint32_t g = 0; g < G; ++g)
    if (u32() != uint32_t(d.layers[size_t(gated[g])].n_units)) bad("units of gated layer " + std::to_string(g) + " differ");
  need(32);
  if (std::memcmp(p + at, hint_prior_identity(d, gated).data(), 32) != 0)
    bad("identity mismatch (another model, layout or descriptor)");
  at += 32;
  const uint32_t prov = u32();
  if (prov > 4096) bad("provenance longer than 4096 bytes");
  need(prov);
  at += prov;
  DgPrior pr;
  pr.layers = u32();
  pr.k = u32();
  const uint32_t n = u32();
  if (pr.layers < 1 || pr.layers > G || pr.k < 1 || pr.k > 64) bad("layers / k out of range");
  const uint64_t lk = uint64_t(pr.layers) * pr.k, per = 4 + 4 * lk;
  if (len != at + per * n + 32) bad("the size does not match its header");
  if (std::memcmp(raw(sha256_hex(p, size_t(len - 32))).data(), p + len - 32, 32) != 0) bad("checksum mismatch");
  pr.unit.resize(size_t(n * lk));
  pr.p.resize(size_t(n * lk));
  pr.row.reserve(n);
  for (uint32_t i = 0; i < n; ++i) {
    const uint64_t r = at + per * i;
    const int32_t sym = int32_t(uint32_t(p[r]) | uint32_t(p[r + 1]) << 8 | uint32_t(p[r + 2]) << 16 |
                                uint32_t(p[r + 3]) << 24);
    if (!pr.row.emplace(sym, i).second) bad("symbol " + std::to_string(sym) + " appears twice");
    for (uint64_t j = 0; j < lk; ++j) {
      const uint32_t u = uint32_t(p[r + 4 + 2 * j]) | uint32_t(p[r + 5 + 2 * j]) << 8;
      const uint32_t q = uint32_t(p[r + 4 + 2 * lk + 2 * j]) | uint32_t(p[r + 5 + 2 * lk + 2 * j]) << 8;
      const uint32_t g = uint32_t(j / pr.k);
      if (u != 0xFFFFu && u >= uint32_t(d.layers[size_t(gated[g])].n_units))
        bad("unit " + std::to_string(u) + " out of gated layer " + std::to_string(g));
      pr.unit[size_t(i * lk + j)] = u == 0xFFFFu ? 0xFFFFFFFFu : u;
      pr.p[size_t(i * lk + j)] = float(q) / 65535.0f;
    }
  }
  return pr;
}

}  // namespace ts
