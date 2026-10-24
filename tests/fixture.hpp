// Synthetic model on disk for the runtime tests: a dense layer, a GGUF-like routed layer (three stacked tensors at
// unaligned offsets), a record-file routed layer (one aligned record per expert) and a large blob layer.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "xxh64.hpp"

namespace sstest {

inline uint8_t byte_at(uint64_t file_id, uint64_t i) {
  uint64_t x = (i + 1) * 0x9E3779B97F4A7C15ULL ^ (file_id + 7) * 0xC2B2AE3D27D4EB4FULL;
  x ^= x >> 29;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 32;
  return uint8_t(x);
}

struct Fixture {
  std::filesystem::path dir;
  std::string json;
  // expected[layer][unit] = engine bytes
  std::vector<std::vector<std::vector<uint8_t>>> expected;
  int n_layers = 5;  // layer 4 = a second GGUF-like layer with layer 1's shape (shares its ring)
  static constexpr int kE1 = 16, kE2 = 8;
  static constexpr uint64_t kGate = 3000, kUp = 3000, kDown = 5000;  // layer 1 parts (unaligned)
  static constexpr uint64_t kRec = 8192;                              // layer 2 records (aligned)
  static constexpr uint64_t kBlob = (1u << 20) + 17;                  // layer 3 blob

  static std::filesystem::path temp_root() {
    if (const char* e = std::getenv("SS_TEST_DIR")) return e;
    return std::filesystem::temp_directory_path();
  }

  void write_file(const std::filesystem::path& p, uint64_t id, uint64_t size) {
    std::vector<char> buf(static_cast<size_t>(size));
    for (uint64_t i = 0; i < size; ++i) buf[size_t(i)] = char(byte_at(id, i));
    std::ofstream f(p, std::ios::binary);
    f.write(buf.data(), std::streamsize(size));
  }

  std::vector<uint8_t> slice(uint64_t id, uint64_t off, uint64_t n) {
    std::vector<uint8_t> v(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) v[size_t(i)] = byte_at(id, off + i);
    return v;
  }

  static std::string hex(uint64_t v) {
    char b[17];
    std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
    return b;
  }

  explicit Fixture(const std::string& tag, bool corrupt_hash = false) {
    dir = temp_root() / ("ss-test-" + tag);
    std::filesystem::create_directories(dir);
    const uint64_t g0 = 1000, u0 = g0 + kE1 * kGate, d0 = u0 + kE1 * kUp;
    const uint64_t rec0 = 192 * 1024;  // aligned records after the stacked tensors (which end at 177,000)
    const uint64_t size0 = rec0 + kE2 * kRec + 512;
    write_file(dir / "experts.bin", 0, size0);
    write_file(dir / "blob.bin", 1, 4096 + kBlob + 100);

    expected.assign(size_t(n_layers), {});
    expected[1].resize(kE1);
    for (int e = 0; e < kE1; ++e) {
      auto& v = expected[1][size_t(e)];
      auto a = slice(0, g0 + e * kGate, kGate), b = slice(0, u0 + e * kUp, kUp), c = slice(0, d0 + e * kDown, kDown);
      v.insert(v.end(), a.begin(), a.end());
      v.insert(v.end(), b.begin(), b.end());
      v.insert(v.end(), c.begin(), c.end());
    }
    expected[2].resize(kE2);
    for (int e = 0; e < kE2; ++e) expected[2][size_t(e)] = slice(0, rec0 + e * kRec, kRec);
    expected[3].resize(1);
    expected[3][0] = slice(1, 4096, kBlob);
    expected[4] = expected[1];

    auto hashes = [&](int L) {
      std::string s = "[";
      for (size_t u = 0; u < expected[size_t(L)].size(); ++u) {
        uint64_t h = ss::xxh64(expected[size_t(L)][u].data(), expected[size_t(L)][u].size());
        if (corrupt_hash && (L == 1 || L == 4) && u == 3) h ^= 1;
        s += (u ? ",\"" : "\"") + hex(h) + "\"";
      }
      return s + "]";
    };
    auto p = [](const std::filesystem::path& x) {
      std::string s = x.generic_string();
      return s;
    };
    json = R"({"schema":"strix-stream/desc-v1","model":"synthetic","engine":"test",
      "fixed_device":[{"name":"embed","bytes":65536}],
      "sources":[{"name":"experts","path":")" + p(dir / "experts.bin") + R"(","align":4096},
                 {"name":"blob","path":")" + p(dir / "blob.bin") + R"(","align":4096}],
      "order":"ranked","tiers":"pinned|disk","unit_align":256,
      "layers":[
        {"index":0,"kind":"dense","fixed_bytes":100000},
        {"index":1,"kind":"moe","fixed_bytes":20000,"n_units":16,"top_k":4,
         "router":{"tensor":"blk.1.ffn_gate_inp.weight","format":"F32","score":"sigmoid","rows":16,"cols":64},
         "parts":[{"name":"gate","format":"IQ2_XXS","source":0,"offset":)" + std::to_string(g0) +
           R"(,"stride":3000,"bytes":3000},
                  {"name":"up","format":"IQ2_XXS","source":0,"offset":)" + std::to_string(u0) +
           R"(,"stride":3000,"bytes":3000},
                  {"name":"down","format":"Q2_K","source":0,"offset":)" + std::to_string(d0) +
           R"(,"stride":5000,"bytes":5000}],
         "hash":)" + hashes(1) + R"(},
        {"index":2,"kind":"moe","n_units":8,"top_k":2,
         "parts":[{"name":"record","format":"EXL3","source":0,"offset":)" + std::to_string(rec0) +
           R"(,"stride":8192,"bytes":8192}],
         "hash":)" + hashes(2) + R"(},
        {"index":3,"kind":"blob","n_units":1,
         "parts":[{"name":"layer","format":"mixed","source":1,"offset":4096,"bytes":)" + std::to_string(kBlob) +
           R"(}],
         "hash":)" + hashes(3) + R"(},
        {"index":4,"kind":"moe","n_units":16,"top_k":4,
         "parts":[{"name":"gate","format":"IQ2_XXS","source":0,"offset":)" + std::to_string(g0) +
           R"(,"stride":3000,"bytes":3000},
                  {"name":"up","format":"IQ2_XXS","source":0,"offset":)" + std::to_string(u0) +
           R"(,"stride":3000,"bytes":3000},
                  {"name":"down","format":"Q2_K","source":0,"offset":)" + std::to_string(d0) +
           R"(,"stride":5000,"bytes":5000}],
         "hash":)" + hashes(4) + R"(}]})";
  }
};

}  // namespace sstest
