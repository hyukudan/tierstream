#include "sha256.hpp"

#include "errors.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#endif

namespace ts {
namespace {

constexpr uint32_t kK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

void block(uint32_t h[8], const unsigned char* p) {
  uint32_t w[64];
  for (int i = 0; i < 16; ++i)
    w[i] =
        uint32_t(p[4 * i]) << 24 | uint32_t(p[4 * i + 1]) << 16 | uint32_t(p[4 * i + 2]) << 8 | uint32_t(p[4 * i + 3]);
  for (int i = 16; i < 64; ++i) {
    uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
    uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
    w[i] = w[i - 16] + s0 + w[i - 7] + s1;
  }
  uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + kK[i] + w[i];
    uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;
}

}  // namespace

std::string sha256_hex(const void* data, size_t n) {
  uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t full = n / 64 * 64;
  for (size_t i = 0; i < full; i += 64) block(h, p + i);
  unsigned char tail[128] = {};
  size_t r = n - full;
  for (size_t i = 0; i < r; ++i) tail[i] = p[full + i];
  tail[r] = 0x80;
  size_t len = r + 1 + 8 <= 64 ? 64 : 128;
  uint64_t bits = uint64_t(n) * 8;
  for (int i = 0; i < 8; ++i) tail[len - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
  for (size_t i = 0; i < len; i += 64) block(h, tail + i);
  std::string out;
  char buf[9];
  for (uint32_t v : h) {
    std::snprintf(buf, sizeof buf, "%08x", static_cast<unsigned>(v));
    out += buf;
  }
  return out;
}

namespace {
std::string hex_of(const uint32_t h[8]) {
  std::string out;
  char buf[9];
  for (int i = 0; i < 8; ++i) {
    std::snprintf(buf, sizeof buf, "%08x", static_cast<unsigned>(h[i]));
    out += buf;
  }
  return out;
}
}  // namespace

Sha256::Sha256() {
  static const uint32_t init[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::memcpy(h_, init, sizeof h_);
#if defined(_WIN32)
  BCRYPT_HASH_HANDLE hh = nullptr;
  if (BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &hh, nullptr, 0, nullptr, 0, 0) == 0) cng_ = hh;
#endif
}

Sha256::~Sha256() {
#if defined(_WIN32)
  if (cng_) BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(cng_));
#endif
}

void Sha256::update(const void* data, size_t n) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
#if defined(_WIN32)
  if (cng_) {
    while (n) {
      ULONG part = ULONG(n > (size_t(1) << 30) ? (size_t(1) << 30) : n);
      require(BCryptHashData(static_cast<BCRYPT_HASH_HANDLE>(cng_), const_cast<PUCHAR>(p), part, 0) == 0, SS_E_IO,
              "SHA-256: the system provider failed");
      p += part;
      n -= part;
    }
    return;
  }
#endif
  total_ += n;
  if (buf_len_) {
    size_t take = std::min<size_t>(64 - buf_len_, n);
    std::memcpy(buf_ + buf_len_, p, take);
    buf_len_ += take;
    p += take;
    n -= take;
    if (buf_len_ < 64) return;
    block(h_, buf_);
    buf_len_ = 0;
  }
  for (; n >= 64; p += 64, n -= 64) block(h_, p);
  std::memcpy(buf_, p, n);
  buf_len_ = n;
}

std::string Sha256::hex() {
#if defined(_WIN32)
  if (cng_) {
    UCHAR d[32];
    const long st = BCryptFinishHash(static_cast<BCRYPT_HASH_HANDLE>(cng_), d, 32, 0);
    BCryptDestroyHash(static_cast<BCRYPT_HASH_HANDLE>(cng_));
    cng_ = nullptr;
    require(st == 0, SS_E_IO, "SHA-256: the system provider failed");
    std::string out;
    char buf[3];
    for (UCHAR b : d) {
      std::snprintf(buf, sizeof buf, "%02x", unsigned(b));
      out += buf;
    }
    return out;
  }
#endif
  unsigned char tail[128] = {};
  std::memcpy(tail, buf_, buf_len_);
  tail[buf_len_] = 0x80;
  size_t len = buf_len_ + 1 + 8 <= 64 ? 64 : 128;
  uint64_t bits = total_ * 8;
  for (int i = 0; i < 8; ++i) tail[len - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
  for (size_t i = 0; i < len; i += 64) block(h_, tail + i);
  return hex_of(h_);
}

}  // namespace ts
