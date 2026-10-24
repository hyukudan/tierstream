#include "xxh64.hpp"

#include <cstring>

namespace ss {
namespace {

constexpr uint64_t P1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t P2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t P3 = 0x165667B19E3779F9ULL;
constexpr uint64_t P4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t P5 = 0x27D4EB2F165667C5ULL;

inline uint64_t rotl(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }

inline uint64_t rd64(const unsigned char* p) {
  uint64_t v = 0;
  for (int k = 7; k >= 0; --k) v = (v << 8) | p[k];  // little-endian, independent of host order
  return v;
}

inline uint32_t rd32(const unsigned char* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

inline uint64_t round(uint64_t acc, uint64_t lane) {
  acc += lane * P2;
  acc = rotl(acc, 31);
  return acc * P1;
}

inline uint64_t merge(uint64_t acc, uint64_t v) {
  acc ^= round(0, v);
  return acc * P1 + P4;
}

}  // namespace

Xxh64::Xxh64(uint64_t seed) : seed_(seed) {
  v_[0] = seed + P1 + P2;
  v_[1] = seed + P2;
  v_[2] = seed;
  v_[3] = seed - P1;
}

void Xxh64::update(const void* data, size_t len) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  total_ += len;
  if (buf_len_ + len < 32) {
    std::memcpy(buf_ + buf_len_, p, len);
    buf_len_ += len;
    return;
  }
  if (buf_len_) {
    size_t take = 32 - buf_len_;
    std::memcpy(buf_ + buf_len_, p, take);
    for (int k = 0; k < 4; ++k) v_[k] = round(v_[k], rd64(buf_ + 8 * k));
    p += take;
    len -= take;
    buf_len_ = 0;
  }
  while (len >= 32) {
    for (int k = 0; k < 4; ++k) v_[k] = round(v_[k], rd64(p + 8 * k));
    p += 32;
    len -= 32;
  }
  std::memcpy(buf_, p, len);
  buf_len_ = len;
}

uint64_t Xxh64::digest() const {
  uint64_t h;
  if (total_ >= 32) {
    h = rotl(v_[0], 1) + rotl(v_[1], 7) + rotl(v_[2], 12) + rotl(v_[3], 18);
    for (int k = 0; k < 4; ++k) h = merge(h, v_[k]);
  } else {
    h = seed_ + P5;
  }
  h += total_;
  const unsigned char* p = buf_;
  size_t len = buf_len_;
  while (len >= 8) {
    h ^= round(0, rd64(p));
    h = rotl(h, 27) * P1 + P4;
    p += 8;
    len -= 8;
  }
  if (len >= 4) {
    h ^= uint64_t(rd32(p)) * P1;
    h = rotl(h, 23) * P2 + P3;
    p += 4;
    len -= 4;
  }
  while (len) {
    h ^= uint64_t(*p) * P5;
    h = rotl(h, 11) * P1;
    ++p;
    --len;
  }
  h ^= h >> 33;
  h *= P2;
  h ^= h >> 29;
  h *= P3;
  h ^= h >> 32;
  return h;
}

uint64_t xxh64(const void* data, size_t len, uint64_t seed) {
  Xxh64 x(seed);
  x.update(data, len);
  return x.digest();
}

}  // namespace ss
