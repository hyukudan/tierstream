// SHA-256 (FIPS 180-4), hex digest. Recipes bind profiles to the exact recipe bytes with it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ts {

std::string sha256_hex(const void* data, size_t n);

// Incremental SHA-256 (multi-gigabyte files). On Windows the system provider (CNG) does the rounds.
class Sha256 {
 public:
  Sha256();
  ~Sha256();
  Sha256(const Sha256&) = delete;
  Sha256& operator=(const Sha256&) = delete;
  void update(const void* data, size_t n);
  std::string hex();  // finishes the hash

 private:
  uint32_t h_[8];
  unsigned char buf_[64];
  size_t buf_len_ = 0;
  uint64_t total_ = 0;
  void* cng_ = nullptr;
};

}  // namespace ts
