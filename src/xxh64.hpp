// XXH64 (seed 0 by default), written from the published algorithm description. Used for the per-unit byte identity
// check: the descriptor carries xxh64 of each unit's engine bytes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ss {

class Xxh64 {
 public:
  explicit Xxh64(uint64_t seed = 0);
  void update(const void* data, size_t len);
  uint64_t digest() const;

 private:
  uint64_t v_[4];
  uint64_t seed_;
  uint64_t total_ = 0;
  unsigned char buf_[32];
  size_t buf_len_ = 0;
};

uint64_t xxh64(const void* data, size_t len, uint64_t seed = 0);

}  // namespace ss
