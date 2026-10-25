// Bounded startup loader shared by CPU fixtures and the CUDA context.
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include "desc.hpp"
namespace ts {
struct DgLoadUnit {
  int layer = 0, unit = 0;
  int32_t slot = -1;
  char* host = nullptr;
};
struct DgLoadCopy { int32_t slot; size_t part; const char* data; uint64_t bytes; };
struct DgLoadStats {
  uint64_t payload_bytes = 0, read_bytes = 0, read_calls = 0, bounce_bytes = 0, h2d_bytes = 0;
  // read_seconds is wall time through the last completed batch (includes hashing
  // and buffer waits); hash_seconds sums reader CPU wall times. bounce_bytes
  // counts payload routed through fallback storage, excluding alignment padding.
  double read_seconds = 0, hash_seconds = 0;
};
// Buffers are caller-owned. recycle waits for the previous transfer from this buffer;
// submit must enqueue all copies before returning. Workers never publish placement.
DgLoadStats dg_load_pipeline(const Desc& d, const std::vector<DgLoadUnit>& units,
    const std::vector<char*>& buffers, uint64_t buffer_bytes, uint32_t readers,
    bool direct, bool verify, const std::function<void(size_t)>& recycle,
    const std::function<void(size_t, const std::vector<DgLoadCopy>&)>& submit);
}  // namespace ts
