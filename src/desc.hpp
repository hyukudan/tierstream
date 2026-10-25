// Model descriptor desc-v1 (docs/design.md §2): the only thing the library knows about a model.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"

namespace ts {

enum class LayerKind { Dense, Moe, Blob };
enum class Tiers { Pinned, Disk, PinnedDisk, Pair };  // where a unit may live when it is not resident
enum class Order { Prefix, Ranked };

struct Named {
  std::string name;
  uint64_t bytes = 0;
};

struct Source {
  std::string name;
  std::string path;
  uint64_t align = 4096;  // unbuffered read granularity (sector / page); power of two
};

// One piece of a unit's engine bytes: unit u's part comes from source `source` at offset + u * stride, `bytes` long.
// Parts are back to back in the unit in the listed order; their sum is the unit's bytes.
struct Part {
  std::string name;
  std::string format;  // GGUF type (Q2_K, IQ2_XXS, MXFP4, Q8_0, ...), EXL3, FP8_E4M3, BF16, F16, F32, x-<custom>
  int source = 0;
  uint64_t offset = 0;
  uint64_t stride = 0;
  uint64_t bytes = 0;
};

struct Router {
  std::string tensor;  // the engine's router weight (read by the engine for lookahead; named for adapters/tools)
  std::string bias;
  std::string format;
  std::string score;  // softmax | sigmoid | sqrt_softplus | x-<custom>
  uint32_t rows = 0;  // experts
  uint32_t cols = 0;  // hidden
};

struct Layer {
  int index = 0;
  LayerKind kind = LayerKind::Dense;
  uint64_t fixed_bytes = 0;  // always-device bytes of this layer (attention, norms, shared expert, router)
  int n_units = 0;           // routed experts (Moe), 1 (Blob), 0 (Dense)
  int top_k = 0;
  int n_shared = 0;
  uint64_t unit_bytes = 0;   // engine bytes per unit (sum of parts)
  std::vector<Part> parts;
  Router router;
  bool has_router = false;
  std::vector<uint64_t> hash;      // xxh64 per unit (empty = not given)
  std::vector<double> benefit;     // per unit (empty = descriptor default: uniform 1 per unit)
  bool streamable = true;          // false: always resident (counted in the planner as fixed)
};

struct LookupTable {
  std::string name;
  int source = 0;
  uint64_t offset = 0;
  uint64_t rows = 0;
  uint64_t row_bytes = 0;
  std::string placement;  // file | pinned | device (the engine reads rows itself; listed for planners)
};

struct Desc {
  std::string model;
  std::string engine;
  std::vector<Named> fixed_device;
  std::vector<Named> host_fixed;
  uint64_t state_bytes_per_position = 0;
  uint64_t workspace_bytes_per_row = 0;
  uint64_t unit_align = 256;  // device alignment of a unit inside the pool / ring
  std::vector<Source> sources;
  Order order = Order::Ranked;
  Tiers tiers = Tiers::PinnedDisk;
  uint32_t ring_slots = 0;    // slots per stream ring; 0 = 2 x the most units of a layer of that unit size
  std::string routes;         // routes file for hot ranking (planners); not read by the library
  std::vector<Layer> layers;
  std::vector<LookupTable> lookup_tables;

  // Derived (validate fills them).
  std::vector<int64_t> unit_base;  // global index of (layer, 0)
  int64_t total_units = 0;
  uint64_t max_unit_bytes = 0;     // over streamable layers
  int max_units_per_layer = 0;     // over streamable layers

  uint64_t unit_stride(int layer) const;  // unit_bytes rounded up to unit_align
  uint64_t fixed_total() const;           // fixed_device + every layer's fixed bytes + non-streamable units
};

Desc desc_from_json(const Json& j);
Json desc_to_json(const Desc& d);
void desc_validate(Desc& d);  // throws SS_E_DESC; fills derived fields

const char* tiers_name(Tiers t);
const char* order_name(Order o);

}  // namespace ts
