// Standalone content identity and measured-cost conversion (placement contract sections 2.4 and 3.9).
#pragma once

#include <functional>
#include <string>
#include <vector>

#include "json.hpp"

namespace ts {
// read must return exactly n consecutive bytes. The parser never asks for bytes past the header/tensor table.
// header_bytes != 0 explicitly supplies the header extent for formats other than GGUF and safetensors.
std::string content_header_sha256(const std::function<void(void*, size_t)>& read, uint64_t size,
                                  uint64_t header_bytes = 0);
Json content_manifest(std::vector<std::string> paths, bool full, uint64_t header_bytes = 0);
Json content_manifest_verify(const Json& manifest, const std::vector<std::string>& paths, bool full,
                             uint64_t header_bytes = 0);
Json convert_rtx_costs(const Json& report);
}  // namespace ts
