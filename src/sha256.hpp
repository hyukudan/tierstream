// SHA-256 (FIPS 180-4), hex digest. Recipes bind profiles to the exact recipe bytes with it.
#pragma once

#include <cstddef>
#include <string>

namespace ts {

std::string sha256_hex(const void* data, size_t n);

}  // namespace ts
