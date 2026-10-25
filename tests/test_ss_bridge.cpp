// The ss:: bridge of the rename (wave4/wstream/rename/design.md §2): a translation unit built against the library's
// src/ (engine branch minimax/rtx-prefill-ss) still calls ss::xxh64.
#include <cstdio>

#include "xxh64.hpp"

int main() {
  const char data[] = "tierstream";
  const uint64_t a = ss::xxh64(data, sizeof data - 1);
  const uint64_t b = ts::xxh64(data, sizeof data - 1);
  ss::Xxh64 h;
  h.update(data, sizeof data - 1);
  const bool ok = a == b && h.digest() == a;
  std::printf("%s\n", ok ? "0 failure(s)" : "1 failure(s): ss::xxh64 differs from ts::xxh64");
  return ok ? 0 : 1;
}
