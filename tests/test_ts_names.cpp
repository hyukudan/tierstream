// The tierstream names, from C++ (design notes §2.1): ts.h first, then ss.h (the other order is
// tests/test_ts_names.c). Every ts_ function has exactly its ss_ counterpart's type, every ts_ type is the ss_ type
// (so the C++ mangling of a consumer's ss_budget* parameter does not change), and a consumer's own members named after
// either survive.
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "tierstream/ts.h"
#include "strix_stream/ss.h"

// Every generated pair: same function type, same type.
#define TS_FN_PAIR(name) \
  static_assert(std::is_same<decltype(&ss_##name), decltype(&ts_##name)>::value, "ts_" #name " has another type");
#define TS_TYPE_PAIR(name) static_assert(std::is_same<ss_##name, ts_##name>::value, "ts_" #name " is another type");
#include "ts_pairs.inc"
#undef TS_FN_PAIR
#undef TS_TYPE_PAIR

namespace {

int failures = 0;
#define EXPECT(c)                                                          \
  do {                                                                     \
    if (!(c)) {                                                            \
      std::printf("  %s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #c); \
      ++failures;                                                          \
    }                                                                      \
  } while (0)

struct consumer {
  int ss_budget = 1;
  int ts_budget = 2;
};

// A consumer function on the old type, called with the new name: one type, one mangled name.
int take_old(const ss_budget* b) { return int(b->struct_size); }

}  // namespace

int main() {
  consumer k;
  EXPECT(k.ss_budget + k.ts_budget == 3);
  ts_budget b;
  ts_budget_defaults(&b);
  EXPECT(take_old(&b) == int(sizeof(ss_budget)));
  // the exported ts_ symbols are distinct functions that forward (one release: the ss_ ones stay the definitions)
  EXPECT(reinterpret_cast<const void*>(&ts_abi_version) != reinterpret_cast<const void*>(&ss_abi_version));
  EXPECT(ts_abi_version() == ss_abi_version());
  std::printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
