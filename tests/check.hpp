// Tiny test harness: CHECK records a failure and continues; main returns the failure count.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

namespace sstest {

template <class T>
std::string str(const T& v) {
  if constexpr (std::is_arithmetic_v<T>) return std::to_string(v);
  else return std::string(v);
}

inline int& failures() {
  static int n = 0;
  return n;
}

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& cases() {
  static std::vector<Case> v;
  return v;
}

struct Reg {
  Reg(const char* n, std::function<void()> f) { cases().push_back({n, std::move(f)}); }
};

// only: run the cases whose name contains it (nullptr = all).
inline int run_all(const char* only = nullptr) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  for (const Case& c : cases()) {
    if (only && std::string(c.name).find(only) == std::string::npos) continue;
    int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      std::printf("  exception: %s\n", e.what());
      ++failures();
    }
    std::printf("%s %s\n", failures() == before ? "PASS" : "FAIL", c.name);
  }
  std::printf("%d failure(s)\n", failures());
  return failures() ? 1 : 0;
}

}  // namespace sstest

#define SS_CAT2(a, b) a##b
#define SS_CAT(a, b) SS_CAT2(a, b)
#define TEST(name)                                                    \
  static void name();                                                 \
  static sstest::Reg SS_CAT(reg_, name)(#name, name);                 \
  static void name()

#define CHECK(cond)                                                                   \
  do {                                                                                \
    if (!(cond)) {                                                                    \
      std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);          \
      ++sstest::failures();                                                           \
    }                                                                                 \
  } while (0)

#define CHECK_EQ(a, b)                                                                                    \
  do {                                                                                                    \
    auto va_ = (a);                                                                                       \
    auto vb_ = (b);                                                                                       \
    if (!(va_ == vb_)) {                                                                                  \
      std::printf("  %s:%d: CHECK_EQ(%s, %s) failed: %s vs %s\n", __FILE__, __LINE__, #a, #b,             \
                  sstest::str(va_).c_str(), sstest::str(vb_).c_str());                              \
      ++sstest::failures();                                                                               \
    }                                                                                                     \
  } while (0)

#define CHECK_OK(expr)                                                                          \
  do {                                                                                          \
    ss_status st_ = (expr);                                                                     \
    if (st_ != SS_OK) {                                                                         \
      std::printf("  %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #expr, int(st_), ss_last_error()); \
      ++sstest::failures();                                                                     \
    }                                                                                           \
  } while (0)
