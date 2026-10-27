// CPU host ranking + boundary cost, excluding route generation, learning and all device work.
// Not a CTest: run ts-bench-dg-scored in a Release build.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>

#include "dg_core.hpp"

namespace {
class NullTransport final : public ts::DgTransport {
 public:
  void h2d(Stream, int32_t, int32_t) override {}
  void d2h(int32_t, int32_t) override {}
  void d2d(int32_t, int32_t) override {}
  void h2h(int32_t, int32_t) override {}
  void d2h_part(Stream, int32_t, int32_t, uint32_t) override {}
  uint32_t parts() const override { return 1; }
  uint64_t part_bytes(uint32_t) const override { return 4096; }
  void write_entry(Stream, uint32_t, int32_t) override {}
  void signal_ready(uint32_t, uint32_t) override {}
  void* event_create() override { return nullptr; }
  void event_destroy(void*) override {}
  void record(void*, Stream) override {}
  void wait(Stream, void*) override {}
  bool done(void*) override { return true; }
  void event_sync(void*) override {}
  void sync(Stream) override {}
  void upload_table(const int32_t*, uint32_t) override {}
};

double run(uint32_t mode, uint32_t width, uint64_t& checksum) {
  ts::DgConfig c;
  c.n_units.assign(40, 384);
  c.persistent_slots = 8350;
  c.bank_slots = 64;
  c.spare_target = 96;
  c.blocks = 40 * 384;
  c.scored = mode != 0;
  c.boundary_score = mode >= 2;
  c.scored_auto = mode >= 3;
  NullTransport t;
  ts::DgCore core(c, &t);
  std::mt19937 rng(1234);
  std::vector<uint32_t> order(40 * 384), ids(384);
  std::iota(order.begin(), order.end(), 0u);
  std::iota(ids.begin(), ids.end(), 0u);
  std::shuffle(order.begin(), order.end(), rng);
  core.place(order);
  double us = 0;
  constexpr uint32_t warmup = 64, cycles = 512;
  for (uint32_t tok = 1; tok <= warmup + cycles; ++tok) {
    core.begin_token(tok);
    for (uint32_t g = 0; g < 40; ++g) {
      std::shuffle(ids.begin(), ids.end(), rng);
      uint64_t mask = 0;
      for (uint32_t i = 0; i < width; ++i)
        if (core.table()[core.key(g, ids[i])] < 0) mask |= uint64_t(1) << i;
      core.serve(g, ids.data(), width, mask);
    }
    const auto start = std::chrono::steady_clock::now();
    for (uint32_t g = 0; g < 40; ++g) core.score_layer(g);
    core.end_token({}, mode == 4 ? 6 : 1);
    const auto stop = std::chrono::steady_clock::now();
    if (tok > warmup) us += std::chrono::duration<double, std::micro>(stop - start).count();
    for (uint32_t k : core.last_victims()) checksum = checksum * 1099511628211ull + k;
  }
  return us / cycles;
}
}  // namespace

int main() {
  std::puts("CPU ranking + boundary, 40 x 384, 8350 slots, spare target 96, median of 3 x (64 warmup + 512 cycles)");
  for (uint32_t width : {6u, 20u, 40u}) {
    std::array<std::array<double, 3>, 5> times{};
    for (uint32_t rep = 0; rep < 3; ++rep) {
      std::array<uint64_t, 5> sums{};
      for (uint32_t i = 0; i < 5; ++i) {  // rotate order to reduce systematic warmup / clock bias
        const uint32_t mode = (i + rep * 2) % 5;
        times[mode][rep] = run(mode, width, sums[mode]);
      }
      if (sums[1] != sums[2] || sums[0] != sums[3] || sums[2] != sums[4]) return 1;
    }
    for (auto& t : times) std::sort(t.begin(), t.end());
    std::printf("union %2u: LRU %.2f / public scored %.2f / heap %.2f / auto R1 %.2f / auto R6 %.2f us/cycle; "
                "same victims 1/1/1\n", width, times[0][1], times[1][1], times[2][1], times[3][1], times[4][1]);
  }
}
