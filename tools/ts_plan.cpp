// ts-plan (also built as ss-plan for one release): validate a desc-v1 manifest and print its tier plan as JSON
// (what hyperloom's planner adapter compares).
//   ts-plan DESC.json --device-gib X [--host-gib Y] [--workspace-gib W] [--ring N] [--dynamic N] [--desc]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "strix_stream/ss.h"

#ifndef TS_TOOL_NAME
#define TS_TOOL_NAME "ts-plan"
#endif

// Every ts-plan diagnostic starts with "ts-plan: ". The old ss-plan prints exactly what strix-stream's ss-plan printed
// (no prefix), so scripts that parse it see the same bytes.
static const char* const kPrefix = std::strcmp(TS_TOOL_NAME, "ss-plan") == 0 ? "" : TS_TOOL_NAME ": ";

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "%susage: " TS_TOOL_NAME " DESC.json --device-gib X [--host-gib Y] [--workspace-gib W] [--ring N] "
                         "[--dynamic N] [--desc]\n", kPrefix);
    return 2;
  }
  ss_budget b;
  ss_budget_defaults(&b);
  bool print_desc = false;
  const double gib = double(uint64_t(1) << 30);
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s%s needs a value\n", kPrefix, a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--device-gib") b.device_bytes = uint64_t(std::atof(next()) * gib);
    else if (a == "--host-gib") b.host_pinned_bytes = uint64_t(std::atof(next()) * gib);
    else if (a == "--workspace-gib") b.workspace_bytes = uint64_t(std::atof(next()) * gib);
    else if (a == "--ring") b.ring_slots = uint32_t(std::atoi(next()));
    else if (a == "--dynamic") b.dynamic_slots_per_layer = uint32_t(std::atoi(next()));
    else if (a == "--desc") print_desc = true;
    else {
      std::fprintf(stderr, "%sunknown argument %s\n", kPrefix, a.c_str());
      return 2;
    }
  }
  ss_desc* d = nullptr;
  if (ss_desc_from_file(argv[1], &d) != SS_OK) {
    std::fprintf(stderr, "%sdescriptor: %s\n", kPrefix, ss_last_error());
    return 1;
  }
  char* out = nullptr;
  if (print_desc) {
    ss_desc_to_json(d, &out, nullptr);
    std::printf("%s\n", out);
    ss_free(out);
    ss_desc_free(d);
    return 0;
  }
  ss_plan* p = nullptr;
  if (ss_plan_make(d, &b, nullptr, 0, &p) != SS_OK) {
    std::fprintf(stderr, "%splan: %s\n", kPrefix, ss_last_error());
    ss_desc_free(d);
    return 1;
  }
  ss_plan_to_json(p, &out, nullptr);
  std::printf("%s\n", out);
  ss_free(out);
  ss_plan_free(p);
  ss_desc_free(d);
  return 0;
}
