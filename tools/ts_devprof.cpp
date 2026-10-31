// ts-devprof: the effective device profile of this machine (strix/device-profile-v1, ss.h "Device profiles").
//   ts-devprof [--device-profile FILE] [--root DIR] [--no-runtime] [--no-cache] [--no-env]
//              [--budget] [--probe-report HOST] [--expect-class NAME]
// Prints the canonical profile JSON (default), or the whole-device caps it gives device 0 (--budget), or the recipes'
// probe report (--probe-report). Notices go to stderr. --expect-class: exit 3 unless the class is NAME (node checks).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "strix_stream/ss.h"

namespace {
constexpr const char* kUsage =
    "usage: ts-devprof [--device-profile FILE] [--root DIR] [--no-runtime] [--no-cache] [--no-env] [--budget] "
    "[--probe-report HOST] [--expect-class NAME]\n";
}

int main(int argc, char** argv) {
  ss_devprof_detect_opts o;
  ss_devprof_detect_opts_defaults(&o);
  bool budget = false;
  const char* host = nullptr;
  const char* expect = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "ts-devprof: %s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--device-profile") o.override_path = next();
    else if (a == "--root") o.root = next();
    else if (a == "--no-runtime") o.flags |= SS_DEVPROF_NO_RUNTIME;
    else if (a == "--no-cache") o.flags |= SS_DEVPROF_NO_CACHE;
    else if (a == "--no-env") o.flags |= SS_DEVPROF_NO_ENV;
    else if (a == "--budget") budget = true;
    else if (a == "--probe-report") host = next();
    else if (a == "--expect-class") expect = next();
    else {
      std::fprintf(stderr, "ts-devprof: unknown argument %s\n%s", a.c_str(), kUsage);
      return 2;
    }
  }
  ss_devprof* p = nullptr;
  if (ss_devprof_detect(&o, &p) != SS_OK) {
    std::fprintf(stderr, "ts-devprof: %s\n", ss_last_error());
    return 1;
  }
  ss_devprof_info info{};
  info.struct_size = sizeof info;
  ss_devprof_get(p, &info);
  char* text = nullptr;
  size_t len = 0;
  int rc = 0;
  if (budget) {
    ss_budget_in in;
    ss_budget_in_defaults(&in);
    if (info.n_devices == 0) {
      std::fprintf(stderr, "ts-devprof: no device in the profile\n");
      rc = 1;
    } else if (ss_devprof_budget(p, 0, &in) != SS_OK) {
      std::fprintf(stderr, "ts-devprof: %s\n", ss_last_error());
      rc = 1;
    } else {
      std::printf("{\"class\":\"%s\",\"device_profile\":\"%s\",\"vram_total_bytes\":%llu,\"ram_total_bytes\":%llu,"
                  "\"unified\":%s}\n",
                  info.class_name, info.hash, static_cast<unsigned long long>(in.vram_total_bytes),
                  static_cast<unsigned long long>(in.ram_total_bytes), in.unified ? "true" : "false");
    }
  } else if (host) {
    if (ss_devprof_probe_report(p, host, &text, &len) != SS_OK) {
      std::fprintf(stderr, "ts-devprof: %s\n", ss_last_error());
      rc = 1;
    }
  } else if (ss_devprof_to_json(p, &text, &len) != SS_OK) {
    std::fprintf(stderr, "ts-devprof: %s\n", ss_last_error());
    rc = 1;
  }
  if (text) {
    std::fwrite(text, 1, len, stdout);
    std::fputc('\n', stdout);
    std::free(text);
  }
  std::fprintf(stderr, "ts-devprof: class %s%s, %u device(s), unified %u, hash %s\n", info.class_name,
               info.generic ? " (generic)" : "", info.n_devices, info.unified, info.hash);
  if (expect && std::strcmp(expect, info.class_name) != 0) {
    std::fprintf(stderr, "ts-devprof: expected class %s\n", expect);
    rc = rc ? rc : 3;
  }
  ss_devprof_free(p);
  return rc;
}
