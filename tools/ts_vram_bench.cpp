// ts-vram-bench / ts-read-bench: how fast weight units get from NVMe files into host memory and, in the CUDA build
// (ts-vram-bench), into device memory, through the multi-location reader (ss_mfile_*).
//   --mode read      units read into host slots (pinned with cudaHostAlloc under --pinned in the CUDA build)
//   --mode h2d       copy-engine baseline: pinned slots -> device, no disk on the timed path (CUDA build)
//   --mode read+h2d  a ring of pinned slots: each unit is read straight into a slot, copied to the device on that
//                    slot's stream, and the slot is recycled once its copy event completes (CUDA build)
// One summary line on stdout; --json writes the full result. Exit codes: 0 ok, 1 error, 2 usage, 3 --check FAIL.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "strix_stream/ss.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/resource.h>
#endif
#if defined(TS_VRAM_CUDA)
#include <cuda_runtime.h>
#endif

#ifndef TS_TOOL_NAME
#define TS_TOOL_NAME "ts-read-bench"
#endif

namespace {
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
constexpr uint64_t kMiB = uint64_t(1) << 20;
constexpr uint64_t kAlign = 4096;

const char* const kUsage =
    "usage: " TS_TOOL_NAME " --file PATH [--weight-dirs \"DIR;DIR\"] [--drive-readers 4] [--verify cached|always]\n"
    "         [--buffered] [--primary-sha256 HEX] [--open-timeout S]\n"
    "         [--unit-bytes 16777216] [--units 64] [--pattern seq|random|ranges:FILE] [--span-offset B]\n"
    "         [--span-bytes B] [--seed 1] [--depth 4] [--batch] [--mode read|h2d|read+h2d] [--pinned] [--ring K]\n"
    "         [--device-uuid UUID] [--check N] [--json PATH]";

struct Args {
  std::string file, weight_dirs, pattern = "seq", json, mode = "read", device_uuid, primary_sha256;
  uint32_t drive_readers = 4, depth = 4, ring = 0;
  int verify = SS_MFILE_VERIFY_CACHED;
  bool buffered = false, batch = false, pinned = false;
  uint64_t unit_bytes = 16 * kMiB, units = 0, span_offset = 0, span_bytes = 0, seed = 1, check = 0;
  double open_timeout = 0;
};

struct Unit {
  uint64_t offset = 0, bytes = 0;  // aligned to kAlign (the ss_mfile read contract)
};

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }
uint64_t now_ns() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count());
}

[[noreturn]] void usage_error(const std::string& msg) {
  std::fprintf(stderr, "%s: %s\n%s\n", TS_TOOL_NAME, msg.c_str(), kUsage);
  throw std::invalid_argument(msg);
}

uint64_t u64_arg(const std::string& name, const char* s) {
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (!*s || *end || errno || *s == '-') usage_error(name + " needs an unsigned integer, got '" + s + "'");
  return uint64_t(v);
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) usage_error(k + " needs a value");
      return argv[++i];
    };
    if (k == "--file") a.file = next();
    else if (k == "--weight-dirs") a.weight_dirs = next();
    else if (k == "--drive-readers") a.drive_readers = uint32_t(u64_arg(k, next()));
    else if (k == "--verify") {
      const std::string v = next();
      if (v == "cached") a.verify = SS_MFILE_VERIFY_CACHED;
      else if (v == "always") a.verify = SS_MFILE_VERIFY_ALWAYS;
      else usage_error("--verify must be cached or always");
    } else if (k == "--buffered") a.buffered = true;
    else if (k == "--primary-sha256") a.primary_sha256 = next();
    else if (k == "--open-timeout") a.open_timeout = double(u64_arg(k, next()));
    else if (k == "--unit-bytes") a.unit_bytes = u64_arg(k, next());
    else if (k == "--units") a.units = u64_arg(k, next());
    else if (k == "--pattern") a.pattern = next();
    else if (k == "--span-offset") a.span_offset = u64_arg(k, next());
    else if (k == "--span-bytes") a.span_bytes = u64_arg(k, next());
    else if (k == "--seed") a.seed = u64_arg(k, next());
    else if (k == "--depth") a.depth = uint32_t(u64_arg(k, next()));
    else if (k == "--batch") a.batch = true;
    else if (k == "--mode") a.mode = next();
    else if (k == "--pinned") a.pinned = true;
    else if (k == "--ring") a.ring = uint32_t(u64_arg(k, next()));
    else if (k == "--device-uuid") a.device_uuid = next();
    else if (k == "--check") a.check = u64_arg(k, next());
    else if (k == "--json") a.json = next();
    else usage_error("unknown argument " + k);
  }
  if (a.file.empty()) usage_error("--file is required");
  if (a.mode != "read" && a.mode != "h2d" && a.mode != "read+h2d") usage_error("--mode must be read, h2d or read+h2d");
#if !defined(TS_VRAM_CUDA)
  if (a.mode != "read") usage_error("--mode " + a.mode + " needs the CUDA build (ts-vram-bench)");
  if (a.pinned) usage_error("--pinned needs the CUDA build (ts-vram-bench)");
#endif
  if (!a.unit_bytes || a.unit_bytes % kAlign) usage_error("--unit-bytes must be a positive multiple of 4096");
  if (a.span_offset % kAlign) usage_error("--span-offset must be a multiple of 4096");
  if (a.depth < 1 || a.depth > 1024) usage_error("--depth must be in 1..1024");
  if (a.mode != "read") {
    if (!a.ring) a.ring = 2 * a.depth;
    if (a.ring < a.depth) usage_error("--ring must be at least --depth");
  }
  return a;
}

// --- the unit list --------------------------------------------------------------------------------------------------

std::vector<Unit> make_units(const Args& a, uint64_t file_size) {
  const uint64_t end = align_up(file_size);  // reads may extend past EOF up to the aligned end
  std::vector<Unit> units;
  if (a.pattern.rfind("ranges:", 0) == 0) {
    const std::string path = a.pattern.substr(7);
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) throw std::runtime_error("cannot open ranges file " + path);
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const ts::Json j = ts::json_parse(text);
    if (!j.is_array()) throw std::runtime_error("ranges file: expected a JSON list of [offset, bytes] pairs");
    for (size_t i = 0; i < j.arr.size(); ++i) {
      const std::string where = "ranges[" + std::to_string(i) + "]";
      const ts::Json& p = j.arr[i];
      if (!p.is_array() || p.arr.size() != 2) throw std::runtime_error(where + ": expected [offset, bytes]");
      const uint64_t off = ts::json_u64(p.arr[0], where + "[0]"), n = ts::json_u64(p.arr[1], where + "[1]");
      if (!n || off >= file_size || n > file_size - off) throw std::runtime_error(where + ": outside the file");
      Unit u;
      u.offset = off / kAlign * kAlign;
      u.bytes = align_up(off + n) - u.offset;
      units.push_back(u);
    }
    if (a.units && a.units < units.size()) units.resize(size_t(a.units));
    if (units.empty()) throw std::runtime_error("ranges file: no ranges");
    return units;
  }
  const uint64_t count = a.units ? a.units : 64;
  if (a.span_offset >= file_size) throw std::runtime_error("--span-offset is past the end of the file");
  const uint64_t span = a.span_bytes ? std::min(a.span_bytes, end - a.span_offset) : end - a.span_offset;
  if (a.pattern == "seq") {
    const uint64_t per_span = std::max<uint64_t>(1, span / a.unit_bytes);  // wraps around within the span
    for (uint64_t i = 0; i < count; ++i) {
      Unit u;
      u.offset = a.span_offset + (i % per_span) * a.unit_bytes;
      u.bytes = std::min(a.unit_bytes, end - u.offset);
      units.push_back(u);
    }
  } else if (a.pattern == "random") {
    if (span < a.unit_bytes) throw std::runtime_error("--pattern random: the span is smaller than one unit");
    const uint64_t positions = (span - a.unit_bytes) / kAlign + 1;
    std::mt19937_64 rng(a.seed);
    for (uint64_t i = 0; i < count; ++i) {
      Unit u;
      u.offset = a.span_offset + (rng() % positions) * kAlign;
      u.bytes = a.unit_bytes;
      units.push_back(u);
    }
  } else {
    usage_error("--pattern must be seq, random or ranges:FILE");
  }
  return units;
}

// Evenly spaced sample of n units (the --check set).
std::vector<uint64_t> sample_units(uint64_t units, uint64_t n) {
  n = std::min(n, units);
  std::vector<uint64_t> s;
  for (uint64_t k = 0; k < n; ++k) s.push_back(k * units / n);
  return s;
}

// --- memory ---------------------------------------------------------------------------------------------------------

class HostBlock {
 public:
  HostBlock(uint64_t bytes, bool pinned) : bytes_(bytes), pinned_(pinned) {
#if defined(TS_VRAM_CUDA)
    if (pinned) {
      if (cudaHostAlloc(&raw_, size_t(bytes + kAlign), cudaHostAllocDefault) != cudaSuccess) raw_ = nullptr;
      if (!raw_) throw std::runtime_error("cudaHostAlloc of " + std::to_string(bytes) + " bytes failed");
      p_ = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw_) + kAlign - 1) / kAlign * kAlign);
      return;
    }
#endif
#if defined(_WIN32)
    raw_ = VirtualAlloc(nullptr, size_t(bytes), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
    if (posix_memalign(&raw_, kAlign, size_t(bytes))) raw_ = nullptr;
#endif
    if (!raw_) throw std::runtime_error("host allocation of " + std::to_string(bytes) + " bytes failed");
    p_ = static_cast<char*>(raw_);
  }
  ~HostBlock() {
#if defined(TS_VRAM_CUDA)
    if (pinned_) {
      cudaFreeHost(raw_);
      return;
    }
#endif
#if defined(_WIN32)
    VirtualFree(raw_, 0, MEM_RELEASE);
#else
    std::free(raw_);
#endif
  }
  HostBlock(const HostBlock&) = delete;
  HostBlock& operator=(const HostBlock&) = delete;
  char* data() const { return p_; }
  uint64_t bytes() const { return bytes_; }
  bool pinned() const { return pinned_; }

 private:
  void* raw_ = nullptr;
  char* p_ = nullptr;
  uint64_t bytes_;
  bool pinned_;
};

double cpu_seconds() {
#if defined(_WIN32)
  FILETIME c, e, k, u;
  if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
  auto t = [](const FILETIME& f) { return double((uint64_t(f.dwHighDateTime) << 32) | f.dwLowDateTime) * 1e-7; };
  return t(k) + t(u);
#else
  rusage r{};
  getrusage(RUSAGE_SELF, &r);
  return double(r.ru_utime.tv_sec + r.ru_stime.tv_sec) + double(r.ru_utime.tv_usec + r.ru_stime.tv_usec) * 1e-6;
#endif
}

// --- exactness ------------------------------------------------------------------------------------------------------

// A plain buffered read of the primary (no ss_mfile), zero-filled past EOF as the ss_mfile contract does.
std::vector<char> plain_read(const std::string& path, uint64_t offset, uint64_t bytes) {
  std::vector<char> b(size_t(bytes), 0);
  std::ifstream f(fs::u8path(path), std::ios::binary);
  if (!f) throw std::runtime_error("check: cannot open " + path);
  f.seekg(std::streamoff(offset));
  f.read(b.data(), std::streamsize(bytes));
  return b;
}

bool same_bytes(const std::string& path, uint64_t offset, const char* got, uint64_t bytes) {
  const std::vector<char> ref = plain_read(path, offset, bytes);
  return std::memcmp(ref.data(), got, size_t(bytes)) == 0;
}

// --- CUDA -----------------------------------------------------------------------------------------------------------

#if defined(TS_VRAM_CUDA)
void cuda_ok(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// --device-uuid ("GPU-xxxxxxxx-...", dashes optional) or the first device the runtime shows (CUDA_VISIBLE_DEVICES).
int pick_device(const std::string& uuid) {
  if (uuid.empty()) return 0;
  std::string want;
  for (char c : uuid.rfind("GPU-", 0) == 0 ? uuid.substr(4) : uuid)
    if (c != '-') want += char(std::tolower(static_cast<unsigned char>(c)));
  if (want.size() != 32) throw std::runtime_error("--device-uuid: expected 32 hex digits");
  int n = 0;
  cuda_ok(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
  for (int d = 0; d < n; ++d) {
    cudaDeviceProp p{};
    cuda_ok(cudaGetDeviceProperties(&p, d), "cudaGetDeviceProperties");
    char hex[33] = {};
    for (int i = 0; i < 16; ++i)
      std::snprintf(hex + 2 * i, 3, "%02x", static_cast<unsigned>(static_cast<unsigned char>(p.uuid.bytes[i])));
    if (want == hex) return d;
  }
  throw std::runtime_error("--device-uuid: no visible device has UUID " + uuid);
}

struct DeviceBlock {
  void* p = nullptr;
  explicit DeviceBlock(uint64_t bytes) { cuda_ok(cudaMalloc(&p, size_t(bytes)), "cudaMalloc"); }
  ~DeviceBlock() { cudaFree(p); }
  DeviceBlock(const DeviceBlock&) = delete;
  DeviceBlock& operator=(const DeviceBlock&) = delete;
};
#endif

// --- the run --------------------------------------------------------------------------------------------------------

struct Result {
  uint64_t bytes = 0;
  double wall_s = 0, cpu_s = 0, open_s = 0;
  std::vector<double> host_ms, device_ms;  // per unit (device_ms empty in read mode)
  ss_mfile_counters before{}, after{};
  std::string check = "SKIPPED";
  std::vector<uint64_t> check_failed;
  uint64_t check_samples = 0;
  int device = -1;
};

class FirstError {
 public:
  void set(const std::string& m) {
    std::lock_guard<std::mutex> l(mu_);
    if (msg_.empty()) msg_ = m;
    failed_.store(true);
  }
  bool failed() const { return failed_.load(); }
  void rethrow() const {
    if (failed_.load()) throw std::runtime_error(msg_);
  }

 private:
  std::mutex mu_;
  std::string msg_;
  std::atomic<bool> failed_{false};
};

void read_ok(ss_status s, const char* what) {
  if (s != SS_OK) throw std::runtime_error(std::string(what) + ": " + ss_last_error());
}

// --mode read: --depth threads each calling ss_mfile_pread into its own slot, or (--batch) one ss_mfile_preadv per
// group of --depth units. Latency = submit -> the call returned (data in host memory).
void run_read(ss_mfile* f, const Args& a, const std::vector<Unit>& units, uint64_t slot_bytes,
              const std::vector<int64_t>& sample_of, std::vector<std::vector<char>>& kept, Result& r) {
#if defined(TS_VRAM_CUDA)
  if (a.pinned) cuda_ok(cudaSetDevice(r.device = pick_device(a.device_uuid)), "cudaSetDevice");
#endif
  HostBlock mem(slot_bytes * a.depth, a.pinned);
  r.host_ms.assign(units.size(), 0);
  FirstError err;
  auto keep = [&](size_t i, const char* p) {
    if (sample_of[i] >= 0) std::memcpy(kept[size_t(sample_of[i])].data(), p, size_t(units[i].bytes));
  };
  const uint64_t t0 = now_ns();
  if (a.batch) {
    std::vector<ss_mfile_io> ios(a.depth);
    for (size_t g = 0; g < units.size() && !err.failed(); g += a.depth) {
      const size_t n = std::min<size_t>(a.depth, units.size() - g);
      for (size_t k = 0; k < n; ++k) ios[k] = {mem.data() + k * slot_bytes, units[g + k].offset, units[g + k].bytes};
      const uint64_t s = now_ns();
      read_ok(ss_mfile_preadv(f, ios.data(), uint32_t(n)), "ss_mfile_preadv");
      const double ms = double(now_ns() - s) * 1e-6;
      for (size_t k = 0; k < n; ++k) {
        r.host_ms[g + k] = ms;
        keep(g + k, mem.data() + k * slot_bytes);
      }
    }
  } else {
    std::atomic<size_t> next{0};
    std::vector<std::thread> threads;
    for (uint32_t w = 0; w < a.depth; ++w) {
      threads.emplace_back([&, w] {
        char* slot = mem.data() + w * slot_bytes;
        try {
          while (!err.failed()) {
            const size_t i = next.fetch_add(1);
            if (i >= units.size()) break;
            uint64_t got = 0;
            const uint64_t s = now_ns();
            read_ok(ss_mfile_pread(f, slot, units[i].bytes, units[i].offset, &got), "ss_mfile_pread");
            r.host_ms[i] = double(now_ns() - s) * 1e-6;
            keep(i, slot);
          }
        } catch (const std::exception& e) {
          err.set(e.what());
        }
      });
    }
    for (auto& t : threads) t.join();
  }
  r.wall_s = double(now_ns() - t0) * 1e-9;
  err.rethrow();
}

#if defined(TS_VRAM_CUDA)
// --mode read+h2d / h2d: a ring of --ring pinned slots. read+h2d: a unit is read straight into a free slot (--depth
// reader threads, or --batch: one preadv per group of --depth slots), then copied to the device on that slot's stream;
// a completion thread waits the slots' copy events in submission order and recycles them. h2d: the slots are filled
// once before timing (the data of the first --ring units) and the timed loop only issues copies.
// Host latency = submit -> read returned; device latency = submit -> copy event observed complete.
void run_ring(ss_mfile* f, const Args& a, const std::vector<Unit>& units, uint64_t slot_bytes,
              const std::vector<int64_t>& sample_of, std::vector<std::vector<char>>& kept,
              std::vector<uint64_t>& check_offset, Result& r) {
  const bool read = a.mode == "read+h2d";
  const int dev = pick_device(a.device_uuid);
  r.device = dev;
  cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
  const uint32_t K = a.ring;
  HostBlock ring(slot_bytes * K, true);
  std::vector<std::unique_ptr<DeviceBlock>> dslot, dcheck;
  std::vector<cudaStream_t> streams(K);
  std::vector<cudaEvent_t> events(K);
  for (uint32_t s = 0; s < K; ++s) {
    dslot.push_back(std::make_unique<DeviceBlock>(slot_bytes));
    cuda_ok(cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking), "cudaStreamCreate");
    cuda_ok(cudaEventCreateWithFlags(&events[s], cudaEventDisableTiming), "cudaEventCreate");
  }
  for (size_t k = 0; k < kept.size(); ++k) dcheck.push_back(std::make_unique<DeviceBlock>(slot_bytes));
  std::vector<uint64_t> slot_src(K, 0);  // h2d: the file offset whose bytes a slot holds
  if (!read) {
    const uint64_t end = align_up(ss_mfile_size(f));
    for (uint32_t s = 0; s < K; ++s) {
      const Unit& u = units[s % units.size()];
      uint64_t got = 0;
      std::memset(ring.data() + s * slot_bytes, 0, size_t(slot_bytes));
      read_ok(ss_mfile_pread(f, ring.data() + s * slot_bytes, std::min(slot_bytes, end - u.offset), u.offset, &got),
              "ss_mfile_pread (h2d fill)");
      slot_src[s] = u.offset;
    }
    read_ok(ss_mfile_get_counters(f, &r.before), "ss_mfile_get_counters");
  }
  r.host_ms.assign(units.size(), 0);
  r.device_ms.assign(units.size(), 0);

  std::mutex mu;
  std::condition_variable cv_free, cv_done;
  std::deque<uint32_t> free_slots;
  for (uint32_t s = 0; s < K; ++s) free_slots.push_back(s);
  struct Done { uint32_t slot; size_t unit; uint64_t t0; };
  std::deque<Done> done;
  FirstError err;
  auto acquire = [&]() -> int64_t {
    std::unique_lock<std::mutex> l(mu);
    cv_free.wait(l, [&] { return !free_slots.empty() || err.failed(); });
    if (err.failed()) return -1;
    const uint32_t s = free_slots.front();
    free_slots.pop_front();
    return s;
  };
  auto copy = [&](uint32_t s, size_t i, uint64_t t0) {
    void* dst = sample_of[i] >= 0 ? dcheck[size_t(sample_of[i])]->p : dslot[s]->p;
    if (sample_of[i] >= 0) check_offset[size_t(sample_of[i])] = read ? units[i].offset : slot_src[s];
    cuda_ok(cudaMemcpyAsync(dst, ring.data() + s * slot_bytes, size_t(units[i].bytes), cudaMemcpyHostToDevice,
                            streams[s]), "cudaMemcpyAsync");
    cuda_ok(cudaEventRecord(events[s], streams[s]), "cudaEventRecord");
    std::lock_guard<std::mutex> l(mu);
    done.push_back({s, i, t0});
    cv_done.notify_one();
  };
  auto fail = [&](const char* what) {
    err.set(what);
    std::lock_guard<std::mutex> l(mu);
    cv_free.notify_all();
    cv_done.notify_all();
  };

  const uint64_t t0 = now_ns();
  std::thread completer([&] {
    try {
      cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
      for (size_t n = 0; n < units.size(); ++n) {
        Done d;
        {
          std::unique_lock<std::mutex> l(mu);
          cv_done.wait(l, [&] { return !done.empty() || err.failed(); });
          if (err.failed()) return;
          d = done.front();
          done.pop_front();
        }
        cuda_ok(cudaEventSynchronize(events[d.slot]), "cudaEventSynchronize");
        r.device_ms[d.unit] = double(now_ns() - d.t0) * 1e-6;
        std::lock_guard<std::mutex> l(mu);
        free_slots.push_back(d.slot);
        cv_free.notify_one();
      }
    } catch (const std::exception& e) {
      fail(e.what());
    }
  });
  std::vector<std::thread> issuers;
  if (!read) {
    issuers.emplace_back([&] {
      try {
        cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
        for (size_t i = 0; i < units.size(); ++i) {
          const int64_t s = acquire();
          if (s < 0) return;
          copy(uint32_t(s), i, now_ns());
        }
      } catch (const std::exception& e) {
        fail(e.what());
      }
    });
  } else if (a.batch) {
    issuers.emplace_back([&] {
      try {
        cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
        std::vector<ss_mfile_io> ios(a.depth);
        std::vector<uint32_t> slots(a.depth);
        for (size_t g = 0; g < units.size(); g += a.depth) {
          const size_t n = std::min<size_t>(a.depth, units.size() - g);
          for (size_t k = 0; k < n; ++k) {
            const int64_t s = acquire();
            if (s < 0) return;
            slots[k] = uint32_t(s);
            ios[k] = {ring.data() + slots[k] * slot_bytes, units[g + k].offset, units[g + k].bytes};
          }
          const uint64_t s0 = now_ns();
          read_ok(ss_mfile_preadv(f, ios.data(), uint32_t(n)), "ss_mfile_preadv");
          const double ms = double(now_ns() - s0) * 1e-6;
          for (size_t k = 0; k < n; ++k) {
            r.host_ms[g + k] = ms;
            copy(slots[k], g + k, s0);
          }
        }
      } catch (const std::exception& e) {
        fail(e.what());
      }
    });
  } else {
    std::atomic<size_t> next{0};
    for (uint32_t w = 0; w < a.depth; ++w) {
      issuers.emplace_back([&] {
        try {
          cuda_ok(cudaSetDevice(dev), "cudaSetDevice");
          while (!err.failed()) {
            const size_t i = next.fetch_add(1);
            if (i >= units.size()) break;
            const int64_t s = acquire();
            if (s < 0) return;
            uint64_t got = 0;
            const uint64_t s0 = now_ns();
            read_ok(ss_mfile_pread(f, ring.data() + uint64_t(s) * slot_bytes, units[i].bytes, units[i].offset, &got),
                    "ss_mfile_pread");
            r.host_ms[i] = double(now_ns() - s0) * 1e-6;
            copy(uint32_t(s), i, s0);
          }
        } catch (const std::exception& e) {
          fail(e.what());
        }
      });
    }
  }
  for (auto& t : issuers) t.join();
  completer.join();
  r.wall_s = double(now_ns() - t0) * 1e-9;
  for (uint32_t s = 0; s < K; ++s) {
    cudaStreamSynchronize(streams[s]);
    cudaStreamDestroy(streams[s]);
    cudaEventDestroy(events[s]);
  }
  err.rethrow();
  for (size_t k = 0; k < kept.size(); ++k)  // the check copies the sampled units back from the device
    cuda_ok(cudaMemcpy(kept[k].data(), dcheck[k]->p, kept[k].size(), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
}
#endif

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t i = size_t(std::max(0.0, std::ceil(p * double(v.size())) - 1));
  return v[std::min(i, v.size() - 1)];
}

ts::Json latency_json(const std::vector<double>& ms) {
  ts::Json j = ts::Json::object();
  j["p50_ms"] = ts::Json::number(percentile(ms, 0.50));
  j["p99_ms"] = ts::Json::number(percentile(ms, 0.99));
  j["max_ms"] = ts::Json::number(percentile(ms, 1.0));
  return j;
}

int bench_main(int argc, char** argv) {
  Args a;
  try {
    a = parse_args(argc, argv);
  } catch (const std::invalid_argument&) {
    return 2;
  }
  try {
    ss_mfile_options o;
    ss_mfile_options_defaults(&o);
    o.direct_io = a.buffered ? 0 : 1;
    o.drive_readers = a.drive_readers;
    o.verify_mode = a.verify;
    o.primary_sha256 = a.primary_sha256.empty() ? nullptr : a.primary_sha256.c_str();
    Result r;
    r.before.struct_size = r.after.struct_size = sizeof r.before;
    ss_mfile* f = nullptr;
    // --open-timeout: a first open of a stripe without cached verification hashes the whole primary and stripe;
    // give up instead of silently reading hundreds of GiB.
    std::mutex open_mu;
    std::condition_variable open_cv;
    bool opened = false;
    std::thread watchdog;
    if (a.open_timeout > 0) {
      watchdog = std::thread([&] {
        std::unique_lock<std::mutex> l(open_mu);
        if (!open_cv.wait_for(l, std::chrono::duration<double>(a.open_timeout), [&] { return opened; })) {
          std::fprintf(stderr, "%s: open did not finish within %.0f s (uncached verification?); giving up\n",
                       TS_TOOL_NAME, a.open_timeout);
          std::fflush(stderr);
          std::_Exit(1);
        }
      });
    }
    const uint64_t o0 = now_ns();
    const ss_status st = ss_mfile_open(a.file.c_str(), a.weight_dirs.c_str(), &o, &f);
    r.open_s = double(now_ns() - o0) * 1e-9;
    {
      std::lock_guard<std::mutex> l(open_mu);
      opened = true;
    }
    open_cv.notify_all();
    if (watchdog.joinable()) watchdog.join();
    read_ok(st, "ss_mfile_open");
    std::unique_ptr<ss_mfile, void (*)(ss_mfile*)> file(f, ss_mfile_close);

    const std::vector<Unit> units = make_units(a, ss_mfile_size(f));
    uint64_t slot_bytes = 0;
    for (const Unit& u : units) slot_bytes = std::max(slot_bytes, u.bytes);
    const std::vector<uint64_t> samples = sample_units(units.size(), a.check);
    std::vector<int64_t> sample_of(units.size(), -1);
    std::vector<std::vector<char>> kept;
    std::vector<uint64_t> check_offset;
    for (size_t k = 0; k < samples.size(); ++k) {
      sample_of[size_t(samples[k])] = int64_t(k);
      kept.emplace_back(size_t(units[size_t(samples[k])].bytes));
      check_offset.push_back(units[size_t(samples[k])].offset);
    }
    read_ok(ss_mfile_get_counters(f, &r.before), "ss_mfile_get_counters");
    const double cpu0 = cpu_seconds();
#if defined(TS_VRAM_CUDA)
    if (a.mode != "read") run_ring(f, a, units, slot_bytes, sample_of, kept, check_offset, r);
    else
#endif
      run_read(f, a, units, slot_bytes, sample_of, kept, r);
    r.cpu_s = cpu_seconds() - cpu0;
    read_ok(ss_mfile_get_counters(f, &r.after), "ss_mfile_get_counters");
    for (const Unit& u : units) r.bytes += u.bytes;

    if (!samples.empty()) {
      r.check_samples = samples.size();
      for (size_t k = 0; k < samples.size(); ++k)
        if (!same_bytes(a.file, check_offset[k], kept[k].data(), kept[k].size())) r.check_failed.push_back(samples[k]);
      r.check = r.check_failed.empty() ? "PASS" : "FAIL";
    }

    const double gb = double(r.bytes) * 1e-9;
    const double gbps = r.wall_s > 0 ? gb / r.wall_s : 0;
    const bool dev = !r.device_ms.empty();
    const uint32_t locs = std::min<uint32_t>(r.after.locations, SS_MFILE_MAX_LOCATIONS);
    std::string loc_text;
    for (uint32_t i = 0; i < locs; ++i) {
      char b[96];
      std::snprintf(b, sizeof b, " loc%u=%.3fGB", i, double(r.after.bytes[i] - r.before.bytes[i]) * 1e-9);
      loc_text += b;
    }
    std::printf("%s: mode=%s pattern=%s units=%zu unit_bytes=%llu depth=%u%s bytes=%llu wall_s=%.3f GBps=%.3f "
                "host_p50_ms=%.2f host_p99_ms=%.2f",
                TS_TOOL_NAME, a.mode.c_str(), a.pattern.c_str(), units.size(),
                static_cast<unsigned long long>(a.unit_bytes), a.depth, a.batch ? " batch" : "",
                static_cast<unsigned long long>(r.bytes), r.wall_s, gbps, percentile(r.host_ms, 0.5),
                percentile(r.host_ms, 0.99));
    if (dev)
      std::printf(" dev_p50_ms=%.2f dev_p99_ms=%.2f", percentile(r.device_ms, 0.5), percentile(r.device_ms, 0.99));
    std::printf(" cpu_s=%.3f cpu_s_per_GB=%.4f open_s=%.2f%s fallback_bytes=%llu verify_fail=%llu check=%s\n",
                r.cpu_s, gb > 0 ? r.cpu_s / gb : 0, r.open_s, loc_text.c_str(),
                static_cast<unsigned long long>(r.after.fallback_bytes - r.before.fallback_bytes),
                static_cast<unsigned long long>(r.after.verify_fail), r.check.c_str());
    if (!r.check_failed.empty()) {
      std::fprintf(stderr, "%s: check FAIL at unit(s)", TS_TOOL_NAME);
      for (uint64_t u : r.check_failed) std::fprintf(stderr, " %llu", static_cast<unsigned long long>(u));
      std::fprintf(stderr, "\n");
    }

    if (!a.json.empty()) {
      ts::Json j = ts::Json::object();
      j["tool"] = ts::Json::string(TS_TOOL_NAME);
      j["mode"] = ts::Json::string(a.mode);
      j["file"] = ts::Json::string(a.file);
      j["weight_dirs"] = ts::Json::string(a.weight_dirs);
      j["pattern"] = ts::Json::string(a.pattern);
      j["unit_bytes"] = ts::Json::number(a.unit_bytes);
      j["units"] = ts::Json::number(uint64_t(units.size()));
      j["depth"] = ts::Json::number(uint64_t(a.depth));
      j["batch"] = ts::Json::boolean(a.batch);
      j["ring"] = ts::Json::number(uint64_t(a.ring));
      j["direct"] = ts::Json::boolean(!a.buffered);
      j["pinned"] = ts::Json::boolean(a.pinned || a.mode != "read");
      j["drive_readers"] = ts::Json::number(uint64_t(a.drive_readers));
      j["verify"] = ts::Json::string(a.verify == SS_MFILE_VERIFY_ALWAYS ? "always" : "cached");
      j["seed"] = ts::Json::number(a.seed);
      if (r.device >= 0) j["device"] = ts::Json::number(int64_t(r.device));
      j["bytes"] = ts::Json::number(r.bytes);
      j["wall_s"] = ts::Json::number(r.wall_s);
      j["gbps"] = ts::Json::number(gbps);
      j["open_s"] = ts::Json::number(r.open_s);
      j["cpu_s"] = ts::Json::number(r.cpu_s);
      j["cpu_s_per_gb"] = ts::Json::number(gb > 0 ? r.cpu_s / gb : 0.0);
      if (a.mode != "h2d") j["latency_host"] = latency_json(r.host_ms);  // h2d has no read
      if (dev) j["latency_device"] = latency_json(r.device_ms);
      ts::Json c = ts::Json::object();
      c["multi"] = ts::Json::boolean(ss_mfile_multi(f) != 0);
      c["open_report"] = ts::Json::string(ss_mfile_open_report(f));
      c["locations"] = ts::Json::array();
      for (uint32_t i = 0; i < locs; ++i) {
        ts::Json l = ts::Json::object();
        const char* p = ss_mfile_location_path(f, int(i));
        l["path"] = ts::Json::string(p ? p : "");
        l["bytes"] = ts::Json::number(r.after.bytes[i] - r.before.bytes[i]);
        l["reads"] = ts::Json::number(r.after.reads[i] - r.before.reads[i]);
        l["busy_s"] = ts::Json::number(double(r.after.busy_ns[i] - r.before.busy_ns[i]) * 1e-9);
        l["disabled"] = ts::Json::boolean(((r.after.disabled_mask >> i) & 1) != 0);
        l["direct"] = ts::Json::boolean(((r.after.direct_mask >> i) & 1) != 0);
        c["locations"].push(std::move(l));
      }
      c["fallback_bytes"] = ts::Json::number(r.after.fallback_bytes - r.before.fallback_bytes);
      c["read_errors"] = ts::Json::number(r.after.read_errors - r.before.read_errors);
      c["demoted_chunks"] = ts::Json::number(r.after.demoted_chunks);
      c["verify_fail"] = ts::Json::number(r.after.verify_fail);
      c["verified_bytes"] = ts::Json::number(r.after.verified_bytes);
      c["verify_s"] = ts::Json::number(double(r.after.verify_ns) * 1e-9);
      j["counters"] = std::move(c);
      ts::Json ck = ts::Json::object();
      ck["samples"] = ts::Json::number(r.check_samples);
      ck["result"] = ts::Json::string(r.check);
      ck["failed_units"] = ts::Json::array();
      for (uint64_t u : r.check_failed) ck["failed_units"].push(ts::Json::number(u));
      j["check"] = std::move(ck);
      std::ofstream out(fs::u8path(a.json), std::ios::binary | std::ios::trunc);
      out << ts::json_write(j) << "\n";
      if (!out) throw std::runtime_error("cannot write " + a.json);
    }
    return r.check == "FAIL" ? 3 : 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s: %s\n", TS_TOOL_NAME, e.what());
    return 1;
  }
}

}  // namespace

#if !defined(TS_VRAM_BENCH_TESTING)
int main(int argc, char** argv) { return bench_main(argc, argv); }
#endif
