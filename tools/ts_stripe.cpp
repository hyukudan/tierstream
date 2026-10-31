// Build, verify and collect immutable secondary-drive stripes.
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include "errors.hpp"
#include "json.hpp"
#include "raw_file.hpp"
#include "sha256.hpp"
#include "stripe_manifest.hpp"
#include "xxh64.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
namespace fs = std::filesystem;
using ts::require;
constexpr uint64_t kMiB = uint64_t(1) << 20;
constexpr size_t kDepth = 3;

std::string path_text(const fs::path& p) { return p.u8string(); }
fs::path path_of(const std::string& s) { return fs::u8path(s); }

std::string hex16(uint64_t n) {
  char s[17];
  std::snprintf(s, sizeof s, "%016llx", static_cast<unsigned long long>(n));
  return s;
}

const ts::Json& field(const ts::Json& j, const char* key) {
  const ts::Json* v = j.find(key);
  require(v != nullptr, SS_E_DESC, std::string("missing ") + key);
  return *v;
}

std::string read_text(const fs::path& path) {
  ts::RawFile f(path_text(path), false);
  auto size = f.size();
  require(size <= (uint64_t(256) << 20), SS_E_DESC, "JSON exceeds 256 MiB");
  std::string text(size_t(size), '\0');
  require(!size || f.read(text.data(), size, 0) == size, SS_E_IO, "read " + path_text(path));
  return text;
}

struct AlignedFree {
  void operator()(char* p) const {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};
using Buffer = std::unique_ptr<char, AlignedFree>;
Buffer buffer(uint64_t bytes) {
  void* p = nullptr;
#if defined(_WIN32)
  p = _aligned_malloc(size_t(bytes), 4096);
#else
  if (posix_memalign(&p, 4096, size_t(bytes))) p = nullptr;
#endif
  require(p != nullptr, SS_E_NOMEM, "aligned buffer allocation failed");
  return Buffer(static_cast<char*>(p));
}

class Handle {
 public:
#if defined(_WIN32)
  HANDLE value = INVALID_HANDLE_VALUE;
  ~Handle() { if (value != INVALID_HANDLE_VALUE) CloseHandle(value); }
#else
  int value = -1;
  ~Handle() { if (value >= 0) ::close(value); }
#endif
  Handle() = default;
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
};

// A persistent lock file serializes make and gc, including partial cleanup and manifest publication.
class OutputLock {
 public:
  OutputLock(const fs::path& dir, const std::string& base) {
    const auto path = dir / path_of(base + ".stripe.lock");
#if defined(_WIN32)
    h_.value = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    require(h_.value != INVALID_HANDLE_VALUE, SS_E_IO, "stripe directory is busy or not writable");
#else
    h_.value = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    require(h_.value >= 0 && flock(h_.value, LOCK_EX | LOCK_NB) == 0, SS_E_IO,
            "stripe directory is busy or not writable");
#endif
  }
 private:
  Handle h_;
};

uint64_t directory_volume(const fs::path& path) {
#if defined(_WIN32)
  Handle h;
  h.value = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  BY_HANDLE_FILE_INFORMATION info{};
  require(h.value != INVALID_HANDLE_VALUE && GetFileInformationByHandle(h.value, &info), SS_E_IO,
          "cannot identify output volume");
  return info.dwVolumeSerialNumber;
#else
  struct stat st{};
  require(::stat(path.c_str(), &st) == 0, SS_E_IO, "cannot identify output volume");
  return uint64_t(st.st_dev);
#endif
}

class StripeWriter {
 public:
  StripeWriter(const fs::path& path, uint64_t size) {
#if defined(_WIN32)
    h_.value = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_FLAG_OVERLAPPED | FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    require(h_.value != INVALID_HANDLE_VALUE, SS_E_IO, "create stripe partial: error " + std::to_string(GetLastError()));
    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = LONGLONG(size);
    FILE_END_OF_FILE_INFO eof{};
    eof.EndOfFile.QuadPart = LONGLONG(size);
    require(SetFileInformationByHandle(h_.value, FileAllocationInfo, &allocation, sizeof allocation) &&
                SetFileInformationByHandle(h_.value, FileEndOfFileInfo, &eof, sizeof eof),
            SS_E_IO, "preallocate stripe: error " + std::to_string(GetLastError()));
#else
    h_.value = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_DIRECT | O_CLOEXEC, 0600);
    require(h_.value >= 0, SS_E_IO, "create stripe partial");
    require((!size || posix_fallocate(h_.value, 0, off_t(size)) == 0) && ftruncate(h_.value, off_t(size)) == 0,
            SS_E_IO, "preallocate stripe");
#endif
  }
  void write(const char* src, uint64_t bytes, uint64_t offset) const {
#if defined(_WIN32)
    Handle event;
    event.value = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    require(event.value != nullptr, SS_E_IO, "create write event");
    OVERLAPPED ov{};
    ov.Offset = DWORD(offset);
    ov.OffsetHigh = DWORD(offset >> 32);
    ov.hEvent = event.value;
    const BOOL started = WriteFile(h_.value, src, DWORD(bytes), nullptr, &ov);
    const DWORD error = started ? ERROR_SUCCESS : GetLastError();
    require(started || error == ERROR_IO_PENDING, SS_E_IO, "stripe write: error " + std::to_string(error));
    DWORD put = 0;
    require(GetOverlappedResult(h_.value, &ov, &put, TRUE) && put == bytes, SS_E_IO, "short or failed stripe write");
#else
    ssize_t put;
    do { put = pwrite(h_.value, src, size_t(bytes), off_t(offset)); } while (put < 0 && errno == EINTR);
    require(put >= 0 && uint64_t(put) == bytes, SS_E_IO, "short or failed stripe write");
#endif
  }
  void flush() const {
#if defined(_WIN32)
    require(FlushFileBuffers(h_.value) != 0, SS_E_IO, "flush stripe");
#else
    require(fsync(h_.value) == 0, SS_E_IO, "flush stripe");
#endif
  }
 private:
  Handle h_;
};

class IoPool {
 public:
  IoPool() {
    try {
      for (size_t i = 0; i < kDepth; ++i) threads_.emplace_back([this]() {
        for (;;) {
          std::packaged_task<void()> job;
          {
            std::unique_lock<std::mutex> lock(mu_);
            ready_.wait(lock, [&]() { return stop_ || !jobs_.empty(); });
            if (jobs_.empty()) return;
            job = std::move(jobs_.front());
            jobs_.pop();
          }
          job();
        }
      });
    } catch (...) { close(); throw; }
  }
  ~IoPool() { close(); }
  template<class F> std::future<void> submit(F f) {
    std::packaged_task<void()> job(std::move(f));
    auto result = job.get_future();
    {
      std::lock_guard<std::mutex> lock(mu_);
      jobs_.push(std::move(job));
    }
    ready_.notify_one();
    return result;
  }
 private:
  void close() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    ready_.notify_all();
    for (auto& thread : threads_) if (thread.joinable()) thread.join();
  }
  std::mutex mu_;
  std::condition_variable ready_;
  std::queue<std::packaged_task<void()>> jobs_;
  std::vector<std::thread> threads_;
  bool stop_ = false;
};

bool starts(const std::string& s, const std::string& prefix) { return s.compare(0, prefix.size(), prefix) == 0; }
bool ends(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void clean_partials(const fs::path& dir, const std::string& base) {
  for (const auto& e : fs::directory_iterator(dir)) {
    const auto name = path_text(e.path().filename());
    if ((starts(name, base + ".stripe-") && (ends(name, ".partial") || ends(name, ".tmp"))) ||
        name == ts::stripe_manifest_name(base) + ".tmp") {
      require(e.is_regular_file() && !e.is_symlink(), SS_E_IO, "unexpected partial file type");
      fs::remove(e.path());
    }
  }
}

std::vector<uint64_t> placement(const fs::path& ranges, const std::string& base, uint64_t size, uint64_t chunk) {
  const auto j = ts::json_parse(read_text(ranges));
  require(ts::json_str(field(j, "schema"), "schema") == "tierstream/stripe-ranges-v1", SS_E_DESC,
          "invalid stripe ranges schema");
  require(ts::json_str(field(j, "primary"), "primary") == base &&
              ts::json_u64(field(j, "size"), "size") == size, SS_E_DESC, "ranges primary basename or size mismatch");
  require(size <= ts::kStripeMaxPrimary, SS_E_DESC, "primary too large");
  const uint64_t count = (size + chunk - 1) / chunk;
  require(count <= ts::kStripeMaxChunks, SS_E_DESC, "too many primary chunks");
  const auto& a = field(j, "ranges");
  require(a.is_array(), SS_E_DESC, "ranges: expected array");
  // Difference arrays make overlapping ranges linear in the number of chunks and ranges.
  // interleave: odd chunks; interleave-even: even chunks (a second stripe drive takes the chunks the first does not);
  // blocks:K / blocks-even:K: odd / even runs of K chunks, so that a read of a few chunks mostly has one stripe home
  struct Rule {
    uint64_t run;
    uint64_t parity;
    std::vector<int64_t> diff;
  };
  std::vector<int64_t> all(size_t(count + 1));
  std::vector<Rule> rules;
  auto rule = [&](uint64_t run, uint64_t parity) -> std::vector<int64_t>& {
    for (auto& r : rules) if (r.run == run && r.parity == parity) return r.diff;
    rules.push_back(Rule{run, parity, std::vector<int64_t>(size_t(count + 1))});
    return rules.back().diff;
  };
  for (const auto& r : a.arr) {
    ts::json_str(field(r, "name"), "range.name");
    const uint64_t offset = ts::json_u64(field(r, "offset"), "range.offset");
    const uint64_t bytes = ts::json_u64(field(r, "bytes"), "range.bytes");
    const auto policy = ts::json_str(field(r, "policy"), "range.policy");
    uint64_t run = 1, parity = 1;
    if (policy == "interleave-even") parity = 0;
    else if (policy.rfind("blocks:", 0) == 0 || policy.rfind("blocks-even:", 0) == 0) {
      const std::string k = policy.substr(policy.find(':') + 1);
      require(!k.empty() && k.size() <= 6 && k.find_first_not_of("0123456789") == std::string::npos, SS_E_DESC,
              "range policy blocks:K needs a run length K of 1..999999 chunks");
      run = std::stoull(k);
      require(run >= 1, SS_E_DESC, "range policy blocks:K needs a run length K of 1..999999 chunks");
      parity = policy[6] == ':' ? 1 : 0;
    } else require(policy == "all" || policy == "interleave", SS_E_DESC, "unknown range policy");
    require(offset <= size && bytes <= size - offset, SS_E_DESC, "range exceeds primary");
    if (!bytes) continue;
    auto& diff = policy == "all" ? all : rule(run, parity);
    ++diff[size_t(offset / chunk)];
    --diff[size_t((offset + bytes - 1) / chunk + 1)];
  }
  std::vector<uint64_t> chunks;
  int64_t na = 0;
  std::vector<int64_t> n(rules.size(), 0);
  for (uint64_t c = 0; c < count; ++c) {
    na += all[size_t(c)];
    bool placed = na != 0;
    for (size_t i = 0; i < rules.size(); ++i) {
      n[i] += rules[i].diff[size_t(c)];
      if (n[i] && (c / rules[i].run) % 2 == rules[i].parity) placed = true;
    }
    if (placed) chunks.push_back(c);
  }
  return chunks;
}

ts::StripeManifest load_manifest(const fs::path& dir, const std::string& base) {
  auto m = ts::stripe_manifest_parse(read_text(dir / path_of(ts::stripe_manifest_name(base))));
  require(m.primary_basename == base, SS_E_DESC, "manifest primary basename mismatch");
  return m;
}

struct Progress {
  const char* phase;
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now(), last = start;
  explicit Progress(const char* name) : phase(name) {}
  void print(uint64_t read, uint64_t written, uint64_t total) {
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last).count() < 10) return;
    const double seconds = std::chrono::duration<double>(now - start).count();
    const double rate = double(read) / seconds;
    std::fprintf(stderr, "ts-stripe: %s %.2f/%.2f GB, read %.2f GB/s, written %.2f GB/s, ETA %.0f s\n",
                 phase, double(read) / 1e9, double(total) / 1e9, rate / 1e9, double(written) / seconds / 1e9,
                 rate ? double(total - read) / rate : 0);
    last = now;
  }
};

bool verify_slots(const fs::path& path, const ts::StripeManifest& m, const ts::RawFile* primary = nullptr,
                  bool keep_going = false, bool check_primary = false) {
  ts::RawFile data(path_text(path), true);
  require(data.direct(), SS_E_IO, "stripe verification requires unbuffered reads");
  require(data.size() == m.data_size, SS_E_IO, "stripe data size mismatch");
  const auto before = data.identity();
  require(!primary || before.volume != primary->identity().volume || before.index != primary->identity().index,
          SS_E_DESC, "stripe aliases primary");
  auto b = buffer(m.chunk_bytes);
  auto p = check_primary ? buffer(m.chunk_bytes) : Buffer();
  bool ok = true;
  Progress progress("verify data");
  for (const auto& run : m.runs) {
    for (uint64_t i = 0; i < run.count; ++i) {
      const uint64_t c = run.first_chunk + i, slot = run.first_slot + i;
      const uint64_t logical = std::min(m.chunk_bytes, m.primary_size - c * m.chunk_bytes);
      bool match = data.read(b.get(), m.chunk_bytes, slot * m.chunk_bytes) == m.chunk_bytes &&
                   ts::xxh64(b.get(), size_t(logical)) == m.hashes[size_t(slot)];
      // Padding is part of the physical format, even though it is excluded from the logical hash.
      if (match) {
        for (uint64_t k = logical; k < m.chunk_bytes; ++k) if (b.get()[k]) { match = false; break; }
      }
      if (check_primary) {
        match = primary->read(p.get(), (logical + 4095) / 4096 * 4096, c * m.chunk_bytes) >= logical &&
                ts::xxh64(p.get(), size_t(logical)) == m.hashes[size_t(slot)] && match;
      }
      if (!match) {
        std::fprintf(stderr, "ts-stripe: hash mismatch at slot %llu (chunk %llu)\n",
                     static_cast<unsigned long long>(slot), static_cast<unsigned long long>(c));
        ok = false;
        if (!keep_going) return false;
      }
      progress.print((slot + 1) * m.chunk_bytes, 0, m.data_size);
    }
  }
  require(data.identity() == before, SS_E_IO, "stripe changed during verification");
  return ok;
}

#if defined(TS_STRIPE_TESTING)
// Only the in-process test can bypass the volume guard; the installed CLI has no bypass.
bool test_allow_same_volume = false;
#endif

void make_stripe(const fs::path& primary_path, const fs::path& ranges, const fs::path& dir,
                 uint64_t chunk, bool keep_going) {
  ts::RawFile primary(path_text(primary_path), true);
  require(primary.direct(), SS_E_IO, "make requires unbuffered primary reads");
  const auto before = primary.identity();
  const auto base = path_text(primary_path.filename());
  // Digest components contain only hex digits; validate the derived name before any copy or output mutation.
  ts::stripe_data_name_validate(base, base + ".stripe-0000000000000000-00000000.data");
  auto chunks = placement(ranges, base, primary.size(), chunk);
  fs::create_directories(dir);
  require(!fs::equivalent(dir, fs::absolute(primary_path).parent_path()), SS_E_INVALID,
          "output directory is the primary directory");
  bool same_volume = directory_volume(dir) == before.volume;
#if defined(TS_STRIPE_TESTING)
  if (test_allow_same_volume) same_volume = false;
#endif
  require(!same_volume, SS_E_INVALID, "output directory is on the primary volume");
  OutputLock lock(dir, base);
  clean_partials(dir, base);
  const uint64_t data_size = uint64_t(chunks.size()) * chunk;
  require(fs::space(dir).available >= data_size + (uint64_t(1) << 30), SS_E_IO,
          "insufficient output free space (1 GiB margin required)");

  ts::StripeManifest m;
  m.chunk_bytes = chunk;
  m.primary_basename = base;
  m.primary_size = primary.size();
  m.data_size = data_size;
  // Hash the canonical chunk list incrementally; include the grid size so different grids cannot reuse a name.
  ts::Xxh64 list_hash;
  const auto prefix = "{\"chunk_bytes\":" + std::to_string(chunk) + ",\"chunks\":[";
  list_hash.update(prefix.data(), prefix.size());
  for (size_t slot = 0; slot < chunks.size(); ++slot) {
    const uint64_t c = chunks[slot];
    const auto item = (slot ? "," : "") + std::to_string(c);
    list_hash.update(item.data(), item.size());
    if (!m.runs.empty() && m.runs.back().first_chunk + m.runs.back().count == c) ++m.runs.back().count;
    else {
      require(m.runs.size() < ts::kStripeMaxRuns, SS_E_DESC, "too many stripe runs");
      m.runs.push_back({c, 1, uint64_t(slot)});
    }
  }
  list_hash.update("]}", 2);
  const auto placement_id = hex16(list_hash.digest()).substr(0, 8);
  auto partial = dir / path_of(base + ".stripe-pending-" + placement_id + ".data.partial");
  Progress progress("copy");
  ts::Sha256 sha;
  const char* inject = std::getenv("TS_STRIPE_FAIL_AFTER_CHUNKS");
  const uint64_t fail_after = inject ? ts::json_u64(ts::json_parse(inject), "TS_STRIPE_FAIL_AFTER_CHUNKS") : UINT64_MAX;
  uint64_t completed = 0, written = 0;
  std::string first_error;
  auto record_error = [&](const std::exception& e) {
    if (!keep_going) throw;
    if (first_error.empty()) first_error = e.what();
    std::fprintf(stderr, "ts-stripe: %s\n", e.what());
  };
  {
    StripeWriter writer(partial, data_size);
    IoPool read_pool, write_pool;
    struct Pending {
      Buffer bytes;
      std::future<void> io;  // drained before the buffer is destroyed, including on exceptions
      ~Pending() { if (io.valid()) io.wait(); }
    };
    std::array<Pending, kDepth> reads, writes;
    for (auto& r : reads) r.bytes = buffer(chunk);
    for (auto& w : writes) w.bytes = buffer(chunk);
    const uint64_t count = (primary.size() + chunk - 1) / chunk;
    auto start_read = [&](uint64_t c, Pending& r) {
      char* dst = r.bytes.get();
      const uint64_t logical = std::min(chunk, primary.size() - c * chunk);
      r.io = read_pool.submit([&, c, logical, dst]() {
        require(primary.read(dst, (logical + 4095) / 4096 * 4096, c * chunk) >= logical,
                SS_E_IO, "short primary read at chunk " + std::to_string(c));
      });
    };
    for (uint64_t c = 0; c < std::min<uint64_t>(count, kDepth); ++c) start_read(c, reads[size_t(c)]);
    size_t slot = 0;
    for (uint64_t c = 0; c < count; ++c) {
      require(c < fail_after, SS_E_IO, "injected failure after " + std::to_string(c) + " chunks");
      auto& r = reads[size_t(c % kDepth)];
      const uint64_t logical = std::min(chunk, primary.size() - c * chunk);
      bool read_ok = true;
      try { r.io.get(); } catch (const std::exception& e) { read_ok = false; record_error(e); }
      if (read_ok) sha.update(r.bytes.get(), size_t(logical));
      if (slot < chunks.size() && chunks[slot] == c) {
        auto& w = writes[slot % kDepth];
        if (w.io.valid()) {
          try { w.io.get(); written += chunk; } catch (const std::exception& e) { record_error(e); }
        }
        m.hashes.push_back(read_ok ? ts::xxh64(r.bytes.get(), size_t(logical)) : 0);
        if (read_ok) {
          std::memcpy(w.bytes.get(), r.bytes.get(), size_t(logical));
          std::memset(w.bytes.get() + logical, 0, size_t(chunk - logical));
          const char* src = w.bytes.get();
          const uint64_t offset = uint64_t(slot) * chunk;
          w.io = write_pool.submit([&, src, offset]() { writer.write(src, chunk, offset); });
        }
        ++slot;
      }
      completed += logical;
      progress.print(completed, written, primary.size());
      if (c + kDepth < count) start_read(c + kDepth, r);
    }
    for (auto& w : writes) if (w.io.valid()) {
      try { w.io.get(); } catch (const std::exception& e) { record_error(e); }
    }
    require(count < fail_after, SS_E_IO, "injected failure after " + std::to_string(count) + " chunks");
    require(first_error.empty(), SS_E_IO, "copy failed; manifest not published: " + first_error);
    writer.flush();
  }
  require(primary.identity() == before && ts::identity_of(path_text(primary_path)) == before, SS_E_IO,
          "primary changed during copy");
  m.primary_sha256 = sha.hex();
  m.data_name = base + ".stripe-" + m.primary_sha256.substr(0, 16) + "-" + placement_id + ".data";
  const auto named_partial = dir / path_of(m.data_name + ".partial");
  fs::rename(partial, named_partial);
  require(verify_slots(named_partial, m, &primary, keep_going), SS_E_IO, "stripe readback verification failed");
  const auto final = dir / path_of(m.data_name);
  if (fs::exists(final)) {
    require(verify_slots(final, m, &primary, keep_going), SS_E_IO, "existing versioned stripe failed verification");
    fs::remove(named_partial);
  } else {
#if defined(_WIN32)
    require(MoveFileExW(named_partial.c_str(), final.c_str(), MOVEFILE_WRITE_THROUGH) != 0, SS_E_IO,
            "publish stripe data: error " + std::to_string(GetLastError()));
#else
    require(chmod(named_partial.c_str(), 0444) == 0, SS_E_IO, "protect stripe data");
    fs::rename(named_partial, final);
#endif
  }
  require(primary.identity() == before && ts::identity_of(path_text(primary_path)) == before, SS_E_IO,
          "primary changed before publication");
  ts::write_file_atomic(path_text(dir / path_of(ts::stripe_manifest_name(base))), ts::stripe_manifest_write(m));
  std::printf("ts-stripe: made %s, %llu slots, %llu bytes\n", m.data_name.c_str(),
               static_cast<unsigned long long>(m.slots()), static_cast<unsigned long long>(data_size));
}

int verify_stripe(const fs::path& dir, const fs::path& primary_path, bool full) {
  const auto m = load_manifest(dir, path_text(primary_path.filename()));
  ts::RawFile primary(path_text(primary_path), true);
  const auto before = primary.identity();
  require(primary.size() == m.primary_size, SS_E_DESC, "primary size mismatch");
  bool ok = verify_slots(dir / path_of(m.data_name), m, &primary, true, full);
  if (full) {
    require(primary.direct(), SS_E_IO, "full verification requires unbuffered primary reads");
    ts::Sha256 sha;
    auto b = buffer(uint64_t(16) << 20);
    Progress progress("verify primary");
    for (uint64_t offset = 0; offset < primary.size(); offset += uint64_t(16) << 20) {
      const uint64_t logical = std::min(uint64_t(16) << 20, primary.size() - offset);
      require(primary.read(b.get(), (logical + 4095) / 4096 * 4096, offset) >= logical, SS_E_IO, "short primary read");
      sha.update(b.get(), size_t(logical));
      progress.print(offset + logical, 0, primary.size());
    }
    if (sha.hex() != m.primary_sha256) { std::fprintf(stderr, "ts-stripe: primary SHA-256 mismatch\n"); ok = false; }
  }
  require(primary.identity() == before && ts::identity_of(path_text(primary_path)) == before, SS_E_IO,
          "primary changed during verification");
  std::printf("ts-stripe: verify %s, %llu slots%s\n", ok ? "OK" : "FAILED",
               static_cast<unsigned long long>(m.slots()), full ? ", full primary SHA-256 checked" : "");
  return ok ? 0 : 1;
}

void gc_stripes(const fs::path& dir, const std::string& base, bool dry_run) {
  ts::stripe_manifest_name(base);
  OutputLock lock(dir, base);
  // A missing or invalid manifest is an error: gc never guesses which data is live.
  const auto m = load_manifest(dir, base);
  std::vector<fs::path> garbage;
  for (const auto& e : fs::directory_iterator(dir)) {
    const auto name = path_text(e.path().filename());
    bool current = name == m.data_name;
#if defined(_WIN32)
    const auto live_name = path_of(m.data_name).wstring();
    const auto entry_name = e.path().filename().wstring();
    current = CompareStringOrdinal(live_name.c_str(), -1, entry_name.c_str(), -1, TRUE) == CSTR_EQUAL;
#endif
    if (starts(name, base + ".stripe-") && ends(name, ".data") && !current) {
      require(e.is_regular_file() && !e.is_symlink(), SS_E_IO, "unexpected stripe data file type");
      garbage.push_back(e.path());
    }
  }
  std::sort(garbage.begin(), garbage.end());
  for (const auto& path : garbage) {
    std::printf("%s\n", path_text(path.filename()).c_str());
    if (!dry_run) {
      std::error_code ec;
      fs::remove(path, ec);
      if (ec) {
#if defined(_WIN32)
        if (ec.value() == ERROR_SHARING_VIOLATION || ec.value() == ERROR_LOCK_VIOLATION) {
          std::fprintf(stderr, "ts-stripe: skipped busy file %s: %s\n",
                       path_text(path.filename()).c_str(), ec.message().c_str());
          continue;
        }
#endif
        ts::fail(SS_E_IO, "remove stripe data: " + ec.message());
      }
    }
  }
}

void show_stripe(const fs::path& dir, const std::string& base) {
  const auto m = load_manifest(dir, base);
  std::printf("chunk_bytes: %llu\nslots: %llu\nbytes: %llu\ndata: %s\nruns: %llu\n",
               static_cast<unsigned long long>(m.chunk_bytes), static_cast<unsigned long long>(m.slots()),
               static_cast<unsigned long long>(m.data_size), m.data_name.c_str(),
               static_cast<unsigned long long>(m.runs.size()));
  for (size_t i = 0; i < std::min<size_t>(m.runs.size(), 8); ++i) {
    const auto& r = m.runs[i];
    std::printf("  [%llu, %llu, %llu]\n", static_cast<unsigned long long>(r.first_chunk),
                 static_cast<unsigned long long>(r.count), static_cast<unsigned long long>(r.first_slot));
  }
}

int stripe_main(int argc, char** argv) {
  try {
    require(argc >= 4, SS_E_INVALID,
            "usage: ts-stripe make PRIMARY RANGES.json OUT_DIR [--chunk-mib 16] [--keep-going]\n"
            "       ts-stripe verify DIR PRIMARY [--full]\n"
            "       ts-stripe gc DIR PRIMARY_BASENAME [--dry-run]\n"
            "       ts-stripe show DIR PRIMARY_BASENAME");
    const std::string command = argv[1];
    if (command == "make") {
      require(argc >= 5, SS_E_INVALID, "make needs PRIMARY RANGES.json OUT_DIR");
      uint64_t mib = 16;
      bool keep_going = false;
      for (int i = 5; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--keep-going") keep_going = true;
        else if (a == "--chunk-mib") {
          require(i + 1 < argc, SS_E_INVALID, "--chunk-mib needs a value");
          mib = ts::json_u64(ts::json_parse(argv[++i]), "--chunk-mib");
        } else ts::fail(SS_E_INVALID, "unknown argument " + a);
      }
      require(mib && mib <= 256 && !(mib & (mib - 1)), SS_E_INVALID,
              "--chunk-mib must be a power of two in [1, 256]");
      make_stripe(path_of(argv[2]), path_of(argv[3]), path_of(argv[4]), mib * kMiB, keep_going);
    } else if (command == "verify") {
      require(argc == 4 || (argc == 5 && std::string(argv[4]) == "--full"), SS_E_INVALID, "unknown verify argument");
      return verify_stripe(path_of(argv[2]), path_of(argv[3]), argc == 5);
    } else if (command == "gc") {
      require(argc == 4 || (argc == 5 && std::string(argv[4]) == "--dry-run"), SS_E_INVALID, "unknown gc argument");
      gc_stripes(path_of(argv[2]), argv[3], argc == 5);
    } else if (command == "show") {
      require(argc == 4, SS_E_INVALID, "unknown show argument");
      show_stripe(path_of(argv[2]), argv[3]);
    } else ts::fail(SS_E_INVALID, "unknown command " + command);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ts-stripe: %s%s\n", argc > 1 && std::string(argv[1]) == "verify" ? "verify FAILED: " : "",
                 e.what());
    return 1;
  }
}

}  // namespace

#if !defined(TS_STRIPE_TESTING)
int main(int argc, char** argv) { return stripe_main(argc, argv); }
#endif
