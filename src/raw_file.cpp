#include "raw_file.hpp"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>

#include "errors.hpp"
#include "sha256.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ts {

namespace {

constexpr uint64_t kHashPiece = uint64_t(64) << 20;

struct AlignedFree {
  void operator()(char* p) const {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};
std::unique_ptr<char, AlignedFree> aligned_buffer(size_t bytes) {
#if defined(_WIN32)
  char* p = static_cast<char*>(_aligned_malloc(bytes, 4096));
#else
  void* raw = nullptr;
  if (posix_memalign(&raw, 4096, bytes) != 0) raw = nullptr;
  char* p = static_cast<char*>(raw);
#endif
  require(p != nullptr, SS_E_NOMEM, "aligned buffer of " + std::to_string(bytes) + " bytes");
  return std::unique_ptr<char, AlignedFree>(p);
}

std::string hex16(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

std::mutex bound_hook_mu;
std::function<void(const std::string&)> bound_hook;
std::atomic<bool> bound_hook_set{false};

// Test hook: runs between the resolve and the bound walk.
void bound_resolved(const std::string& real) {
  if (!bound_hook_set.load()) return;
  std::function<void(const std::string&)> hook;
  {
    std::lock_guard<std::mutex> lock(bound_hook_mu);
    hook = bound_hook;
  }
  if (hook) hook(real);
}

std::atomic<uint64_t> g_fail_piece{0};
std::atomic<uint32_t> g_max_in_flight{0};

// read_many's test hooks: the n-th piece from now fails; the most pieces one batch had in flight
bool test_fail_piece_hit() {
  uint64_t v = g_fail_piece.load(std::memory_order_relaxed);
  while (v) {
    if (g_fail_piece.compare_exchange_weak(v, v - 1)) return v == 1;
  }
  return false;
}

void note_in_flight(uint32_t n) {
  uint32_t m = g_max_in_flight.load(std::memory_order_relaxed);
  while (n > m && !g_max_in_flight.compare_exchange_weak(m, n)) {
  }
}

}  // namespace

void test_bound_resolved_hook(std::function<void(const std::string&)> hook) {
  std::lock_guard<std::mutex> lock(bound_hook_mu);
  bound_hook_set.store(bool(hook));
  bound_hook = std::move(hook);
}
void RawFile::test_fail_piece(uint64_t n) { g_fail_piece.store(n); }
uint32_t RawFile::test_max_in_flight() { return g_max_in_flight.load(); }

#if defined(_WIN32)

namespace {
std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), &w[0], n);
  return w;
}

uint64_t ft(const FILETIME& t) { return uint64_t(t.dwHighDateTime) << 32 | t.dwLowDateTime; }

}  // namespace

FileIdentity identity_of_handle(void* h, const std::string& path) {
  BY_HANDLE_FILE_INFORMATION bi;
  require(GetFileInformationByHandle(h, &bi) != 0, SS_E_IO, "identity of " + path + ": error " +
                                                               std::to_string(GetLastError()));
  FileIdentity id;
  id.volume = bi.dwVolumeSerialNumber;
  id.index = uint64_t(bi.nFileIndexHigh) << 32 | bi.nFileIndexLow;
  id.size = uint64_t(bi.nFileSizeHigh) << 32 | bi.nFileSizeLow;
  id.mtime = ft(bi.ftLastWriteTime);
  // creation time, not the NTFS change time: metadata-only events (another hard link removed, attributes) move the
  // change time and would invalidate verified caches of unchanged bytes; content writes move the last-write time
  id.ctime = ft(bi.ftCreationTime);
  return id;
}

namespace {
// Paths past MAX_PATH need the \\?\ prefix (the path is already normalised: nothing is lost).
std::wstring long_path(const std::wstring& w) {
  if (w.size() < 240 || w.rfind(L"\\\\?\\", 0) == 0) return w;
  if (w.rfind(L"\\\\", 0) == 0) return L"\\\\?\\UNC\\" + w.substr(2);
  return L"\\\\?\\" + w;
}

DWORD attributes_of(HANDLE h) {
  FILE_ATTRIBUTE_TAG_INFO tag{};
  if (GetFileInformationByHandleEx(h, FileAttributeTagInfo, &tag, sizeof tag)) return tag.FileAttributes;
  BY_HANDLE_FILE_INFORMATION bi;
  if (GetFileInformationByHandle(h, &bi)) return bi.dwFileAttributes;
  return INVALID_FILE_ATTRIBUTES;
}

std::wstring final_path(HANDLE h) {
  std::wstring out(512, L'\0');
  for (int tries = 0; tries < 2; ++tries) {
    DWORD n = GetFinalPathNameByHandleW(h, &out[0], DWORD(out.size()), FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (!n) return std::wstring();
    if (n < out.size()) {
      out.resize(n);
      return out;
    }
    out.assign(size_t(n) + 1, L'\0');
  }
  return std::wstring();
}

std::string narrow(const std::wstring& w) {
  if (w.empty()) return std::string();
  int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), &s[0], n, nullptr, nullptr);
  return s;
}
}  // namespace

BoundDir::BoundDir(const std::string& file_path) {
  namespace fs = std::filesystem;
  // resolve once: the real final path (links followed, short names expanded), then the bound walk of that path
  std::wstring w;
  {
    fs::path given_path = fs::absolute(fs::u8path(file_path)).lexically_normal();
    const std::wstring given = long_path(given_path.make_preferred().wstring());
    HANDLE h = CreateFileW(given.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    require(h != INVALID_HANDLE_VALUE, SS_E_IO, "open " + file_path + ": error " + std::to_string(GetLastError()));
    w = final_path(h);
    const DWORD e = GetLastError();
    CloseHandle(h);
    require(!w.empty(), SS_E_IO, "open " + file_path + ": final path: error " + std::to_string(e));
  }
  bound_resolved(narrow(w));
  fs::path p;
  // a \\?\ path is walked in its DOS form (its root is then the drive or the share, not the device namespace)
  if (w.rfind(L"\\\\?\\UNC\\", 0) == 0) w = L"\\\\" + w.substr(8);
  else if (w.rfind(L"\\\\?\\", 0) == 0) w = w.substr(4);
  p = fs::path(w);
  path_ = p.u8string();
  require(p.has_filename() && p.has_root_name() && p.has_root_directory(), SS_E_IO,
          "open " + file_path + ": not an absolute file path");
  leaf_ = p.filename().u8string();
  require(leaf_ != "." && leaf_ != "..", SS_E_IO, "open " + path_ + ": not a file path");
  wpath_ = long_path(w);
  try {
    fs::path dir = p.root_path();
    std::vector<fs::path> chain{dir};
    for (const auto& part : p.relative_path().parent_path()) {
      require(part != L"." && part != L"..", SS_E_IO, "open " + path_ + ": relative component");
      dir /= part;
      chain.push_back(dir);
    }
    // each directory from the volume root down: held without FILE_SHARE_DELETE (no rename or delete while held), a
    // reparse point (junction, symlink, mount point) refused. FILE_TRAVERSE counts for share access; attributes alone
    // would not.
    for (const fs::path& d : chain) {
      const std::wstring dw = long_path(d.wstring());
      const DWORD share = FILE_SHARE_READ | FILE_SHARE_WRITE;
      const DWORD flags = FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT;
      HANDLE h = CreateFileW(dw.c_str(), FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE, share, nullptr,
                             OPEN_EXISTING, flags, nullptr);
      if (h == INVALID_HANDLE_VALUE && GetLastError() == ERROR_ACCESS_DENIED)
        h = CreateFileW(dw.c_str(), FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES | SYNCHRONIZE, share, nullptr,
                        OPEN_EXISTING, flags, nullptr);
      require(h != INVALID_HANDLE_VALUE, SS_E_IO,
              "open " + path_ + ": directory " + d.u8string() + ": error " + std::to_string(GetLastError()));
      dirs_.push_back(h);
      const DWORD a = attributes_of(h);
      require(a != INVALID_FILE_ATTRIBUTES, SS_E_IO,
              "open " + path_ + ": attributes of " + d.u8string() + ": error " + std::to_string(GetLastError()));
      require(!(a & FILE_ATTRIBUTE_REPARSE_POINT), SS_E_IO,
              "open " + path_ + ": " + d.u8string() + " is a reparse point (junction or link)");
      require((a & FILE_ATTRIBUTE_DIRECTORY) != 0, SS_E_IO, "open " + path_ + ": " + d.u8string() + " is not a directory");
    }
    dir_final_ = final_path(static_cast<HANDLE>(dirs_.back()));
    require(!dir_final_.empty(), SS_E_IO,
            "open " + path_ + ": final path of its directory: error " + std::to_string(GetLastError()));
  } catch (...) {
    for (void* h : dirs_) CloseHandle(static_cast<HANDLE>(h));
    dirs_.clear();
    throw;
  }
}

BoundDir::~BoundDir() {
  for (void* h : dirs_) CloseHandle(static_cast<HANDLE>(h));
}

void* BoundDir::open_leaf(unsigned long access, unsigned long share, unsigned long flags) const {
  return CreateFileW(wpath_.c_str(), DWORD(access), DWORD(share), nullptr, OPEN_EXISTING,
                     DWORD(flags) | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
}

void BoundDir::check_leaf(void* handle) const {
  HANDLE h = static_cast<HANDLE>(handle);
  const DWORD a = attributes_of(h);
  require(a != INVALID_FILE_ATTRIBUTES, SS_E_IO, "attributes of " + path_ + ": error " + std::to_string(GetLastError()));
  require(!(a & FILE_ATTRIBUTE_REPARSE_POINT), SS_E_IO, "open " + path_ + ": a reparse point (link)");
  require(!(a & FILE_ATTRIBUTE_DIRECTORY), SS_E_IO, "open " + path_ + ": not a regular file");
  // the leaf must be the entry of the held directory: its final path is the directory's plus the leaf name
  std::wstring expect = dir_final_;
  if (expect.back() != L'\\') expect += L'\\';
  expect += std::filesystem::u8path(leaf_).wstring();
  const std::wstring got = final_path(h);
  require(!got.empty() && CompareStringOrdinal(got.c_str(), int(got.size()), expect.c_str(), int(expect.size()),
                                               TRUE) == CSTR_EQUAL,
          SS_E_IO, "open " + path_ + ": its final path " + narrow(got) + " is not in the held directory");
}

RawFile::RawFile(const std::string& path, bool direct, bool bound) : path_(path) {
  std::wstring w = widen(path);
  if (bound) dirs_ = std::make_unique<BoundDir>(path);
  auto open = [&](DWORD flags) {
    return dirs_ ? static_cast<HANDLE>(dirs_->open_leaf(GENERIC_READ, FILE_SHARE_READ, flags))
                 : CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
  };
  // share mode read-only: no writer can open the file while this handle lives (and an open writer makes this fail)
  DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  HANDLE h = open(flags);
  if (h == INVALID_HANDLE_VALUE && direct && GetLastError() == ERROR_INVALID_PARAMETER) {
    h = open(FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED);
    direct = false;
  }
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "open " + path + ": error " + std::to_string(GetLastError()));
  if (dirs_) {
    try {
      dirs_->check_leaf(h);
    } catch (...) {
      CloseHandle(h);
      throw;
    }
  }
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(h, &sz)) {
    DWORD e = GetLastError();
    CloseHandle(h);
    fail(SS_E_IO, "size of " + path + ": error " + std::to_string(e));
  }
  h_ = h;
  direct_ = direct;
  size_ = uint64_t(sz.QuadPart);
}

RawFile::~RawFile() {
  if (h_) CloseHandle(static_cast<HANDLE>(h_));
}

FileIdentity RawFile::identity() const { return identity_of_handle(static_cast<HANDLE>(h_), path_); }

uint64_t RawFile::read(void* dst, uint64_t n, uint64_t offset) const {
  // one event per reading thread, closed when the thread exits
  struct ThreadEvent {
    HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~ThreadEvent() { if (h) CloseHandle(h); }
  };
  thread_local ThreadEvent te;
  HANDLE event = te.h;
  require(event != nullptr, SS_E_IO, "read event: error " + std::to_string(GetLastError()));
  uint64_t done = 0;
  char* p = static_cast<char*>(dst);
  while (done < n) {
    DWORD want = DWORD(std::min<uint64_t>(n - done, uint64_t(1) << 30));
    uint64_t off = offset + done;
    OVERLAPPED ov{};
    ov.Offset = DWORD(off & 0xFFFFFFFFu);
    ov.OffsetHigh = DWORD(off >> 32);
    ov.hEvent = event;
    DWORD got = 0;
    if (!ReadFile(static_cast<HANDLE>(h_), p + done, want, nullptr, &ov)) {
      DWORD e = GetLastError();
      if (e == ERROR_HANDLE_EOF) break;
      if (e != ERROR_IO_PENDING) fail(SS_E_IO, "read " + path_ + " at " + std::to_string(off) + ": error " + std::to_string(e));
    }
    if (!GetOverlappedResult(static_cast<HANDLE>(h_), &ov, &got, TRUE)) {
      DWORD e = GetLastError();
      if (e == ERROR_HANDLE_EOF) break;
      fail(SS_E_IO, "read " + path_ + " at " + std::to_string(off) + ": error " + std::to_string(e));
    }
    done += got;
    if (got < want) break;  // end of file
  }
  return done;
}

uint64_t RawFile::read_split(void* dst, uint64_t n, uint64_t offset, uint64_t part,
                            const std::function<void(const char*, uint64_t)>& on_part) const {
  struct Event {
    HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~Event() { if (h) CloseHandle(h); }
  };
  thread_local Event events[kMaxSplit];
  // checked before anything is issued: nothing below may throw while a read is in flight
  for (const Event& e : events) require(e.h != nullptr, SS_E_IO, "read event: error " + std::to_string(GetLastError()));
  char* p = static_cast<char*>(dst);
  uint64_t done = 0;
  while (done < n) {
    // one batch of up to kMaxSplit parts in flight, completed (and handed to on_part) in order
    OVERLAPPED ov[kMaxSplit];
    DWORD want[kMaxSplit], got[kMaxSplit];
    uint32_t issued = 0;
    bool eof = false;
    DWORD error = 0;  // the first failure, reported once every issued read has completed
    uint64_t error_at = 0;
    for (uint64_t at = done; at < n && issued < kMaxSplit; at += part) {
      want[issued] = DWORD(std::min(part, n - at));
      uint64_t off = offset + at;
      ov[issued] = OVERLAPPED{};
      ov[issued].Offset = DWORD(off & 0xFFFFFFFFu);
      ov[issued].OffsetHigh = DWORD(off >> 32);
      ov[issued].hEvent = events[issued].h;
      if (!ReadFile(static_cast<HANDLE>(h_), p + at, want[issued], nullptr, &ov[issued])) {
        DWORD e = GetLastError();
        if (e == ERROR_HANDLE_EOF) {  // nothing was queued: this part and the rest are past the end of the file
          eof = true;
          break;
        }
        if (e != ERROR_IO_PENDING) {
          error = e;
          error_at = off;
          break;
        }
      }
      ++issued;
    }
    // parts are handed to on_part in order as each one completes, while the later ones are still being read; every
    // issued read is waited for before anything is thrown (none may outlive this call)
    const uint64_t batch_start = done;
    bool stopped = false;  // a short part (end of file), an error or an on_part exception: no more callbacks
    std::exception_ptr thrown;
    for (uint32_t i = 0; i < issued; ++i) {
      got[i] = 0;
      if (!GetOverlappedResult(static_cast<HANDLE>(h_), &ov[i], &got[i], TRUE)) {
        DWORD e = GetLastError();
        if (e != ERROR_HANDLE_EOF && !error) {
          error = e;
          error_at = offset + batch_start + uint64_t(i) * part;
        }
      }
      if (stopped || error) continue;
      try {
        on_part(p + done, got[i]);
      } catch (...) {
        thrown = std::current_exception();
        stopped = true;
        continue;
      }
      done += got[i];
      if (got[i] < want[i]) stopped = eof = true;  // end of file
    }
    if (error) fail(SS_E_IO, "read " + path_ + " at " + std::to_string(error_at) + ": error " + std::to_string(error));
    if (thrown) std::rethrow_exception(thrown);
    if (eof || !issued) return done;
  }
  return done;
}

void RawFile::read_many(const Piece* reads, uint32_t n) const {
  struct Event {
    HANDLE h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    ~Event() { if (h) CloseHandle(h); }
  };
  thread_local Event events[kMaxSplit];
  // checked before anything is issued: nothing below may throw while a read is in flight
  for (const Event& e : events) require(e.h != nullptr, SS_E_IO, "read event: error " + std::to_string(GetLastError()));
  DWORD error = 0;  // the first failure, reported once every issued read has completed
  uint64_t error_at = 0;
  for (uint32_t first = 0; first < n && !error; first += kMaxSplit) {
    OVERLAPPED ov[kMaxSplit];
    bool issued[kMaxSplit] = {};
    const uint32_t count = std::min<uint32_t>(kMaxSplit, n - first);
    for (uint32_t i = 0; i < count; ++i) {
      const Piece& r = reads[first + i];
      ov[i] = OVERLAPPED{};
      ov[i].Offset = DWORD(r.offset & 0xFFFFFFFFu);
      ov[i].OffsetHigh = DWORD(r.offset >> 32);
      ov[i].hEvent = events[i].h;
      if (test_fail_piece_hit()) {
        if (!error) {
          error = ERROR_READ_FAULT;
          error_at = r.offset;
        }
        continue;
      }
      if (r.bytes > 0xFFFFFFFFull || r.offset + r.bytes > size_) {
        if (!error) {
          error = ERROR_INVALID_PARAMETER;
          error_at = r.offset;
        }
        continue;
      }
      if (!ReadFile(static_cast<HANDLE>(h_), r.dst, DWORD(r.bytes), nullptr, &ov[i])) {
        const DWORD e = GetLastError();
        if (e != ERROR_IO_PENDING) {
          if (!error) {
            error = e;
            error_at = r.offset;
          }
          continue;
        }
      }
      issued[i] = true;
    }
    uint32_t in_flight = 0;
    for (uint32_t i = 0; i < count; ++i) in_flight += issued[i] ? 1u : 0u;
    note_in_flight(in_flight);
    for (uint32_t i = 0; i < count; ++i) {
      if (!issued[i]) continue;
      DWORD got = 0;
      const BOOL ok = GetOverlappedResult(static_cast<HANDLE>(h_), &ov[i], &got, TRUE);
      if ((!ok || got != reads[first + i].bytes) && !error) {
        error = ok ? DWORD(ERROR_HANDLE_EOF) : GetLastError();
        error_at = reads[first + i].offset;
      }
    }
  }
  if (error) fail(SS_E_IO, "read " + path_ + " at " + std::to_string(error_at) + ": error " + std::to_string(error));
}

FileIdentity identity_of(const std::string& path) {
  HANDLE h = CreateFileW(widen(path).c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "open " + path + ": error " + std::to_string(GetLastError()));
  FileIdentity id;
  try {
    id = identity_of_handle(h, path);
  } catch (...) {
    CloseHandle(h);
    throw;
  }
  CloseHandle(h);
  return id;
}

std::string cache_dir() {
  const char* base = std::getenv("LOCALAPPDATA");
  if (!base || !*base) return std::string();
  std::string d = std::string(base) + "\\tierstream";
  CreateDirectoryW(widen(d).c_str(), nullptr);
  return d;
}

void write_file_atomic(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp";
  HANDLE h = CreateFileW(widen(tmp).c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "create " + tmp + ": error " + std::to_string(GetLastError()));
  DWORD put = 0;
  bool ok = WriteFile(h, text.data(), DWORD(text.size()), &put, nullptr) && put == text.size() && FlushFileBuffers(h);
  CloseHandle(h);
  require(ok, SS_E_IO, "write " + tmp);
  require(MoveFileExW(widen(tmp).c_str(), widen(path).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0,
          SS_E_IO, "replace " + path + ": error " + std::to_string(GetLastError()));
}

#else

namespace {
FileIdentity identity_of_stat(const struct stat& st) {
  FileIdentity id;
  id.volume = uint64_t(st.st_dev);
  id.index = uint64_t(st.st_ino);
  id.size = uint64_t(st.st_size);
  id.mtime = uint64_t(st.st_mtim.tv_sec) * 1000000000ull + uint64_t(st.st_mtim.tv_nsec);
  id.ctime = uint64_t(st.st_ctim.tv_sec) * 1000000000ull + uint64_t(st.st_ctim.tv_nsec);
  return id;
}
}  // namespace

BoundDir::BoundDir(const std::string& file_path) {
  namespace fs = std::filesystem;
  // resolve once: the real path (links followed), then the bound walk of that path
  char* real = ::realpath(fs::absolute(fs::u8path(file_path)).c_str(), nullptr);
  require(real != nullptr, SS_E_IO, "open " + file_path + ": " + std::strerror(errno));
  fs::path p(real);
  std::free(real);
  bound_resolved(p.u8string());
  path_ = p.u8string();
  require(p.has_filename() && p.has_root_directory(), SS_E_IO, "open " + file_path + ": not an absolute file path");
  leaf_ = p.filename().u8string();
  require(leaf_ != "." && leaf_ != "..", SS_E_IO, "open " + path_ + ": not a file path");
  try {
    int fd = ::open("/", O_PATH | O_DIRECTORY | O_CLOEXEC);
    require(fd >= 0, SS_E_IO, "open " + path_ + ": /: " + std::strerror(errno));
    dirs_.push_back(fd);
    // each component relative to its held parent, never following a symlink (ELOOP / ENOTDIR on a link)
    for (const auto& part : p.relative_path().parent_path()) {
      const std::string name = part.u8string();
      require(name != "." && name != "..", SS_E_IO, "open " + path_ + ": relative component");
      fd = ::openat(dirs_.back(), name.c_str(), O_PATH | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      require(fd >= 0, SS_E_IO,
              "open " + path_ + ": directory " + name + " (a link is refused): " + std::strerror(errno));
      dirs_.push_back(fd);
      struct stat st;
      require(fstat(fd, &st) == 0 && S_ISDIR(st.st_mode), SS_E_IO,
              "open " + path_ + ": " + name + " is not a directory (a link is refused)");
    }
  } catch (...) {
    for (int fd : dirs_) ::close(fd);
    dirs_.clear();
    throw;
  }
}

BoundDir::~BoundDir() {
  for (int fd : dirs_) ::close(fd);
}

int BoundDir::open_leaf(int flags) const {
  return ::openat(dirs_.back(), leaf_.c_str(), flags | O_NOFOLLOW | O_CLOEXEC);
}

void BoundDir::check_leaf(int fd) const {
  struct stat st;
  require(fstat(fd, &st) == 0, SS_E_IO, "stat " + path_ + ": " + std::strerror(errno));
  require(S_ISREG(st.st_mode), SS_E_IO, "not a regular file: " + path_);
}

RawFile::RawFile(const std::string& path, bool direct, bool bound) : path_(path) {
  if (bound) dirs_ = std::make_unique<BoundDir>(path);
  auto open = [&](int flags) { return dirs_ ? dirs_->open_leaf(flags) : ::open(path.c_str(), flags); };
  int fd = -1;
  if (direct) {
    fd = open(O_RDONLY | O_DIRECT | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) direct = false;
    else if (fd < 0) {
      const int e = errno;
      fail(SS_E_IO, "open " + path + ": " + std::strerror(e) + (e == ELOOP ? " (a link)" : ""));
    }
  }
  if (fd < 0) {
    fd = open(O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0) {
      const int e = errno;
      fail(SS_E_IO, "open " + path + ": " + std::strerror(e) + (e == ELOOP ? " (a link)" : ""));
    }
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    int e = errno;
    ::close(fd);
    fail(SS_E_IO, "stat " + path + ": " + std::strerror(e));
  }
  if (!S_ISREG(st.st_mode)) {
    ::close(fd);
    fail(SS_E_IO, "not a regular file: " + path);
  }
  fd_ = fd;
  direct_ = direct;
  size_ = uint64_t(st.st_size);
}

RawFile::~RawFile() {
  if (fd_ >= 0) ::close(fd_);
}

FileIdentity identity_of_fd(int fd, const std::string& path) {
  struct stat st;
  require(fstat(fd, &st) == 0, SS_E_IO, "stat " + path + ": " + std::strerror(errno));
  return identity_of_stat(st);
}

FileIdentity RawFile::identity() const { return identity_of_fd(fd_, path_); }

uint64_t RawFile::read(void* dst, uint64_t n, uint64_t offset) const {
  uint64_t done = 0;
  char* p = static_cast<char*>(dst);
  while (done < n) {
    size_t want = size_t(std::min<uint64_t>(n - done, uint64_t(1) << 30));
    ssize_t got = ::pread(fd_, p + done, want, off_t(offset + done));
    if (got < 0) {
      if (errno == EINTR) continue;
      fail(SS_E_IO, "read " + path_ + " at " + std::to_string(offset + done) + ": " + std::strerror(errno));
    }
    done += uint64_t(got);
    if (size_t(got) < want) break;  // end of file
  }
  return done;
}

uint64_t RawFile::read_split(void* dst, uint64_t n, uint64_t offset, uint64_t part,
                            const std::function<void(const char*, uint64_t)>& on_part) const {
  // positional reads in order (the pools give the queue depth here)
  char* p = static_cast<char*>(dst);
  uint64_t done = 0;
  while (done < n) {
    uint64_t want = std::min(part, n - done), got = read(p + done, want, offset + done);
    on_part(p + done, got);
    done += got;
    if (got < want) break;
  }
  return done;
}

void RawFile::read_many(const Piece* reads, uint32_t n) const {
  // positional reads in order (no overlapped handle here); the same all-or-throw contract
  uint64_t error_at = UINT64_MAX;
  for (uint32_t i = 0; i < n; ++i) {
    const Piece& r = reads[i];
    if (test_fail_piece_hit()) {
      if (error_at == UINT64_MAX) error_at = r.offset;
      continue;
    }
    if (error_at != UINT64_MAX) continue;
    if (read(r.dst, r.bytes, r.offset) != r.bytes) error_at = r.offset;
  }
  note_in_flight(n ? 1u : 0u);
  if (error_at != UINT64_MAX) fail(SS_E_IO, "read " + path_ + " at " + std::to_string(error_at) + ": short or failed");
}

FileIdentity identity_of(const std::string& path) {
  struct stat st;
  require(::stat(path.c_str(), &st) == 0, SS_E_IO, "stat " + path + ": " + std::strerror(errno));
  return identity_of_stat(st);
}

std::string cache_dir() {
  std::string base;
  if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) base = x;
  else if (const char* h = std::getenv("HOME"); h && *h) base = std::string(h) + "/.cache";
  else return std::string();
  ::mkdir(base.c_str(), 0700);
  std::string d = base + "/tierstream";
  ::mkdir(d.c_str(), 0700);
  return d;
}

void write_file_atomic(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp";
  int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
  require(fd >= 0, SS_E_IO, "create " + tmp + ": " + std::strerror(errno));
  bool ok = ::write(fd, text.data(), text.size()) == ssize_t(text.size()) && ::fsync(fd) == 0;
  ::close(fd);
  require(ok, SS_E_IO, "write " + tmp);
  require(::rename(tmp.c_str(), path.c_str()) == 0, SS_E_IO, "replace " + path + ": " + std::strerror(errno));
}

#endif

void check_identity(const FileIdentity& id, const FileIdentity& expected, const std::string& what) {
  if (!expected.volume && !expected.index && !expected.size) return;
  require(id.volume == expected.volume && id.index == expected.index && id.size == expected.size, SS_E_DESC,
          what + ": file identity " + std::to_string(id.volume) + ":" + std::to_string(id.index) + " (" +
              std::to_string(id.size) + " bytes) is not the expected " + std::to_string(expected.volume) + ":" +
              std::to_string(expected.index) + " (" + std::to_string(expected.size) + " bytes)");
}

std::string sha256_file(const std::string& path) { return sha256_of(RawFile(path, true)); }

std::string sha256_of(const RawFile& f) {
  const std::string& path = f.path();
  // two buffers: the next piece is read while the current one is hashed
  auto a = aligned_buffer(size_t(kHashPiece)), b = aligned_buffer(size_t(kHashPiece));
  Sha256 h;
  const FileIdentity before = f.identity();
  for (uint64_t off = 0; off < f.size(); off += kHashPiece) {
    uint64_t want = std::min(kHashPiece, f.size() - off);
    uint64_t got = f.read(a.get(), (want + 4095) / 4096 * 4096, off);
    require(got >= want, SS_E_IO, "short read hashing " + path);
    h.update(a.get(), size_t(want));
    std::swap(a, b);
  }
  require(f.identity() == before, SS_E_IO, path + " changed while it was hashed");
  return h.hex();
}

std::string file_sha256_cached(const RawFile& f) {
  const FileIdentity id = f.identity();
  const std::string dir = cache_dir();
  std::string key;
  if (!dir.empty()) {
    key = dir +
#if defined(_WIN32)
          "\\file-sha256-" +
#else
          "/file-sha256-" +
#endif
          hex16(id.volume) + "-" + hex16(id.index);
    if (FILE* fp = std::fopen(key.c_str(), "rb")) {
      char line[256] = {};
      size_t n = std::fread(line, 1, sizeof line - 1, fp);
      std::fclose(fp);
      line[n] = 0;
      unsigned long long sz = 0, mt = 0, ct = 0;
      char hex[65] = {};
      if (std::sscanf(line, "%llu %llu %llu %64s", &sz, &mt, &ct, hex) == 4 && sz == id.size && mt == id.mtime &&
          ct == id.ctime && std::strlen(hex) == 64)
        return hex;
    }
  }
  const std::string digest = sha256_of(f);
  if (!dir.empty() && f.identity() == id) {
    char line[256];
    std::snprintf(line, sizeof line, "%llu %llu %llu %s\n", static_cast<unsigned long long>(id.size),
                  static_cast<unsigned long long>(id.mtime), static_cast<unsigned long long>(id.ctime), digest.c_str());
    try {
      write_file_atomic(key, line);
    } catch (const Error&) {
      // a cache that cannot be written only costs the next open another full read
    }
  }
  return digest;
}

}  // namespace ts
