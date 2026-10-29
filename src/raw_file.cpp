#include "raw_file.hpp"

#include <algorithm>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

}  // namespace

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

FileIdentity identity_of_handle(HANDLE h, const std::string& path) {
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
}  // namespace

RawFile::RawFile(const std::string& path, bool direct) : path_(path) {
  std::wstring w = widen(path);
  // share mode read-only: no writer can open the file while this handle lives (and an open writer makes this fail)
  DWORD flags = FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED | (direct ? FILE_FLAG_NO_BUFFERING : 0);
  HANDLE h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, flags, nullptr);
  if (h == INVALID_HANDLE_VALUE && direct && GetLastError() == ERROR_INVALID_PARAMETER) {
    h = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    direct = false;
  }
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "open " + path + ": error " + std::to_string(GetLastError()));
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

RawFile::RawFile(const std::string& path, bool direct) : path_(path) {
  int fd = -1;
  if (direct) {
    fd = ::open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC | O_NONBLOCK);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) direct = false;
    else if (fd < 0) fail(SS_E_IO, "open " + path + ": " + std::strerror(errno));
  }
  if (fd < 0) {
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK);
    require(fd >= 0, SS_E_IO, "open " + path + ": " + std::strerror(errno));
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

FileIdentity RawFile::identity() const {
  struct stat st;
  require(fstat(fd_, &st) == 0, SS_E_IO, "stat " + path_ + ": " + std::strerror(errno));
  return identity_of_stat(st);
}

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

std::string sha256_file(const std::string& path) {
  RawFile f(path, true);
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

std::string file_sha256_cached(const std::string& path) {
  const FileIdentity id = identity_of(path);
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
  const std::string digest = sha256_file(path);
  if (!dir.empty() && identity_of(path) == id) {
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
