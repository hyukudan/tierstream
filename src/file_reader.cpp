#include "file_reader.hpp"

#include "errors.hpp"

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
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ts {

#if defined(_WIN32)

namespace {
std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), &w[0], n);
  return w;
}
}  // namespace

File::File(const std::string& path, bool direct) : path_(path), dirs_(std::make_unique<BoundDir>(path)) {
  DWORD flags = FILE_ATTRIBUTE_NORMAL | (direct ? FILE_FLAG_NO_BUFFERING : FILE_FLAG_SEQUENTIAL_SCAN);
  HANDLE h = static_cast<HANDLE>(dirs_->open_leaf(GENERIC_READ, FILE_SHARE_READ, flags));
  if (h == INVALID_HANDLE_VALUE && direct) {
    h = static_cast<HANDLE>(dirs_->open_leaf(GENERIC_READ, FILE_SHARE_READ, FILE_ATTRIBUTE_NORMAL));
    direct = false;
  }
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "open " + path + ": error " + std::to_string(GetLastError()));
  try {
    dirs_->check_leaf(h);
  } catch (...) {
    CloseHandle(h);
    throw;
  }
  LARGE_INTEGER sz;
  if (!GetFileSizeEx(h, &sz)) {
    CloseHandle(h);
    fail(SS_E_IO, "size of " + path + ": error " + std::to_string(GetLastError()));
  }
  h_ = h;
  direct_ = direct;
  size_ = uint64_t(sz.QuadPart);
}

File::~File() {
  if (h_) CloseHandle(static_cast<HANDLE>(h_));
}

FileIdentity File::identity() const { return identity_of_handle(h_, path_); }

uint64_t File::pread(void* dst, uint64_t n, uint64_t offset) const {
  uint64_t done = 0;
  char* p = static_cast<char*>(dst);
  while (done < n) {
    DWORD want = DWORD(n - done > (1u << 30) ? (1u << 30) : n - done);
    OVERLAPPED ov{};
    uint64_t off = offset + done;
    ov.Offset = DWORD(off & 0xFFFFFFFFu);
    ov.OffsetHigh = DWORD(off >> 32);
    DWORD got = 0;
    if (!ReadFile(static_cast<HANDLE>(h_), p + done, want, &got, &ov)) {
      DWORD e = GetLastError();
      if (e == ERROR_HANDLE_EOF) break;
      fail(SS_E_IO, "read " + path_ + " at " + std::to_string(off) + ": error " + std::to_string(e));
    }
    done += got;
    if (got < want) break;  // end of file (an unbuffered read past it would fail on the unaligned offset)
  }
  return done;
}

#else

File::File(const std::string& path, bool direct) : path_(path), dirs_(std::make_unique<BoundDir>(path)) {
  int fd = -1;
  if (direct) {
    fd = dirs_->open_leaf(O_RDONLY | O_DIRECT | O_NONBLOCK);
    if (fd < 0 && (errno == EINVAL || errno == EOPNOTSUPP)) direct = false;
    else if (fd < 0) fail(SS_E_IO, "open " + path + ": " + std::strerror(errno));
  }
  if (fd < 0) {
    fd = dirs_->open_leaf(O_RDONLY | O_NONBLOCK);
    require(fd >= 0, SS_E_IO, "open " + path + ": " + std::strerror(errno));
#if defined(POSIX_FADV_RANDOM)
    posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
#endif
  }
  try {
    dirs_->check_leaf(fd);
  } catch (...) {
    ::close(fd);
    throw;
  }
  struct stat st;
  if (fstat(fd, &st) != 0) {
    ::close(fd);
    fail(SS_E_IO, "stat " + path + ": " + std::strerror(errno));
  }
  fd_ = fd;
  direct_ = direct;
  size_ = uint64_t(st.st_size);
}

File::~File() {
  if (fd_ >= 0) ::close(fd_);
}

FileIdentity File::identity() const { return identity_of_fd(fd_, path_); }

uint64_t File::pread(void* dst, uint64_t n, uint64_t offset) const {
  uint64_t done = 0;
  char* p = static_cast<char*>(dst);
  while (done < n) {
    size_t want = size_t(n - done > (uint64_t(1) << 30) ? (uint64_t(1) << 30) : n - done);
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

#endif

}  // namespace ts
