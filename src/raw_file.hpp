// Positional file reads for multi-location sessions: overlapped handles (real queue depth from many threads), share
// mode read-only (no writer can open the file while it is held), file identity for caches and reuse checks.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ts {

struct FileIdentity {
  uint64_t volume = 0, index = 0, size = 0, mtime = 0, ctime = 0;
  bool operator==(const FileIdentity& o) const {
    return volume == o.volume && index == o.index && size == o.size && mtime == o.mtime && ctime == o.ctime;
  }
  bool operator!=(const FileIdentity& o) const { return !(*this == o); }
};

// Positional reads from any thread at real queue depth (Windows: overlapped handle, one event per read).
class RawFile {
 public:
  // direct: unbuffered; falls back to buffered when the filesystem refuses (direct() reports it). Throws SS_E_IO.
  RawFile(const std::string& path, bool direct);
  ~RawFile();
  RawFile(const RawFile&) = delete;
  RawFile& operator=(const RawFile&) = delete;
  const std::string& path() const { return path_; }
  bool direct() const { return direct_; }
  uint64_t size() const { return size_; }
  FileIdentity identity() const;
  // Reads up to n bytes at offset (fewer only at the end of the file); throws on an IO error.
  uint64_t read(void* dst, uint64_t n, uint64_t offset) const;
  // The same bytes as concurrent reads of `part` bytes (at most kMaxSplit in flight): on_part(bytes, got) is called for
  // each part in order as it completes, so the caller can hash while later parts are still being read. Stops at the
  // first short part (end of file); returns the bytes read.
  static constexpr uint32_t kMaxSplit = 16;
  uint64_t read_split(void* dst, uint64_t n, uint64_t offset, uint64_t part,
                      const std::function<void(const char*, uint64_t)>& on_part) const;
#if defined(_WIN32)
  void* handle() const { return h_; }
#endif

 private:
  std::string path_;
  bool direct_ = false;
  uint64_t size_ = 0;
#if defined(_WIN32)
  void* h_ = nullptr;
#else
  int fd_ = -1;
#endif
};

// Identity of the file at path now (opened briefly, read-only); throws SS_E_IO when it cannot be opened.
FileIdentity identity_of(const std::string& path);
// The user's tierstream cache directory (%LOCALAPPDATA%	ierstream, $XDG_CACHE_HOME/tierstream or
// ~/.cache/tierstream), created on demand; empty when none can be determined.
std::string cache_dir();
// Writes text to path atomically (temporary file + flush + replace).
void write_file_atomic(const std::string& path, const std::string& text);

// Full-file SHA-256 (hex), cached by file identity under the user's cache directory (tierstream/file-sha256).
std::string file_sha256_cached(const std::string& path);
std::string sha256_file(const std::string& path);  // uncached, unbuffered read

}  // namespace ts
