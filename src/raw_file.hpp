// Positional file reads for multi-location sessions: overlapped handles (real queue depth from many threads), share
// mode read-only (no writer can open the file while it is held), file identity for caches and reuse checks.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
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

// Bound opens, resolve then bind: the given path is resolved once to its real final path (Windows: opened normally and
// GetFinalPathNameByHandleW, which also expands 8.3 short names; Linux: realpath), so a link that exists then is
// followed once. That real path is then opened through held handles of its directory and of every ancestor
// directory, from the root of its volume down, and no held component may be a link (one swapped in after the resolve
// is refused): Windows opens each directory with FILE_FLAG_OPEN_REPARSE_POINT and rejects a reparse point (junction,
// symlink, mount point), and holds it without FILE_SHARE_DELETE, so no held directory can be renamed or deleted while
// it is held; Linux walks openat(O_PATH | O_DIRECTORY | O_NOFOLLOW) from "/" (it cannot deny renames; the directory
// fds pin the inodes the walk used). The leaf is opened relative to the held directory (Windows: by its full path with
// FILE_FLAG_OPEN_REPARSE_POINT, then its final path must be the held directory's final path plus the leaf name; Linux:
// openat(dirfd, leaf, O_NOFOLLOW)) and must be a regular file, not a link. Every failure throws SS_E_IO naming the
// component. The held handles live as long as the object.
class BoundDir {
 public:
  explicit BoundDir(const std::string& file_path);
  ~BoundDir();
  BoundDir(const BoundDir&) = delete;
  BoundDir& operator=(const BoundDir&) = delete;
  const std::string& path() const { return path_; }  // the absolute, normalised file path
#if defined(_WIN32)
  // CreateFileW of the leaf with FILE_FLAG_OPEN_REPARSE_POINT added (OPEN_EXISTING): INVALID_HANDLE_VALUE and
  // GetLastError on failure, then check_leaf before use.
  void* open_leaf(unsigned long access, unsigned long share, unsigned long flags) const;
  void check_leaf(void* h) const;  // throws SS_E_IO: a reparse point, a directory, or not in the held directory
#else
  int open_leaf(int flags) const;  // openat(dirfd, leaf, flags | O_NOFOLLOW | O_CLOEXEC); -1 and errno on failure
  void check_leaf(int fd) const;   // throws SS_E_IO unless a regular file
#endif

 private:
  std::string path_, leaf_;
#if defined(_WIN32)
  std::wstring wpath_, dir_final_;
  std::vector<void*> dirs_;
#else
  std::vector<int> dirs_;
#endif
};

// Test hook: called with the resolved real path between the resolve and the bound walk (empty function = none).
void test_bound_resolved_hook(std::function<void(const std::string&)> hook);

// Positional reads from any thread at real queue depth (Windows: overlapped handle, one event per read).
class RawFile {
 public:
  // direct: unbuffered; falls back to buffered when the filesystem refuses (direct() reports it). Throws SS_E_IO.
  // bound: opened through a BoundDir held for the object's lifetime (false: by path, for cache files only).
  RawFile(const std::string& path, bool direct, bool bound = true);
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
  // Scattered reads, all in flight together (Windows: one OVERLAPPED each on the overlapped handle, up to kMaxSplit at a
  // time; elsewhere one after another): reads[i] = {dst, offset, bytes}; every one must be read whole (inside the file)
  // or it throws SS_E_IO, after every issued read completed (none outlives the call).
  struct Piece {
    void* dst;
    uint64_t offset, bytes;
  };
  void read_many(const Piece* reads, uint32_t n) const;
  // Test hooks: the n-th later piece of read_many fails (not issued, the call throws once the others completed), 0 =
  // off; the most pieces one read_many batch had in flight.
  static void test_fail_piece(uint64_t n);
  static uint32_t test_max_in_flight();
#if defined(_WIN32)
  void* handle() const { return h_; }
#endif

 private:
  std::string path_;
  bool direct_ = false;
  uint64_t size_ = 0;
  std::unique_ptr<BoundDir> dirs_;
#if defined(_WIN32)
  void* h_ = nullptr;
#else
  int fd_ = -1;
#endif
};

// SS_E_DESC naming `what` unless id has the expected volume, index and size (an expectation of all zeros is none).
void check_identity(const FileIdentity& id, const FileIdentity& expected, const std::string& what);

// Identity of an open file (Windows: a HANDLE).
#if defined(_WIN32)
FileIdentity identity_of_handle(void* h, const std::string& path);
#else
FileIdentity identity_of_fd(int fd, const std::string& path);
#endif
// Identity of the file at path now (opened briefly, read-only); throws SS_E_IO when it cannot be opened.
FileIdentity identity_of(const std::string& path);
// The user's tierstream cache directory (%LOCALAPPDATA%	ierstream, $XDG_CACHE_HOME/tierstream or
// ~/.cache/tierstream), created on demand; empty when none can be determined.
std::string cache_dir();
// Writes text to path atomically (temporary file + flush + replace).
void write_file_atomic(const std::string& path, const std::string& text);

// Full-file SHA-256 (hex), cached by file identity under the user's cache directory (tierstream/file-sha256).
// Read through the open file f (never reopened by path).
std::string file_sha256_cached(const RawFile& f);
std::string sha256_of(const RawFile& f);           // uncached
std::string sha256_file(const std::string& path);  // uncached, unbuffered read

}  // namespace ts
