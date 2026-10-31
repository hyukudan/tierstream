// Unbuffered positional reads (Linux O_DIRECT, Windows FILE_FLAG_NO_BUFFERING): the page cache is never filled with
// streamed weights. On unified memory the page cache is GPU memory, so a buffered read would evict residents.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "raw_file.hpp"

namespace ts {

class File {
 public:
  // direct = true asks for unbuffered IO; if the filesystem refuses (tmpfs, network shares), the file is opened
  // buffered and direct() reports false. Opened through a BoundDir (raw_file.hpp) held for the object's lifetime.
  File(const std::string& path, bool direct);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  bool direct() const { return direct_; }
  uint64_t size() const { return size_; }
  // Reads exactly n bytes at offset into dst unless the file ends first; returns the bytes read. For a direct file,
  // offset, n and dst must be multiples of the source alignment (the caller reads the covering aligned range).
  uint64_t pread(void* dst, uint64_t n, uint64_t offset) const;
  FileIdentity identity() const;  // of the open file

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

}  // namespace ts
