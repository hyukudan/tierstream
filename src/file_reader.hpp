// Unbuffered positional reads (Linux O_DIRECT, Windows FILE_FLAG_NO_BUFFERING): the page cache is never filled with
// streamed weights. On unified memory the page cache is GPU memory, so a buffered read would evict residents.
#pragma once

#include <cstdint>
#include <string>

namespace ts {

class File {
 public:
  // direct = true asks for unbuffered IO; if the filesystem refuses (tmpfs, network shares), the file is opened
  // buffered and direct() reports false.
  File(const std::string& path, bool direct);
  ~File();
  File(const File&) = delete;
  File& operator=(const File&) = delete;

  bool direct() const { return direct_; }
  uint64_t size() const { return size_; }
  // Reads exactly n bytes at offset into dst unless the file ends first; returns the bytes read. For a direct file,
  // offset, n and dst must be multiples of the source alignment (the caller reads the covering aligned range).
  uint64_t pread(void* dst, uint64_t n, uint64_t offset) const;

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

}  // namespace ts
