// Descriptor sources retain File reads unless a secondary directory has a stripe manifest.
#pragma once

#include <filesystem>
#include <memory>
#include <string>

#include "file_reader.hpp"
#include "mfile.hpp"

namespace ts {

class SourceFile {
 public:
  virtual ~SourceFile() = default;
  virtual uint64_t pread(void* dst, uint64_t n, uint64_t off) const = 0;
  virtual uint64_t size() const = 0;
  virtual bool direct() const = 0;  // requires aligned covering reads
  virtual bool physical_direct() const { return direct(); }
  virtual uint64_t alignment(uint64_t source_align) const { return source_align; }
  virtual std::shared_ptr<MFile> session() const { return {}; }
};

class FileSource final : public SourceFile {
 public:
  FileSource(const std::string& path, bool direct) : file_(path, direct) {}
  uint64_t pread(void* dst, uint64_t n, uint64_t off) const override { return file_.pread(dst, n, off); }
  uint64_t size() const override { return file_.size(); }
  bool direct() const override { return file_.direct(); }

 private:
  File file_;
};

class MFileSource final : public SourceFile {
 public:
  explicit MFileSource(std::shared_ptr<MFile> file) : file_(std::move(file)) {}
  uint64_t pread(void* dst, uint64_t n, uint64_t off) const override { return file_->pread(dst, n, off); }
  uint64_t size() const override { return file_->size(); }
  bool direct() const override { return true; }
  bool physical_direct() const override {
    ss_mfile_counters c{};
    c.struct_size = sizeof c;
    file_->counters(&c);
    const uint32_t all = (1u << c.locations) - 1;
    return c.locations != 0 && (c.direct_mask & all) == all;
  }
  // Larger descriptor alignments must not extend a covering read past the session's aligned EOF.
  uint64_t alignment(uint64_t) const override { return MFile::kAlign; }
  std::shared_ptr<MFile> session() const override { return file_; }

 private:
  std::shared_ptr<MFile> file_;
};

inline std::unique_ptr<SourceFile> open_source(const std::string& path, bool direct,
                                             std::shared_ptr<MFile> session) {
  if (session) return std::make_unique<MFileSource>(std::move(session));
  return std::make_unique<FileSource>(path, direct);
}

inline std::unique_ptr<SourceFile> open_source(const std::string& path, const ss_options& o) {
  // Do not even canonicalise the primary on the default path.
  if (o.weight_dirs && *o.weight_dirs) {
    const auto dirs = MFile::secondary_dirs(path, o.weight_dirs);
    const auto name = std::filesystem::u8path(path).filename().u8string() + ".stripe.json";
    for (const auto& dir : dirs) {
      std::error_code ec;
      if (!std::filesystem::is_regular_file(std::filesystem::u8path(dir) / std::filesystem::u8path(name), ec)) continue;
      MFile::Options opt;
      opt.direct = o.direct_io != 0;
      opt.drive_readers = o.drive_readers ? o.drive_readers : 4;
      opt.verify_mode = o.weight_verify;
      return std::make_unique<MFileSource>(MFile::open(path, o.weight_dirs, opt));
    }
  }
  return std::make_unique<FileSource>(path, o.direct_io != 0);
}

}  // namespace ts
