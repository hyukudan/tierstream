// Descriptor sources retain File reads unless a secondary directory has a stripe manifest. Every source is opened
// through held directory handles (BoundDir, raw_file.hpp) and may be checked against an expected file identity.
#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "errors.hpp"
#include "file_reader.hpp"
#include "mfile.hpp"
#include "raw_file.hpp"

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
  virtual FileIdentity identity() const = 0;  // of the file read (a session: its primary)
  // Scattered whole reads, in flight together where the source can (each piece inside the file, aligned like a pread):
  // throws SS_E_IO when one is short or fails, once none is in flight. Default: one pread after another.
  virtual void read_pieces(const std::vector<RawFile::Piece>& pieces) const {
    for (const RawFile::Piece& p : pieces)
      require(pread(p.dst, p.bytes, p.offset) == p.bytes, SS_E_IO, "short read");
  }
  // The grid a piece must not cross (a multi-location session's chunk; 0 = none).
  virtual uint64_t piece_grid() const { return 0; }
};

class FileSource final : public SourceFile {
 public:
  FileSource(const std::string& path, bool direct) : file_(path, direct), path_(path) {}
  uint64_t pread(void* dst, uint64_t n, uint64_t off) const override { return file_.pread(dst, n, off); }
  uint64_t size() const override { return file_.size(); }
  bool direct() const override { return file_.direct(); }
  FileIdentity identity() const override { return file_.identity(); }
  // On a second, overlapped handle (opened on first use, in the same direct mode): every piece in flight together.
  // If that handle cannot be opened in the same mode, the pieces are read one after another as before.
  void read_pieces(const std::vector<RawFile::Piece>& pieces) const override {
    std::call_once(raw_once_, [&] {
      try {
        auto raw = std::make_unique<RawFile>(path_, file_.direct());
        // the same file as the bound source only (a path swapped since the source opened is never read)
        if (raw->direct() == file_.direct() && raw->identity() == file_.identity()) raw_ = std::move(raw);
      } catch (const Error&) {
      }
    });
    if (!raw_) return SourceFile::read_pieces(pieces);
    raw_->read_many(pieces.data(), uint32_t(pieces.size()));
  }

 private:
  File file_;
  std::string path_;
  mutable std::once_flag raw_once_;
  mutable std::unique_ptr<RawFile> raw_;
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
  FileIdentity identity() const override { return file_->primary_identity(); }
  // one preadv: medium pieces (>= 1 MiB) are routed and balanced across the homes of their chunks
  void read_pieces(const std::vector<RawFile::Piece>& pieces) const override {
    std::vector<ss_mfile_io> ios;
    ios.reserve(pieces.size());
    for (const RawFile::Piece& p : pieces) ios.push_back({p.dst, p.offset, p.bytes});
    file_->preadv(ios.data(), uint32_t(ios.size()));
  }
  uint64_t piece_grid() const override { return file_->chunk_bytes(); }

 private:
  std::shared_ptr<MFile> file_;
};

// SS_E_DESC naming the source unless the opened file has the expected identity (volume, index, size; all zero = none).
inline std::unique_ptr<SourceFile> checked_source(std::unique_ptr<SourceFile> f, const FileIdentity& expected,
                                                  const std::string& path) {
  check_identity(f->identity(), expected, "source " + path);
  return f;
}

// expected: the identity the file must have (a worker's reopen passes the identity of the context's open source).
inline std::unique_ptr<SourceFile> open_source(const std::string& path, bool direct, std::shared_ptr<MFile> session,
                                             const FileIdentity& expected = FileIdentity()) {
  if (session) return checked_source(std::make_unique<MFileSource>(std::move(session)), expected, path);
  return checked_source(std::make_unique<FileSource>(path, direct), expected, path);
}

// The expected identity of desc source `index` from ss_options.source_identities (none past source_identity_n).
inline FileIdentity expected_source_identity(const ss_options& o, size_t index) {
  FileIdentity id;
  if (o.source_identities && index < o.source_identity_n) {
    const ss_file_identity& e = o.source_identities[index];
    id.volume = e.volume;
    id.index = e.index;
    id.size = e.size;
  }
  return id;
}

inline std::unique_ptr<SourceFile> open_source(const std::string& path, const ss_options& o,
                                             const FileIdentity& expected = FileIdentity()) {
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
      return checked_source(std::make_unique<MFileSource>(MFile::open(path, o.weight_dirs, opt)), expected, path);
    }
  }
  return checked_source(std::make_unique<FileSource>(path, o.direct_io != 0), expected, path);
}

}  // namespace ts
