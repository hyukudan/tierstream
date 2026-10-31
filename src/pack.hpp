// Indexed prepared pack tierstream/pack-v1 (docs/pack.md): one directory per model holding every tensor / expert part
// in the engine's final device format, whole and contiguous, plus a binary index and a manifest. The library never
// produces device bytes: an engine's repacker (or ts-pack build from an engine-provided layout) hands them to
// PackWriter; Pack gives each app exactly its records, row ranges and units, read through the same bound sources
// (SourceFile: held directories, expected identities, stripe sessions) as every descriptor source.
#pragma once

#include <atomic>
#include <cstdint>
#include <climits>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "raw_file.hpp"
#include "sha256.hpp"
#include "source_file.hpp"
#include "xxh64.hpp"

namespace ts {

constexpr const char* kPackSchema = "tierstream/pack-v1";
constexpr const char* kPackManifestName = "pack.json";
constexpr const char* kPackIndexName = "pack.index";
constexpr uint64_t kPackAlign = 4096;
// Limits checked before allocation.
constexpr uint64_t kPackMaxRecords = uint64_t(1) << 24;
constexpr uint64_t kPackMaxSegments = uint64_t(1) << 24;
constexpr uint64_t kPackMaxStrings = uint64_t(1) << 26;  // string table bytes
constexpr uint64_t kPackMaxManifest = uint64_t(16) << 20;
constexpr uint32_t kPackMaxFiles = 4096;

// Row segment of a record: rows [row0, row1) are the bytes [offset, offset + bytes) of the record (offsets relative to
// the record), with their own xxh64. Segments tile the record's rows and bytes in order.
struct PackSegment {
  uint32_t row0 = 0, row1 = 0;
  uint64_t offset = 0, bytes = 0, xxh64 = 0;
};

// One record of the index. Strings are ids into the index's string table. Records are stored in (file, offset) order.
struct PackRecord {
  int32_t layer = -1, expert = -1;
  uint32_t name = 0, format = 0;
  uint16_t file = 0;
  uint16_t kind = 0;        // 0 = stored bytes (the only kind of v1; others are reserved and refused)
  uint32_t rows = 0;        // 0 = not row-addressable
  uint64_t row_bytes = 0;   // uniform rows (rows x row_bytes = bytes); 0 with segments = non-uniform rows
  uint64_t offset = 0, bytes = 0, xxh64 = 0;
  uint32_t seg_first = 0, seg_count = 0;
};

struct PackIndex {
  std::vector<PackRecord> records;
  std::vector<PackSegment> segments;
  std::vector<std::string> strings;
};
// Binary codec (little endian, frozen for v1): "TSPACKI1", u32 version 1, u32 record size 64, u32 segment size 32,
// u32 string count, u64 records, u64 segments, u64 string bytes, 16 reserved zero bytes; the records, the segments,
// then per string u32 length + bytes. parse checks sizes before allocating and every field against the limits.
std::string pack_index_write(const PackIndex& ix);
PackIndex pack_index_parse(const std::string& bytes);

struct PackFileEntry {
  std::string name;
  uint64_t size = 0;
  std::string sha256;
};
struct PackSourceEntry {  // a model file the records were produced from (its identity, not its path)
  std::string name;
  uint64_t size = 0;
  std::string sha256;
};
struct PackFormatEntry {  // how the bytes of a format tag were produced
  std::string transform, version, tool;
};

struct PackManifest {
  std::string model;
  uint64_t alignment = kPackAlign;
  std::vector<PackSourceEntry> sources;
  std::map<std::string, PackFormatEntry> formats;
  std::vector<PackFileEntry> files;
  PackFileEntry index;
  uint64_t records = 0, segments = 0;
  std::vector<std::string> layouts;  // optional layouts the build included (sorted, unique)
  Json meta;     // engine notes (object; x-* free form)
  Json receipt;  // creation receipt (object)
};
std::string pack_manifest_write(const PackManifest& m);  // canonical JSON
PackManifest pack_manifest_parse(const std::string& text);  // throws SS_E_DESC naming the field

// Validation shared by the reader and the verifier: string ids, files, alignment, (file, offset) order without
// overlap, records inside their file, unique keys, row geometry and segment tiling, unit-major adjacency.
void pack_index_validate(const PackIndex& ix, const PackManifest& m);

// ---- writer -------------------------------------------------------------------------------------------------------

struct PackWriterOptions {
  std::string model;
  std::vector<PackSourceEntry> sources;
  std::map<std::string, PackFormatEntry> formats;  // every record's format must be listed
  uint64_t split_bytes = 0;  // 0 = one data file; else a new file starts at the first layer boundary past this size
  std::vector<std::string> layouts;  // optional layouts this build includes (recorded in pack.json)
  Json meta = Json::object(), receipt = Json::object();
};

struct PackRecordSpec {
  int32_t layer = -1, expert = -1;
  std::string name, format;
  uint64_t bytes = 0;
  uint32_t rows = 0;
  uint64_t row_bytes = 0;
  // Optional row segments, as boundaries: seg_rows[0] = 0 < ... < seg_rows[n] = rows. With uniform rows (row_bytes > 0)
  // seg_bytes may be empty; otherwise seg_bytes[i] is segment i's bytes (their sum = bytes).
  std::vector<uint32_t> seg_rows;
  std::vector<uint64_t> seg_bytes;
};

// Streaming writer: records in unit-major order (a layer's records together, layers ascending; inside a layer the
// records of one unit (layer, expert >= 0) adjacent), each 4 KiB aligned with zero padding. The pack is built in
// DIR.incomplete (created exclusively) and published by renaming it to DIR after the index and the manifest were
// written and every file flushed: nothing is visible until complete. Destruction without finish removes it. A refused
// record spec leaves the writer as it was; any other error (I/O, a data-file split, finish) poisons it: DIR.incomplete is
// removed at once and every later call fails.
class PackWriter {
 public:
  PackWriter(const std::string& dir, PackWriterOptions o);
  ~PackWriter();
  PackWriter(const PackWriter&) = delete;
  PackWriter& operator=(const PackWriter&) = delete;
  void begin(const PackRecordSpec& r);
  void write(const void* data, uint64_t n);
  void end();
  void add(const PackRecordSpec& r, const void* data) { begin(r); write(data, r.bytes); end(); }
  // Returns the SHA-256 of pack.json (the pin).
  std::string finish();
  uint64_t data_bytes() const { return total_; }
  bool failed() const { return failed_; }

 private:
  void live() const;  // throws once the writer failed or finished
  void poison();      // after an error past the checks: drops the open file and removes DIR.incomplete
  void begin_checked(const PackRecordSpec& r, const std::vector<PackSegment>& segs, bool new_layer);
  std::string finish_checked();
  void open_file();
  void close_file();
  bool split_bytes_reached() const;
  std::string dir_, tmp_;
  PackWriterOptions o_;
  PackIndex ix_;
  std::map<std::string, uint32_t> string_ids_;
  std::vector<PackFileEntry> files_;
  struct Out;
  std::unique_ptr<Out> out_;
  bool open_rec_ = false, finished_ = false, failed_ = false;
  bool tmp_left_ = false;  // poison() could not remove DIR.incomplete (the destructor tries once more)
  PackRecord rec_;
  uint64_t rec_written_ = 0;
  std::vector<uint64_t> seg_end_;
  size_t seg_i_ = 0;
  Xxh64 rec_hash_, seg_hash_;
  int32_t last_layer_ = INT32_MIN, last_expert_ = -1;
  bool layer_has_units_ = false;
  std::set<int32_t> closed_units_;  // units of the current layer already finished
  std::set<std::pair<std::pair<int32_t, int32_t>, std::string>> keys_;
  uint64_t total_ = 0, string_bytes_ = 0;
  uint32_t intern(const std::string& s);
};

// Test seam (fault injection; the library never sets it): the next data-file creation fails, or a data-file write that
// would take that file past fail_write_at bytes fails (0 = never), both with SS_E_IO.
struct PackWriterFaults {
  bool fail_create = false;
  uint64_t fail_write_at = 0;
};
extern PackWriterFaults g_pack_writer_faults;

// ---- reader -------------------------------------------------------------------------------------------------------

enum class PackVerify { Cached = 0, Always = 1, Off = 2 };

struct PackOptions {
  std::string pin_sha256;          // SHA-256 of pack.json (empty = not pinned)
  PackVerify verify = PackVerify::Cached;
  std::string weight_dirs;         // stripe directories of the data files (ss_options.weight_dirs rules)
  uint32_t drive_readers = 4;
  bool direct = true;
  std::vector<std::string> source_sha256;  // the app's authenticated sources, in manifest order (empty = no check)
  std::vector<std::string> require_layouts;  // optional layouts the app reads (a pack without one is refused)
  bool persist_verified = true;    // Cached: keep the verified records per data-file identities in the cache dir
};

struct PackUnit {
  uint32_t file = 0;
  uint64_t offset = 0, bytes = 0;     // the unit's contiguous span (its records and the padding between them)
  uint64_t first = 0, count = 0;      // its records
};

struct PackRead {
  uint64_t record = 0;
  uint32_t row0 = 0, row1 = 0;  // row1 == 0: the whole record
  void* dst = nullptr;
};

struct PackStats {
  uint64_t reads = 0, bytes = 0, bounce_bytes = 0, verified_records = 0, verified_segments = 0, verified_bytes = 0;
  uint64_t verify_ns = 0;
};

class Pack {
 public:
  static std::unique_ptr<Pack> open(const std::string& dir, const PackOptions& o);
  ~Pack();
  Pack(const Pack&) = delete;
  Pack& operator=(const Pack&) = delete;

  const PackManifest& manifest() const { return m_; }
  const PackIndex& index() const { return ix_; }
  const std::string& manifest_sha256() const { return manifest_sha_; }
  const std::string& dir() const { return dir_; }
  const std::string& name(const PackRecord& r) const { return ix_.strings[r.name]; }
  const std::string& format(const PackRecord& r) const { return ix_.strings[r.format]; }
  std::string file_path(uint32_t file) const;
  const SourceFile& file(uint32_t f) const { return *files_[f]; }
  // -1 when absent
  int64_t find(int32_t layer, int32_t expert, const std::string& name) const;
  bool unit(int32_t layer, int32_t expert, PackUnit* out) const;
  // Byte span of rows [row0, row1) of a record (row1 == 0: whole). Throws SS_E_INVALID for a range a record cannot give.
  void row_span(const PackRecord& r, uint32_t row0, uint32_t row1, uint64_t* off, uint64_t* bytes) const;
  // Reads, all in flight together per data file (bounces only for unaligned heads / tails or destinations), verified
  // per the verify mode: a whole record against its xxh64; a row range against the xxh64 of the segments it covers, or
  // (no covering segments) by reading the whole record once. Throws SS_E_IO / SS_E_HASH.
  void readv(const PackRead* reads, size_t n);
  void read(uint64_t record, uint32_t row0, uint32_t row1, void* dst) {
    PackRead r{record, row0, row1, dst};
    readv(&r, 1);
  }
  // A unit's whole span (records and padding) into dst (unit.bytes long), records verified as whole reads.
  void read_unit(int32_t layer, int32_t expert, void* dst);
  // Full check: every data file's size and SHA-256 against the manifest and every record / segment xxh64.
  Json verify_full();
  // desc-v1 for the device-gated cache from the engine's own descriptor: the pack's data files are appended to its
  // sources and each part of a routed layer (part name = record name of each unit) points into the pack (offset = unit
  // 0's record, stride = the distance between consecutive units, uniform). Parts must equal their records' bytes. A
  // routed layer without unit hashes gets them from the pack (unless verify is Off): every unit's records are read
  // once with this pack's verification and the xxh64 of the unit's parts back to back is listed, so the cache's
  // verify_hash checks every landing against verified bytes; the lists are cached per selection and data-file
  // identities.
  Json dg_descriptor(const Json& base);
  Json info() const;
  PackStats stats() const;
  FileIdentity file_identity(uint32_t f) const { return files_[f]->identity(); }

 private:
  Pack() = default;
  std::vector<uint64_t> unit_hashes(int32_t layer, int n_units, const std::vector<std::string>& parts);
  bool record_verified(uint64_t i) const;
  bool segment_verified(uint64_t i) const;
  void persist();
  std::string dir_, manifest_sha_, verified_path_;
  PackOptions o_;
  PackManifest m_;
  PackIndex ix_;
  std::unique_ptr<RawFile> manifest_file_, index_file_;
  std::vector<std::unique_ptr<SourceFile>> files_;
  std::vector<std::unique_ptr<RawFile>> held_;  // a held handle per data file: its live identity (stripe sessions
                                                // report the one of their open)
  std::vector<FileIdentity> open_ids_;  // the data files' identities at open (the verified cache's key)
  std::vector<uint8_t> striped_;        // data file served by a multi-location session (hashed on every read)
  bool files_unchanged(const std::vector<uint8_t>* only = nullptr) const;  // only: the data files to check (1 = check)
  void forget_verified();  // a data file changed: no cached result is trusted or kept from now on
  struct KeyHash {
    size_t operator()(const std::pair<std::pair<int32_t, int32_t>, std::string>& k) const;
  };
  std::unordered_map<std::pair<std::pair<int32_t, int32_t>, std::string>, uint64_t, KeyHash> by_key_;
  std::map<std::pair<int32_t, int32_t>, PackUnit> units_;
  mutable std::mutex mu_;
  std::vector<uint8_t> rec_ok_, seg_ok_;
  bool dirty_ = false, changed_ = false;
  PackStats stats_;
};

}  // namespace ts
