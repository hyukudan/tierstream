// tierstream/pack-v1: index codec, manifest, writer, indexed reader (docs/pack.md).
#include "pack.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <future>
#include <set>
#include <thread>

#include "desc.hpp"
#include "errors.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ts {
namespace fs = std::filesystem;

namespace {

// ---- little-endian codec ------------------------------------------------------------------------------------------

void put32(std::string& s, uint32_t v) {
  for (int i = 0; i < 4; ++i) s.push_back(char((v >> (8 * i)) & 0xff));
}
void put64(std::string& s, uint64_t v) {
  for (int i = 0; i < 8; ++i) s.push_back(char((v >> (8 * i)) & 0xff));
}
void put16(std::string& s, uint16_t v) {
  s.push_back(char(v & 0xff));
  s.push_back(char(v >> 8));
}
struct Cursor {
  const std::string& s;
  size_t at = 0;
  void need(uint64_t n) const { require(n <= s.size() - at, SS_E_DESC, "pack index: truncated"); }
  uint64_t get(int bytes) {
    need(uint64_t(bytes));
    uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) v |= uint64_t(uint8_t(s[at + size_t(i)])) << (8 * i);
    at += size_t(bytes);
    return v;
  }
};

constexpr char kIndexMagic[8] = {'T', 'S', 'P', 'A', 'C', 'K', 'I', '1'};
constexpr uint32_t kRecordSize = 64, kSegmentSize = 32, kHeaderSize = 64;

std::string path_join(const std::string& dir, const std::string& name) {
  return (fs::u8path(dir) / fs::u8path(name)).u8string();
}

bool is_hex64(const std::string& s) {
  return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(uint8_t(c)));
  return s;
}
std::string data_name(size_t i) {
  char b[32];
  std::snprintf(b, sizeof b, "pack-%04u.data", unsigned(i));
  return b;
}
uint64_t pad_up(uint64_t v) { return (v + kPackAlign - 1) / kPackAlign * kPackAlign; }

struct Aligned {
  char* p = nullptr;
  explicit Aligned(uint64_t n) {
    const size_t sz = size_t(pad_up(n ? n : 1));
#if defined(_WIN32)
    p = static_cast<char*>(_aligned_malloc(sz, kPackAlign));
#else
    p = static_cast<char*>(std::aligned_alloc(kPackAlign, sz));
#endif
    require(p != nullptr, SS_E_NOMEM, "pack: bounce allocation");
  }
  ~Aligned() {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
  Aligned(const Aligned&) = delete;
  Aligned& operator=(const Aligned&) = delete;
};

std::string read_whole(const RawFile& f, uint64_t limit, const std::string& what) {
  require(f.size() <= limit, SS_E_DESC, what + " is too large");
  std::string s(size_t(f.size()), '\0');
  if (!s.empty()) require(f.read(&s[0], s.size(), 0) == s.size(), SS_E_IO, "short read of " + what);
  return s;
}

}  // namespace

// ---- index codec --------------------------------------------------------------------------------------------------

std::string pack_index_write(const PackIndex& ix) {
  uint64_t sbytes = 0;
  for (const std::string& s : ix.strings) sbytes += 4 + s.size();
  std::string out;
  out.reserve(size_t(kHeaderSize + ix.records.size() * kRecordSize + ix.segments.size() * kSegmentSize + sbytes));
  out.append(kIndexMagic, 8);
  put32(out, 1);
  put32(out, kRecordSize);
  put32(out, kSegmentSize);
  put32(out, uint32_t(ix.strings.size()));
  put64(out, ix.records.size());
  put64(out, ix.segments.size());
  put64(out, sbytes);
  out.append(16, '\0');
  for (const PackRecord& r : ix.records) {
    put32(out, uint32_t(r.layer));
    put32(out, uint32_t(r.expert));
    put32(out, r.name);
    put32(out, r.format);
    put16(out, r.file);
    put16(out, r.kind);
    put32(out, r.rows);
    put64(out, r.row_bytes);
    put64(out, r.offset);
    put64(out, r.bytes);
    put64(out, r.xxh64);
    put32(out, r.seg_first);
    put32(out, r.seg_count);
  }
  for (const PackSegment& g : ix.segments) {
    put32(out, g.row0);
    put32(out, g.row1);
    put64(out, g.offset);
    put64(out, g.bytes);
    put64(out, g.xxh64);
  }
  for (const std::string& s : ix.strings) {
    put32(out, uint32_t(s.size()));
    out += s;
  }
  return out;
}

PackIndex pack_index_parse(const std::string& bytes) {
  Cursor c{bytes};
  c.need(kHeaderSize);
  require(std::memcmp(bytes.data(), kIndexMagic, 8) == 0, SS_E_DESC, "pack index: bad magic");
  c.at = 8;
  require(c.get(4) == 1, SS_E_DESC, "pack index: unsupported version");
  require(c.get(4) == kRecordSize && c.get(4) == kSegmentSize, SS_E_DESC, "pack index: unexpected record sizes");
  const uint64_t n_strings = c.get(4), n_records = c.get(8), n_segments = c.get(8), sbytes = c.get(8);
  for (int i = 0; i < 16; ++i) require(bytes[c.at + size_t(i)] == 0, SS_E_DESC, "pack index: reserved bytes not zero");
  c.at += 16;
  require(n_records <= kPackMaxRecords && n_segments <= kPackMaxSegments && sbytes <= kPackMaxStrings &&
              n_strings <= sbytes / 4,
          SS_E_DESC, "pack index: counts over the limits");
  require(bytes.size() == kHeaderSize + n_records * kRecordSize + n_segments * kSegmentSize + sbytes, SS_E_DESC,
          "pack index: size does not match its counts");
  PackIndex ix;
  ix.records.resize(size_t(n_records));
  for (PackRecord& r : ix.records) {
    r.layer = int32_t(uint32_t(c.get(4)));
    r.expert = int32_t(uint32_t(c.get(4)));
    r.name = uint32_t(c.get(4));
    r.format = uint32_t(c.get(4));
    r.file = uint16_t(c.get(2));
    r.kind = uint16_t(c.get(2));
    r.rows = uint32_t(c.get(4));
    r.row_bytes = c.get(8);
    r.offset = c.get(8);
    r.bytes = c.get(8);
    r.xxh64 = c.get(8);
    r.seg_first = uint32_t(c.get(4));
    r.seg_count = uint32_t(c.get(4));
  }
  ix.segments.resize(size_t(n_segments));
  for (PackSegment& g : ix.segments) {
    g.row0 = uint32_t(c.get(4));
    g.row1 = uint32_t(c.get(4));
    g.offset = c.get(8);
    g.bytes = c.get(8);
    g.xxh64 = c.get(8);
  }
  ix.strings.resize(size_t(n_strings));
  for (std::string& s : ix.strings) {
    const uint64_t len = c.get(4);
    c.need(len);
    s.assign(bytes, c.at, size_t(len));
    c.at += size_t(len);
  }
  require(c.at == bytes.size(), SS_E_DESC, "pack index: string table size mismatch");
  return ix;
}

// ---- manifest -----------------------------------------------------------------------------------------------------

namespace {

void only_keys(const Json& o, const std::string& where, std::initializer_list<const char*> keys) {
  require(o.is_object(), SS_E_DESC, where + ": expected an object");
  for (const auto& kv : o.obj) {
    if (kv.first.rfind("x-", 0) == 0) continue;
    bool ok = false;
    for (const char* k : keys) ok = ok || kv.first == k;
    require(ok, SS_E_DESC, where + ": unknown key " + kv.first);
  }
}
const Json& need_key(const Json& o, const char* k, const std::string& where) {
  const Json* v = o.find(k);
  require(v != nullptr, SS_E_DESC, where + ": missing " + k);
  return *v;
}
Json file_json(const PackFileEntry& f) {
  Json j = Json::object();
  j["name"] = Json::string(f.name);
  j["size"] = Json::number(f.size);
  j["sha256"] = Json::string(f.sha256);
  return j;
}
PackFileEntry file_parse(const Json& j, const std::string& where) {
  PackFileEntry f;
  f.name = json_str(need_key(j, "name", where), where + ".name");
  f.size = json_u64(need_key(j, "size", where), where + ".size");
  f.sha256 = json_str(need_key(j, "sha256", where), where + ".sha256");
  require(is_hex64(f.sha256), SS_E_DESC, where + ".sha256: expected 64 lowercase hex digits");
  return f;
}

}  // namespace

std::string pack_manifest_write(const PackManifest& m) {
  Json j = Json::object();
  j["schema"] = Json::string(kPackSchema);
  j["model"] = Json::string(m.model);
  j["alignment"] = Json::number(m.alignment);
  Json src = Json::array();
  for (const PackSourceEntry& s : m.sources) {
    Json e = Json::object();
    e["name"] = Json::string(s.name);
    e["size"] = Json::number(s.size);
    e["sha256"] = Json::string(s.sha256);
    src.push(e);
  }
  j["sources"] = src;
  Json fm = Json::object();
  for (const auto& kv : m.formats) {
    Json e = Json::object();
    e["transform"] = Json::string(kv.second.transform);
    e["version"] = Json::string(kv.second.version);
    e["tool"] = Json::string(kv.second.tool);
    fm[kv.first] = e;
  }
  j["formats"] = fm;
  Json files = Json::array();
  for (const PackFileEntry& f : m.files) files.push(file_json(f));
  j["files"] = files;
  Json ix = file_json(m.index);
  ix["records"] = Json::number(m.records);
  ix["segments"] = Json::number(m.segments);
  j["index"] = ix;
  Json lay = Json::array();
  for (const std::string& l : m.layouts) lay.push(Json::string(l));
  j["layouts"] = lay;
  j["meta"] = m.meta.is_null() ? Json::object() : m.meta;
  j["receipt"] = m.receipt.is_null() ? Json::object() : m.receipt;
  return json_write(j);
}

PackManifest pack_manifest_parse(const std::string& text) {
  const Json j = json_parse(text);
  const std::string top = "pack.json";
  only_keys(j, "pack.json",
            {"schema", "model", "alignment", "sources", "formats", "files", "index", "layouts", "meta", "receipt"});
  require(json_str(need_key(j, "schema", "pack.json"), "schema") == kPackSchema, SS_E_DESC,
          "pack.json: schema is not tierstream/pack-v1");
  PackManifest m;
  m.model = json_str(need_key(j, "model", "pack.json"), "model");
  m.alignment = json_u64(need_key(j, "alignment", "pack.json"), "alignment");
  require(m.alignment == kPackAlign, SS_E_DESC, "pack.json: alignment must be 4096");
  const Json& src = need_key(j, "sources", top);
  require(src.is_array(), SS_E_DESC, "sources: expected an array");
  for (size_t i = 0; i < src.arr.size(); ++i) {
    const std::string w = "sources[" + std::to_string(i) + "]";
    only_keys(src.arr[i], w, {"name", "size", "sha256"});
    PackSourceEntry s;
    s.name = json_str(need_key(src.arr[i], "name", w), w + ".name");
    s.size = json_u64(need_key(src.arr[i], "size", w), w + ".size");
    s.sha256 = json_str(need_key(src.arr[i], "sha256", w), w + ".sha256");
    require(is_hex64(s.sha256), SS_E_DESC, w + ".sha256: expected 64 lowercase hex digits");
    m.sources.push_back(s);
  }
  const Json& fm = need_key(j, "formats", top);
  require(fm.is_object(), SS_E_DESC, "formats: expected an object");
  for (const auto& kv : fm.obj) {
    const std::string w = "formats." + kv.first;
    only_keys(kv.second, w, {"transform", "version", "tool"});
    PackFormatEntry e;
    e.transform = json_str(need_key(kv.second, "transform", w), w + ".transform");
    e.version = json_str(need_key(kv.second, "version", w), w + ".version");
    e.tool = json_str(need_key(kv.second, "tool", w), w + ".tool");
    m.formats[kv.first] = e;
  }
  const Json& files = need_key(j, "files", top);
  require(files.is_array() && !files.arr.empty() && files.arr.size() <= kPackMaxFiles, SS_E_DESC,
          "files: expected 1.." + std::to_string(kPackMaxFiles) + " entries");
  for (size_t i = 0; i < files.arr.size(); ++i) {
    const std::string w = "files[" + std::to_string(i) + "]";
    only_keys(files.arr[i], w, {"name", "size", "sha256"});
    m.files.push_back(file_parse(files.arr[i], w));
    require(m.files.back().name == data_name(i), SS_E_DESC, w + ".name: expected " + data_name(i));
    require(m.files.back().size % kPackAlign == 0, SS_E_DESC, w + ".size: not a multiple of 4096");
  }
  const Json& ix = need_key(j, "index", top);
  only_keys(ix, "index", {"name", "size", "sha256", "records", "segments"});
  m.index = file_parse(ix, "index");
  require(m.index.name == kPackIndexName, SS_E_DESC, "index.name: expected pack.index");
  m.records = json_u64(need_key(ix, "records", "index"), "index.records");
  m.segments = json_u64(need_key(ix, "segments", "index"), "index.segments");
  if (const Json* v = j.find("layouts")) {
    require(v->is_array(), SS_E_DESC, "layouts: expected an array");
    for (const Json& l : v->arr) m.layouts.push_back(json_str(l, "layouts"));
    require(std::is_sorted(m.layouts.begin(), m.layouts.end()) &&
                std::adjacent_find(m.layouts.begin(), m.layouts.end()) == m.layouts.end(),
            SS_E_DESC, "layouts: expected sorted unique names");
  }
  if (const Json* v = j.find("meta")) {
    require(v->is_object(), SS_E_DESC, "meta: expected an object");
    m.meta = *v;
  }
  if (const Json* v = j.find("receipt")) {
    require(v->is_object(), SS_E_DESC, "receipt: expected an object");
    m.receipt = *v;
  }
  return m;
}

void pack_index_validate(const PackIndex& ix, const PackManifest& m) {
  require(ix.records.size() == m.records && ix.segments.size() == m.segments, SS_E_DESC,
          "pack index: counts differ from pack.json");
  {  // keys compare names by id: every string appears once
    std::set<std::string> seen;
    for (const std::string& s : ix.strings)
      require(seen.insert(s).second, SS_E_DESC, "pack index: duplicate string \"" + s.substr(0, 64) + "\"");
  }
  std::set<std::pair<std::pair<int32_t, int32_t>, uint32_t>> keys;
  std::set<std::pair<int32_t, int32_t>> closed;
  std::pair<int32_t, int32_t> cur_unit{INT32_MIN, -1};
  int32_t last_layer = INT32_MIN;
  bool layer_has_units = false;
  uint64_t seg_next = 0;
  for (size_t i = 0; i < ix.records.size(); ++i) {
    const PackRecord& r = ix.records[i];
    const std::string w = "pack index record " + std::to_string(i);
    require(r.name < ix.strings.size() && r.format < ix.strings.size() && !ix.strings[r.name].empty(), SS_E_DESC,
            w + ": bad string id");
    require(m.formats.count(ix.strings[r.format]) == 1, SS_E_DESC, w + ": format not listed in pack.json");
    require(r.kind == 0, SS_E_DESC, w + ": record kind " + std::to_string(r.kind) +
                                        " is not supported by this reader (pack-v1 reads stored records only)");
    require(r.file < m.files.size(), SS_E_DESC, w + ": bad file");
    require(r.layer >= -1 && r.expert >= -1, SS_E_DESC, w + ": bad key");
    require(r.bytes > 0 && r.offset % kPackAlign == 0 && r.offset <= m.files[r.file].size &&
                r.bytes <= m.files[r.file].size - r.offset,
            SS_E_DESC, w + ": outside its data file or unaligned");
    if (i > 0) {
      const PackRecord& p = ix.records[i - 1];
      require(r.file > p.file || (r.file == p.file && r.offset >= pad_up(p.offset + p.bytes)), SS_E_DESC,
              w + ": records not in (file, offset) order or overlapping");
      require(r.file == p.file || r.layer != p.layer, SS_E_DESC, w + ": a layer spans two data files");
    }
    require(keys.insert({{r.layer, r.expert}, r.name}).second, SS_E_DESC, w + ": duplicate key");
    // unit-major: layers ascending, a layer's dense records first, a unit's records adjacent
    require(r.layer >= last_layer, SS_E_DESC, w + ": layers not ascending");
    if (r.layer != last_layer) {
      last_layer = r.layer;
      layer_has_units = false;
    }
    if (r.expert < 0) {
      require(!layer_has_units, SS_E_DESC, w + ": a dense record after the layer's units");
    } else {
      layer_has_units = true;
      const std::pair<int32_t, int32_t> u{r.layer, r.expert};
      if (u != cur_unit) {
        if (cur_unit.second >= 0) closed.insert(cur_unit);
        require(closed.count(u) == 0, SS_E_DESC, w + ": a unit's records are not adjacent");
        cur_unit = u;
      }
    }
    // rows
    if (r.rows == 0) {
      require(r.row_bytes == 0 && r.seg_count == 0, SS_E_DESC, w + ": row geometry without rows");
    } else if (r.seg_count == 0) {
      require(r.row_bytes > 0 && r.bytes / r.row_bytes == r.rows && r.bytes % r.row_bytes == 0, SS_E_DESC,
              w + ": rows x row_bytes != bytes");
    }
    if (r.seg_count) {
      require(r.rows > 0 && r.seg_first == seg_next && uint64_t(r.seg_first) + r.seg_count <= ix.segments.size(),
              SS_E_DESC, w + ": bad segment range");
      uint32_t row = 0;
      uint64_t off = 0;
      for (uint32_t k = 0; k < r.seg_count; ++k) {
        const PackSegment& g = ix.segments[r.seg_first + k];
        require(g.row0 == row && g.row1 > g.row0 && g.row1 <= r.rows && g.offset == off && g.bytes > 0 &&
                    g.bytes <= r.bytes - off,
                SS_E_DESC, w + ": segments do not tile the record");
        const uint64_t n_rows = g.row1 - g.row0;
        if (r.row_bytes) require(g.bytes % n_rows == 0 && g.bytes / n_rows == r.row_bytes, SS_E_DESC,
                                 w + ": segment bytes differ from its uniform rows");
        row = g.row1;
        off += g.bytes;
      }
      require(row == r.rows && off == r.bytes, SS_E_DESC, w + ": segments do not tile the record");
      seg_next += r.seg_count;
    }
  }
  require(seg_next == ix.segments.size(), SS_E_DESC, "pack index: segments not owned by records");
}

// ---- writer -------------------------------------------------------------------------------------------------------

PackWriterFaults g_pack_writer_faults;

// A data file being written. sha and size count the bytes accepted (on disk or in buf); after a failed drain the file
// is unusable (failed) and the writer that owns it is poisoned.
struct PackWriter::Out {
  std::string path;
  Sha256 sha;
  uint64_t size = 0, on_disk = 0;
  std::vector<char> buf;
  size_t fill = 0;
  bool failed = false;
#if defined(_WIN32)
  HANDLE h = INVALID_HANDLE_VALUE;
#else
  int fd = -1;
  uint64_t advised = 0;
#endif
  explicit Out(const std::string& p) : path(p), buf(size_t(8) << 20) {
    if (g_pack_writer_faults.fail_create) {
      g_pack_writer_faults.fail_create = false;
      fail(SS_E_IO, "create " + p + ": injected failure");
    }
#if defined(_WIN32)
    h = CreateFileW(fs::u8path(p).wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    require(h != INVALID_HANDLE_VALUE, SS_E_IO, "create " + p + ": error " + std::to_string(GetLastError()));
#else
    fd = ::open(p.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    require(fd >= 0, SS_E_IO, "create " + p + ": " + std::strerror(errno));
#endif
  }
  ~Out() {
#if defined(_WIN32)
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
#else
    if (fd >= 0) ::close(fd);
#endif
  }
  void drain() {
    require(!failed, SS_E_IO, "write " + path + ": an earlier write failed");
    failed = true;  // until this drain completes
    const uint64_t at_fault = g_pack_writer_faults.fail_write_at;
    require(at_fault == 0 || on_disk + fill <= at_fault, SS_E_IO, "write " + path + ": injected failure");
    size_t at = 0;
    while (at < fill) {
#if defined(_WIN32)
      DWORD put = 0;
      const DWORD n = DWORD(std::min<size_t>(fill - at, size_t(1) << 30));
      require(WriteFile(h, buf.data() + at, n, &put, nullptr) && put > 0, SS_E_IO,
              "write " + path + ": error " + std::to_string(GetLastError()));
#else
      const ssize_t put = ::write(fd, buf.data() + at, fill - at);
      require(put > 0, SS_E_IO, "write " + path + ": " + std::strerror(errno));
#endif
      at += size_t(put);
    }
    on_disk += fill;
    fill = 0;
#if !defined(_WIN32)
    // a pack is hundreds of GiB: written pages leave the page cache (unified memory) once they reached the disk
    if (on_disk - advised >= (uint64_t(256) << 20)) {
      require(::fdatasync(fd) == 0, SS_E_IO, "sync " + path);
      (void)::posix_fadvise(fd, off_t(advised), off_t(on_disk - advised), POSIX_FADV_DONTNEED);
      advised = on_disk;
    }
#endif
    failed = false;
  }
  // A full buffer drains before more bytes are taken, so a failed drain leaves sha / size / fill matching what was taken.
  void put(const void* data, uint64_t n) {
    require(!failed, SS_E_IO, "write " + path + ": an earlier write failed");
    const char* p = static_cast<const char*>(data);
    while (n) {
      if (fill == buf.size()) drain();
      const size_t take = size_t(std::min<uint64_t>(n, buf.size() - fill));
      std::memcpy(buf.data() + fill, p, take);
      sha.update(p, take);
      size += take;
      fill += take;
      p += take;
      n -= take;
    }
  }
  void close() {
    drain();
#if defined(_WIN32)
    require(FlushFileBuffers(h) != 0, SS_E_IO, "flush " + path);
    CloseHandle(h);
    h = INVALID_HANDLE_VALUE;
#else
    require(::fsync(fd) == 0, SS_E_IO, "sync " + path);
    (void)::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    fd = -1;
#endif
  }
};

namespace {
void write_small_file(const std::string& path, const std::string& bytes) {
#if defined(_WIN32)
  HANDLE h = CreateFileW(fs::u8path(path).wstring().c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                         FILE_ATTRIBUTE_NORMAL, nullptr);
  require(h != INVALID_HANDLE_VALUE, SS_E_IO, "create " + path + ": error " + std::to_string(GetLastError()));
  DWORD put = 0;
  const bool ok = WriteFile(h, bytes.data(), DWORD(bytes.size()), &put, nullptr) && put == bytes.size() &&
                  FlushFileBuffers(h);
  CloseHandle(h);
#else
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0444);
  require(fd >= 0, SS_E_IO, "create " + path + ": " + std::strerror(errno));
  const bool ok = ::write(fd, bytes.data(), bytes.size()) == ssize_t(bytes.size()) && ::fsync(fd) == 0;
  ::close(fd);
#endif
  require(ok, SS_E_IO, "write " + path);
}
void sync_dir(const std::string& dir) {
#if !defined(_WIN32)
  const int fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (fd >= 0) {
    (void)::fsync(fd);
    ::close(fd);
  }
#else
  (void)dir;
#endif
}
}  // namespace

PackWriter::PackWriter(const std::string& dir, PackWriterOptions o) : dir_(dir), o_(std::move(o)) {
  require(!dir.empty(), SS_E_INVALID, "pack writer: empty directory");
  std::error_code ec;
  require(!fs::exists(fs::u8path(dir), ec), SS_E_INVALID, "pack writer: " + dir + " exists (use a new directory)");
  for (const PackSourceEntry& s : o_.sources)
    require(!s.name.empty() && is_hex64(s.sha256), SS_E_INVALID, "pack writer: source " + s.name +
                                                                     " needs a name and a lowercase SHA-256");
  for (const auto& kv : o_.formats) require(!kv.first.empty(), SS_E_INVALID, "pack writer: empty format tag");
  std::sort(o_.layouts.begin(), o_.layouts.end());
  o_.layouts.erase(std::unique(o_.layouts.begin(), o_.layouts.end()), o_.layouts.end());
  for (const std::string& l : o_.layouts) require(!l.empty(), SS_E_INVALID, "pack writer: empty layout name");
  require(o_.meta.is_object() && o_.receipt.is_object(), SS_E_INVALID, "pack writer: meta / receipt not objects");
  tmp_ = dir + ".incomplete";
  const fs::path t = fs::u8path(tmp_);
  if (t.has_parent_path()) fs::create_directories(t.parent_path(), ec);
  // an exclusive creation: two writers never share a temporary directory
  require(fs::create_directory(t, ec) && !ec, SS_E_IO, "pack writer: cannot create " + tmp_ +
                                                           " (another build in progress, or a stale one to remove)");
  try {  // no destructor runs for a failed constructor: DIR.incomplete is removed here
    open_file();
  } catch (...) {
    poison();
    throw;
  }
}

PackWriter::~PackWriter() {
  // a poisoned writer already removed its directory (another build may own it now), unless that removal failed: then
  // the directory is still there (no other build could create it) and one more attempt is made
  if (finished_ || (failed_ && !tmp_left_)) return;
  poison();
}

void PackWriter::live() const {
  require(!failed_, SS_E_IO, "pack writer: an earlier error failed this build (abort it and start again)");
  require(!finished_, SS_E_INVALID, "pack writer: already finished");
}

void PackWriter::poison() {
  failed_ = true;
  open_rec_ = false;
  out_.reset();
  std::error_code ec;
  fs::remove_all(fs::u8path(tmp_), ec);
  tmp_left_ = ec && !tmp_left_;  // retried once, by the destructor
}

uint32_t PackWriter::intern(const std::string& s) {
  auto it = string_ids_.find(s);
  if (it != string_ids_.end()) return it->second;
  require(string_bytes_ + 4 + s.size() <= kPackMaxStrings, SS_E_INVALID,
          "pack writer: the index string table would exceed its limit");
  string_bytes_ += 4 + s.size();
  const uint32_t id = uint32_t(ix_.strings.size());
  ix_.strings.push_back(s);
  string_ids_[s] = id;
  return id;
}

void PackWriter::open_file() {
  require(files_.size() < kPackMaxFiles, SS_E_INVALID, "pack writer: too many data files");
  files_.push_back({data_name(files_.size()), 0, ""});
  out_ = std::make_unique<Out>(path_join(tmp_, files_.back().name));
}

void PackWriter::close_file() {
  out_->close();
  files_.back().size = out_->size;
  files_.back().sha256 = out_->sha.hex();
  out_.reset();
}

void PackWriter::begin(const PackRecordSpec& r) {
  live();
  require(!open_rec_, SS_E_INVALID, "pack writer: begin inside a record");
  const std::string w = "pack writer: record (" + std::to_string(r.layer) + ", " + std::to_string(r.expert) + ", " +
                        r.name + ")";
  require(!r.name.empty() && r.name.size() <= 4096 && r.bytes > 0 && r.layer >= -1 && r.expert >= -1, SS_E_INVALID,
          w + ": needs a name, bytes and a key >= -1");
  require(o_.formats.count(r.format) == 1, SS_E_INVALID, w + ": format " + r.format + " not declared");
  require(ix_.records.size() < kPackMaxRecords, SS_E_INVALID, "pack writer: too many records");
  if (r.rows == 0) {
    require(r.row_bytes == 0 && r.seg_rows.empty() && r.seg_bytes.empty(), SS_E_INVALID, w + ": row geometry without rows");
  } else if (r.row_bytes) {
    require(r.bytes % r.row_bytes == 0 && r.bytes / r.row_bytes == r.rows, SS_E_INVALID, w + ": rows x row_bytes != bytes");
  } else {
    require(!r.seg_rows.empty(), SS_E_INVALID, w + ": non-uniform rows need segments");
  }
  seg_end_.clear();
  std::vector<PackSegment> segs;
  if (!r.seg_rows.empty()) {
    require(r.rows > 0 && r.seg_rows.size() >= 2 && r.seg_rows.front() == 0 && r.seg_rows.back() == r.rows,
            SS_E_INVALID, w + ": segment boundaries must run from 0 to rows");
    const size_t n = r.seg_rows.size() - 1;
    require(r.row_bytes ? (r.seg_bytes.empty() || r.seg_bytes.size() == n) : r.seg_bytes.size() == n, SS_E_INVALID,
            w + ": segment bytes count");
    require(ix_.segments.size() + n <= kPackMaxSegments, SS_E_INVALID, "pack writer: too many segments");
    uint64_t off = 0;
    for (size_t k = 0; k < n; ++k) {
      require(r.seg_rows[k + 1] > r.seg_rows[k], SS_E_INVALID, w + ": segment boundaries not increasing");
      // rows x row_bytes == bytes was checked above, so a sub-range's product cannot overflow
      const uint64_t b = r.row_bytes ? uint64_t(r.seg_rows[k + 1] - r.seg_rows[k]) * r.row_bytes : r.seg_bytes[k];
      if (r.row_bytes && !r.seg_bytes.empty()) require(r.seg_bytes[k] == b, SS_E_INVALID, w + ": segment bytes");
      require(b > 0 && b <= r.bytes - off, SS_E_INVALID, w + ": segment bytes outside the record");
      segs.push_back({r.seg_rows[k], r.seg_rows[k + 1], off, b, 0});
      off += b;
      seg_end_.push_back(off);
    }
    require(off == r.bytes, SS_E_INVALID, w + ": segment bytes do not sum to the record's bytes");
  }
  {  // the string table budget, before any state changes
    uint64_t more = 0;
    if (!string_ids_.count(r.name)) more += 4 + r.name.size();
    if (!string_ids_.count(r.format) && r.format != r.name) more += 4 + r.format.size();
    require(string_bytes_ + more <= kPackMaxStrings, SS_E_INVALID,
            "pack writer: the index string table would exceed its limit");
  }
  // unit-major order: every rule is checked before any state changes (a refused record leaves the writer as it was)
  require(r.layer >= last_layer_, SS_E_INVALID, w + ": layers must be added in ascending order");
  const bool new_layer = r.layer != last_layer_;
  if (r.expert < 0) {
    require(new_layer || !layer_has_units_, SS_E_INVALID, w + ": a layer's dense records come before its units");
  } else if (!new_layer && r.expert != last_expert_) {
    require(closed_units_.count(r.expert) == 0, SS_E_INVALID, w + ": a unit's records must be adjacent");
  }
  require(keys_.count({{r.layer, r.expert}, r.name}) == 0, SS_E_INVALID, w + ": duplicate key");
  try {  // past the checks: any error (a data-file split) poisons the writer
    begin_checked(r, segs, new_layer);
  } catch (...) {
    poison();
    throw;
  }
}

void PackWriter::begin_checked(const PackRecordSpec& r, const std::vector<PackSegment>& segs, bool new_layer) {
  if (new_layer) {
    closed_units_.clear();
    if (split_bytes_reached()) {
      close_file();
      open_file();
    }
    last_layer_ = r.layer;
    last_expert_ = -1;
    layer_has_units_ = false;
  }
  if (r.expert >= 0) {
    if (r.expert != last_expert_) {
      if (last_expert_ >= 0) closed_units_.insert(last_expert_);
      last_expert_ = r.expert;
    }
    layer_has_units_ = true;
  }
  keys_.insert({{r.layer, r.expert}, r.name});
  rec_ = PackRecord();
  rec_.layer = r.layer;
  rec_.expert = r.expert;
  rec_.name = intern(r.name);
  rec_.format = intern(r.format);
  rec_.file = uint16_t(files_.size() - 1);
  rec_.rows = r.rows;
  rec_.row_bytes = r.row_bytes;
  rec_.offset = out_->size;
  rec_.bytes = r.bytes;
  rec_.seg_first = uint32_t(ix_.segments.size());
  rec_.seg_count = uint32_t(segs.size());
  for (const PackSegment& g : segs) ix_.segments.push_back(g);
  rec_written_ = 0;
  seg_i_ = 0;
  rec_hash_ = Xxh64();
  seg_hash_ = Xxh64();
  open_rec_ = true;
}

bool PackWriter::split_bytes_reached() const {
  return o_.split_bytes && out_->size > 0 && out_->size >= o_.split_bytes;
}

void PackWriter::write(const void* data, uint64_t n) {
  live();
  require(open_rec_, SS_E_INVALID, "pack writer: write outside a record");
  require(n <= rec_.bytes - rec_written_, SS_E_INVALID, "pack writer: more bytes than the record declared");
  const char* p = static_cast<const char*>(data);
  try {
    out_->put(p, n);
  } catch (...) {  // some of the bytes may be in the file: no retry can make the record whole again
    poison();
    throw;
  }
  rec_hash_.update(p, size_t(n));
  // segment hashes: each byte range goes to the segment it belongs to
  uint64_t at = rec_written_, left = n;
  while (left && seg_i_ < seg_end_.size()) {
    const uint64_t take = std::min(left, seg_end_[seg_i_] - at);
    seg_hash_.update(p, size_t(take));
    p += take;
    at += take;
    left -= take;
    if (at == seg_end_[seg_i_]) {
      ix_.segments[rec_.seg_first + seg_i_].xxh64 = seg_hash_.digest();
      seg_hash_ = Xxh64();
      ++seg_i_;
    }
  }
  rec_written_ += n;
}

void PackWriter::end() {
  live();
  require(open_rec_, SS_E_INVALID, "pack writer: end outside a record");
  require(rec_written_ == rec_.bytes, SS_E_INVALID, "pack writer: record " + ix_.strings[rec_.name] + " got " +
                                                        std::to_string(rec_written_) + " of " +
                                                        std::to_string(rec_.bytes) + " bytes");
  rec_.xxh64 = rec_hash_.digest();
  static const char zeros[kPackAlign] = {};
  const uint64_t pad = pad_up(out_->size) - out_->size;
  try {
    if (pad) out_->put(zeros, pad);
    ix_.records.push_back(rec_);
  } catch (...) {
    poison();
    throw;
  }
  total_ += rec_.bytes;
  open_rec_ = false;
}

std::string PackWriter::finish() {
  live();
  require(!open_rec_, SS_E_INVALID, "pack writer: finish inside a record");
  require(!ix_.records.empty(), SS_E_INVALID, "pack writer: no records");
  try {
    return finish_checked();
  } catch (...) {
    if (!finished_) poison();
    throw;
  }
}

namespace {
// Renames the finished directory to dir, never over an existing one: renameat2(RENAME_NOREPLACE) on Linux when the
// kernel and the file system have it (no window between the check and the rename), else a check then the rename (on
// Windows a directory never replaces an existing one; elsewhere an empty one appearing in between is the window left).
void publish_dir(const std::string& tmp, const std::string& dir) {
#if defined(__linux__) && defined(RENAME_NOREPLACE)
  if (::renameat2(AT_FDCWD, tmp.c_str(), AT_FDCWD, dir.c_str(), RENAME_NOREPLACE) == 0) return;
  const int e = errno;
  require(e == ENOSYS || e == EINVAL, SS_E_IO,
          "pack writer: publish " + dir + ": " +
              (e == EEXIST ? std::string("it appeared during the build") : std::string(std::strerror(e))));
#endif
  std::error_code ec;
  require(!fs::exists(fs::u8path(dir), ec), SS_E_IO, "pack writer: " + dir + " appeared during the build");
  fs::rename(fs::u8path(tmp), fs::u8path(dir), ec);
  require(!ec, SS_E_IO, "pack writer: publish " + dir + ": " + ec.message());
}
}  // namespace

std::string PackWriter::finish_checked() {
  close_file();
  PackManifest m;
  m.model = o_.model;
  m.sources = o_.sources;
  m.formats = o_.formats;
  m.files = files_;
  m.layouts = o_.layouts;
  m.meta = o_.meta;
  m.receipt = o_.receipt;
  const std::string index_bytes = pack_index_write(ix_);
  m.index = {kPackIndexName, index_bytes.size(), sha256_hex(index_bytes.data(), index_bytes.size())};
  m.records = ix_.records.size();
  m.segments = ix_.segments.size();
  pack_index_validate(ix_, m);  // the reader's rules, before anything is published
  write_small_file(path_join(tmp_, kPackIndexName), index_bytes);
  const std::string text = pack_manifest_write(m);
  require(text.size() <= kPackMaxManifest, SS_E_INVALID, "pack writer: pack.json would exceed its 16 MiB limit");
  write_small_file(path_join(tmp_, kPackManifestName), text);
  sync_dir(tmp_);
  const std::string pin = sha256_hex(text.data(), text.size());
  publish_dir(tmp_, dir_);
  finished_ = true;
  // the pack is published: a failure to make its rename durable does not fail the build
  try {
    const fs::path parent = fs::u8path(dir_).parent_path();
    sync_dir(parent.empty() ? "." : parent.u8string());
  } catch (...) {
  }
  return pin;
}

// ---- reader -------------------------------------------------------------------------------------------------------

size_t Pack::KeyHash::operator()(const std::pair<std::pair<int32_t, int32_t>, std::string>& k) const {
  return std::hash<std::string>()(k.second) ^ (size_t(uint32_t(k.first.first)) * 0x9E3779B97F4A7C15ull) ^
         (size_t(uint32_t(k.first.second)) << 1);
}

namespace {
// Verified-record cache (Cached mode): "TSPACKV1", u32 files, per file the data file identity (5 x u64), u64 records,
// u64 segments, then a byte per record and per segment (1 = verified). Bound to the manifest digest by its file name.
std::string verified_blob(const std::vector<FileIdentity>& ids, const std::vector<uint8_t>& rec,
                          const std::vector<uint8_t>& seg) {
  std::string s("TSPACKV1", 8);
  put32(s, uint32_t(ids.size()));
  for (const FileIdentity& id : ids) {
    put64(s, id.volume);
    put64(s, id.index);
    put64(s, id.size);
    put64(s, id.mtime);
    put64(s, id.ctime);
  }
  put64(s, rec.size());
  put64(s, seg.size());
  s.append(reinterpret_cast<const char*>(rec.data()), rec.size());
  s.append(reinterpret_cast<const char*>(seg.data()), seg.size());
  return s;
}
bool verified_parse(const std::string& s, const std::vector<FileIdentity>& ids, std::vector<uint8_t>* rec,
                    std::vector<uint8_t>* seg) {
  try {
    Cursor c{s};
    c.need(8);
    if (s.compare(0, 8, "TSPACKV1") != 0) return false;
    c.at = 8;
    if (c.get(4) != ids.size()) return false;
    for (const FileIdentity& id : ids) {
      FileIdentity got;
      got.volume = c.get(8);
      got.index = c.get(8);
      got.size = c.get(8);
      got.mtime = c.get(8);
      got.ctime = c.get(8);
      if (got != id) return false;
    }
    if (c.get(8) != rec->size() || c.get(8) != seg->size()) return false;
    c.need(rec->size() + seg->size());
    if (c.at + rec->size() + seg->size() != s.size()) return false;
    for (size_t i = 0; i < rec->size(); ++i) (*rec)[i] |= uint8_t(s[c.at + i] == 1);
    c.at += rec->size();
    for (size_t i = 0; i < seg->size(); ++i) (*seg)[i] |= uint8_t(s[c.at + i] == 1);
    return true;
  } catch (const Error&) {
    return false;
  }
}
std::string read_small(const std::string& path) {
  std::string s;
  std::FILE* f = nullptr;
#if defined(_WIN32)
  f = _wfopen(fs::u8path(path).wstring().c_str(), L"rb");
#else
  f = std::fopen(path.c_str(), "rb");
#endif
  if (!f) return s;
  char b[65536];
  size_t n;
  while ((n = std::fread(b, 1, sizeof b, f)) > 0 && s.size() < (uint64_t(1) << 30)) s.append(b, n);
  std::fclose(f);
  return s;
}
}  // namespace

std::unique_ptr<Pack> Pack::open(const std::string& dir, const PackOptions& o) {
  std::unique_ptr<Pack> p(new Pack());
  p->dir_ = dir;
  p->o_ = o;
  // pack.json and pack.index: bound opens, held (share-read) for the pack's lifetime, read through their handles
  p->manifest_file_ = std::make_unique<RawFile>(path_join(dir, kPackManifestName), false, true);
  const std::string text = read_whole(*p->manifest_file_, kPackMaxManifest, "pack.json");
  p->manifest_sha_ = sha256_hex(text.data(), text.size());
  if (!o.pin_sha256.empty())
    require(lower(o.pin_sha256) == p->manifest_sha_, SS_E_DESC,
            "pack " + dir + ": pack.json SHA-256 " + p->manifest_sha_ + " differs from the pinned " + o.pin_sha256);
  p->m_ = pack_manifest_parse(text);
  if (!o.source_sha256.empty()) {
    require(o.source_sha256.size() == p->m_.sources.size(), SS_E_DESC,
            "pack " + dir + ": built from " + std::to_string(p->m_.sources.size()) + " sources, the app has " +
                std::to_string(o.source_sha256.size()));
    for (size_t i = 0; i < o.source_sha256.size(); ++i)
      require(lower(o.source_sha256[i]) == p->m_.sources[i].sha256, SS_E_DESC,
              "pack " + dir + ": source " + p->m_.sources[i].name + " SHA-256 " + p->m_.sources[i].sha256 +
                  " differs from the app's " + o.source_sha256[i]);
  }
  for (const std::string& l : o.require_layouts)
    if (!std::binary_search(p->m_.layouts.begin(), p->m_.layouts.end(), l)) {
      std::string have;
      for (const std::string& h : p->m_.layouts) have += (have.empty() ? "" : ", ") + h;
      fail(SS_E_DESC, "pack " + dir + " has no layout " + l + " (built with: " + (have.empty() ? "none" : have) + ")");
    }
  p->index_file_ = std::make_unique<RawFile>(path_join(dir, kPackIndexName), false, true);
  require(p->index_file_->size() == p->m_.index.size, SS_E_DESC,
          "pack " + dir + ": pack.index size differs from pack.json");  // before anything is allocated for it
  const std::string ib = read_whole(*p->index_file_, kHeaderSize + kPackMaxRecords * kRecordSize +
                                                         kPackMaxSegments * kSegmentSize + kPackMaxStrings,
                                    "pack.index");
  require(ib.size() == p->m_.index.size && sha256_hex(ib.data(), ib.size()) == p->m_.index.sha256, SS_E_DESC,
          "pack " + dir + ": pack.index differs from pack.json");
  p->ix_ = pack_index_parse(ib);
  pack_index_validate(p->ix_, p->m_);
  // data files: descriptor-source rules (bound opens, stripes of each data file through weight_dirs)
  ss_options so{};
  so.struct_size = sizeof so;
  so.direct_io = o.direct ? 1 : 0;
  so.weight_dirs = o.weight_dirs.empty() ? nullptr : o.weight_dirs.c_str();
  so.drive_readers = o.drive_readers;
  so.weight_verify = SS_MFILE_VERIFY_CACHED;
  for (size_t i = 0; i < p->m_.files.size(); ++i) {
    const std::string path = path_join(dir, p->m_.files[i].name);
    p->files_.push_back(open_source(path, so));
    require(p->files_.back()->size() == p->m_.files[i].size, SS_E_DESC,
            "pack " + dir + ": " + p->m_.files[i].name + " size differs from pack.json");
  }
  for (uint32_t f = 0; f < p->files_.size(); ++f) {
    const std::shared_ptr<MFile> s = p->files_[f]->session();
    p->striped_.push_back(s && s->multi() ? 1 : 0);
    p->held_.push_back(std::make_unique<RawFile>(p->file_path(f), false, true));
    p->open_ids_.push_back(p->held_.back()->identity());
    check_identity(p->files_[f]->identity(), p->open_ids_.back(), "pack data file " + p->m_.files[f].name);
  }
  for (size_t i = 0; i < p->ix_.records.size(); ++i) {
    const PackRecord& r = p->ix_.records[i];
    p->by_key_[{{r.layer, r.expert}, p->ix_.strings[r.name]}] = i;
    if (r.expert < 0) continue;
    auto it = p->units_.find({r.layer, r.expert});
    if (it == p->units_.end()) {
      p->units_[{r.layer, r.expert}] = PackUnit{r.file, r.offset, r.bytes, i, 1};
    } else {
      it->second.bytes = r.offset + r.bytes - it->second.offset;
      ++it->second.count;
    }
  }
  p->rec_ok_.assign(p->ix_.records.size(), 0);
  p->seg_ok_.assign(p->ix_.segments.size(), 0);
  if (o.verify == PackVerify::Cached && o.persist_verified) {
    const std::string cd = cache_dir();
    if (!cd.empty()) {
      const std::string vd = path_join(cd, "pack-verified");
      std::error_code ec;
      fs::create_directories(fs::u8path(vd), ec);
      p->verified_path_ = path_join(vd, p->manifest_sha_ + ".bits");
      verified_parse(read_small(p->verified_path_), p->open_ids_, &p->rec_ok_, &p->seg_ok_);
    }
  }
  return p;
}

Pack::~Pack() { persist(); }

// Best effort: the cache only saves later checks, so a failure to write it (another process writing it at the same
// time, a read-only cache directory) is not an error.
void Pack::persist() {
  std::lock_guard<std::mutex> lk(mu_);
  if (!dirty_ || verified_path_.empty()) return;
  try {
    // keyed by the identities seen at open: a data file changed since (Linux cannot deny writers) is never vouched for
    if (!files_unchanged()) return;
    std::vector<uint8_t> rec = rec_ok_, seg = seg_ok_;
    verified_parse(read_small(verified_path_), open_ids_, &rec, &seg);  // keep another process's results
    write_file_atomic(verified_path_, verified_blob(open_ids_, rec, seg));
    dirty_ = false;
  } catch (const std::exception&) {
  }
}

bool Pack::files_unchanged(const std::vector<uint8_t>* only) const {
  for (size_t f = 0; f < held_.size(); ++f)
    if ((!only || (*only)[f]) && held_[f]->identity() != open_ids_[f]) return false;
  return true;
}

void Pack::forget_verified() {  // under mu_
  std::fill(rec_ok_.begin(), rec_ok_.end(), uint8_t(0));
  std::fill(seg_ok_.begin(), seg_ok_.end(), uint8_t(0));
  verified_path_.clear();
  dirty_ = false;
  changed_ = true;
}

std::string Pack::file_path(uint32_t file) const { return path_join(dir_, m_.files[file].name); }

int64_t Pack::find(int32_t layer, int32_t expert, const std::string& name) const {
  auto it = by_key_.find({{layer, expert}, name});
  return it == by_key_.end() ? -1 : int64_t(it->second);
}

bool Pack::unit(int32_t layer, int32_t expert, PackUnit* out) const {
  auto it = units_.find({layer, expert});
  if (it == units_.end()) return false;
  if (out) *out = it->second;
  return true;
}

void Pack::row_span(const PackRecord& r, uint32_t row0, uint32_t row1, uint64_t* off, uint64_t* bytes) const {
  if (row1 == 0 && row0 == 0) {
    *off = r.offset;
    *bytes = r.bytes;
    return;
  }
  const std::string w = "pack: rows [" + std::to_string(row0) + ", " + std::to_string(row1) + ") of " + name(r);
  require(r.rows > 0 && row0 < row1 && row1 <= r.rows, SS_E_INVALID, w + ": not a row range of the record");
  if (r.row_bytes) {
    *off = r.offset + uint64_t(row0) * r.row_bytes;
    *bytes = uint64_t(row1 - row0) * r.row_bytes;
    return;
  }
  // non-uniform rows: the range must be whole segments
  uint64_t a = UINT64_MAX, b = 0;
  for (uint32_t k = 0; k < r.seg_count; ++k) {
    const PackSegment& g = ix_.segments[r.seg_first + k];
    if (g.row0 == row0) a = g.offset;
    if (g.row1 == row1) b = g.offset + g.bytes;
  }
  require(a != UINT64_MAX && b > a, SS_E_INVALID, w + ": non-uniform rows are read by whole segments");
  *off = r.offset + a;
  *bytes = b - a;
}

bool Pack::record_verified(uint64_t i) const { return rec_ok_[i] != 0; }
bool Pack::segment_verified(uint64_t i) const { return seg_ok_[i] != 0; }

namespace {
struct Piece {  // one read of the plan: in place or into a bounce, then copied out
  const SourceFile* file;
  uint64_t offset, bytes;
  char* dst;
};
struct CopyOut {
  char* to;
  const char* from;
  uint64_t bytes;
};
// Adds the pieces that land [F, F + n) of `file` at out: an unbuffered source reads in place when out is aligned like
// F (whole aligned blocks), its unaligned head and tail blocks through bounce blocks; otherwise the covering aligned
// span goes through one bounce. Pieces never cross the source's piece grid.
void plan_span(const SourceFile& file, uint64_t F, uint64_t n, char* out, std::vector<Piece>* pieces,
               std::vector<CopyOut>* copies, std::vector<std::unique_ptr<Aligned>>* bounces, uint64_t* bounce_bytes) {
  const uint64_t al = file.direct() ? std::max<uint64_t>(file.alignment(kPackAlign), 1) : 1;
  const uint64_t grid = file.piece_grid();
  auto add = [&](uint64_t off, uint64_t len, char* dst) {
    while (len) {
      uint64_t take = len;
      if (grid) take = std::min(take, grid - off % grid);
      pieces->push_back({&file, off, take, dst});
      off += take;
      dst += take;
      len -= take;
    }
  };
  if (al == 1) {
    add(F, n, out);
    return;
  }
  const uint64_t a0 = F / al * al, a1 = (F + n + al - 1) / al * al;
  const uint64_t head = F - a0;
  const bool co = (uintptr_t(out) + al - head % al) % al == 0;
  const uint64_t m0 = (F + al - 1) / al * al, m1 = (F + n) / al * al;  // whole aligned blocks inside the span
  if (co && m1 > m0) {
    if (F < m0) {  // head block
      bounces->push_back(std::make_unique<Aligned>(al));
      add(a0, al, bounces->back()->p);
      copies->push_back({out, bounces->back()->p + head, m0 - F});
      *bounce_bytes += m0 - F;
    }
    add(m0, m1 - m0, out + (m0 - F));
    if (m1 < F + n) {  // tail block
      bounces->push_back(std::make_unique<Aligned>(al));
      add(m1, al, bounces->back()->p);
      copies->push_back({out + (m1 - F), bounces->back()->p, F + n - m1});
      *bounce_bytes += F + n - m1;
    }
    return;
  }
  bounces->push_back(std::make_unique<Aligned>(a1 - a0));
  add(a0, a1 - a0, bounces->back()->p);
  copies->push_back({out, bounces->back()->p + head, n});
  *bounce_bytes += n;
}
}  // namespace

void Pack::readv(const PackRead* reads, size_t n) {
  struct Job {
    uint64_t record;
    uint64_t off, bytes;  // file span read
    char* out;
    // verification after the read: the whole record, or segments [s0, s1) (record-relative offsets base)
    bool check_record = false;
    uint32_t s0 = 0, s1 = 0;
    // a range read verified through its whole record: the record lands in tmp, the range is copied to dst
    std::unique_ptr<Aligned> tmp;
    char* dst = nullptr;
    uint64_t dst_off = 0, dst_bytes = 0;
  };
  std::vector<Job> jobs(n);
  // Cached: results vouch only for the bytes of the files as opened (Linux cannot deny writers); only the data files
  // this call reads are checked (a cached result is relied on only by a read of its own file)
  const bool cached = o_.verify == PackVerify::Cached;
  std::vector<uint8_t> touched(files_.size(), 0);
  for (size_t i = 0; i < n; ++i) {
    require(reads[i].record < ix_.records.size() && reads[i].dst != nullptr, SS_E_INVALID, "pack: bad read");
    touched[ix_.records[reads[i].record].file] = 1;
  }
  if (cached) {
    std::lock_guard<std::mutex> lk(mu_);
    if (!changed_ && !files_unchanged(&touched)) forget_verified();
  }
  bool relied = false;  // a job skipped hashing because of a cached result
  for (size_t i = 0; i < n; ++i) {
    const PackRead& rd = reads[i];
    const PackRecord& r = ix_.records[rd.record];
    Job& j = jobs[i];
    j.record = rd.record;
    row_span(r, rd.row0, rd.row1, &j.off, &j.bytes);
    j.out = static_cast<char*>(rd.dst);
    if (o_.verify == PackVerify::Off) continue;
    std::lock_guard<std::mutex> lk(mu_);
    // a striped data file is hashed on every read: its stripe copies have no live identity check here
    const bool always = o_.verify == PackVerify::Always || changed_ || striped_[r.file];
    if (j.off == r.offset && j.bytes == r.bytes) {
      j.check_record = always || !record_verified(rd.record);
      relied = relied || !j.check_record;
      continue;
    }
    if (!always && record_verified(rd.record)) {
      relied = true;
      continue;
    }
    // whole segments covering exactly the range?
    uint32_t s0 = UINT32_MAX, s1 = UINT32_MAX;
    for (uint32_t k = 0; k < r.seg_count; ++k) {
      const PackSegment& g = ix_.segments[r.seg_first + k];
      if (r.offset + g.offset == j.off) s0 = k;
      if (r.offset + g.offset + g.bytes == j.off + j.bytes) s1 = k + 1;
    }
    if (s0 != UINT32_MAX && s1 != UINT32_MAX && s1 > s0) {
      bool all = !always;
      for (uint32_t k = s0; all && k < s1; ++k) all = segment_verified(r.seg_first + k);
      if (!all) {
        j.s0 = s0;
        j.s1 = s1;
      }
      relied = relied || all;
      continue;
    }
    // otherwise the whole record is read and checked once, the range copied out of it
    j.tmp = std::make_unique<Aligned>(r.bytes);
    j.dst = j.out;
    j.dst_off = j.off - r.offset;
    j.dst_bytes = j.bytes;
    j.out = j.tmp->p;
    j.off = r.offset;
    j.bytes = r.bytes;
    j.check_record = true;
  }
  std::vector<Piece> pieces;
  std::vector<CopyOut> copies;
  std::vector<std::unique_ptr<Aligned>> bounces;
  uint64_t bounce_bytes = 0, bytes = 0;
  for (Job& j : jobs) {
    plan_span(*files_[ix_.records[j.record].file], j.off, j.bytes, j.out, &pieces, &copies, &bounces, &bounce_bytes);
    bytes += j.bytes;
  }
  // all pieces of one data file in flight together (overlapped reads, or one preadv on a stripe session)
  for (uint32_t f = 0; f < files_.size(); ++f) {
    std::vector<RawFile::Piece> mine;
    for (const Piece& p : pieces)
      if (p.file == files_[f].get()) mine.push_back({p.dst, p.offset, p.bytes});
    if (!mine.empty()) files_[f]->read_pieces(mine);
  }
  for (const CopyOut& c : copies) std::memcpy(c.to, c.from, size_t(c.bytes));
  // verification (in parallel over jobs when there is enough to hash)
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<size_t> todo;
  for (size_t i = 0; i < jobs.size(); ++i)
    if (jobs[i].check_record || jobs[i].s1 > jobs[i].s0) todo.push_back(i);
  std::vector<std::string> errors(jobs.size());
  auto check = [&](size_t i) {
    const Job& j = jobs[i];
    const PackRecord& r = ix_.records[j.record];
    if (j.check_record) {
      if (xxh64(j.out, size_t(j.bytes)) != r.xxh64) errors[i] = "record " + name(r);
      return;
    }
    for (uint32_t k = j.s0; k < j.s1; ++k) {
      const PackSegment& g = ix_.segments[r.seg_first + k];
      const char* at = j.out + (r.offset + g.offset - j.off);
      if (xxh64(at, size_t(g.bytes)) != g.xxh64) {
        errors[i] = "rows [" + std::to_string(g.row0) + ", " + std::to_string(g.row1) + ") of " + name(r);
        return;
      }
    }
  };
  uint64_t hash_bytes = 0;
  for (size_t i : todo) hash_bytes += jobs[i].bytes;
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency() / 2);
  if (todo.size() > 1 && hash_bytes >= (uint64_t(64) << 20) && hw > 1) {
    std::atomic<size_t> next{0};
    std::mutex em;
    std::exception_ptr err;  // the first exception of a hashing thread, rethrown after every thread joined
    {
      std::vector<std::thread> ts;
      struct Join {  // also when starting a thread throws: no running std::thread is destroyed
        std::vector<std::thread>& ts;
        ~Join() {
          for (std::thread& t : ts)
            if (t.joinable()) t.join();
        }
      } join{ts};
      for (unsigned t = 0; t < std::min<size_t>(hw, todo.size()); ++t)
        ts.emplace_back([&] {
          try {
            for (size_t k; (k = next.fetch_add(1)) < todo.size();) check(todo[k]);
          } catch (...) {
            next = todo.size();
            std::lock_guard<std::mutex> lk(em);
            if (!err) err = std::current_exception();
          }
        });
    }
    if (err) std::rethrow_exception(err);
  } else {
    for (size_t i : todo) check(i);
  }
  const uint64_t ns =
      uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
  for (size_t i = 0; i < jobs.size(); ++i)
    require(errors[i].empty(), SS_E_HASH, "pack " + dir_ + ": xxh64 mismatch in " + errors[i]);
  if (relied && !files_unchanged(&touched)) {  // changed while bytes vouched for by the cache were read: they are not
    std::lock_guard<std::mutex> lk(mu_);
    forget_verified();
    fail(SS_E_IO, "pack " + dir_ + ": a data file changed while it was read");
  }
  for (Job& j : jobs)
    if (j.tmp) std::memcpy(j.dst, j.out + j.dst_off, size_t(j.dst_bytes));
  std::lock_guard<std::mutex> lk(mu_);
  for (size_t i : todo) {
    const Job& j = jobs[i];
    const PackRecord& r = ix_.records[j.record];
    if (j.check_record) {
      dirty_ = dirty_ || !rec_ok_[j.record];
      rec_ok_[j.record] = 1;
      ++stats_.verified_records;
    } else {
      for (uint32_t k = j.s0; k < j.s1; ++k) {
        dirty_ = dirty_ || !seg_ok_[r.seg_first + k];
        seg_ok_[r.seg_first + k] = 1;
        ++stats_.verified_segments;
      }
    }
    stats_.verified_bytes += j.bytes;
  }
  stats_.verify_ns += ns;
  stats_.reads += n;
  stats_.bytes += bytes;
  stats_.bounce_bytes += bounce_bytes;
}

void Pack::read_unit(int32_t layer, int32_t expert, void* dst) {
  PackUnit u;
  require(unit(layer, expert, &u), SS_E_INVALID,
          "pack: no unit (" + std::to_string(layer) + ", " + std::to_string(expert) + ")");
  // the padding between records is zero by construction; it is read as is
  std::vector<PackRead> rs;
  char* base = static_cast<char*>(dst);
  for (uint64_t i = u.first; i < u.first + u.count; ++i) {
    const PackRecord& r = ix_.records[i];
    rs.push_back({i, 0, 0, base + (r.offset - u.offset)});
  }
  // zero the gaps (padding) so the span equals the file bytes
  uint64_t at = u.offset;
  for (uint64_t i = u.first; i < u.first + u.count; ++i) {
    const PackRecord& r = ix_.records[i];
    if (r.offset > at) std::memset(base + (at - u.offset), 0, size_t(r.offset - at));
    at = r.offset + r.bytes;
  }
  readv(rs.data(), rs.size());
}

Json Pack::verify_full() {
  Json rep = Json::object();
  Json files = Json::array();
  bool ok_all = true;
  constexpr uint64_t kChunk = uint64_t(16) << 20;
  Aligned buf(kChunk);
  for (uint32_t f = 0; f < files_.size(); ++f) {
    const SourceFile& sf = *files_[f];
    Sha256 sha;
    // records of this file in offset order with their running hashes
    std::vector<uint64_t> recs;
    for (uint64_t i = 0; i < ix_.records.size(); ++i)
      if (ix_.records[i].file == f) recs.push_back(i);
    size_t ri = 0;
    Xxh64 rh, sh;
    uint32_t sk = 0;
    uint64_t bad = 0;
    std::vector<std::string> bad_names;
    for (uint64_t at = 0; at < m_.files[f].size; at += kChunk) {
      const uint64_t len = std::min(kChunk, m_.files[f].size - at);
      require(sf.pread(buf.p, len, at) >= len, SS_E_IO, "pack: short read of " + m_.files[f].name);
      sha.update(buf.p, size_t(len));
      uint64_t pos = at;
      while (ri < recs.size() && pos < at + len) {
        const PackRecord& r = ix_.records[recs[ri]];
        if (pos < r.offset) {
          pos = std::min(r.offset, at + len);
          continue;
        }
        const uint64_t end = std::min(r.offset + r.bytes, at + len);
        const char* p = buf.p + (pos - at);
        rh.update(p, size_t(end - pos));
        // segments
        uint64_t sp = pos;
        while (sk < r.seg_count && sp < end) {
          const PackSegment& g = ix_.segments[r.seg_first + sk];
          const uint64_t gend = std::min(r.offset + g.offset + g.bytes, end);
          sh.update(buf.p + (sp - at), size_t(gend - sp));
          sp = gend;
          if (gend == r.offset + g.offset + g.bytes) {
            const bool good = sh.digest() == g.xxh64;
            {
              std::lock_guard<std::mutex> lk(mu_);
              if (good && !seg_ok_[r.seg_first + sk]) dirty_ = true;
              seg_ok_[r.seg_first + sk] = good ? 1 : 0;
            }
            if (!good) ++bad;
            sh = Xxh64();
            ++sk;
          }
        }
        pos = end;
        if (end == r.offset + r.bytes) {
          const bool good = rh.digest() == r.xxh64;
          {
            std::lock_guard<std::mutex> lk(mu_);
            if (good && !rec_ok_[recs[ri]]) dirty_ = true;
            rec_ok_[recs[ri]] = good ? 1 : 0;
          }
          if (!good) {
            ++bad;
            if (bad_names.size() < 16) bad_names.push_back(name(r));
          }
          rh = Xxh64();
          sk = 0;
          ++ri;
        }
      }
    }
    const std::string got = sha.hex();
    Json e = Json::object();
    e["name"] = Json::string(m_.files[f].name);
    e["sha256_ok"] = Json::boolean(got == m_.files[f].sha256);
    e["bad_records"] = Json::number(bad);
    Json names = Json::array();
    for (const std::string& s : bad_names) names.push(Json::string(s));
    e["bad"] = names;
    ok_all = ok_all && got == m_.files[f].sha256 && bad == 0;
    files.push(e);
  }
  rep["files"] = files;
  rep["ok"] = Json::boolean(ok_all);
  rep["manifest_sha256"] = Json::string(manifest_sha_);
  if (ok_all) persist();
  return rep;
}

namespace {
std::string ids_text(const std::vector<FileIdentity>& ids) {
  std::string s;
  for (const FileIdentity& id : ids)
    s += std::to_string(id.volume) + "," + std::to_string(id.index) + "," + std::to_string(id.size) + "," +
         std::to_string(id.mtime) + "," + std::to_string(id.ctime) + ";";
  return s;
}
}  // namespace

std::vector<uint64_t> Pack::unit_hashes(int32_t layer, int n_units, const std::vector<std::string>& parts) {
  std::vector<uint64_t> out(static_cast<size_t>(n_units));
  uint64_t unit_bytes = 0;
  std::vector<std::vector<uint64_t>> recs(static_cast<size_t>(n_units));
  for (int u = 0; u < n_units; ++u)
    for (const std::string& name : parts) {
      const int64_t k = find(layer, u, name);
      require(k >= 0, SS_E_DESC, "pack: no record (" + std::to_string(layer) + ", " + std::to_string(u) + ", " + name + ")");
      recs[size_t(u)].push_back(uint64_t(k));
      if (u == 0) unit_bytes += ix_.records[size_t(k)].bytes;
    }
  constexpr int kBatch = 8;  // units in flight together (8 x one unit's selected parts of host memory)
  std::vector<std::unique_ptr<Aligned>> bufs;
  for (int i = 0; i < kBatch; ++i) bufs.push_back(std::make_unique<Aligned>(unit_bytes + kPackAlign * parts.size()));
  for (int u0 = 0; u0 < n_units; u0 += kBatch) {
    const int n = std::min(kBatch, n_units - u0);
    std::vector<PackRead> rs;
    std::vector<std::vector<std::pair<char*, uint64_t>>> spans(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) {
      char* at = bufs[size_t(k)]->p;
      for (uint64_t ri : recs[size_t(u0 + k)]) {
        const uint64_t b = ix_.records[size_t(ri)].bytes;
        rs.push_back({ri, 0, 0, at});
        spans[size_t(k)].push_back({at, b});
        at += pad_up(b);  // each part lands aligned; the hash runs over the parts back to back
      }
    }
    readv(rs.data(), rs.size());
    for (int k = 0; k < n; ++k) {
      Xxh64 h;
      for (const auto& sp : spans[size_t(k)]) h.update(sp.first, size_t(sp.second));
      out[size_t(u0 + k)] = h.digest();
    }
  }
  return out;
}

Json Pack::dg_descriptor(const Json& base) {
  Desc probe = desc_from_json(base);  // the engine's descriptor must be valid as given
  desc_validate(probe);
  Json out = base;
  Json* sources = &out["sources"];
  require(sources->is_array(), SS_E_DESC, "base descriptor: sources");
  const size_t first = sources->arr.size();
  // desc-v1 addresses at most 1024 sources (desc.cpp): refused here by name rather than by the descriptor check
  require(first + files_.size() <= 1024, SS_E_UNSUPPORTED,
          "pack: the base descriptor's " + std::to_string(first) + " sources and the pack's " +
              std::to_string(files_.size()) + " data files exceed desc-v1's 1024 sources");
  for (uint32_t f = 0; f < files_.size(); ++f) {
    Json s = Json::object();
    s["name"] = Json::string("pack:" + m_.files[f].name);
    s["path"] = Json::string(file_path(f));
    s["align"] = Json::number(uint64_t(kPackAlign));
    sources->push(s);
  }
  Json& layers = out["layers"];
  for (size_t li = 0; li < probe.layers.size(); ++li) {
    const Layer& L = probe.layers[li];
    if (L.parts.empty() || L.n_units == 0) continue;
    Json& lj = layers.arr[li];
    Json& parts = lj["parts"];
    for (size_t pi = 0; pi < L.parts.size(); ++pi) {
      const Part& part = L.parts[pi];
      const std::string w = "layer " + std::to_string(L.index) + " part " + part.name;
      const int64_t r0 = find(L.index, 0, part.name);
      require(r0 >= 0, SS_E_DESC, w + ": no pack record (" + std::to_string(L.index) + ", 0, " + part.name + ")");
      const PackRecord& a = ix_.records[size_t(r0)];
      uint64_t stride = 0, prev = 0;
      for (int u = 0; u < L.n_units; ++u) {
        const int64_t ri = find(L.index, u, part.name);
        require(ri >= 0, SS_E_DESC, w + ": no pack record for unit " + std::to_string(u));
        const PackRecord& r = ix_.records[size_t(ri)];
        require(r.bytes == part.bytes, SS_E_DESC, w + ": unit " + std::to_string(u) + " has " +
                                                      std::to_string(r.bytes) + " bytes in the pack, the engine " +
                                                      std::to_string(part.bytes));
        require(r.file == a.file, SS_E_DESC, w + ": units in different data files");
        // ascending offsets: a descending order would wrap the stride
        require(u == 0 || r.offset > prev, SS_E_UNSUPPORTED, w + ": units are not placed in ascending order");
        prev = r.offset;
        if (u == 1) stride = r.offset - a.offset;
        require(r.offset == a.offset + uint64_t(u) * stride, SS_E_UNSUPPORTED,
                w + ": units are not evenly placed in the pack (desc-v1 needs one stride)");
      }
      Json& pj = parts.arr[pi];
      pj["source"] = Json::number(uint64_t(first + a.file));
      pj["offset"] = Json::number(a.offset);
      pj["stride"] = Json::number(L.n_units > 1 ? stride : pad_up(a.bytes));
    }
    if (!L.hash.empty() || o_.verify == PackVerify::Off) continue;
    // unit hashes from verified pack bytes, cached by (pack, layer, parts, data-file identities)
    std::vector<std::string> names;
    Json selj = Json::object();
    selj["layer"] = Json::number(int64_t(L.index));
    selj["units"] = Json::number(uint64_t(L.n_units));
    Json pn = Json::array();
    for (const Part& part : L.parts) {
      names.push_back(part.name);
      pn.push(Json::string(part.name));
    }
    selj["parts"] = pn;
    const std::string sel = json_write(selj);
    std::string cache;
    {
      std::lock_guard<std::mutex> lk(mu_);
      bool striped = false;
      for (uint8_t s : striped_) striped = striped || s;
      if (!verified_path_.empty() && !changed_ && !striped)
        cache = verified_path_.substr(0, verified_path_.size() - 5) + "-units-" +
                sha256_hex(sel.data(), sel.size()).substr(0, 32) + ".json";
    }
    std::vector<uint64_t> hashes;
    if (!cache.empty()) {
      try {
        const Json c = json_parse(read_small(cache));
        const Json* ids = c.is_object() ? c.find("ids") : nullptr;
        const Json* seen = c.is_object() ? c.find("selection") : nullptr;
        const Json* hj = c.is_object() ? c.find("hash") : nullptr;
        if (ids && seen && hj && hj->is_array() && json_str(*ids, "ids") == ids_text(open_ids_) &&
            json_str(*seen, "selection") == sel)
          for (const Json& h : hj->arr) hashes.push_back(std::stoull(json_str(h, "hash"), nullptr, 16));
      } catch (const std::exception&) {
        hashes.clear();
      }
      if (hashes.size() != size_t(L.n_units)) hashes.clear();
    }
    const bool computed = hashes.empty();
    if (computed) hashes = unit_hashes(L.index, L.n_units, names);
    Json hl = Json::array();
    for (uint64_t h : hashes) {
      char b[17];
      std::snprintf(b, sizeof b, "%016llx", (unsigned long long)h);
      hl.push(Json::string(b));
    }
    lj["hash"] = hl;
    if (computed && !cache.empty()) {
      if (files_unchanged()) {
        Json c = Json::object();
        c["ids"] = Json::string(ids_text(open_ids_));
        c["selection"] = Json::string(sel);
        c["hash"] = hl;
        try {
          write_file_atomic(cache, json_write(c));
        } catch (const std::exception&) {
        }
      }
    }
  }
  Desc check = desc_from_json(out);
  desc_validate(check);
  return desc_to_json(check);
}

Json Pack::info() const {
  Json j = Json::object();
  j["schema"] = Json::string(kPackSchema);
  j["dir"] = Json::string(dir_);
  j["model"] = Json::string(m_.model);
  j["manifest_sha256"] = Json::string(manifest_sha_);
  j["records"] = Json::number(uint64_t(ix_.records.size()));
  j["segments"] = Json::number(uint64_t(ix_.segments.size()));
  j["units"] = Json::number(uint64_t(units_.size()));
  uint64_t data = 0, payload = 0;
  for (const PackFileEntry& f : m_.files) data += f.size;
  std::map<std::string, std::pair<uint64_t, uint64_t>> per;
  int32_t lo = INT32_MAX, hi = INT32_MIN;
  for (const PackRecord& r : ix_.records) {
    payload += r.bytes;
    auto& e = per[format(r)];
    ++e.first;
    e.second += r.bytes;
    lo = std::min(lo, r.layer);
    hi = std::max(hi, r.layer);
  }
  j["data_bytes"] = Json::number(data);
  j["payload_bytes"] = Json::number(payload);
  j["padding_bytes"] = Json::number(data - payload);
  j["layers"] = Json::string(std::to_string(lo) + ".." + std::to_string(hi));
  Json fm = Json::object();
  for (const auto& kv : per) {
    Json e = Json::object();
    e["records"] = Json::number(kv.second.first);
    e["bytes"] = Json::number(kv.second.second);
    const auto it = m_.formats.find(kv.first);
    if (it != m_.formats.end()) {
      e["transform"] = Json::string(it->second.transform);
      e["version"] = Json::string(it->second.version);
      e["tool"] = Json::string(it->second.tool);
    }
    fm[kv.first] = e;
  }
  j["formats"] = fm;
  Json files = Json::array();
  for (uint32_t f = 0; f < files_.size(); ++f) {
    Json e = Json::object();
    e["name"] = Json::string(m_.files[f].name);
    e["size"] = Json::number(m_.files[f].size);
    e["sha256"] = Json::string(m_.files[f].sha256);
    e["direct"] = Json::boolean(files_[f]->direct());
    e["striped"] = Json::boolean(bool(files_[f]->session()));
    files.push(e);
  }
  j["files"] = files;
  Json src = Json::array();
  for (const PackSourceEntry& s : m_.sources) {
    Json e = Json::object();
    e["name"] = Json::string(s.name);
    e["size"] = Json::number(s.size);
    e["sha256"] = Json::string(s.sha256);
    src.push(e);
  }
  j["sources"] = src;
  Json lay = Json::array();
  for (const std::string& l : m_.layouts) lay.push(Json::string(l));
  j["layouts"] = lay;
  j["verify"] = Json::string(o_.verify == PackVerify::Always ? "always" : o_.verify == PackVerify::Off ? "off" : "cached");
  j["meta"] = m_.meta.is_null() ? Json::object() : m_.meta;
  j["receipt"] = m_.receipt.is_null() ? Json::object() : m_.receipt;
  const PackStats s = stats();
  Json st = Json::object();
  st["reads"] = Json::number(s.reads);
  st["bytes"] = Json::number(s.bytes);
  st["bounce_bytes"] = Json::number(s.bounce_bytes);
  st["verified_records"] = Json::number(s.verified_records);
  st["verified_segments"] = Json::number(s.verified_segments);
  st["verified_bytes"] = Json::number(s.verified_bytes);
  st["verify_ns"] = Json::number(s.verify_ns);
  j["stats"] = st;
  return j;
}

PackStats Pack::stats() const {
  std::lock_guard<std::mutex> lk(mu_);
  return stats_;
}

}  // namespace ts
