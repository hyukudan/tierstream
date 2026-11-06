#include "placement_io.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <set>

#include "errors.hpp"
#include "file_reader.hpp"
#include "sha256.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/stat.h>
#endif

namespace ts {
namespace {
void schema(bool ok, const std::string& why) { require(ok, SS_E_RECIPE, "PL_E_SCHEMA " + why); }
void stale(bool ok, const std::string& why) {
  require(ok, SS_E_RECIPE, "PL_E_STALE " + why + "; refresh with ts-recipe manifest --full");
}

struct Header {
  const std::function<void(void*, size_t)>& read;
  uint64_t remaining;
  Sha256 hash;
  void bytes(uint64_t n) {
    schema(n <= remaining, "content header exceeds file size");
    unsigned char buf[4096];
    while (n) {
      const size_t count = size_t(std::min<uint64_t>(n, sizeof buf));
      read(buf, count);
      hash.update(buf, count);
      remaining -= count;
      n -= count;
    }
  }
  uint64_t integer(size_t n) {
    schema(n <= remaining, "truncated content header");
    unsigned char buf[8] = {};
    read(buf, n);
    remaining -= n;
    hash.update(buf, n);
    uint64_t v = 0;
    for (size_t i = 0; i < n; ++i) v |= uint64_t(buf[i]) << (8 * i);
    return v;
  }
  void string() { bytes(integer(8)); }
  void value(uint64_t type, unsigned depth = 0) {
    schema(depth <= 16, "GGUF metadata nesting exceeds limit");
    static const size_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    schema(type < sizeof sizes / sizeof sizes[0], "unknown GGUF metadata type");
    if (type == 8) {
      string();
    } else if (type == 9) {
      const uint64_t subtype = integer(4), count = integer(8);
      schema(subtype < 13 && subtype != 9, "invalid GGUF array type");
      if (subtype != 8) {
        schema(count <= remaining / sizes[subtype], "GGUF array exceeds header");
        bytes(count * sizes[subtype]);
      } else {
        schema(count <= remaining / 8, "GGUF string array exceeds header");
        for (uint64_t i = 0; i < count; ++i) value(subtype, depth + 1);
      }
    } else {
      bytes(sizes[type]);
    }
  }
};

std::string path_of(const std::string& path) {
  return std::filesystem::absolute(std::filesystem::u8path(path)).lexically_normal().generic_u8string();
}

Json file_stat(const std::string& path) {
  Json out = Json::object();
  out["path"] = Json::string(path);
#if defined(_WIN32)
  WIN32_FILE_ATTRIBUTE_DATA info;
  const auto wpath = std::filesystem::u8path(path).wstring();
  stale(GetFileAttributesExW(wpath.c_str(), GetFileExInfoStandard, &info) != 0 &&
            !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY), path + ": cannot stat file");
  const uint64_t ticks = (uint64_t(info.ftLastWriteTime.dwHighDateTime) << 32) | info.ftLastWriteTime.dwLowDateTime;
  constexpr uint64_t epoch = 116444736000000000ULL;
  schema(ticks >= epoch && ticks - epoch <= uint64_t(INT64_MAX) / 100, path + ": mtime out of range");
  out["mtime_ns"] = Json::number((ticks - epoch) * 100);
  out["size"] = Json::number((uint64_t(info.nFileSizeHigh) << 32) | info.nFileSizeLow);
#else
  struct stat info;
  stale(stat(path.c_str(), &info) == 0 && S_ISREG(info.st_mode), path + ": cannot stat file");
#if defined(__APPLE__)
  const auto mt = info.st_mtimespec;
#else
  const auto mt = info.st_mtim;
#endif
  schema(mt.tv_sec >= 0 && uint64_t(mt.tv_sec) <= uint64_t(INT64_MAX) / 1000000000, path + ": mtime out of range");
  out["mtime_ns"] = Json::number(uint64_t(mt.tv_sec) * 1000000000 + uint64_t(mt.tv_nsec));
  out["size"] = Json::number(uint64_t(info.st_size));
#endif
  return out;
}

Json scan(const std::string& path, bool full, uint64_t extent) {
  Json out = file_stat(path);
  // Positional OS reads issue exactly the requested range. CRT fread can read ahead even with _IONBF.
  File file(path, false);
  uint64_t offset = 0;
  auto read = [&](void* buf, size_t n) {
    stale(file.pread(buf, n, offset) == n, path + ": truncated content header");
    offset += n;
  };
  out["header_sha256"] = Json::string(content_header_sha256(read, json_u64(*out.find("size"), "size"), extent));
  if (extent) out["header_bytes"] = Json::number(extent);
  if (full) {
    Sha256 hash;
    unsigned char buf[65536];
    uint64_t at = 0;
    const uint64_t size = json_u64(*out.find("size"), "size");
    try {
      while (at < size) {
        const size_t n = size_t(std::min<uint64_t>(size - at, sizeof buf));
        stale(file.pread(buf, n, at) == n, path + ": truncated full read");
        hash.update(buf, n);
        at += n;
      }
    } catch (const Error& e) {
      fail(SS_E_RECIPE, "PL_E_STALE " + path + ": full read failed: " + e.what());
    }
    out["sha256"] = Json::string(hash.hex());
  }
  const Json after = file_stat(path);
  for (const char* key : {"size", "mtime_ns"})
    stale(json_write(*out.find(key)) == json_write(*after.find(key)), path + "." + key + ": changed during scan");
  return out;
}

uint64_t decimal_milli(const Json& value, const std::string& field) {
  const double d = json_f64(value, field);
  schema(d >= 0, field + ": expected nonnegative number");
  char buf[64];
  const auto text = std::to_chars(buf, buf + sizeof buf, d, std::chars_format::general);
  schema(text.ec == std::errc(), field + ": cannot format number");
  std::string s(buf, text.ptr), digits;
  const size_t e = s.find_first_of("eE");
  int exponent = e == std::string::npos ? 0 : std::stoi(s.substr(e + 1));
  const std::string mantissa = s.substr(0, e);
  const size_t dot = mantissa.find('.');
  if (dot != std::string::npos) exponent -= int(mantissa.size() - dot - 1);
  for (char c : mantissa) if (c >= '0' && c <= '9') digits += c;
  exponent += 3;
  const int keep = int(digits.size()) + exponent;
  uint64_t result = 0;
  schema(keep <= 19, field + ": converted integer overflows");
  for (int i = 0; i < keep; ++i) {
    const unsigned digit = i < int(digits.size()) ? unsigned(digits[size_t(i)] - '0') : 0;
    schema(result <= (uint64_t(INT64_MAX) - digit) / 10, field + ": converted integer overflows");
    result = result * 10 + digit;
  }
  if (keep >= 0 && keep < int(digits.size()) && digits[size_t(keep)] >= '5') {
    schema(result < uint64_t(INT64_MAX), field + ": rounded integer overflows");
    ++result;
  }
  return result;
}

Json convert_values(const Json& in, const std::string& field) {
  if (in.is_null()) return Json();
  if (in.is_object()) {
    Json out = Json::object();
    for (const auto& kv : in.obj) {
      Json v = convert_values(kv.second, field + "." + kv.first);
      if (!v.is_null()) out[kv.first] = std::move(v);
    }
    return out;
  }
  schema(in.is_number(), field + ": expected milliseconds or a keyed table");
  if (json_f64(in, field) < 0) return Json();
  return Json::number(decimal_milli(in, field));
}
}  // namespace

std::string content_header_sha256(const std::function<void(void*, size_t)>& read, uint64_t size,
                                  uint64_t header_bytes) {
  Header h{read, size, {}};
  if (header_bytes) {
    h.bytes(header_bytes);
    return h.hash.hex();
  }
  const uint64_t magic = h.integer(4);
  if (magic == 0x46554747) {  // GGUF v2/v3: metadata, then every tensor info; no tensor-data alignment padding.
    const uint64_t version = h.integer(4);
    schema(version == 2 || version == 3, "unsupported GGUF version");
    const uint64_t tensors = h.integer(8), metadata = h.integer(8);
    schema(metadata <= h.remaining / 12 && tensors <= h.remaining / 24, "GGUF counts exceed file size");
    for (uint64_t i = 0; i < metadata; ++i) {
      h.string();
      h.value(h.integer(4));
    }
    for (uint64_t i = 0; i < tensors; ++i) {
      h.string();
      const uint64_t dims = h.integer(4);
      schema(dims >= 1 && dims <= 4, "invalid GGUF tensor dimension count");
      h.bytes(dims * 8 + 4 + 8);
    }
  } else {  // safetensors: 64-bit little-endian JSON-header length, then exactly that header.
    const uint64_t length = magic | (h.integer(4) << 32);
    schema(length >= 2 && length <= h.remaining && length <= 100000000, "unknown format; supply --header-bytes");
    schema(h.integer(1) == '{', "unknown format; supply --header-bytes");
    h.bytes(length - 1);
  }
  return h.hash.hex();
}

Json content_manifest(std::vector<std::string> paths, bool full, uint64_t header_bytes) {
  for (auto& p : paths) p = path_of(p);
  std::sort(paths.begin(), paths.end());
  schema(!paths.empty() && std::adjacent_find(paths.begin(), paths.end()) == paths.end(), "manifest needs unique files");
  Json out = Json::object();
  out["schema"] = Json::string("tierstream/content-manifest-v1");
  out["files"] = Json::array();
  for (const auto& path : paths) out["files"].push(scan(path, full, header_bytes));
  return out;
}

Json content_manifest_verify(const Json& manifest, const std::vector<std::string>& paths, bool full,
                             uint64_t header_bytes) {
  const Json* schema_value = manifest.find("schema"), *files = manifest.find("files");
  schema(schema_value && schema_value->is_string() && schema_value->text == "tierstream/content-manifest-v1" &&
             files && files->is_array() && !files->arr.empty(), "expected content-manifest-v1 with files");
  std::set<std::string> selected, cached;
  for (const auto& p : paths) schema(selected.insert(path_of(p)).second, "duplicate selected file");
  for (const auto& file : files->arr) {
    const Json* p = file.find("path");
    schema(p && p->is_string(), "manifest file.path missing");
    const std::string path = path_of(p->text);
    schema(cached.insert(path).second, "duplicate manifest file: " + path);
    const Json stat = file_stat(path);
    // Refuse stale metadata before opening any content.
    for (const char* key : {"size", "mtime_ns"}) {
      const Json* v = file.find(key);
      stale(v && json_u64(*v, key) == json_u64(*stat.find(key), key), path + "." + key);
    }
    uint64_t extent = header_bytes;
    if (const Json* v = file.find("header_bytes")) {
      const uint64_t saved = json_u64(*v, "header_bytes");
      schema(!extent || extent == saved, "header_bytes differs from manifest");
      extent = saved;
    }
    Json actual;
    try { actual = scan(path, full, extent); }
    catch (const Error& e) {
      fail(SS_E_RECIPE, "PL_E_STALE " + path + ".header_sha256: " + e.what());
    }
    for (const char* key : {"header_sha256", "sha256"}) {
      if (std::string(key) == "sha256" && !full) continue;
      const Json* v = file.find(key);
      stale(v && v->is_string() && v->text == actual.find(key)->text, path + "." + key);
    }
  }
  stale(selected.empty() || selected == cached, "files: selected files differ from manifest");
  Json out = manifest;
  std::sort(out["files"].arr.begin(), out["files"].arr.end(), [](const Json& a, const Json& b) {
    return a.find("path")->text < b.find("path")->text;
  });
  return out;
}

Json convert_rtx_costs(const Json& report) {
  const Json* s = report.find("schema");
  schema(s && s->is_string() && (s->text == "wdsrtx/rtx-unit-costs-v1" || s->text == "rtx-unit-costs-v1"),
         "expected rtx-unit-costs-v1");
  Json out = Json::object();
  out["schema"] = Json::string("tierstream/solver-costs-v1");
  for (const char* name : {"allhit", "verify", "draft", "miss_pinned", "miss_file", "boundary", "engram"}) {
    const std::string from = std::string(name) + "_ms", to = std::string(name) + "_us";
    if (const Json* v = report.find(from)) {
      Json converted = convert_values(*v, from);
      if (!converted.is_null()) out[to] = std::move(converted);
    }
  }
  if (const Json* v = report.find("accepted")) {
    Json converted = convert_values(*v, "accepted");
    if (!converted.is_null()) out["accepted_milli"] = std::move(converted);
  }
  for (const char* key : {"accepted_milli", "file_staging_bytes", "applicability", "xfer_measured"})
    if (const Json* v = report.find(key)) out[key] = *v;
  if (const Json* provenance = report.find("provenance")) {
    schema(provenance->is_object(), "provenance must be a table");
    out["provenance"] = Json::object();
    for (const auto& kv : provenance->obj) {
      std::string key = kv.first;
      if (key.size() >= 3 && key.substr(key.size() - 3) == "_ms") key.replace(key.size() - 3, 3, "_us");
      if (key == "accepted") key = "accepted_milli";
      out["provenance"][key] = kv.second;
    }
  }
  return out;
}
}  // namespace ts
