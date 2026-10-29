#include "stripe_manifest.hpp"

#include <algorithm>
#include <cstdio>

#include "errors.hpp"
#include "json.hpp"
#include "xxh64.hpp"

namespace ts {
namespace {

void check(bool ok, const std::string& why) { require(ok, SS_E_DESC, "stripe: " + why); }

std::string hex16(uint64_t v) {
  char b[17];
  std::snprintf(b, sizeof b, "%016llx", static_cast<unsigned long long>(v));
  return b;
}

uint64_t hex(const Json& j, size_t n, const char* where) {
  const std::string s = json_str(j, where);
  check(s.size() == n && s.find_first_not_of("0123456789abcdef") == std::string::npos,
        std::string(where) + ": expected lowercase hex");
  uint64_t v = 0;
  for (char c : s) v = (v << 4) | uint64_t(c <= '9' ? c - '0' : c - 'a' + 10);
  return v;
}

const Json& field(const Json& j, const char* name) {
  const Json* p = j.find(name);
  check(p != nullptr, std::string("missing ") + name);
  return *p;
}

uint64_t number(const Json& j, const char* where) {
  uint64_t v = json_u64(j, where);
  check(v < (uint64_t(1) << 53), std::string(where) + ": integer must be below 2^53");
  return v;
}

bool basename(const std::string& s) {
  return !s.empty() && s != "." && s != ".." &&
         s.find_first_of("/\\:") == std::string::npos &&
         std::none_of(s.begin(), s.end(), [](unsigned char c) { return c < 32; });
}

std::string name_key(std::string s) {
#if defined(_WIN32)
  for (char& c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
#endif
  return s;
}

uint64_t header(const Json& j) {
  check(j.is_object() && j.obj.size() == 7, "expected schema, chunk_bytes, primary, data, runs, hashes, digest");
  check(json_str(field(j, "schema"), "schema") == "tierstream/stripe-v1", "unsupported schema");
  uint64_t c = number(field(j, "chunk_bytes"), "chunk_bytes");
  check(c >= (uint64_t(1) << 20) && c <= (uint64_t(256) << 20) && !(c & (c - 1)), "invalid chunk_bytes");
  const Json& p = field(j, "primary");
  const Json& d = field(j, "data");
  check(p.is_object() && p.obj.size() == 3 && d.is_object() && d.obj.size() == 2, "invalid primary or data object");
  std::string base = json_str(field(p, "basename"), "primary.basename");
  std::string data = json_str(field(d, "name"), "data.name");
  stripe_data_name_validate(base, data);
  hex(field(p, "sha256"), 64, "primary.sha256");
  uint64_t size = number(field(p, "size"), "primary.size");
  check(size <= kStripeMaxPrimary, "primary size exceeds limit");
  uint64_t chunks = size / c + uint64_t(size % c != 0);
  check(chunks <= kStripeMaxChunks, "chunk count exceeds limit");
  number(field(d, "size"), "data.size");
  hex(field(j, "digest"), 16, "digest");
  return chunks;
}

// Inspect array cardinalities without constructing Json nodes. Parse only the small header first, so all format
// limits are checked before the general JSON parser allocates the runs and hashes arrays.
struct Scan {
  const std::string& s;
  size_t i = 0;
  void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i; }
  void string() {
    check(i < s.size() && s[i++] == '"', "expected string");
    while (i < s.size()) {
      char c = s[i++];
      if (c == '"') return;
      if (c == '\\') { check(i < s.size(), "unterminated escape"); ++i; }
    }
    check(false, "unterminated string");
  }
  uint64_t value(unsigned depth = 0) {
    check(depth <= 64, "nesting too deep");
    ws();
    check(i < s.size(), "unexpected end");
    if (s[i] == '"') { string(); return 0; }
    if (s[i] == '[' || s[i] == '{') {
      char end = s[i++] == '[' ? ']' : '}';
      uint64_t count = 0;
      ws();
      while (i < s.size() && s[i] != end) {
        if (end == '}') { string(); ws(); check(i < s.size() && s[i++] == ':', "expected colon"); }
        value(depth + 1);
        ++count;
        ws();
        if (i < s.size() && s[i] == ',') { ++i; ws(); }
        else break;
      }
      check(i < s.size() && s[i++] == end, "unterminated array or object");
      return count;
    }
    size_t start = i;
    while (i < s.size() && s[i] != ',' && s[i] != ']' && s[i] != '}' && s[i] != ' ' &&
           s[i] != '\n' && s[i] != '\r' && s[i] != '\t') ++i;
    check(i != start, "expected value");
    return 0;
  }
};

void preflight(const std::string& text) {
  check(text.size() <= (size_t(256) << 20), "manifest exceeds text limit");
  Scan s{text};
  s.ws();
  check(s.i < text.size() && text[s.i++] == '{', "expected object");
  std::string small = "{";
  uint64_t runs = 0, hashes = 0;
  bool first = true;
  s.ws();
  while (s.i < text.size() && text[s.i] != '}') {
    size_t key_start = s.i;
    s.string();
    check(s.i - key_start <= 128, "oversized key");
    std::string key = json_str(json_parse(text.substr(key_start, s.i - key_start)), "key");
    s.ws();
    check(s.i < text.size() && text[s.i++] == ':', "expected colon");
    s.ws();
    size_t start = s.i;
    uint64_t count = 0;
    if (key == "runs" || key == "hashes") {
      check(s.i < text.size() && text[s.i++] == '[', key + ": expected array");
      s.ws();
      while (s.i < text.size() && text[s.i] != ']') {
        size_t begin = s.i;
        s.value();
        // Each run has exactly three scalar integers; each hash is one short hex string. Reject malformed shapes
        // here as well, before they can expand into a large tree in the general parser.
        check(s.i - begin <= 128, "oversized run or hash");
        Json entry = json_parse(text.substr(begin, s.i - begin));
        if (key == "runs") {
          check(entry.is_array() && entry.arr.size() == 3, "run must have three integers");
          for (const auto& v : entry.arr) number(v, "run integer");
        } else hex(entry, 16, "hash");
        check(++count <= (key == "runs" ? kStripeMaxRuns : kStripeMaxChunks), key + " count exceeds limit");
        s.ws();
        if (s.i < text.size() && text[s.i] == ',') { ++s.i; s.ws(); }
        else break;
      }
      check(s.i < text.size() && text[s.i++] == ']', "unterminated array");
    } else s.value();
    if (!first) small += ',';
    first = false;
    small += text.substr(key_start, start - key_start);
    if (key == "runs" || key == "hashes") {
      check(text[start] == '[', key + ": expected array");
      small += "[]";
      if (key == "runs") { runs = count; check(runs <= kStripeMaxRuns, "run count exceeds limit"); }
      else { hashes = count; check(hashes <= kStripeMaxChunks, "hash count exceeds limit"); }
    } else {
      check(s.i - start <= 4096, "header value too large");
      small += text.substr(start, s.i - start);
    }
    s.ws();
    if (s.i < text.size() && text[s.i] == ',') { ++s.i; s.ws(); }
    else break;
  }
  check(s.i < text.size() && text[s.i++] == '}', "expected object end");
  s.ws();
  check(s.i == text.size(), "trailing characters");
  small += '}';
  check(hashes <= header(json_parse(small)), "hash count exceeds primary chunks");
}

}  // namespace

void stripe_data_name_validate(const std::string& primary_basename, const std::string& data_name) {
  check(basename(primary_basename) && basename(data_name) && data_name.find("..") == std::string::npos &&
        name_key(primary_basename) != name_key(data_name), "data and primary must be distinct bare names");
}

StripeManifest stripe_manifest_parse(const std::string& text) {
  preflight(text);
  Json j = json_parse(text);
  uint64_t chunks = header(j);
  StripeManifest m;
  m.chunk_bytes = number(field(j, "chunk_bytes"), "chunk_bytes");
  const Json& p = field(j, "primary");
  m.primary_basename = json_str(field(p, "basename"), "primary.basename");
  m.primary_size = number(field(p, "size"), "primary.size");
  m.primary_sha256 = json_str(field(p, "sha256"), "primary.sha256");
  m.data_name = json_str(field(field(j, "data"), "name"), "data.name");
  m.data_size = number(field(field(j, "data"), "size"), "data.size");
  const Json& runs = field(j, "runs");
  const Json& hashes = field(j, "hashes");
  check(runs.is_array() && runs.arr.size() <= kStripeMaxRuns && hashes.is_array() && hashes.arr.size() <= chunks,
        "invalid arrays");
  uint64_t end = 0, slots = 0;
  m.runs.reserve(runs.arr.size());
  for (const Json& r : runs.arr) {
    check(r.is_array() && r.arr.size() == 3, "run must have three integers");
    StripeRun run{number(r.arr[0], "first_chunk"), number(r.arr[1], "count"), number(r.arr[2], "first_slot")};
    check(run.count && run.first_chunk >= end && run.first_chunk < chunks && run.count <= chunks - run.first_chunk,
          "runs must be sorted, non-overlapping and within the primary");
    check(run.first_slot == slots, "slots must be sorted and dense");
    check(run.count <= chunks - slots, "slot count exceeds limit");
    slots += run.count;
    end = run.first_chunk + run.count;
    m.runs.push_back(run);
  }
  check(hashes.arr.size() == slots, "hash count must equal slot count");
  check(m.data_size == slots * m.chunk_bytes, "data size must equal slots * chunk_bytes");
  m.hashes.reserve(size_t(slots));
  for (const Json& h : hashes.arr) m.hashes.push_back(hex(h, 16, "hash"));
  uint64_t digest = hex(field(j, "digest"), 16, "digest");
  j.obj.erase("digest");
  std::string canonical = json_write(j);
  check(digest == xxh64(canonical.data(), canonical.size()), "digest mismatch");
  return m;
}

std::string stripe_manifest_write(const StripeManifest& m) {
  check(m.runs.size() <= kStripeMaxRuns && m.hashes.size() <= kStripeMaxChunks, "array limits exceeded");
  check(m.primary_size <= kStripeMaxPrimary, "primary size exceeds limit");
  Json j = Json::object();
  j["schema"] = Json::string("tierstream/stripe-v1");
  j["chunk_bytes"] = Json::number(m.chunk_bytes);
  j["primary"]["basename"] = Json::string(m.primary_basename);
  j["primary"]["size"] = Json::number(m.primary_size);
  j["primary"]["sha256"] = Json::string(m.primary_sha256);
  j["data"]["name"] = Json::string(m.data_name);
  j["data"]["size"] = Json::number(m.data_size);
  j["runs"] = Json::array();
  j["hashes"] = Json::array();
  for (const StripeRun& r : m.runs) {
    Json a = Json::array();
    a.push(Json::number(r.first_chunk));
    a.push(Json::number(r.count));
    a.push(Json::number(r.first_slot));
    j["runs"].push(std::move(a));
  }
  for (uint64_t h : m.hashes) j["hashes"].push(Json::string(hex16(h)));
  std::string text = json_write(j);
  j["digest"] = Json::string(hex16(xxh64(text.data(), text.size())));
  text = json_write(j);
  stripe_manifest_parse(text);
  return text;
}

std::string stripe_manifest_name(const std::string& primary_basename) { return primary_basename + ".stripe.json"; }

}  // namespace ts
