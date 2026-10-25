#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

#include "errors.hpp"

namespace ts {

Json Json::number(uint64_t v) {
  Json j;
  j.kind = Kind::Number;
  j.text = std::to_string(v);
  return j;
}

Json Json::number(int64_t v) {
  Json j;
  j.kind = Kind::Number;
  j.text = std::to_string(v);
  return j;
}

Json Json::number(double v) {
  require(std::isfinite(v), SS_E_INVALID, "json: non-finite number");
  char buf[40];
  std::snprintf(buf, sizeof buf, "%.17g", v);
  Json j;
  j.kind = Kind::Number;
  j.text = buf;
  return j;
}

Json Json::string(std::string v) {
  Json j;
  j.kind = Kind::String;
  j.text = std::move(v);
  return j;
}

Json Json::boolean(bool v) {
  Json j;
  j.kind = Kind::Bool;
  j.b = v;
  return j;
}

Json Json::array() {
  Json j;
  j.kind = Kind::Array;
  return j;
}

Json Json::object() {
  Json j;
  j.kind = Kind::Object;
  return j;
}

const Json* Json::find(const std::string& key) const {
  if (kind != Kind::Object) return nullptr;
  auto it = obj.find(key);
  return it == obj.end() ? nullptr : &it->second;
}

namespace {

constexpr size_t kMaxDepth = 64;
constexpr size_t kMaxText = size_t(256) << 20;

struct Parser {
  const std::string& s;
  size_t i = 0;

  [[noreturn]] void bad(const char* what) const {
    fail(SS_E_DESC, std::string("json: ") + what + " at offset " + std::to_string(i));
  }
  void ws() {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
  }
  bool lit(const char* w) {
    size_t n = 0;
    while (w[n]) ++n;
    if (s.compare(i, n, w) != 0) return false;
    i += n;
    return true;
  }
  static void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += char(cp);
    } else if (cp < 0x800) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }
  uint32_t hex4() {
    if (i + 4 > s.size()) bad("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else bad("bad \\u escape");
    }
    return v;
  }
  std::string str() {
    if (s[i] != '"') bad("expected a string");
    ++i;
    std::string out;
    while (true) {
      if (i >= s.size()) bad("unterminated string");
      char c = s[i++];
      if (c == '"') break;
      if (static_cast<unsigned char>(c) < 0x20) bad("control character in string");
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) bad("unterminated escape");
      char e = s[i++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp < 0xDC00) {
            if (!(i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u')) bad("lone surrogate");
            i += 2;
            uint32_t lo = hex4();
            if (lo < 0xDC00 || lo >= 0xE000) bad("bad surrogate pair");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp < 0xE000) {
            bad("lone surrogate");
          }
          utf8(out, cp);
          break;
        }
        default: bad("bad escape");
      }
    }
    return out;
  }
  Json num() {
    size_t b = i;
    if (s[i] == '-') ++i;
    if (i >= s.size() || !(s[i] >= '0' && s[i] <= '9')) bad("bad number");
    if (s[i] == '0') ++i;
    else while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    if (i < s.size() && s[i] == '.') {
      ++i;
      if (i >= s.size() || !(s[i] >= '0' && s[i] <= '9')) bad("bad fraction");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
      ++i;
      if (i < s.size() && (s[i] == '+' || s[i] == '-')) ++i;
      if (i >= s.size() || !(s[i] >= '0' && s[i] <= '9')) bad("bad exponent");
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') ++i;
    }
    Json j;
    j.kind = Json::Kind::Number;
    j.text = s.substr(b, i - b);
    return j;
  }
  Json value(size_t depth) {
    if (depth > kMaxDepth) bad("nesting too deep");
    ws();
    if (i >= s.size()) bad("unexpected end");
    char c = s[i];
    if (c == '{') {
      ++i;
      Json j = Json::object();
      ws();
      if (i < s.size() && s[i] == '}') {
        ++i;
        return j;
      }
      while (true) {
        ws();
        if (i >= s.size()) bad("unexpected end");
        std::string k = str();
        ws();
        if (i >= s.size() || s[i] != ':') bad("expected ':'");
        ++i;
        if (j.obj.count(k)) bad("duplicate key");
        j.obj[k] = value(depth + 1);
        ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == '}') {
          ++i;
          return j;
        }
        bad("expected ',' or '}'");
      }
    }
    if (c == '[') {
      ++i;
      Json j = Json::array();
      ws();
      if (i < s.size() && s[i] == ']') {
        ++i;
        return j;
      }
      while (true) {
        j.arr.push_back(value(depth + 1));
        ws();
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (i < s.size() && s[i] == ']') {
          ++i;
          return j;
        }
        bad("expected ',' or ']'");
      }
    }
    if (c == '"') return Json::string(str());
    if (lit("true")) {
      Json j;
      j.kind = Json::Kind::Bool;
      j.b = true;
      return j;
    }
    if (lit("false")) {
      Json j;
      j.kind = Json::Kind::Bool;
      return j;
    }
    if (lit("null")) return Json();
    return num();
  }
};

void write_str(std::string& out, const std::string& v) {
  out += '"';
  for (unsigned char c : v) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", c);
          out += buf;
        } else {
          out += char(c);
        }
    }
  }
  out += '"';
}

void write(std::string& out, const Json& v) {
  switch (v.kind) {
    case Json::Kind::Null: out += "null"; break;
    case Json::Kind::Bool: out += v.b ? "true" : "false"; break;
    case Json::Kind::Number: out += v.text; break;
    case Json::Kind::String: write_str(out, v.text); break;
    case Json::Kind::Array: {
      out += '[';
      for (size_t k = 0; k < v.arr.size(); ++k) {
        if (k) out += ',';
        write(out, v.arr[k]);
      }
      out += ']';
      break;
    }
    case Json::Kind::Object: {
      out += '{';
      bool first = true;
      for (const auto& kv : v.obj) {
        if (!first) out += ',';
        first = false;
        write_str(out, kv.first);
        out += ':';
        write(out, kv.second);
      }
      out += '}';
      break;
    }
  }
}

}  // namespace

Json json_parse(const std::string& text) {
  require(text.size() <= kMaxText, SS_E_DESC, "json: text larger than 256 MiB");
  Parser p{text};
  Json v = p.value(0);
  p.ws();
  if (p.i != text.size()) p.bad("trailing characters");
  return v;
}

std::string json_write(const Json& v) {
  std::string out;
  write(out, v);
  return out;
}

uint64_t json_u64(const Json& v, const std::string& where) {
  require(v.is_number(), SS_E_DESC, where + ": expected an unsigned integer");
  const std::string& t = v.text;
  require(!t.empty() && t.find_first_not_of("0123456789") == std::string::npos, SS_E_DESC,
          where + ": expected an unsigned integer, got " + t);
  require(t.size() <= 20, SS_E_DESC, where + ": integer out of range");
  uint64_t r = 0;
  for (char c : t) {
    uint64_t d = uint64_t(c - '0');
    require(r <= (std::numeric_limits<uint64_t>::max() - d) / 10, SS_E_DESC, where + ": integer out of range");
    r = r * 10 + d;
  }
  return r;
}

int64_t json_i64(const Json& v, const std::string& where) {
  require(v.is_number(), SS_E_DESC, where + ": expected an integer");
  bool neg = !v.text.empty() && v.text[0] == '-';
  Json m = v;
  if (neg) m.text = v.text.substr(1);
  uint64_t u = json_u64(m, where);
  require(u <= uint64_t(std::numeric_limits<int64_t>::max()), SS_E_DESC, where + ": integer out of range");
  return neg ? -int64_t(u) : int64_t(u);
}

double json_f64(const Json& v, const std::string& where) {
  require(v.is_number(), SS_E_DESC, where + ": expected a number");
  char* end = nullptr;
  double d = std::strtod(v.text.c_str(), &end);
  require(end && *end == 0 && std::isfinite(d), SS_E_DESC, where + ": bad number " + v.text);
  return d;
}

const std::string& json_str(const Json& v, const std::string& where) {
  require(v.is_string(), SS_E_DESC, where + ": expected a string");
  return v.text;
}

bool json_bool(const Json& v, const std::string& where) {
  require(v.kind == Json::Kind::Bool, SS_E_DESC, where + ": expected true/false");
  return v.b;
}

}  // namespace ts
