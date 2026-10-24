#include "toml.hpp"

#include <cstdint>
#include <set>
#include <vector>

#include "errors.hpp"

namespace ss {
namespace {

constexpr size_t kMaxDepth = 64;
constexpr size_t kMaxText = size_t(16) << 20;

bool bare_char(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
}

bool digit(char c) { return c >= '0' && c <= '9'; }

// Table bookkeeping is by path ("a\x1fb\x1e3\x1fc": keys separated by \x1f, array-of-tables elements by \x1e + index)
// so it survives the vectors that hold array-of-tables elements growing.
std::string child(const std::string& path, const std::string& key) { return path + '\x1f' + key; }

class Reader {
 public:
  explicit Reader(const std::string& s) : s_(s) {}

  Json run() {
    require(s_.size() <= kMaxText, SS_E_RECIPE, "toml: document larger than 16 MiB");
    if (s_.compare(0, 3, "\xEF\xBB\xBF") == 0) i_ = 3;
    while (true) {
      skip_blank_lines();
      if (i_ >= s_.size()) break;
      if (s_[i_] == '[') {
        bool aot = i_ + 1 < s_.size() && s_[i_ + 1] == '[';
        i_ += aot ? 2 : 1;
        std::vector<std::string> keys = key();
        if (aot) {
          expect("]]", "expected ']]' after an array-of-tables header");
          open_array_table(keys);
        } else {
          expect("]", "expected ']' after a table header");
          open_table(keys);
        }
      } else {
        std::vector<std::string> keys = key();
        spaces();
        expect("=", "expected '=' after a key");
        spaces();
        Json v = value(0);
        assign(keys, std::move(v));
      }
      end_of_line();
    }
    return std::move(root_);
  }

 private:
  struct Node {
    Json* j;
    std::string path;
  };

  const std::string& s_;
  size_t i_ = 0;
  Json root_ = Json::object();
  std::vector<std::string> cur_;
  std::set<std::string> defined_, dotted_, frozen_, aot_;

  [[noreturn]] void bad(const std::string& what) const {
    size_t line = 1;
    for (size_t k = 0; k < i_ && k < s_.size(); ++k) line += s_[k] == '\n';
    fail(SS_E_RECIPE, "toml: line " + std::to_string(line) + ": " + what);
  }

  bool at(const char* w) const { return s_.compare(i_, std::char_traits<char>::length(w), w) == 0; }

  void expect(const char* w, const char* what) {
    if (!at(w)) bad(what);
    i_ += std::char_traits<char>::length(w);
  }

  void spaces() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t')) ++i_;
  }

  void comment() {
    if (i_ < s_.size() && s_[i_] == '#')
      while (i_ < s_.size() && s_[i_] != '\n') ++i_;
  }

  bool newline() {
    if (at("\r\n")) {
      i_ += 2;
      return true;
    }
    if (i_ < s_.size() && s_[i_] == '\n') {
      ++i_;
      return true;
    }
    return false;
  }

  void skip_blank_lines() {
    while (true) {
      spaces();
      comment();
      if (!newline()) return;
    }
  }

  // Whitespace, comments and newlines inside an array.
  void skip_array_space() {
    while (true) {
      spaces();
      comment();
      if (!newline()) return;
    }
  }

  void end_of_line() {
    spaces();
    comment();
    if (i_ < s_.size() && !newline()) bad("expected the end of the line");
  }

  // ---- keys ---------------------------------------------------------------------------------------------------
  std::string simple_key() {
    if (i_ >= s_.size()) bad("expected a key");
    if (s_[i_] == '"') {
      if (at("\"\"\"")) bad("a key cannot be a multi-line string");
      return basic_string();
    }
    if (s_[i_] == '\'') {
      if (at("'''")) bad("a key cannot be a multi-line string");
      return literal_string();
    }
    size_t b = i_;
    while (i_ < s_.size() && bare_char(s_[i_])) ++i_;
    if (i_ == b) bad("expected a key");
    return s_.substr(b, i_ - b);
  }

  std::vector<std::string> key() {
    std::vector<std::string> keys;
    spaces();
    keys.push_back(simple_key());
    while (true) {
      spaces();
      if (i_ >= s_.size() || s_[i_] != '.') break;
      ++i_;
      spaces();
      keys.push_back(simple_key());
    }
    return keys;
  }

  static std::string dotted(const std::vector<std::string>& keys, size_t n) {
    std::string out;
    for (size_t k = 0; k < n; ++k) out += (k ? "." : "") + keys[k];
    return out;
  }

  // ---- tables -------------------------------------------------------------------------------------------------
  bool is_frozen(const std::string& path) const {
    for (size_t k = 0; k <= path.size(); ++k)
      if ((k == path.size() || path[k] == '\x1f' || path[k] == '\x1e') && frozen_.count(path.substr(0, k))) return true;
    return false;
  }

  // Into an existing or new table: array-of-tables entries resolve to their last element.
  Node enter(Node n, const std::string& k, const std::vector<std::string>& keys, size_t upto, bool mark_dotted) {
    std::string p = child(n.path, k);
    Json* next;
    auto it = n.j->obj.find(k);
    if (it == n.j->obj.end()) {
      next = &((*n.j)[k] = Json::object());
      if (mark_dotted) dotted_.insert(p);
    } else {
      next = &it->second;
      if (is_frozen(p)) bad("'" + dotted(keys, upto) + "' is an inline table or array and cannot be extended");
      if (aot_.count(p)) {
        if (mark_dotted) bad("'" + dotted(keys, upto) + "' is an array of tables");
        p += '\x1e' + std::to_string(next->arr.size() - 1);
        next = &next->arr.back();
      } else if (!next->is_object()) {
        bad("'" + dotted(keys, upto) + "' already has a value");
      } else if (mark_dotted && defined_.count(p)) {
        bad("'" + dotted(keys, upto) + "' is a table defined by a header; extend it there");
      }
    }
    return {next, p};
  }

  Node walk(const std::vector<std::string>& keys, size_t n) {
    Node node{&root_, ""};
    for (size_t k = 0; k < n; ++k) node = enter(node, keys[k], keys, k + 1, false);
    return node;
  }

  void open_table(const std::vector<std::string>& keys) {
    Node parent = walk(keys, keys.size() - 1);
    const std::string& k = keys.back();
    std::string p = child(parent.path, k);
    auto it = parent.j->obj.find(k);
    if (it != parent.j->obj.end()) {
      if (aot_.count(p)) bad("[" + dotted(keys, keys.size()) + "] is an array of tables");
      if (!it->second.is_object() || is_frozen(p)) bad("'" + dotted(keys, keys.size()) + "' already has a value");
      if (defined_.count(p) || dotted_.count(p)) bad("table [" + dotted(keys, keys.size()) + "] defined twice");
    } else {
      (*parent.j)[k] = Json::object();
    }
    defined_.insert(p);
    cur_ = keys;
  }

  void open_array_table(const std::vector<std::string>& keys) {
    Node parent = walk(keys, keys.size() - 1);
    const std::string& k = keys.back();
    std::string p = child(parent.path, k);
    auto it = parent.j->obj.find(k);
    if (it == parent.j->obj.end()) {
      (*parent.j)[k] = Json::array();
      aot_.insert(p);
    } else if (!aot_.count(p)) {
      bad("'" + dotted(keys, keys.size()) + "' is not an array of tables");
    }
    (*parent.j)[k].push(Json::object());
    defined_.insert(p + '\x1e' + std::to_string((*parent.j)[k].arr.size() - 1));
    cur_ = keys;
  }

  void assign(const std::vector<std::string>& keys, Json v) {
    Node t = walk(cur_, cur_.size());
    std::vector<std::string> full = cur_;
    full.insert(full.end(), keys.begin(), keys.end());
    for (size_t k = 0; k + 1 < keys.size(); ++k) t = enter(t, keys[k], full, cur_.size() + k + 1, true);
    const std::string& k = keys.back();
    if (t.j->obj.count(k)) bad("duplicate key '" + dotted(full, full.size()) + "'");
    bool freeze = v.is_array() || v.is_object();
    (*t.j)[k] = std::move(v);
    if (freeze) frozen_.insert(child(t.path, k));
  }

  // ---- values -------------------------------------------------------------------------------------------------
  void put_utf8(std::string& out, uint32_t cp) {
    if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) bad("escape is not a Unicode scalar value");
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

  uint32_t hex(int n) {
    uint32_t v = 0;
    for (int k = 0; k < n; ++k) {
      if (i_ >= s_.size()) bad("short unicode escape");
      char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f')
        v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        v |= uint32_t(c - 'A' + 10);
      else
        bad("bad unicode escape");
    }
    return v;
  }

  void escape(std::string& out) {
    if (i_ >= s_.size()) bad("unterminated string");
    char c = s_[i_++];
    switch (c) {
      case 'b':
        out += '\b';
        break;
      case 't':
        out += '\t';
        break;
      case 'n':
        out += '\n';
        break;
      case 'f':
        out += '\f';
        break;
      case 'r':
        out += '\r';
        break;
      case '"':
        out += '"';
        break;
      case '\\':
        out += '\\';
        break;
      case 'u':
        put_utf8(out, hex(4));
        break;
      case 'U':
        put_utf8(out, hex(8));
        break;
      default:
        bad(std::string("unknown escape \\") + c);
    }
  }

  void control(char c, bool multiline) {
    unsigned char u = static_cast<unsigned char>(c);
    if ((u < 0x20 && c != '\t' && !(multiline && (c == '\n' || c == '\r'))) || u == 0x7F)
      bad("control character in a string");
  }

  std::string basic_string() {
    ++i_;
    std::string out;
    while (true) {
      if (i_ >= s_.size() || s_[i_] == '\n') bad("unterminated string");
      char c = s_[i_++];
      if (c == '"') return out;
      if (c == '\\') {
        escape(out);
        continue;
      }
      control(c, false);
      out += c;
    }
  }

  std::string literal_string() {
    ++i_;
    size_t b = i_;
    while (i_ < s_.size() && s_[i_] != '\'') {
      if (s_[i_] == '\n') bad("unterminated string");
      control(s_[i_], false);
      ++i_;
    }
    if (i_ >= s_.size()) bad("unterminated string");
    return s_.substr(b, i_++ - b);
  }

  // Multi-line strings: a newline right after the opening quotes is dropped; up to two quotes may end the content.
  std::string multiline(char q) {
    i_ += 3;
    newline();
    std::string out;
    while (true) {
      if (i_ >= s_.size()) bad("unterminated multi-line string");
      if (s_[i_] == q && i_ + 2 < s_.size() && s_[i_ + 1] == q && s_[i_ + 2] == q) {
        size_t n = 3;
        while (n < 5 && i_ + n < s_.size() && s_[i_ + n] == q) ++n;
        out.append(n - 3, q);
        i_ += n;
        return out;
      }
      char c = s_[i_++];
      if (q == '"' && c == '\\') {
        // Line-ending backslash: drop it and the whitespace and newlines that follow.
        size_t k = i_;
        while (k < s_.size() && (s_[k] == ' ' || s_[k] == '\t')) ++k;
        if (k < s_.size() && (s_[k] == '\n' || (s_[k] == '\r' && k + 1 < s_.size() && s_[k + 1] == '\n'))) {
          i_ = k;
          while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
          continue;
        }
        escape(out);
        continue;
      }
      control(c, true);
      out += c;
    }
  }

  // Digits with single '_' between digits; returns them without the separators.
  std::string digits(const std::string& t, size_t& k, bool (*ok)(char)) {
    std::string out;
    if (k >= t.size() || !ok(t[k])) bad("bad number '" + t + "'");
    while (k < t.size() && (ok(t[k]) || t[k] == '_')) {
      if (t[k] == '_') {
        if (k + 1 >= t.size() || !ok(t[k + 1])) bad("bad '_' in number '" + t + "'");
      } else {
        out += t[k];
      }
      ++k;
    }
    return out;
  }

  Json number() {
    size_t b = i_;
    while (i_ < s_.size() && (bare_char(s_[i_]) || s_[i_] == '+' || s_[i_] == '.' || s_[i_] == ':')) ++i_;
    std::string t = s_.substr(b, i_ - b);
    if (t.empty()) bad("expected a value");
    if (t.find(':') != std::string::npos ||
        (t.size() >= 10 && digit(t[0]) && digit(t[1]) && digit(t[2]) && digit(t[3]) && t[4] == '-'))
      bad("dates and times are not supported in recipes");
    if (t == "inf" || t == "+inf" || t == "-inf" || t == "nan" || t == "+nan" || t == "-nan")
      bad("inf and nan are not allowed");
    Json j;
    j.kind = Json::Kind::Number;
    if (t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'o' || t[1] == 'b')) {
      const int base = t[1] == 'x' ? 16 : t[1] == 'o' ? 8 : 2;
      auto ok = base == 16  ? +[](char c) { return digit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
                : base == 8 ? +[](char c) { return c >= '0' && c <= '7'; }
                            : +[](char c) { return c == '0' || c == '1'; };
      size_t k = 2;
      std::string d = digits(t, k, ok);
      if (k != t.size()) bad("bad number '" + t + "'");
      uint64_t v = 0;
      for (char c : d) {
        uint64_t x = digit(c) ? uint64_t(c - '0') : uint64_t((c | 0x20) - 'a' + 10);
        if (v > (uint64_t(INT64_MAX) - x) / uint64_t(base)) bad("integer out of range '" + t + "'");
        v = v * uint64_t(base) + x;
      }
      j.text = std::to_string(v);
      return j;
    }
    size_t k = 0;
    std::string out;
    if (t[k] == '+' || t[k] == '-') {
      if (t[k] == '-') out += '-';
      ++k;
    }
    std::string ip = digits(t, k, digit);
    if (ip.size() > 1 && ip[0] == '0') bad("leading zero in number '" + t + "'");
    out += ip;
    bool is_float = false;
    if (k < t.size() && t[k] == '.') {
      ++k;
      out += '.' + digits(t, k, digit);
      is_float = true;
    }
    if (k < t.size() && (t[k] == 'e' || t[k] == 'E')) {
      ++k;
      out += 'e';
      if (k < t.size() && (t[k] == '+' || t[k] == '-')) out += t[k++];
      out += digits(t, k, digit);
      is_float = true;
    }
    if (k != t.size()) bad("bad number '" + t + "'");
    if (!is_float) {
      // TOML integers are signed 64-bit.
      const std::string lim = out[0] == '-' ? "9223372036854775808" : "9223372036854775807";
      if (ip.size() > lim.size() || (ip.size() == lim.size() && ip > lim)) bad("integer out of range '" + t + "'");
    }
    j.text = out;
    return j;
  }

  Json inline_table(size_t depth) {
    ++i_;
    Json t = Json::object();
    std::set<std::string> implicit;  // tables made by dotted keys inside this inline table
    spaces();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return t;
    }
    while (true) {
      std::vector<std::string> keys = key();
      spaces();
      expect("=", "expected '=' after a key");
      spaces();
      Json v = value(depth + 1);
      Json* o = &t;
      std::string p;
      for (size_t k = 0; k + 1 < keys.size(); ++k) {
        p = child(p, keys[k]);
        auto it = o->obj.find(keys[k]);
        if (it == o->obj.end()) {
          implicit.insert(p);
          o = &((*o)[keys[k]] = Json::object());
        } else if (implicit.count(p)) {
          o = &it->second;
        } else {
          bad("'" + dotted(keys, k + 1) + "' already has a value");
        }
      }
      if (o->obj.count(keys.back())) bad("duplicate key '" + dotted(keys, keys.size()) + "'");
      (*o)[keys.back()] = std::move(v);
      spaces();
      if (i_ < s_.size() && s_[i_] == ',') {
        ++i_;
        spaces();
        if (i_ < s_.size() && (s_[i_] == '\n' || s_[i_] == '\r')) bad("inline tables are one line (TOML 1.0)");
        continue;
      }
      if (i_ < s_.size() && s_[i_] == '}') {
        ++i_;
        return t;
      }
      bad("expected ',' or '}' in an inline table (inline tables are one line)");
    }
  }

  Json value(size_t depth) {
    if (depth > kMaxDepth) bad("nesting too deep");
    if (i_ >= s_.size()) bad("expected a value");
    char c = s_[i_];
    if (c == '"') return Json::string(at("\"\"\"") ? multiline('"') : basic_string());
    if (c == '\'') return Json::string(at("'''") ? multiline('\'') : literal_string());
    if (c == '{') return inline_table(depth);
    if (c == '[') {
      ++i_;
      Json a = Json::array();
      while (true) {
        skip_array_space();
        if (i_ < s_.size() && s_[i_] == ']') {
          ++i_;
          return a;
        }
        a.push(value(depth + 1));
        skip_array_space();
        if (i_ < s_.size() && s_[i_] == ',') {
          ++i_;
          continue;
        }
        if (i_ < s_.size() && s_[i_] == ']') {
          ++i_;
          return a;
        }
        bad("expected ',' or ']' in an array");
      }
    }
    if (at("true") && (i_ + 4 >= s_.size() || !bare_char(s_[i_ + 4]))) {
      i_ += 4;
      return Json::boolean(true);
    }
    if (at("false") && (i_ + 5 >= s_.size() || !bare_char(s_[i_ + 5]))) {
      i_ += 5;
      return Json::boolean(false);
    }
    return number();
  }
};

}  // namespace

Json toml_parse(const std::string& text) { return Reader(text).run(); }

}  // namespace ss
