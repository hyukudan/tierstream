// Minimal JSON value + parser + canonical writer (desc-v1 manifests, plans, stats). Numbers keep their text so 64-bit
// byte counts and offsets round-trip exactly.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ts {

struct Json {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  std::string text;  // number text or string value
  std::vector<Json> arr;
  std::map<std::string, Json> obj;  // sorted keys: the writer is canonical

  static Json number(uint64_t v);
  static Json number(int64_t v);
  static Json number(double v);
  static Json string(std::string v);
  static Json boolean(bool v);
  static Json array();
  static Json object();

  bool is_null() const { return kind == Kind::Null; }
  bool is_object() const { return kind == Kind::Object; }
  bool is_array() const { return kind == Kind::Array; }
  bool is_string() const { return kind == Kind::String; }
  bool is_number() const { return kind == Kind::Number; }

  const Json* find(const std::string& key) const;
  Json& operator[](const std::string& key) { kind = Kind::Object; return obj[key]; }
  void push(Json v) { kind = Kind::Array; arr.push_back(std::move(v)); }
};

// Throws ts::Error (errors.hpp) with a position on malformed input. Depth and size are bounded.
Json json_parse(const std::string& text);
std::string json_write(const Json& v);

// Typed getters with path-qualified errors ("layers[3].unit_bytes: expected an unsigned integer").
uint64_t json_u64(const Json& v, const std::string& where);
int64_t json_i64(const Json& v, const std::string& where);
double json_f64(const Json& v, const std::string& where);
const std::string& json_str(const Json& v, const std::string& where);
bool json_bool(const Json& v, const std::string& where);

}  // namespace ts
