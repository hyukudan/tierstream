// Format ids across the strix-stream -> tierstream rename: every parser accepts
// "strix-stream/<x>" and "tierstream/<x>" as the same format; every output keeps writing "strix-stream/<x>" for one
// release, so no hashed document changes.
#pragma once

#include <string>

namespace ts {

// `emitted` is the id this build writes ("strix-stream/<x>").
inline bool schema_accepts(const std::string& got, const char* emitted) {
  const std::string e(emitted);
  if (got == e) return true;
  const std::string old_prefix = "strix-stream/";
  return e.compare(0, old_prefix.size(), old_prefix) == 0 && got == "tierstream/" + e.substr(old_prefix.size());
}

// "strix-stream/<x>" or "tierstream/<x>", for refusal messages.
inline std::string schema_both(const char* emitted) {
  const std::string e(emitted);
  const std::string old_prefix = "strix-stream/";
  if (e.compare(0, old_prefix.size(), old_prefix) != 0) return "\"" + e + "\"";
  return "\"" + e + "\" or \"tierstream/" + e.substr(old_prefix.size()) + "\"";
}

}  // namespace ts
