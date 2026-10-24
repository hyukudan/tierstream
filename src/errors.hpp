// Internal error type: thrown inside the library, turned into an ss_status + thread-local message at the C ABI.
#pragma once

#include <stdexcept>
#include <string>

#include "strix_stream/ss.h"

namespace ss {

class Error : public std::runtime_error {
 public:
  Error(ss_status status, const std::string& msg) : std::runtime_error(msg), status_(status) {}
  ss_status status() const { return status_; }

 private:
  ss_status status_;
};

[[noreturn]] inline void fail(ss_status status, const std::string& msg) { throw Error(status, msg); }

inline void require(bool ok, ss_status status, const std::string& msg) {
  if (!ok) fail(status, msg);
}

}  // namespace ss
