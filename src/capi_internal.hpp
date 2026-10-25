// Internal glue between capi.cpp (handles, per-thread error text) and the additive C ABIs built beside it
// (adapt_capi.cpp). Not installed.
#pragma once

#include <cstddef>
#include <functional>
#include <string>

#include "strix_stream/ss.h"

namespace ts {
class Ctx;
struct Desc;
}  // namespace ts

ss_status ts_capi_guard(const std::function<void()>& f);
ss_status ts_capi_text(const std::string& s, char** out, size_t* len);
ts::Ctx* ts_capi_ctx(ss_ctx* c);
const ts::Desc* ts_capi_desc(const ss_desc* d);
