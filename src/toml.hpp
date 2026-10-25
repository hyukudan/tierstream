// TOML 1.0 reader for recipes (docs/recipes.md §1): the document becomes the same Json tree a JSON recipe gives.
// Subset: tables, arrays of tables, dotted and quoted keys, inline tables, arrays, basic / literal strings (single and
// multi-line), integers (decimal, hex, octal, binary, '_' separators), floats, booleans. Dates and times, inf and nan
// are refused. Integers keep their decimal text and floats their text, so integer fields stay integers.
#pragma once

#include <string>

#include "json.hpp"

namespace ts {

// Throws ts::Error(SS_E_RECIPE) with "toml: line N: ..." on malformed input. Depth and size are bounded.
Json toml_parse(const std::string& text);

}  // namespace ts
