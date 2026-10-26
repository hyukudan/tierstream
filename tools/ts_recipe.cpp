// ts-recipe (also built as ss-recipe for one release): validate, explain, match, select and compile tierstream
// recipes (docs/recipes.md §8.4).
//   ts-recipe validate RECIPE [LAYERS]
//   ts-recipe explain  RECIPE [LAYERS] [--json]
//   ts-recipe match    RECIPE PROBE.json [LAYERS]
//   ts-recipe select   RECIPE STATE.json [LAYERS]
//   ts-recipe plan     RECIPE --route R --device-set S [--member M] [--id PROBE_ID] --desc DESC.json [--benefit FILE]
//                      [--ref SET=DESC.json[,BENEFIT]]... [LAYERS]
//   ts-recipe place    RECIPE --route R --probe PROBE.json --components COMPONENTS.json [--desc DESC.json]
//                      [--usage USAGE.json] [--context CONTEXT.json] [--explain] [--json-out FILE] [LAYERS]
//   ts-recipe solve    RECIPE --route R --probe PROBE.json --inputs SOLVER-INPUTS.json [--desc D] [--context X]
//                      [--table] [--json-out FILE] [--pick ROW_ID --emit-overrides FILE] [LAYERS]
//   LAYERS: [--profile P --identity IDENTITY.json] [--overrides O]... [--set path=value]...
// JSON on stdout; exit 0 = success, 1 = refusal (message on stderr), 2 = usage error.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "json.hpp"
#include "strix_stream/ss.h"
#include "tierstream/ts_place.h"

// The tool's own name (ts-recipe; ss-recipe for one release of the rename): every message starts with it.
#ifndef TS_TOOL_NAME
#define TS_TOOL_NAME "ts-recipe"
#endif

namespace {

const char* kUsage =
    "usage: " TS_TOOL_NAME " validate|explain|match|select|plan|place|solve RECIPE [...]\n"
    "  validate RECIPE [LAYERS]\n"
    "  explain  RECIPE [LAYERS] [--json]\n"
    "  match    RECIPE PROBE.json [LAYERS]\n"
    "  select   RECIPE STATE.json [LAYERS]\n"
    "  plan     RECIPE --route R --device-set S [--member M] [--id PROBE_ID] --desc DESC.json [--benefit FILE]\n"
    "           [--ref SET=DESC.json[,BENEFIT]]... [LAYERS]\n"
    "  place    RECIPE --route R --probe PROBE.json --components COMPONENTS.json [--desc DESC.json]\n"
    "           [--usage USAGE.json] [--context CONTEXT.json] [--explain] [--json-out FILE] [LAYERS]\n"
    "  solve    RECIPE --route R --probe PROBE.json --inputs SOLVER-INPUTS.json [--desc D] [--context X]\n"
    "           [--table] [--json-out FILE] [--pick ROW_ID --emit-overrides FILE] [LAYERS]\n"
    "  LAYERS: [--profile P --identity IDENTITY.json] [--overrides O]... [--set path=value]...\n";

[[noreturn]] void usage(const std::string& why) {
  std::fprintf(stderr, TS_TOOL_NAME ": %s\n%s", why.c_str(), kUsage);
  std::exit(2);
}

[[noreturn]] void refuse(const char* what) {
  std::fprintf(stderr, TS_TOOL_NAME ": %s: %s\n", what, ss_last_error());
  std::exit(1);
}

std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, TS_TOOL_NAME ": cannot open %s\n", path.c_str());
    std::exit(1);
  }
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

void write_file(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f << text;
  f.close();
  if (!f) {
    std::fprintf(stderr, TS_TOOL_NAME ": cannot write %s\n", path.c_str());
    std::exit(1);
  }
}

void print_owned(char* s) {
  std::printf("%s\n", s);
  ss_free(s);
}

std::vector<double> read_benefit(const std::string& path) {
  std::vector<double> out;
  try {
    ts::Json j = ts::json_parse(slurp(path));
    if (!j.is_array()) throw std::runtime_error("expected a JSON array of numbers");
    for (const auto& v : j.arr) out.push_back(ts::json_f64(v, path));
  } catch (const std::exception& e) {
    std::fprintf(stderr, TS_TOOL_NAME ": benefit %s: %s\n", path.c_str(), e.what());
    std::exit(1);
  }
  return out;
}

struct DescHolder {
  ss_desc* d = nullptr;
  ~DescHolder() { ss_desc_free(d); }
};

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) usage("missing command or recipe");
  const std::string cmd = argv[1];
  if (cmd != "validate" && cmd != "explain" && cmd != "match" && cmd != "select" && cmd != "plan" && cmd != "place" && cmd != "solve")
    usage("unknown command " + cmd);
  const std::string recipe_path = argv[2];
  int i = 3;
  std::string input;  // PROBE / STATE
  if (cmd == "match" || cmd == "select") {
    if (argc < 4 || argv[3][0] == '-') usage(cmd + " needs a JSON input file");
    input = argv[3];
    i = 4;
  }
  std::string profile, identity, route, set, desc_path, benefit_path, id, probe_path, components_path, usage_path,
      context_path, inputs_path, json_path, pick, emit_path;
  int member = 0;
  bool json_out = false, explain_out = false;
  std::vector<std::string> overrides, sets, refs;
  for (; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) usage(a + " needs a value");
      return argv[++i];
    };
    if (a == "--profile")
      profile = next();
    else if (a == "--identity")
      identity = next();
    else if (a == "--overrides")
      overrides.push_back(next());
    else if (a == "--set")
      sets.push_back(next());
    else if (a == "--json" && cmd == "explain")
      json_out = true;
    else if (a == "--route" && (cmd == "plan" || cmd == "place" || cmd == "solve"))
      route = next();
    else if (a == "--probe" && (cmd == "place" || cmd == "solve"))
      probe_path = next();
    else if (a == "--components" && cmd == "place")
      components_path = next();
    else if (a == "--usage" && cmd == "place")
      usage_path = next();
    else if (a == "--context" && (cmd == "place" || cmd == "solve"))
      context_path = next();
    else if (a == "--explain" && cmd == "place")
      explain_out = true;
    else if (a == "--inputs" && cmd == "solve")
      inputs_path = next();
    else if (a == "--json-out" && (cmd == "place" || cmd == "solve"))
      json_path = next();
    else if (a == "--pick" && cmd == "solve")
      pick = next();
    else if (a == "--emit-overrides" && cmd == "solve")
      emit_path = next();
    else if (a == "--table" && cmd == "solve")
      explain_out = true;
    else if (a == "--id" && cmd == "plan")
      id = next();
    else if (a == "--member" && cmd == "plan")
      member = std::atoi(next().c_str());
    else if (a == "--desc" && (cmd == "plan" || cmd == "place" || cmd == "solve"))
      desc_path = next();
    else if (a == "--benefit" && cmd == "plan")
      benefit_path = next();
    else if (a == "--ref" && cmd == "plan")
      refs.push_back(next());
    else if (a == "--device-set" && cmd == "plan")
      set = next();
    else
      usage("unknown argument " + a);
  }
  // "--set" is the override flag; plan names its device set with --device-set (and accepts S=... in --ref).
  if (!profile.empty() && identity.empty()) usage("--profile needs --identity (the current probe report + workload)");

  if (pick.empty() != emit_path.empty()) usage("--pick and --emit-overrides must be used together");
  if (!json_path.empty() && json_path == emit_path) usage("--json-out and --emit-overrides need different files");

  ss_recipe* r = nullptr;
  if (ss_recipe_from_file(recipe_path.c_str(), &r) != SS_OK) refuse("recipe");
  std::unique_ptr<ss_recipe, void (*)(ss_recipe*)> guard(r, ss_recipe_free);
  if (!profile.empty()) {
    const std::string t = slurp(profile), id_text = slurp(identity);
    if (ss_recipe_add_profile(r, t.data(), t.size(), profile.c_str(), id_text.data(), id_text.size()) != SS_OK)
      refuse("profile");
  }
  for (const auto& o : overrides) {
    const std::string t = slurp(o);
    if (ss_recipe_add_overrides(r, t.data(), t.size(), o.c_str()) != SS_OK) refuse("overrides");
  }
  for (const auto& s : sets) {
    const size_t eq = s.find('=');
    if (eq == std::string::npos) usage("--set needs path=value");
    if (ss_recipe_set(r, s.substr(0, eq).c_str(), s.substr(eq + 1).c_str(), "--set") != SS_OK) refuse("--set");
  }

  char* out = nullptr;
  if (cmd == "validate") {
    if (ss_recipe_to_json(r, &out, nullptr) != SS_OK) refuse("validate");
    print_owned(out);
    return 0;
  }
  if (cmd == "explain") {
    if (ss_recipe_explain(r, &out, nullptr) != SS_OK) refuse("explain");
    if (json_out) {
      print_owned(out);
      return 0;
    }
    ts::Json j = ts::json_parse(out);
    ss_free(out);
    std::printf("recipe %s (sha256 %s)\n", j.find("recipe")->text.c_str(), j.find("recipe_sha256")->text.c_str());
    for (const auto& L : j.find("layers")->arr)
      std::printf("  layer %-8s %s\n", L.find("kind")->text.c_str(), L.find("origin")->text.c_str());
    for (const auto& f : j.find("fields")->arr) {
      std::string from = f.find("from")->text;
      if (const ts::Json* o = f.find("origin")) from += " " + o->text;
      std::printf("%-64s %-24s [%s]\n", f.find("path")->text.c_str(), ts::json_write(*f.find("value")).c_str(),
                  from.c_str());
      if (const ts::Json* sh = f.find("shadowed"))
        for (const auto& s : sh->arr)
          std::printf("%-64s   was %s from %s %s: %s\n", "", ts::json_write(*s.find("value")).c_str(),
                      s.find("from")->text.c_str(), s.find("origin")->text.c_str(), s.find("reason")->text.c_str());
    }
    return 0;
  }
  if (cmd == "match" || cmd == "select") {
    const std::string t = slurp(input);
    ss_status st = cmd == "match" ? ss_recipe_match(r, t.data(), t.size(), &out, nullptr)
                                  : ss_recipe_select(r, t.data(), t.size(), &out, nullptr);
    if (st != SS_OK) refuse(cmd.c_str());
    print_owned(out);
    return 0;
  }

  if (cmd == "solve") {
    if (route.empty() || probe_path.empty() || inputs_path.empty()) usage("solve needs --route, --probe and --inputs");
    const std::string probe = slurp(probe_path);
    std::string inputs = slurp(inputs_path);
    if (!desc_path.empty() || !context_path.empty()) {
      try {
        ts::Json in = ts::json_parse(inputs);
        if (!desc_path.empty()) in["desc"] = ts::Json::string(desc_path);
        if (!context_path.empty()) in["context"] = ts::Json::string(context_path);
        inputs = ts::json_write(in);
      } catch (const std::exception& e) {
        std::fprintf(stderr, TS_TOOL_NAME ": solve: %s\n", e.what());
        return 1;
      }
    }
    char* tab = nullptr;
    char* emitted = nullptr;
    if (ts_solve_pick(r, route.c_str(), probe.data(), probe.size(), inputs.data(), inputs.size(),
                      pick.empty() ? nullptr : pick.c_str(), pick.empty() ? nullptr : &emitted, nullptr,
                      &out, nullptr, explain_out ? &tab : nullptr, nullptr) != SS_OK)
      refuse("solve");
    if (!emit_path.empty()) {
      write_file(emit_path, emitted);
      ss_free(emitted);
    }
    if (!json_path.empty()) write_file(json_path, std::string(out) + "\n");
    if (explain_out) {
      std::printf("%s", tab);
      ss_free(tab);
      ss_free(out);
      return 0;
    }
    print_owned(out);
    return 0;
  }
  if (cmd == "place") {
    if (route.empty() || probe_path.empty() || components_path.empty())
      usage("place needs --route, --probe and --components");
    const std::string probe = slurp(probe_path), comps = slurp(components_path);
    const std::string usage_text = usage_path.empty() ? std::string() : slurp(usage_path);
    const std::string context_text = context_path.empty() ? std::string() : slurp(context_path);
    DescHolder desc;
    if (!desc_path.empty() && ss_desc_from_file(desc_path.c_str(), &desc.d) != SS_OK) refuse("descriptor");
    char* ex = nullptr;
    if (ts_place(r, route.c_str(), probe.data(), probe.size(), comps.data(), comps.size(), desc.d,
                 usage_path.empty() ? nullptr : usage_text.data(), usage_text.size(),
                 context_path.empty() ? nullptr : context_text.data(), context_text.size(), &out, nullptr,
                 explain_out ? &ex : nullptr, nullptr) != SS_OK)
      refuse("place");
    if (!json_path.empty()) write_file(json_path, std::string(out) + "\n");
    if (explain_out) {
      std::printf("%s", ex);
      ss_free(ex);
      ss_free(out);
      return 0;
    }
    print_owned(out);
    return 0;
  }

  // plan
  if (route.empty() || set.empty() || desc_path.empty()) usage("plan needs --route, --device-set and --desc");
  DescHolder desc;
  if (ss_desc_from_file(desc_path.c_str(), &desc.d) != SS_OK) refuse("descriptor");
  std::vector<double> ben;
  if (!benefit_path.empty()) ben = read_benefit(benefit_path);
  // --ref SET=DESC[,BENEFIT], in the route's device order.
  std::vector<std::unique_ptr<DescHolder>> ref_desc;
  std::vector<std::vector<double>> ref_ben;
  std::vector<const ss_desc*> rd;
  std::vector<const double*> rb;
  std::vector<size_t> rn;
  for (const auto& x : refs) {
    const size_t eq = x.find('=');
    if (eq == std::string::npos) usage("--ref needs SET=DESC.json[,BENEFIT]");
    std::string files = x.substr(eq + 1), bpath;
    const size_t comma = files.find(',');
    if (comma != std::string::npos) {
      bpath = files.substr(comma + 1);
      files = files.substr(0, comma);
    }
    ref_desc.push_back(std::make_unique<DescHolder>());
    if (ss_desc_from_file(files.c_str(), &ref_desc.back()->d) != SS_OK) refuse("reference descriptor");
    ref_ben.push_back(bpath.empty() ? std::vector<double>{} : read_benefit(bpath));
  }
  for (size_t k = 0; k < refs.size(); ++k) {
    rd.push_back(ref_desc[k]->d);
    rb.push_back(ref_ben[k].empty() ? nullptr : ref_ben[k].data());
    rn.push_back(ref_ben[k].size());
  }
  ss_budget b;
  ss_budget_defaults(&b);
  ss_options o;
  ss_options_defaults(&o);
  if (ss_recipe_plan(r, route.c_str(), set.c_str(), member, id.empty() ? "unbound" : id.c_str(), desc.d,
                     ben.empty() ? nullptr : ben.data(), ben.size(), rd.empty() ? nullptr : rd.data(),
                     rb.empty() ? nullptr : rb.data(), rn.empty() ? nullptr : rn.data(), rd.size(), &b, &o, nullptr,
                     &out, nullptr) != SS_OK)
    refuse("plan");
  print_owned(out);
  return 0;
}
