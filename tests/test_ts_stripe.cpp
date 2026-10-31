#define TS_STRIPE_TESTING
#include "../tools/ts_stripe.cpp"

#include <functional>

#if defined(_WIN32)
#include <io.h>
#else
#include <sys/wait.h>
#endif

namespace {

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

void rejects(const std::function<void()>& f) {
  bool rejected = false;
  try { f(); } catch (const std::exception&) { rejected = true; }
  check(rejected, "expected rejection");
}

int cli(std::vector<std::string> args) {
  args.insert(args.begin(), "ts-stripe");
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  return stripe_main(int(argv.size()), argv.data());
}

class Capture {
 public:
  explicit Capture(const fs::path& path) {
    std::fflush(stdout);
#if defined(_WIN32)
    saved_ = _dup(_fileno(stdout));
    file_ = _wfopen(path.c_str(), L"w+b");
    check(saved_ >= 0 && file_, "open stdout capture");
    check(_dup2(_fileno(file_), _fileno(stdout)) == 0, "redirect stdout");
#else
    saved_ = dup(fileno(stdout));
    file_ = std::fopen(path.c_str(), "w+b");
    check(saved_ >= 0 && file_, "open stdout capture");
    check(dup2(fileno(file_), fileno(stdout)) >= 0, "redirect stdout");
#endif
  }
  ~Capture() {
    std::fflush(stdout);
#if defined(_WIN32)
    _dup2(saved_, _fileno(stdout));
    _close(saved_);
#else
    dup2(saved_, fileno(stdout));
    close(saved_);
#endif
    std::fclose(file_);
  }
 private:
  int saved_ = -1;
  FILE* file_ = nullptr;
};

void inject(const char* value) {
#if defined(_WIN32)
  check(_putenv_s("TS_STRIPE_FAIL_AFTER_CHUNKS", value) == 0, "set fault injection");
#else
  if (*value) check(setenv("TS_STRIPE_FAIL_AFTER_CHUNKS", value, 1) == 0, "set fault injection");
  else unsetenv("TS_STRIPE_FAIL_AFTER_CHUNKS");
#endif
}

int external_cli(const fs::path& exe, const std::vector<std::string>& args) {
#if defined(_WIN32)
  std::wstring command = L"\"" + exe.wstring() + L"\"";
  for (const auto& a : args) command += L" \"" + path_of(a).wstring() + L"\"";
  STARTUPINFOW si{};
  si.cb = sizeof si;
  PROCESS_INFORMATION pi{};
  check(CreateProcessW(exe.c_str(), command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS,
                       nullptr, nullptr, &si, &pi) != 0, "start tool executable");
  Handle process, thread;
  process.value = pi.hProcess;
  thread.value = pi.hThread;
  check(WaitForSingleObject(process.value, 20000) == WAIT_OBJECT_0, "tool executable timeout");
  DWORD code = 0;
  check(GetExitCodeProcess(process.value, &code) != 0, "tool exit code");
  return int(code);
#else
  const pid_t child = fork();
  check(child >= 0, "fork tool executable");
  if (!child) {
    std::vector<std::string> strings{path_text(exe)};
    strings.insert(strings.end(), args.begin(), args.end());
    std::vector<char*> argv;
    for (auto& a : strings) argv.push_back(a.data());
    argv.push_back(nullptr);
    execv(exe.c_str(), argv.data());
    _exit(127);
  }
  int status = 0;
  check(waitpid(child, &status, 0) == child, "wait tool executable");
  return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
#endif
}

void write_text(const fs::path& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  f << text;
  check(bool(f), "write fixture text");
}

void run(const fs::path& root, const fs::path& exe) {
  const fs::path primary = root / "weights.bin", ranges = root / "ranges.json", out = root / "secondary";
  const uint64_t size = 70 * kMiB - 123;
  {
    std::ofstream f(primary, std::ios::binary);
    std::vector<char> b(size_t(kMiB), 0);
    for (uint64_t c = 0; c < 70; ++c) {
      for (size_t i = 0; i < b.size(); ++i) b[i] = char((c * 71 + i * 13 + (i >> 12)) & 255);
      f.write(b.data(), std::streamsize(std::min(kMiB, size - c * kMiB)));
    }
    check(bool(f), "write synthetic primary");
  }
  const std::string base = "weights.bin";
  ts::Json j = ts::Json::object();
  j["schema"] = ts::Json::string("tierstream/stripe-ranges-v1");
  j["primary"] = ts::Json::string(base);
  j["size"] = ts::Json::number(size);
  j["ranges"] = ts::Json::array();
  auto range = [&](uint64_t offset, uint64_t bytes, const char* policy) {
    ts::Json r = ts::Json::object();
    r["name"] = ts::Json::string("synthetic");
    r["offset"] = ts::Json::number(offset);
    r["bytes"] = ts::Json::number(bytes);
    r["policy"] = ts::Json::string(policy);
    j["ranges"].push(std::move(r));
  };
  range(0, size, "interleave");
  range(2 * kMiB - 4, 8, "all");
  range(68 * kMiB, 1, "all");
  range(size, 0, "all");
  write_text(ranges, ts::json_write(j));
  const auto p = path_text(primary), r = path_text(ranges), d = path_text(out);
  check(cli({"make", p, r, path_text(root), "--chunk-mib", "1"}) == 1, "refuse primary directory");
  check(external_cli(exe, {"make", p, r, d, "--chunk-mib", "1"}) == 1, "production CLI refuses same volume");
  test_allow_same_volume = true;
  for (const char* name : {"bad..bin", "bad."}) {
    const auto invalid_primary = root / name;
    fs::create_hard_link(primary, invalid_primary);
    const auto invalid_out = root / "invalid-name";
    check(cli({"make", path_text(invalid_primary), r, path_text(invalid_out), "--chunk-mib", "1"}) == 1,
          "reject invalid derived data name");
    check(!fs::exists(invalid_out), "invalid derived name created output before validation");
    fs::remove(invalid_primary);
  }
  check(cli({"make", p, r, d, "--chunk-mib", "3"}) == 1, "reject invalid chunk size");
  auto bad = j;
  bad["schema"] = ts::Json::string("wrong");
  write_text(ranges, ts::json_write(bad));
  check(cli({"make", p, r, d, "--chunk-mib", "1"}) == 1, "reject wrong ranges schema");
  bad = j;
  bad["primary"] = ts::Json::string("other.bin");
  write_text(ranges, ts::json_write(bad));
  check(cli({"make", p, r, d, "--chunk-mib", "1"}) == 1, "reject wrong ranges basename");
  bad = j;
  bad["size"] = ts::Json::number(size + 1);
  write_text(ranges, ts::json_write(bad));
  check(cli({"make", p, r, d, "--chunk-mib", "1"}) == 1, "reject wrong ranges size");
  bad = j;
  bad["ranges"].arr[0]["bytes"] = ts::Json::number(UINT64_MAX);
  write_text(ranges, ts::json_write(bad));
  check(cli({"make", p, r, d, "--chunk-mib", "1"}) == 1, "reject range overflow");
  write_text(ranges, ts::json_write(j));
  const auto interrupted = root / "interrupted";
  inject("5");
  check(cli({"make", p, r, path_text(interrupted), "--chunk-mib", "1"}) == 1, "injected interruption");
  inject("");
  check(!fs::exists(interrupted / ts::stripe_manifest_name(base)), "interruption published a manifest");
  size_t partials = 0;
  fs::path stale;
  for (const auto& e : fs::directory_iterator(interrupted)) if (ends(path_text(e.path()), ".partial")) {
    ++partials;
    stale = e.path();
  }
  check(partials == 1, "interruption did not leave one partial");
  write_text(interrupted / (ts::stripe_manifest_name(base) + ".tmp"), "stale");
  check(cli({"make", p, r, path_text(interrupted), "--chunk-mib", "1", "--keep-going"}) == 0,
        "make after interruption");
  check(!fs::exists(stale) && !fs::exists(interrupted / (ts::stripe_manifest_name(base) + ".tmp")),
        "next make did not clean stale files");
  fs::rename(interrupted, out / "ready");
  const auto ready = out / "ready";
  const auto sd = path_text(ready);
  check(external_cli(exe, {"verify", sd, p, "--full"}) == 0, "executable full verify");
  check(cli({"verify", sd, p}) == 0, "verify");
  const auto manifest_path = ready / ts::stripe_manifest_name(base);
  const auto manifest_text = read_text(manifest_path);
  const auto m = ts::stripe_manifest_parse(manifest_text);
  check(m.chunk_bytes == kMiB && m.primary_size == size && m.slots() == 37, "wrong placement sizes");
  check(ts::stripe_manifest_write(m) == manifest_text, "canonical manifest round trip");
  const auto data_path = ready / path_of(m.data_name);
  {
    ts::RawFile source(p, true), data(path_text(data_path), true);
    auto a = buffer(kMiB), b = buffer(kMiB);
    std::vector<uint64_t> actual;
    for (const auto& run : m.runs) for (uint64_t i = 0; i < run.count; ++i) {
      const auto c = run.first_chunk + i, slot = run.first_slot + i;
      actual.push_back(c);
      const uint64_t logical = std::min(kMiB, size - c * kMiB);
      check(source.read(a.get(), kMiB, c * kMiB) >= logical, "short source read");
      check(data.read(b.get(), kMiB, slot * kMiB) == kMiB, "short stripe read");
      check(std::memcmp(a.get(), b.get(), size_t(logical)) == 0, "slot bytes differ from primary");
      for (uint64_t k = logical; k < kMiB; ++k) check(b.get()[k] == 0, "tail not zero padded");
    }
    std::vector<uint64_t> expected;
    for (uint64_t c = 0; c < 70; ++c) if ((c & 1) || c == 2 || c == 68) expected.push_back(c);
    check(actual == expected, "touching ranges or policies placed wrong chunks");
#if defined(_WIN32)
    Handle writer;
    writer.value = CreateFileW(primary.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    check(writer.value == INVALID_HANDLE_VALUE, "primary serving handle permits a writer");
#endif
  }
  check(cli({"show", sd, base}) == 0, "show");
  rejects([&]() { auto invalid = m; invalid.runs[0].first_slot = 1; ts::stripe_manifest_write(invalid); });
  rejects([&]() { auto invalid = m; invalid.data_name = "../bad"; ts::stripe_manifest_write(invalid); });
  rejects([&]() { auto invalid = m; invalid.primary_size = ts::kStripeMaxPrimary + 1; ts::stripe_manifest_write(invalid); });
  rejects([&]() {
    auto invalid = ts::json_parse(manifest_text);
    invalid["digest"] = ts::Json::string("0000000000000000");
    ts::stripe_manifest_parse(ts::json_write(invalid));
  });
  const auto current_id = ts::identity_of(path_text(data_path));
  check(cli({"make", p, r, sd, "--chunk-mib", "1"}) == 0, "reuse existing versioned data");
  check(ts::identity_of(path_text(data_path)) == current_id, "existing data replaced instead of reused");
  {
    OutputLock lock(ready, base);
    check(cli({"make", p, r, sd, "--chunk-mib", "1"}) == 1, "concurrent make ignored output lock");
    check(cli({"gc", sd, base}) == 1, "concurrent gc ignored output lock");
  }
  inject("4");
  check(cli({"make", p, r, sd, "--chunk-mib", "1"}) == 1, "interrupted replacement");
  inject("");
  check(read_text(manifest_path) == manifest_text, "interruption damaged existing manifest");
  check(cli({"verify", sd, p}) == 0, "old manifest no longer valid");
  {
    Capture capture(root / "gc.txt");
    check(cli({"gc", sd, base, "--dry-run"}) == 0, "gc dry-run");
  }
  check(read_text(root / "gc.txt").empty(), "gc lists current data");
#if defined(_WIN32)
  auto case_variant = m;
  for (char& c : case_variant.data_name) if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
  write_text(manifest_path, ts::stripe_manifest_write(case_variant));
  {
    Capture capture(root / "gc.txt");
    check(cli({"gc", sd, base, "--dry-run"}) == 0, "gc case-insensitive live name");
  }
  check(read_text(root / "gc.txt").empty(), "gc lists current data with different filename casing");
  write_text(manifest_path, manifest_text);
#endif
  const auto orphan = ready / "weights.bin.stripe-unused.data";
  write_text(orphan, "orphan");
  {
    Capture capture(root / "gc.txt");
    check(cli({"gc", sd, base, "--dry-run"}) == 0, "gc orphan dry-run");
  }
  check(read_text(root / "gc.txt") == "weights.bin.stripe-unused.data\n" && fs::exists(orphan), "dry-run changed files");
  check(cli({"gc", sd, base}) == 0 && !fs::exists(orphan) && fs::exists(data_path), "gc removed wrong data");
#if defined(_WIN32)
  {
    const auto busy = ready / "weights.bin.stripe-a-busy.data";
    const auto unused = ready / "weights.bin.stripe-z-unused.data";
    write_text(busy, "busy");
    write_text(unused, "unused");
    ts::RawFile held(path_text(busy), false);
    check(cli({"gc", sd, base}) == 0, "gc failed on busy data");
    check(fs::exists(busy) && !fs::exists(unused), "gc did not continue past busy file");
  }
  check(cli({"gc", sd, base}) == 0, "gc after busy file closes");
#endif
  // An unplaced primary change is only detectable by the full-file SHA-256 check.
  {
    std::fstream f(primary, std::ios::binary | std::ios::in | std::ios::out);
    const char changed = 1;
    f.write(&changed, 1);
    check(bool(f), "change unplaced primary byte");
  }
  check(cli({"verify", sd, p}) == 0, "stripe-only verify unexpectedly binds primary contents");
  check(cli({"verify", sd, p, "--full"}) == 1, "full verify ignored primary SHA-256");
  {
    std::fstream f(primary, std::ios::binary | std::ios::in | std::ios::out);
    const char original = 0;
    f.write(&original, 1);
    check(bool(f), "restore primary byte");
  }
  range(0, 1, "all");
  write_text(ranges, ts::json_write(j));
  const auto alternate = root / "alternate";
  check(cli({"make", p, r, path_text(alternate), "--chunk-mib", "1"}) == 0, "alternate placement make");
  const auto alt = load_manifest(alternate, base);
  check(alt.data_name != m.data_name && alt.primary_sha256 == m.primary_sha256 && alt.slots() == m.slots() + 1,
        "placement not bound to data version name");
  j["ranges"].arr.pop_back();
  write_text(ranges, ts::json_write(j));
#if !defined(_WIN32)
  fs::permissions(data_path, fs::perms::owner_write, fs::perm_options::add);
#endif
  {
    std::fstream f(data_path, std::ios::binary | std::ios::in | std::ios::out);
    char byte = 0;
    f.read(&byte, 1);
    byte ^= 1;
    f.seekp(0);
    f.write(&byte, 1);
    check(bool(f), "corrupt stripe fixture");
  }
  check(external_cli(exe, {"verify", sd, p}) == 1, "corrupted stripe accepted");
  check(cli({"make", p, r, sd, "--chunk-mib", "1"}) == 1, "corrupted immutable data was reused or overwritten");
  check(read_text(manifest_path) == manifest_text, "failed reuse changed manifest");
  {
    Capture capture(root / "gc.txt");
    check(cli({"gc", sd, base, "--dry-run"}) == 0, "gc after corruption");
  }
  check(read_text(root / "gc.txt").empty(), "gc lists corrupt current data");
}

}  // namespace

// blocks:K / blocks-even:K place alternating runs of K chunks; the two parities cover each placed range exactly once
void run_block_placement(const fs::path& root) {
  const fs::path ranges = root / "blocks.json";
  const uint64_t chunk = kMiB, size = 20 * chunk - 5;
  auto write = [&](const char* policy, uint64_t offset, uint64_t bytes) {
    ts::Json j = ts::Json::object();
    j["schema"] = ts::Json::string("tierstream/stripe-ranges-v1");
    j["primary"] = ts::Json::string("weights.bin");
    j["size"] = ts::Json::number(size);
    j["ranges"] = ts::Json::array();
    ts::Json r = ts::Json::object();
    r["name"] = ts::Json::string("experts");
    r["offset"] = ts::Json::number(offset);
    r["bytes"] = ts::Json::number(bytes);
    r["policy"] = ts::Json::string(policy);
    j["ranges"].push(std::move(r));
    write_text(ranges, ts::json_write(j));
  };
  write("blocks:4", 0, size);
  check(placement(ranges, "weights.bin", size, chunk) == std::vector<uint64_t>{4, 5, 6, 7, 12, 13, 14, 15},
        "blocks:4 places the odd runs of 4 chunks");
  write("blocks-even:4", 2 * chunk, 15 * chunk);
  check(placement(ranges, "weights.bin", size, chunk) == std::vector<uint64_t>{2, 3, 8, 9, 10, 11, 16},
        "blocks-even:4 places the even runs inside the range");
  write("blocks:1", 0, size);
  check(placement(ranges, "weights.bin", size, chunk).size() == 10, "blocks:1 equals interleave");
  for (const char* bad : {"blocks:", "blocks:0", "blocks:x", "blocks:1234567", "blocks-odd:4", "blocks:-1"}) {
    write(bad, 0, size);
    rejects([&] { placement(ranges, "weights.bin", size, chunk); });
  }
}

int main(int argc, char** argv) {
  fs::path root;
  try {
    check(argc == 3, "test needs fixture parent and executable");
    const auto parent = fs::absolute(path_of(argv[1]));
    fs::create_directories(parent);
    root = parent / ("run-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    check(fs::create_directory(root), "create unique fixture root");
    run(root, fs::absolute(path_of(argv[2])));
    run_block_placement(root);
    fs::remove_all(root);
    std::puts("stripe tool tests passed");
    return 0;
  } catch (const std::exception& e) {
    inject("");
    std::fprintf(stderr, "stripe tool test: %s\n", e.what());
    if (!root.empty()) std::fprintf(stderr, "fixtures retained: %s\n", path_text(root).c_str());
    return 1;
  }
}
