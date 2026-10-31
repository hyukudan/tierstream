// Whole-device budgets (ss.h "Whole-device budgets", docs/design.md §5.1): the planner that gives the expert tiers
// what the caps leave after the fixed parts, and the ledger that tracks real allocations against the same caps.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"
#include "strix_stream/ss.h"

namespace ts {

void budget_in_defaults(ss_budget_in* in);
// Pure arithmetic: writes every field of out (struct_size included); false = refused (out.refuse says why).
bool budget_plan(const ss_budget_in& in, ss_budget_out& out);
// The plan as JSON members (no braces): ss_budget_plan_json.
std::string budget_plan_json(const ss_budget_in& in, const ss_budget_out& out);

// The ledger (struct ss_budget_ledger). add throws SS_E_CAPACITY (cap) / SS_E_INVALID; sub throws SS_E_INVALID.
ss_budget_ledger* ledger_create(uint64_t vram_total_bytes, uint64_t ram_total_bytes);
void ledger_free(ss_budget_ledger* l);
void ledger_add(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes);
void ledger_sub(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes);
void ledger_observe_device(ss_budget_ledger* l, uint64_t used);
// One measured sample (SS_BUDGET_NO_SAMPLE: none on that side); within_caps: every measured peak <= its cap.
void ledger_observe(ss_budget_ledger* l, const char* stage, uint64_t device_used, uint64_t host_used);
bool ledger_within_caps(const ss_budget_ledger* l);
ss_budget_stats ledger_stats(const ss_budget_ledger* l);

// A context's charges on an optional ledger, released together (destructor, or release()). charge() checks the cap
// before the caller allocates. Not thread-safe itself (the ledger is): one owner thread.
class BudgetCharges {
 public:
  explicit BudgetCharges(ss_budget_ledger* l = nullptr) : l_(l) {}
  BudgetCharges(const BudgetCharges&) = delete;
  BudgetCharges& operator=(const BudgetCharges&) = delete;
  ~BudgetCharges() { release(); }
  bool active() const { return l_ != nullptr; }
  void charge(ss_budget_kind kind, const char* name, uint64_t bytes);
  void release() noexcept;
  // The memory stays allocated past the owner (a context leaking it on purpose): keep the charges on the ledger.
  void forget() noexcept { charges_.clear(); }

 private:
  struct Charge {
    ss_budget_kind kind;
    std::string name;
    uint64_t bytes;
  };
  ss_budget_ledger* l_;
  std::vector<Charge> charges_;
};

// Host mode: a backend whose alloc_device / alloc_pinned are charged to the ledger first ("device" / "pinned"
// components named after the context) and released by their frees.
std::unique_ptr<Backend> budget_backend(std::unique_ptr<Backend> inner, ss_budget_ledger* l);

}  // namespace ts
