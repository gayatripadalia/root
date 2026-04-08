## \file
## \ingroup tutorial_dataframe
## \notebook -nodraw
##
## \brief Thread-safe column definitions in Python RDataFrame using `rdfslot_`.
##
## This tutorial shows how to safely use mutable state in multi-threaded
## workflows by assigning per-slot resources, avoiding race conditions without
## locks or mutexes.
##
## In C++, ROOT provides `DefineSlot` and `RedefineSlot`, which automatically
## pass the slot (thread) index as the first argument to a user-defined lambda.
## This allows safe, lock-free access to per-slot resources — for example, one
## random-number generator or one histogram per thread — eliminating data races
## without requiring a mutex.
##
## These APIs are currently not available in PyROOT.
##
## This tutorial demonstrates how to reproduce the same pattern in Python using
## the implicit column `rdfslot_`, which carries the slot index for each entry.
## By forwarding `rdfslot_` as an explicit argument, we can index into a
## per-slot container with exactly the same safety guarantees that `DefineSlot`
## provides in C++.
##
## ### What this tutorial covers
##
##  1. Why shared mutable state is unsafe in a multi-threaded RDataFrame loop
##     and how `DefineSlot` / `rdfslot_` solve the problem.
##  2. **Use-case A** — per-slot random-number generation: smearing true values
##     with Gaussian noise without a shared RNG or a mutex.
##  3. **Use-case B** — per-slot histograms: filling thread-local histograms and
##     merging them at the end (a lock-free alternative to a shared histogram
##     protected by a mutex).
##  4. Verification that the approach is thread-safe and produces
##     statistically correct results.
##
## In short:
## Shared resource (unsafe): one RNG / histogram shared across threads
## Per-slot resource (safe): one RNG / histogram per slot via rdfslot_
##
## \macro_output
## \author Gayatri Padalia
import ROOT
import numpy as np
import ctypes
# Helper: print a section banner so the terminal output is easy to follow
def banner(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)
# Background: DefineSlot in C++ vs the Python workaround
#
# In C++, you would write the following to safely smear values per-thread:
#
#   // One RNG per slot — constructed before the event loop
#   unsigned int nSlots = df.GetNSlots();
#   std::vector<TRandom3> rngs(nSlots);
#   for (unsigned int i = 0; i < nSlots; ++i) rngs[i].SetSeed(i + 1);
#
#   auto df_smeared = df.DefineSlot(
#       "smeared_pt",
#       [&rngs](unsigned int slot, double pt) {
#           // `slot` is guaranteed unique per thread — no data race
#           return pt + rngs[slot].Gaus(0.0, 0.01 * pt);
#       },
#       {"true_pt"}
#   );
#
# DefineSlot injects `slot` automatically.  In Python we replicate this by
# including "rdfslot_" in the column list and receiving it as the first
# argument of our callable.  Everything else is identical.
def main() -> None:
    # Enable implicit multi-threading
    # ROOT will use as many slots as there are logical CPU cores (up to the
    # pool size). Each slot runs on exactly one thread at any given time,
    # so a per-slot resource is inherently thread-local.
    ROOT.ROOT.EnableImplicitMT()
    # Build a synthetic dataset
    #
    # We simulate 200 000 particle-physics events, each with:
    # true_pt   — true transverse momentum (Gaussian, mean=50, sigma=10)
    # true_eta  — pseudo-rapidity           (Gaussian, mean=0,  sigma=2)
    #
    # We use a single-threaded RNG here (before the event loop) so the
    # dataset itself is reproducible.
    N_EVENTS = 200_000
    rng_init = np.random.default_rng(seed=42)
    true_pt  = rng_init.normal(loc=50.0, scale=10.0, size=N_EVENTS).astype(np.float64)
    true_eta = rng_init.normal(loc=0.0,  scale=2.0,  size=N_EVENTS).astype(np.float64)
    # Wrap numpy arrays as ROOT RVecs so RDataFrame can consume them directly
    df = ROOT.RDF.FromNumpy({"true_pt": true_pt, "true_eta": true_eta})
    # Get number of slots used by this dataframe (correct approach)
    n_slots = df.GetNSlots()
    banner(f"Implicit MT enabled  |  slots used = {n_slots}")
    print(f"\n  Dataset: {N_EVENTS:,} events  |  columns: true_pt, true_eta")
    #  USE CASE A: Per-slot random-number generation
    #
    # Goal: apply a Gaussian smearing to true_pt that mimics detector
    # resolution.  The smearing must be applied independently per entry, but
    # the RNG state is mutable — sharing one RNG across threads would cause
    # a data race (non-deterministic results, potential crashes).
    #
    # Wrong approach (DO NOT DO THIS):
    #   shared_rng = np.random.default_rng(seed=0)          # one RNG for all threads
    #   df.Define("smeared_pt",
    #             lambda pt: pt + shared_rng.normal(0, 0.01*pt),  # RACE CONDITION
    #             ["true_pt"])
    #
    # Correct approach — one RNG per slot, indexed via rdfslot_:
    banner("Use case A: per-slot RNG for Gaussian smearing")
    # Allocate one independent RNG per slot.  Each has a distinct seed so the
    # sequences do not overlap, giving us both thread-safety and statistical
    # independence between slots.
    slot_rngs = [
        np.random.default_rng(seed=slot_id + 1)
        for slot_id in range(n_slots)
    ]
    # Relative pT resolution: 1 % of true_pt (a typical inner-tracker value)
    PT_RESOLUTION = 0.01
    def smear_pt(slot: int, pt: float) -> float:
        """Return a smeared pT drawn from N(pt, PT_RESOLUTION*pt).
        `slot` is the RDataFrame slot index forwarded from rdfslot_.
        Indexing into slot_rngs[slot] is safe because each slot
        is assigned to exactly one thread at a time.
        """
        return float(slot_rngs[slot].normal(loc=pt, scale=PT_RESOLUTION * pt))
    # "rdfslot_" must appear first in the column list; it maps to `slot`.
    df_smeared = df.Define("smeared_pt", smear_pt, ["rdfslot_", "true_pt"])
    # Compute summary statistics to verify the smearing is correct
    mean_true    = df_smeared.Mean("true_pt").GetValue()
    mean_smeared = df_smeared.Mean("smeared_pt").GetValue()
    std_true     = df_smeared.StdDev("true_pt").GetValue()
    std_smeared  = df_smeared.StdDev("smeared_pt").GetValue()
    print(f"\n  true_pt    : mean = {mean_true:.4f}  std = {std_true:.4f}")
    print(f"  smeared_pt : mean = {mean_smeared:.4f}  std = {std_smeared:.4f}")
    print(
        f"\n  Resolution check: std(smeared - true) / mean(true) "
        f"= {(std_smeared**2 - std_true**2)**0.5 / mean_true * 100:.2f} %  "
        f"(expected ~{PT_RESOLUTION*100:.1f} %)"
    )
    # Show a few rows so the reader can see rdfslot_ in action
    print("\n  Sample rows (slot | true_pt | smeared_pt):")
    display_a = df_smeared.Define(
        "slot_id", "rdfslot_"          # expose slot index as a named column
    ).Display(["slot_id", "true_pt", "smeared_pt"], nRows=8)
    display_a.Print()
    #  USE CASE B: Per-slot histograms (lock-free filling + manual merge)
    #
    # Goal: build a pT spectrum histogram in a multi-threaded loop without
    # placing a mutex around every Fill() call.
    #
    # Wrong approach (DO NOT DO THIS in a DefineSlot / lambda context):
    #   shared_hist = ROOT.TH1D("h_shared", "", 60, 20, 80)
    #   df.Foreach(lambda pt: shared_hist.Fill(pt), ["smeared_pt"])
    #   # TH1::Fill is NOT thread-safe — concurrent fills corrupt bin counts.
    #
    # Correct approach — one histogram per slot, merged after the loop:
    banner("Use case B: per-slot histograms (lock-free filling)")
    # Allocate one TH1D per slot.  Names must be unique for ROOT's object
    # management; appending the slot index is the standard convention.
    PT_BINS, PT_LO, PT_HI = 60, 20.0, 80.0
    slot_hists = [
        ROOT.TH1D(f"h_pt_slot{s}", f"Slot {s} pT spectrum", PT_BINS, PT_LO, PT_HI)
        for s in range(n_slots)
    ]
    # Prevent ROOT from taking ownership so Python keeps control
    for h in slot_hists:
        ROOT.SetOwnership(h, False)
    def fill_slot_hist(slot: int, pt: float) -> None:
        """Fill the histogram that belongs to this slot.
        Because each slot maps to one thread, slot_hists[slot].Fill()
        is never called concurrently on the same histogram — no mutex needed.
        """
        slot_hists[slot].Fill(pt)
    # Foreach is the natural choice for side-effecting operations like Fill.
    # rdfslot_ is forwarded as the first argument exactly as in DefineSlot.
    df_smeared.Foreach(fill_slot_hist, ["rdfslot_", "smeared_pt"])
    # ---- Merge all per-slot histograms into one ----------------------------
    h_merged = slot_hists[0].Clone("h_pt_merged")
    ROOT.SetOwnership(h_merged, False)
    h_merged.SetTitle("Merged pT spectrum (all slots)")
    for h in slot_hists[1:]:
        h_merged.Add(h)
    total_entries = int(h_merged.GetEntries())
    peak_bin      = h_merged.GetMaximumBin()
    peak_pt       = h_merged.GetBinCenter(peak_bin)
    print(f"\n  Merged histogram: {total_entries:,} entries  |  peak at pT ≈ {peak_pt:.1f} GeV")
    print(f"  Per-slot entry counts:")
    for s, h in enumerate(slot_hists):
        print(f"    slot {s:2d}  →  {int(h.GetEntries()):>8,} entries")
    # Verify no entries were lost in the merge
    slot_total = sum(int(h.GetEntries()) for h in slot_hists)
    assert total_entries == slot_total, (
        f"Entry count mismatch: merged={total_entries}, sum-of-slots={slot_total}"
    )
    print(f"\n  Integrity check PASSED: merged entries == sum of slot entries ({total_entries:,})")
    #  USE CASE C: Per-slot accumulators (thread-safe running sum)
    #
    # A lighter-weight alternative to histograms when you only need a scalar
    # aggregate.  Each slot accumulates its own partial sum; the totals are
    # added after the event loop.
    #
    # This mirrors what ROOT's built-in actions (Sum, Mean, etc.) do
    # internally — exposed here so you can implement custom aggregations.
    banner("Use case C: per-slot scalar accumulators")
    # ctypes arrays are mutable and indexable from Python lambdas
    slot_sums    = (ctypes.c_double * n_slots)(*([0.0] * n_slots))
    slot_counts  = (ctypes.c_longlong * n_slots)(*([0]  * n_slots))
    def accumulate(slot: int, pt: float, eta: float) -> None:
        """Add pt to the running sum for this slot (no mutex required)."""
        slot_sums[slot]   += pt
        slot_counts[slot] += 1
    df_smeared.Foreach(accumulate, ["rdfslot_", "smeared_pt", "true_eta"])
    grand_sum   = sum(slot_sums)
    grand_count = sum(slot_counts)
    computed_mean = grand_sum / grand_count
    # Cross-check against RDataFrame's own Mean action
    ref_mean = df_smeared.Mean("smeared_pt").GetValue()
    print(f"\n  Manual mean (via per-slot accumulators) : {computed_mean:.6f}")
    print(f"  RDataFrame Mean('smeared_pt')           : {ref_mean:.6f}")
    print(f"  Difference                              : {abs(computed_mean - ref_mean):.2e}")
    print(f"\n  Per-slot partial sums:")
    for s in range(n_slots):
        partial_mean = slot_sums[s] / slot_counts[s] if slot_counts[s] else float("nan")
        print(f"    slot {s:2d}  →  {slot_counts[s]:>8,} entries  partial mean = {partial_mean:.4f}")
    #  Summary
    banner("Summary")
    print("""
  The implicit column rdfslot_ is the Python equivalent of the `slot`
  argument provided by C++ DefineSlot / RedefineSlot.  By listing
  "rdfslot_" first in the column list and receiving it as the first
  parameter of any callable (Define or Foreach), you can:
    • Index into per-slot RNG instances → lock-free random smearing
    • Fill per-slot histograms          → lock-free histogram filling
    • Accumulate per-slot partial sums  → lock-free custom aggregations
  In every case the pattern is:
      per_slot_resource = [Resource(seed=s) for s in range(n_slots)]
      def my_func(slot, *columns):
          # per_slot_resource[slot] is owned by exactly one thread
          return per_slot_resource[slot].compute(*columns)
      df.Define("result", my_func, ["rdfslot_", "col_a", "col_b", ...])
  This pattern serves as a practical replacement for DefineSlot until that API
  becomes available in PyROOT.
""")
    print(" This pattern avoids locks and ensures deterministic, thread-safe execution.")
if __name__ == "__main__":
    main()
