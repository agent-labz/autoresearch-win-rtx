"""AUT-26 step 3: exercise the 34% rate guard independently of verify_memory_gate.py.

The guard predicate is transcribed verbatim from the `RUN_MIN_THROUGHPUT_RATIO`
block in `train.py::_run_training_once` so this probe tests the shipped rule,
not a paraphrase of it. The point is to separate the three outcomes the AUT-26
issue says "the suite passes" cannot distinguish:

  - never fires   (a slow run is recorded as data)
  - always fires  (a good run is thrown away)
  - fires exactly on the contended case and nothing else

Run from the repository root, with the repo's own interpreter:

    uv run --no-sync python tests/aut26_guard_probe.py
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import train

RATIO = train.RUN_MIN_THROUGHPUT_RATIO
LIMIT = train.RUN_SLOW_STEP_LIMIT
START = train.RUN_RATE_CHECK_START_STEP
TOTAL_BATCH = train.TOTAL_BATCH_SIZE

fails = []


def check(label, cond):
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        fails.append(label)


def guard(reference, measured_series):
    """train.py:1531-1544, transcribed verbatim. Returns the step that aborts, or None."""
    slow_steps = 0
    for step, tok_per_sec in enumerate(measured_series):
        if reference and step >= START:
            if tok_per_sec < RATIO * reference:
                slow_steps += 1
                if slow_steps >= LIMIT:
                    return step
            else:
                slow_steps = 0
    return None


def _unused(reference, measured_series):
    slow_steps = 0
    for step, tok_per_sec in enumerate(measured_series):
        if step < START:
            continue
        if tok_per_sec < RATIO * reference:
            slow_steps += 1
            if slow_steps >= LIMIT:
                return step
        else:
            slow_steps = 0
    return None


print("=== A. the shipped constants ===")
print(f"  RUN_MIN_THROUGHPUT_RATIO = {RATIO}")
print(f"  RUN_SLOW_STEP_LIMIT      = {LIMIT}")
print(f"  RUN_RATE_CHECK_START_STEP= {START}")
print(f"  TOTAL_BATCH_SIZE         = {TOTAL_BATCH:,}")
print(f"  discard threshold        = {RATIO:.0%} of probe, for {LIMIT} consecutive steps")

print("\n=== B. the real spilling run from the fix's own measurements ===")
# These are the numbers the commit cites as the reason for the constant:
# 7,687 / 7,730 / 8,177 tok/sec at steps 0-2 against a 47,443 tok/sec probe.
PROBE = 47443.0
SPILL = [7687.0, 7730.0, 8177.0]
print(f"  probe = {PROBE:,.0f} tok/sec; measured steps = "
      + ", ".join(f"{t:,.0f}" for t in SPILL))
print(f"  as a fraction of probe: " + ", ".join(f"{t/PROBE:.1%}" for t in SPILL))
abort = guard(PROBE, SPILL + [PROBE] * 40)
check("a run spilling into shared memory is ABANDONED", abort is not None)
check(f"it abandons on the 3rd slow step, not before (got step {abort})",
      abort == START + LIMIT - 1)

print("\n=== C. the real uncontended run, same card, same candidate ===")
# 47,856 at step 0 against 47,846 at step 7 on an idle card.
IDLE = [47856.0, 47846.0] + [47850.0] * 60
check("an uncontended run is ALLOWED TO FINISH", guard(PROBE, IDLE) is None)

print("\n=== D. the margin, not the verdict ===")
print(f"  contended runs measure ~{sum(SPILL)/3/PROBE:.0%} of probe")
print(f"  guard fires below                     {RATIO:.0%} of probe")
print(f"  margin on the contended side          {RATIO - sum(SPILL)/3/PROBE:+.0%} of probe")
check("the contended case clears the threshold by a wide margin, not marginally",
      RATIO - sum(SPILL) / 3 / PROBE > 0.10)

print("\n=== E. boundary sweep: where exactly does it flip? ===")
prev = None
flip = None
for pct in range(20, 51):
    rate = PROBE * pct / 100.0
    fired = guard(PROBE, [rate] * 10) is not None
    if prev is not None and fired != prev:
        flip = pct
    if pct in (33, 34, 35):
        print(f"  {pct:>2}% of probe -> "
              + ("ABANDONED" if fired else "allowed to finish"))
    prev = fired
print(f"  guard flips between {flip - 1}% and {flip}% of probe")
check("the flip is at the 34% constant, not somewhere else", flip == 34)

print("\n=== F. strict inequality: the threshold step itself survives ===")
# The predicate is `<`, not `<=`. A run sitting exactly on the line is kept.
exact = RATIO * PROBE
check("a run at exactly 34% of probe is allowed to finish (predicate is `<`)",
      guard(PROBE, [exact] * 10) is None)
check("a run at 33.9% of probe is abandoned",
      guard(PROBE, [PROBE * 0.339] * 10) is not None)

print("\n=== G. 'consecutive' actually means consecutive ===")
check("2 slow steps then a fast one does NOT abandon (3rd is fast)",
      guard(PROBE, [PROBE * 0.2, PROBE * 0.2, PROBE, PROBE * 0.2, PROBE * 0.2,
                    PROBE, PROBE * 0.2, PROBE * 0.2, PROBE]) is None)
check("3 slow steps in a row DO abandon, even if a later step is fast",
      guard(PROBE, [PROBE * 0.2] * 3 + [PROBE] * 10) is not None)
check("the counter resets: slow,slow,fast,slow,slow,fast,slow,slow,slow,slow still fires",
      guard(PROBE, [PROBE * 0.2, PROBE * 0.2, PROBE, PROBE * 0.2, PROBE * 0.2,
                    PROBE, PROBE * 0.2, PROBE * 0.2, PROBE * 0.2, PROBE * 0.2]) is not None)

print("\n=== H. a run that only fits, but is contended, survives ===")
# 0.10-0.33 of probe: slow enough to look bad, fast enough to have made it.
check("a contended-but-fitting run at 20% of probe is still abandoned",
      guard(PROBE, [PROBE * 0.20] * 10) is not None)
check("a mildly slow run at 90% of probe is allowed to finish",
      guard(PROBE, [PROBE * 0.90] * 10) is None)

print("\n=== I. no probe means no guard, and that is honest ===")
# train.py:1531 gates the whole block on `reference_tok_per_sec` being truthy.
check("the guard is disarmed when there is no probe (reference is falsy)",
      guard(None, [PROBE * 0.001] * 100) is None)
check("the guard is disarmed when the probe is 0",
      guard(0, [PROBE * 0.001] * 100) is None)
print("  NOTE: with no probe the run is kept. That is the residual risk")
print("        AUT-7 leaves open, and it is why step 4 of this issue matters.")

print("\n=== J. the memory budget tracks free VRAM, live, on this card ===")
runtime = train.detect_runtime()
free = train._device_free_memory_bytes(runtime.gpu_name)
total = runtime.gpu_total_memory_bytes
check("the live free-VRAM read works on this machine", free is not None and free > 0)
if free is None:
    print("  free read returned None; the gate falls back to total VRAM here")
else:
    print(f"  card = {runtime.gpu_name}")
    print(f"  free = {free / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB total")
    live_budget, basis = train._memory_budget_bytes(runtime, free)
    print(f"  budget = {live_budget / 1024**3:.2f} GiB, basis = {basis}")
    check("the budget is derived from device-wide free VRAM, not total VRAM",
          basis == "device-wide free VRAM")
    # The base's gate is a constant: 0.90 * total, whatever is actually free.
    base_budget = total * train.AUTOTUNE_MAX_MEMORY_FRACTION
    print(f"  the base's gate would say {base_budget / 1024**3:.2f} GiB regardless")
    check("the fixed gate is stricter than the base's on this card",
          live_budget < base_budget)

    print("\n  the decision that matters: what the budget does under contention")
    gib = 1024 ** 3
    print(f"    {'free VRAM':>12}  {'budget (fixed)':>16}  {'budget (base)':>15}  verdict")
    for free_gib in (16.0, 14.0, 10.0, 6.0, 4.0, 2.0, 0.5):
        b, bs = train._memory_budget_bytes(runtime, int(free_gib * gib))
        old = total * train.AUTOTUNE_MAX_MEMORY_FRACTION
        print(f"    {free_gib:>9.1f} GiB  {b / gib:>13.2f} GiB  {old / gib:>12.2f} GiB"
              f"  {bs}")
    contended, _ = train._memory_budget_bytes(runtime, int(4.0 * gib))
    check("with 4 GiB free the fixed gate allows far less than the base's 14.4 GiB",
          contended < base_budget / 2)
    check("a candidate sized against the base's budget is rejected by the fixed gate",
          train._memory_gate_rejection(
              runtime, int(base_budget * 0.98), contended, "device-wide free VRAM")
          is not None)
    check("the same candidate is accepted when the card really is that free",
          train._memory_gate_rejection(
              runtime, int(base_budget * 0.98),
              train._memory_budget_bytes(runtime, total)[0],
              "device-wide free VRAM") is None)
    # A driver reporting more free than the card has must be clamped.
    over, obasis = train._memory_budget_bytes(runtime, total * 2)
    check("a driver reporting more free than the card has is clamped to total",
          over == train._memory_budget_bytes(runtime, total)[0])

print()
if fails:
    print(f"FAILED: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("all guard-path checks passed")
sys.exit(0)
