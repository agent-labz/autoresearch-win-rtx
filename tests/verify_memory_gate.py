r"""Regression check for the autotune memory gate fix in autoresearch-win-rtx.

Closes AUT-7 (audit finding A8): the autotune gated a candidate on 90% of the
card's *total* VRAM, which cannot see another process, so it could green-light a
candidate sized against 14.4 GiB when 2-10 GiB was actually free -- and on WDDM
an allocation that does not fit does not raise OutOfMemoryError, it spills to
shared memory, runs 5-30x slower, and still produces a val_bpb to log.

Run from the fork worktree:

    uv run --no-sync python tests/verify_memory_gate.py

No training, no eval, no data, no checkpoint written. It calls the fork's own
decision functions, so it proves the shipped behaviour rather than a restatement
of it. Every assertion prints a PASS/FAIL line; the script exits non-zero on any
failure.

What it pins down:

1. `torch.cuda.mem_get_info()` is a per-process view on WDDM and does not move
   when another process takes VRAM. This is the measurement that decided the
   design: the issue's first proposed fix (gate on a fraction of `mem_get_info`'s
   free) cannot work on this platform, so the gate reads the driver instead.
2. The driver read is device-wide, fail-soft, and cheap.
3. The budget is a fraction of device-wide free memory, clamped to total, and
   degrades to total when the driver cannot be asked.
4. The physical ceiling rejects a commit larger than the card, and the
   free-memory budget rejects a commit that does not fit right now.
5. A cache hit is re-validated against the card as it is now. This is the
   common path -- the shipped cache skips measurement entirely.
6. The spilling candidate is rejected on live hardware.
7. On an idle card the new gate accepts everything the shipped gate accepted, so
   the fix costs no throughput when there is nothing to avoid.
8. The run-time guard raises its own type and `main()` catches it before the
   bare `RuntimeError` handler, so a spill drops to a smaller batch instead of
   becoming a hard exit.
"""

import os
import sys
import types

sys.path.insert(0, os.getcwd())

FAILURES = []
CHECKS = 0

GIB = 1024 ** 3
MIB = 1024 ** 2
ADA = (8, 9)

# Measured on this box, 2026-09-27, torch 2.9.1+cu128, RTX 4060 Ti.
MEASURED_IDLE = {
    # (batch, checkpointing): (peak_allocated, peak_reserved, tok_per_sec)
    (32, False): (22_822_291_712, 25_620_905_984, None),
    (16, False): (12_135_700_480, 12_914_262_016, 47_256.4),
    (8, False): (6_238_115_840, 6_635_388_928, 47_848.9),
    (4, False): (3_289_323_520, 3_516_923_904, 48_421.4),
    (32, True): (10_353_905_664, 13_316_915_200, 35_884.0),
    (16, True): (5_286_400_000, 6_807_355_392, 38_288.3),
    (8, True): (2_752_647_168, 3_313_500_160, 38_896.2),
    (4, True): (1_485_770_752, 1_839_202_304, 39_289.1),
}


def check(label, condition, detail=""):
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(': ' + detail) if detail else ''}")
        FAILURES.append(label)


def section(title):
    print(f"\n=== {title} ===")


def stub_runtime(total_bytes, name="NVIDIA GeForce RTX 4060 Ti", cc=ADA, vram_gb=None):
    return types.SimpleNamespace(
        gpu_total_memory_bytes=total_bytes,
        gpu_name=name,
        gpu_cc=cc,
        gpu_vram_gb=total_bytes / GIB if vram_gb is None else vram_gb,
    )


def main():
    import subprocess

    import torch

    import train

    props = torch.cuda.get_device_properties(0)
    total_bytes = int(props.total_memory)
    gpu_name = torch.cuda.get_device_name()
    live = stub_runtime(total_bytes, gpu_name, torch.cuda.get_device_capability())
    source = open("train.py", encoding="utf-8").read()
    main_body = source.split("def main()")[1]
    train_body = source.split("def _run_training_once")[1].split("def main()")[0]
    probe_body = source.split("def _benchmark_train_candidate")[1].split("def _autotune_train_candidate")[0]
    print(f"  info  {gpu_name}, total_memory={total_bytes} B = {total_bytes / GIB:.6f} GiB")

    section("1. torch.cuda.mem_get_info() does not see another process (design input)")
    own_free, mem_total = torch.cuda.mem_get_info()
    check(
        "mem_get_info total matches props.total_memory",
        mem_total == total_bytes,
        f"{mem_total} vs {total_bytes}",
    )
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    smi_used, smi_free = (int(v) for v in smi.split(","))
    print(
        f"  info  this process: mem_get_info free={own_free / GIB:.3f} GiB, "
        f"reserved={torch.cuda.memory_reserved() / MIB:.1f} MiB | "
        f"nvidia-smi: used={smi_used} MiB free={smi_free} MiB"
    )
    # A hog is started by the caller and is holding VRAM right now.
    hog_mib = int(os.environ.get("VERIFY_GB_HOG_MIB", "0"))
    if hog_mib:
        check(
            "another process really is on the card",
            smi_used > hog_mib,
            f"nvidia-smi used={smi_used} MiB, expected more than {hog_mib} MiB",
        )
        driver_free_mib = smi_free
        check(
            "nvidia-smi sees the other process",
            driver_free_mib < total_bytes / MIB - 512,
            f"free={driver_free_mib} MiB of {total_bytes / MIB:.0f} MiB",
        )
        check(
            "mem_get_info is blind to it (measured 0.000 GiB movement on this box)",
            own_free > driver_free_mib * MIB,
            f"mem_get_info free={own_free / GIB:.3f} GiB vs driver free={driver_free_mib * MIB / GIB:.3f} GiB",
        )
        check(
            "the driver read is the lower, honest number the gate must use",
            train._device_free_memory_bytes(gpu_name) == driver_free_mib * MIB,
            f"{train._device_free_memory_bytes(gpu_name)} vs {driver_free_mib * MIB}",
        )
    else:
        print("  info  no hog; set VERIFY_GB_HOG_MIB to the hog's size to run the contended checks")

    section("2. the driver read is device-wide, fail-soft and cheap")
    free_now = train._device_free_memory_bytes(gpu_name)
    check("driver read succeeds on this box", isinstance(free_now, int) and free_now > 0)
    check("driver read is under total", free_now < total_bytes)
    real_run = subprocess.run

    def boom(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    try:
        subprocess.run = boom
        check("a missing nvidia-smi returns None, not a guess", train._device_free_memory_bytes(gpu_name) is None)

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        subprocess.run = timeout
        check("a hanging nvidia-smi returns None", train._device_free_memory_bytes(gpu_name) is None)

        def garbage(*args, **kwargs):
            return types.SimpleNamespace(stdout="not,numbers\n", returncode=0)

        subprocess.run = garbage
        check("unparseable output returns None", train._device_free_memory_bytes(gpu_name) is None)

        def empty(*args, **kwargs):
            return types.SimpleNamespace(stdout="", returncode=0)

        subprocess.run = empty
        check("empty output returns None", train._device_free_memory_bytes(gpu_name) is None)
    finally:
        subprocess.run = real_run
    check("the query has a timeout so a wedged driver cannot hang a run", train.DEVICE_QUERY_TIMEOUT_S > 0)
    import time as _time

    t0 = _time.time()
    for _ in range(5):
        train._device_free_memory_bytes(gpu_name)
    per_query_ms = (_time.time() - t0) / 5 * 1000
    check("the query is cheap enough to run per candidate", per_query_ms < 500, f"{per_query_ms:.0f} ms")
    print(f"  info  {per_query_ms:.0f} ms per query")

    section("3. the budget is a fraction of free memory, clamped, and degrades to total")
    frac = train.AUTOTUNE_MAX_MEMORY_FRACTION
    check("the fraction is unchanged from the shipped gate", frac == 0.90, f"{frac}")
    budget, basis = train._memory_budget_bytes(live, None)
    check("no driver read falls back to total VRAM", basis == "total VRAM")
    check(
        "the fallback budget equals the old gate exactly",
        budget == total_bytes * frac,
        f"{budget} vs {total_bytes * frac}",
    )
    for free_gib in (2.0, 4.9, 9.0, 14.3):
        budget, basis = train._memory_budget_bytes(live, int(free_gib * GIB))
        check(
            f"free={free_gib} GiB -> {free_gib * frac:.2f} GiB budget off free VRAM",
            basis == "device-wide free VRAM" and abs(budget - free_gib * frac * GIB) < 1,
            f"got {budget / GIB:.3f} GiB off {basis}",
        )
    budget, basis = train._memory_budget_bytes(live, total_bytes * 4)
    check(
        "a driver reporting more free than the card has is clamped to total",
        basis == "device-wide free VRAM" and budget == total_bytes * frac,
        f"{budget / GIB:.3f} GiB",
    )
    check(
        "a contended card gets a strictly smaller budget than an idle one",
        train._memory_budget_bytes(live, int(5.0 * GIB))[0] < train._memory_budget_bytes(live, int(15.0 * GIB))[0],
    )

    section("4. the gate rejects a spill and a commit that does not fit now")
    shipped_gate = total_bytes * frac
    for label, peak, want_reject in (
        ("7 GiB commit on a 15.996 GiB card is inside the old gate", 7 * GIB, False),
        ("14.4 GiB commit is exactly at the old gate", shipped_gate, False),
        ("14.5 GiB commit is over the old gate", shipped_gate + 1, True),
        ("24 GiB commit is more than the card physically has", 24 * GIB, True),
    ):
        check(
            f"{label} -> {'rejected' if want_reject else 'accepted'}",
            (train._memory_gate_rejection(live, peak, shipped_gate, "total VRAM") is not None) == want_reject,
        )
    over_card = train._memory_gate_rejection(live, 24 * GIB, 10 * GIB, "device-wide free VRAM")
    check("the physical ceiling is reported before the budget", over_card is not None and "more than" in over_card)
    over_budget = train._memory_gate_rejection(live, 6 * GIB, 4 * GIB, "device-wide free VRAM")
    check(
        "an under-card commit over the free budget is rejected for being over budget",
        over_budget is not None and "budget" in over_budget and "more than the card" not in over_budget,
        f"got {over_budget!r}",
    )
    check(
        "the reason carries no prefix of its own, so the probe and the cache can each label it",
        not over_budget.startswith("Autotune candidate rejected")
        and not over_budget.startswith("Ignoring cached")
        and 'print(f"Autotune candidate rejected: {rejection}")' in probe_body
        and 'print(f"Ignoring cached autotune candidate: {cached_reason}.")' in source,
    )
    check(
        "an under-card commit inside the free budget is accepted",
        train._memory_gate_rejection(live, 6 * GIB, 10 * GIB, "device-wide free VRAM") is None,
    )

    section("5. a cache hit is re-validated against the card as it is now")
    candidates = [(32, False), (16, False), (8, False), (4, False)]
    idle_budget, _ = train._memory_budget_bytes(live, int(15.0 * GIB))
    contended_budget, _ = train._memory_budget_bytes(live, int(5.0 * GIB))
    entry = {
        "train_batch_size": 16,
        "use_activation_checkpointing": False,
        "tok_per_sec": 47256.4,
        "peak_reserved_bytes": 12_914_262_016,
    }
    picked, reason = train._cached_candidate(live, entry, candidates, idle_budget, "device-wide free VRAM")
    check("a cached candidate that still fits is used", picked == (16, False), f"{picked} / {reason}")
    picked, reason = train._cached_candidate(live, entry, candidates, contended_budget, "device-wide free VRAM")
    check(
        "the same cached candidate is refused once the card is 10 GiB busier",
        picked is None and "budget" in reason,
        f"{picked} / {reason}",
    )
    check("the cache is on a version whose entries carry the committed mark", train.AUTOTUNE_CACHE_VERSION == "gpu-profile-v4", train.AUTOTUNE_CACHE_VERSION)
    legacy = {"train_batch_size": 16, "use_activation_checkpointing": False, "peak_memory_bytes": 12_135_700_480}
    picked, reason = train._cached_candidate(live, legacy, candidates, idle_budget, "total VRAM")
    check(
        "a pre-fix entry (no committed mark) cannot pin a decision",
        picked is None and "committed" in reason,
        f"{picked} / {reason}",
    )
    picked, reason = train._cached_candidate(live, {**entry, "train_batch_size": 64}, candidates, idle_budget, "total VRAM")
    check("an out-of-profile cached candidate is refused", picked is None and "candidate set" in reason)
    picked, reason = train._cached_candidate(live, None, candidates, idle_budget, "total VRAM")
    check("a missing entry is refused", picked is None)
    for bogus in ("nope", 7, [1, 2]):
        picked, _ = train._cached_candidate(live, bogus, candidates, idle_budget, "total VRAM")
        check(f"a {type(bogus).__name__} entry is refused", picked is None)

    section("6. the spilling candidate is rejected on live hardware")
    candidates_live = train._build_train_candidates(
        train.RuntimeConfig(
            device=torch.device("cuda"),
            device_type="cuda",
            amp_dtype=torch.bfloat16,
            use_compile=False,
            use_activation_checkpointing=False,
            attention_backend="sdpa",
            gpu_name=gpu_name,
            gpu_vram_gb=total_bytes / GIB,
            gpu_peak_flops=None,
            gpu_cc=torch.cuda.get_device_capability(),
            gpu_total_memory_bytes=total_bytes,
            tf32_enabled=True,
            gpu_profile=train._resolve_gpu_profile(gpu_name, torch.cuda.get_device_capability(), total_bytes / GIB, True),
        )
    )
    print(f"  info  live candidates={candidates_live}")
    check("(32, False) is in the live candidate set", (32, False) in candidates_live)
    measured = MEASURED_IDLE[(32, False)]
    check(
        "B=32 without checkpointing really does commit more than the card (A8, reproduced)",
        measured[1] > total_bytes,
        f"reserved={measured[1] / GIB:.2f} GiB vs card {total_bytes / GIB:.2f} GiB",
    )
    check(
        "the shipped gate would have rejected it, and the physical ceiling still does",
        train._memory_gate_rejection(live, measured[0], shipped_gate, "total VRAM") is not None
        and train._memory_gate_rejection(live, measured[1], shipped_gate, "total VRAM") is not None,
    )
    check(
        "the allocator counters alone cannot see a spill under the card total",
        train._memory_gate_rejection(live, measured[0], shipped_gate, "total VRAM") is not None
        and MEASURED_IDLE[(16, False)][0] < total_bytes,
        "B=16 allocated 11.6 GiB on a 15.996 GiB card and spilled anyway under contention",
    )

    section("7. no throughput is lost on an idle card")
    if hog_mib:
        print("  info  card is contended; section 7 is an idle-card check and is skipped")
        section("8. a contended card rejects what the shipped gate accepted")
        driver_free = train._device_free_memory_bytes(gpu_name)
        contended_budget, basis = train._memory_budget_bytes(live, driver_free)
        print(f"  info  driver free={driver_free / GIB:.3f} GiB -> budget {contended_budget / GIB:.3f} GiB off {basis}")
        newly_rejected = []
        for (batch, ckpt), (alloc, reserved, _tok) in MEASURED_IDLE.items():
            if alloc <= shipped_gate and reserved > contended_budget:
                newly_rejected.append((batch, ckpt))
        check(
            "the shipped gate would have accepted candidates that cannot fit now",
            bool(newly_rejected),
            f"{newly_rejected}",
        )
        check(
            "the new gate refuses all of them",
            all(
                train._memory_gate_rejection(live, MEASURED_IDLE[c][1], contended_budget, basis) is not None
                for c in newly_rejected
            ),
        )
        still_ok = [c for c, (_a, r, _t) in MEASURED_IDLE.items() if r <= contended_budget]
        print(f"  info  still fits at this contention level: {sorted(still_ok)}")
        check("at least one candidate survives, so the run can still produce a number", bool(still_ok))
    else:
        idle_free = train._device_free_memory_bytes(gpu_name)
        idle_budget_real, _ = train._memory_budget_bytes(live, idle_free)
        print(f"  info  device free={idle_free / GIB:.3f} GiB -> budget {idle_budget_real / GIB:.3f} GiB")
        shipped_accepts, new_accepts = [], []
        for (batch, ckpt), (alloc, reserved, _tok) in MEASURED_IDLE.items():
            old_ok = alloc <= shipped_gate
            new_ok = train._memory_gate_rejection(live, reserved, idle_budget_real, "device-wide free VRAM") is None
            shipped_accepts.append((batch, ckpt) if old_ok else None)
            new_accepts.append((batch, ckpt) if new_ok else None)
        lost = [c for c in shipped_accepts if c is not None and c not in new_accepts]
        check(
            "every candidate the shipped gate accepted on an idle card is still accepted",
            not lost,
            f"lost {lost}",
        )
        fastest_old = max(
            (c for c in shipped_accepts if c is not None),
            key=lambda c: MEASURED_IDLE[c][2] if MEASURED_IDLE[c][2] else -1,
        )
        fastest_new = max(
            (c for c in new_accepts if c is not None),
            key=lambda c: MEASURED_IDLE[c][2] if MEASURED_IDLE[c][2] else -1,
        )
        check("the selected candidate is unchanged", fastest_old == fastest_new, f"{fastest_old} vs {fastest_new}")
        print(f"  info  shipped gate accepted {sorted(c for c in shipped_accepts if c)}")
        print(f"  info  new gate accepted     {sorted(c for c in new_accepts if c)}")
        print(f"  info  fastest candidate     {fastest_new}")
        check(
            "gating on reserved rather than allocated costs a candidate only when fragmentation is real",
            all(
                MEASURED_IDLE[c][1] <= idle_budget_real
                for c in new_accepts
                if c is not None
            ),
        )

    section("9. the run-time guard raises its own type and is caught first")
    check(
        "GpuRunNotTrustworthyError is a RuntimeError so it cannot escape the old handlers",
        issubclass(train.GpuRunNotTrustworthyError, RuntimeError),
    )
    spill_handler = main_body.find("except GpuRunNotTrustworthyError as exc:")
    bare_handler = main_body.find("except RuntimeError as exc:")
    oom_handler = main_body.find("except torch.cuda.OutOfMemoryError:")

    check("main() catches the spill type at all", spill_handler > 0)
    check("the spill type is caught before the bare RuntimeError handler", 0 < spill_handler < bare_handler)
    check("the spill type is caught before the OOM handler too", 0 < spill_handler < oom_handler)
    check(
        "the spill type falls through to the next candidate rather than exiting",
        "Abandoned candidate" in main_body and "trying next candidate" in main_body,
    )
    check(
        "a run abandoned for every candidate fails loudly instead of logging nothing",
        "FAIL: training failed for all batch size candidates." in main_body
        and "for reason in untrustworthy_reasons" in main_body,
    )
    check(
        "the physical commit ceiling is checked inside the training loop",
        "committed = torch.cuda.max_memory_reserved()" in train_body
        and "committed > runtime.gpu_total_memory_bytes" in train_body,
    )
    check(
        "the loop check is not the probe's: the probe rejects before it ever returns",
        "committed = torch.cuda.max_memory_reserved()" not in probe_body
        and "max_memory_reserved()" in probe_body,
    )
    check(
        "the run-time guard's abort decision needs no knowledge of other processes",
        "committed > runtime.gpu_total_memory_bytes" in train_body
        and "RUN_MIN_THROUGHPUT_RATIO * reference_tok_per_sec" in train_body,
    )
    check(
        "the driver read inside the training loop feeds reporting, not the abort decision",
        "lowest_free_bytes" in train_body
        and train_body.index("lowest_free_bytes = _device_free_memory_bytes") < train_body.index("committed = torch.cuda.max_memory_reserved()"),
        "the sample is recorded and returned; neither raise reads it",
    )
    check(
        "the guard is armed from that same candidate's own probed rate, not the winner's",
        "reference_tok_per_sec=reference_tok_per_sec" in main_body
        and 'autotuned_rates.get((train_batch_size, use_checkpointing))' in main_body
        and "RUN_MIN_THROUGHPUT_RATIO * reference_tok_per_sec" in train_body,
    )
    check(
        "a candidate with no probed rate is run on the physical ceiling alone",
        "autotuned_rates.get(" in main_body and "reference_tok_per_sec=None" in source,
    )
    check(
        "the rate threshold is well below any plausible clock or thermal drift",
        0.1 < train.RUN_MIN_THROUGHPUT_RATIO < 0.5,
        f"{train.RUN_MIN_THROUGHPUT_RATIO}",
    )
    check("a single slow step cannot abandon a run", train.RUN_SLOW_STEP_LIMIT >= 3)
    check(
        "the guard starts at step 0, because a spilling candidate is slow from step 0",
        train.RUN_RATE_CHECK_START_STEP == 0,
        f"{train.RUN_RATE_CHECK_START_STEP}",
    )
    # Measured tok/sec by step, same candidate (4, False) for the first two:
    #   idle control   47,856 / 47,809 / 47,852 / 47,897 -> flat, no warmup
    #   10 GiB hog     21,450 / 21,566 / 21,428 / 21,427 -> flat 2.2x slower, still fits
    # Then (16, False), which needs 12.03 GiB with 5.0 GiB free, so it spills:
    #                  7,687 / 7,730 / 8,177               -> 5.8x slower, at the START
    idle_flat = [47856, 47809, 47852, 47897]
    fits_slower = [21450, 21566, 21428, 21427]
    spilling = [7687, 7730, 8177]
    threshold = train.RUN_MIN_THROUGHPUT_RATIO * idle_flat[0]
    check("an uncontended run is flat from step 0, so there is no warmup to skip", max(idle_flat) - min(idle_flat) < 0.01 * idle_flat[0], f"{idle_flat}")
    check("contention that still fits is uniform from step 0, not a transient", max(fits_slower) - min(fits_slower) < 0.01 * fits_slower[0], f"{fits_slower}")
    check("a spilling candidate is already slow at step 0, so a warmup allowance would miss it", spilling[0] < threshold, f"{spilling} vs {threshold:.0f}")
    check(
        "a run that only fits but is contended is above the threshold, so it is not abandoned",
        min(fits_slower) > threshold,
        f"{min(fits_slower)} vs {threshold:.0f}",
    )
    check(
        "a spilling run is below the threshold at every step it takes, so it is abandoned",
        max(spilling) < threshold,
        f"{max(spilling)} vs {threshold:.0f}",
    )
    check(
        "the margin between the two cases is wide, not marginal",
        min(fits_slower) / max(spilling) > 2.0,
        f"ratio {min(fits_slower) / max(spilling):.2f}x",
    )
    check(
        "the rate guard is off when there is no probe to compare against",
        "reference_tok_per_sec=None" in source,
    )
    check(
        "contention is printed at startup, at its worst during training, and again at completion",
        source.count("_report_device_contention(") == 4
        and 'prefix="(worst seen during training) "' in source
        and 'prefix="(at completion) "' in source,
    )
    check(
        "contention is sampled during the run, because it can appear after startup",
        "CONTENTION_SAMPLE_EVERY" in source
        and "(step + 1) % CONTENTION_SAMPLE_EVERY == 0" in train_body
        and "lowest_free_memory_bytes" in train_body,
    )
    check(
        "the sample interval is coarse enough not to tax the loop",
        train.CONTENTION_SAMPLE_EVERY >= 10,
        f"{train.CONTENTION_SAMPLE_EVERY}",
    )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
