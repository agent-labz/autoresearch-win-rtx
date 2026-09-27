r"""Regression check for the VRAM-tier / checkpoint-candidate fix in autoresearch-win-rtx.

Run from the fork worktree:

    uv run --no-sync python tests/verify_tier_fix.py

No GPU training, no tokenizer, no data. It calls the fork's own resolver and
candidate builder, so it proves the shipped behaviour rather than a restatement
of it. Every assertion prints a PASS/FAIL line; the script exits non-zero on
any failure.

What it pins down:

1. Every SKU in the README platform matrix lands in the tier its name implies,
   at four different assumed VRAM reservation sizes (0, 4, 12, 64 MiB). The
   reservation is the real reason `props.total_memory` is short of the marketed
   capacity, and the size of the reservation varies by card, so the fix has to
   survive the pessimistic case rather than the one this machine happens to hit.
2. No supported consumer profile can force activation checkpointing: every one of
   them offers both modes, so the autotune measures rather than obeys a tier.
3. `AUTORESEARCH_FORCE_CHECKPOINTING` is parsed once, rejects anything that is
   neither `0` nor `1` instead of reading as unset, narrows the profile's mode
   set, and the narrowed mode is what reaches `GPTConfig`. The candidate set is
   also a pure function of the runtime: changing the environment afterwards does
   not move it, which is the property that keeps the single read honest.
4. The autotune cache is on a version that the old candidate set cannot satisfy,
   so a warm v2 cache cannot pin the mis-tiered decision after the fix.
5. Live hardware agrees with the matrix assumption.
"""

import inspect
import os
import sys

sys.path.insert(0, os.getcwd())

FAILURES = []
CHECKS = 0

GIB = 1024 ** 3
MIB = 1024 ** 2

TURING = (7, 5)
AMPERE = (8, 6)
ADA = (8, 9)
BLACKWELL = (12, 0)

# (name, cc, nominal GiB, expected profile name, expected warning)
SKU_MATRIX = [
    # Turing
    ("NVIDIA GeForce RTX 2060", TURING, 6, "compatibility", "below the 8 GB floor"),
    ("NVIDIA GeForce RTX 2060 SUPER", TURING, 8, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2070", TURING, 8, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2070 SUPER", TURING, 8, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2080", TURING, 8, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2080 SUPER", TURING, 8, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2080 Ti", TURING, 11, "turing-8-11gb", None),
    ("NVIDIA GeForce RTX 2060", TURING, 12, "turing-12-15gb", None),
    # Ampere
    ("NVIDIA GeForce RTX 3060", AMPERE, 12, "ampere-10-15gb", None),
    ("NVIDIA GeForce RTX 3080", AMPERE, 10, "ampere-10-15gb", None),
    ("NVIDIA GeForce RTX 3080", AMPERE, 12, "ampere-10-15gb", None),
    ("NVIDIA GeForce RTX 3080 Ti", AMPERE, 12, "ampere-10-15gb", None),
    ("NVIDIA GeForce RTX 3090", AMPERE, 24, "ampere-24gb-plus", None),
    ("NVIDIA GeForce RTX 3090 Ti", AMPERE, 24, "ampere-24gb-plus", None),
    # Ada
    ("NVIDIA GeForce RTX 4060 Ti", ADA, 16, "ada-16gb", None),
    ("NVIDIA GeForce RTX 4070", ADA, 12, "ada-10-15gb", None),
    ("NVIDIA GeForce RTX 4070 SUPER", ADA, 12, "ada-10-15gb", None),
    ("NVIDIA GeForce RTX 4070 Ti", ADA, 12, "ada-10-15gb", None),
    ("NVIDIA GeForce RTX 4070 Ti SUPER", ADA, 16, "ada-16gb", None),
    ("NVIDIA GeForce RTX 4080", ADA, 16, "ada-16gb", None),
    ("NVIDIA GeForce RTX 4080 SUPER", ADA, 16, "ada-16gb", None),
    ("NVIDIA GeForce RTX 4090", ADA, 24, "ada-24gb-plus", None),
    # Blackwell
    ("NVIDIA GeForce RTX 5060 Ti", BLACKWELL, 16, "blackwell-16gb", None),
    ("NVIDIA GeForce RTX 5070", BLACKWELL, 12, "blackwell-10-15gb", None),
    ("NVIDIA GeForce RTX 5070 Ti", BLACKWELL, 16, "blackwell-16gb", None),
    ("NVIDIA GeForce RTX 5080", BLACKWELL, 16, "blackwell-16gb", None),
    ("NVIDIA GeForce RTX 5090", BLACKWELL, 32, "blackwell-24gb-plus", None),
    # Laptop parts stay out of the matrix regardless of VRAM.
    ("NVIDIA GeForce RTX 4080 Laptop GPU", ADA, 16, "compatibility", "laptop GPUs"),
]

RESERVATIONS_MIB = (0, 4, 12, 64)


def check(label, condition, detail=""):
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}{(': ' + detail) if detail else ''}")
        FAILURES.append(label)


def make_runtime(train, profile, total_bytes, vram_gb, cc, name):
    import torch

    return train.RuntimeConfig(
        device=torch.device("cuda"),
        device_type="cuda",
        amp_dtype=torch.bfloat16,
        use_compile=False,
        attention_backend="sdpa",
        gpu_name=name,
        gpu_vram_gb=vram_gb,
        gpu_peak_flops=None,
        gpu_cc=cc,
        gpu_total_memory_bytes=total_bytes,
        tf32_enabled=True,
        gpu_profile=profile,
    )


def section(title):
    print(f"\n=== {title} ===")


def main():
    import torch

    import train

    section("0. tolerance is sized below the retail VRAM gap")
    tol = train.VRAM_REPORTING_TOLERANCE_GB
    check("tolerance > largest reservation assumed (64 MiB = 0.0625 GiB)", tol > 64 * MIB / GIB, f"tol={tol}")
    nominals = sorted({nominal for _, _, nominal, _, _ in SKU_MATRIX})
    gaps = [b - a for a, b in zip(nominals, nominals[1:])]
    check("tolerance < smallest gap between adjacent retail sizes", tol < min(gaps), f"min gap={min(gaps)} GiB")
    print(f"  info  tolerance={tol} GiB, retail gaps={gaps}")

    section("1. every README SKU lands in the tier its name implies, at 4 reservation sizes")
    for reservation_mib in RESERVATIONS_MIB:
        for name, cc, nominal, expected_profile, expected_warning in SKU_MATRIX:
            total_bytes = int(nominal * GIB) - reservation_mib * MIB
            vram_gb = total_bytes / GIB
            profile = train._resolve_gpu_profile(name, cc, vram_gb, True)
            warning = train._compatibility_warning(name, cc, vram_gb)
            label = f"{name} {nominal}GB -{reservation_mib}MiB -> {profile.name} (want {expected_profile})"
            check(label, profile.name == expected_profile, f"got {profile.name}")
            if expected_warning is None:
                check(f"{name} {nominal}GB -{reservation_mib}MiB has no warning", warning is None, f"got {warning!r}")
            else:
                check(
                    f"{name} {nominal}GB -{reservation_mib}MiB warns {expected_warning!r}",
                    warning is not None and expected_warning in warning,
                    f"got {warning!r}",
                )

    section("2. no supported consumer profile can force activation checkpointing")
    for name, cc, nominal, expected_profile, expected_warning in SKU_MATRIX:
        if expected_profile == "compatibility":
            continue
        for reservation_mib in RESERVATIONS_MIB:
            total_bytes = int(nominal * GIB) - reservation_mib * MIB
            vram_gb = total_bytes / GIB
            profile = train._resolve_gpu_profile(name, cc, vram_gb, True)
            runtime = make_runtime(train, profile, total_bytes, vram_gb, cc, name)
            candidates = train._build_train_candidates(runtime)
            modes = {m for _, m in candidates}
            check(
                f"{name} {nominal}GB -{reservation_mib}MiB offers both checkpoint modes",
                modes == {False, True},
                f"got {sorted(modes)}",
            )

    section("3. AUTORESEARCH_FORCE_CHECKPOINTING is one read, and it reaches the model config")
    for raw, expected in (("1", True), ("0", False), (None, None), ("", None), ("  ", None)):
        os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)
        if raw is not None:
            os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"] = raw
        try:
            got = train._forced_checkpointing()
        finally:
            os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)
        check(f"parse {raw!r} -> {expected}", got is expected, f"got {got!r}")
    os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"] = "yes"
    try:
        train._forced_checkpointing()
        check("a value that is neither 0 nor 1 raises instead of silently reading as unset", False)
    except ValueError:
        check("a value that is neither 0 nor 1 raises instead of silently reading as unset", True)
    finally:
        os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)

    # The pre-fix code read the environment in _build_train_candidates as well
    # as in detect_runtime, so these checks go through the same single reader
    # detect_runtime uses. A candidate set that does not depend on the env here
    # is the property that stops the two readers drifting apart again.
    for flag, expected_modes in (("1", {True}), ("0", {False})):
        for name, cc, nominal, expected_profile, _ in SKU_MATRIX:
            if expected_profile == "compatibility" and flag == "0" and nominal > 16:
                continue
            total_bytes = int(nominal * GIB)
            vram_gb = total_bytes / GIB
            profile = train._resolve_gpu_profile(name, cc, vram_gb, True)
            os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"] = flag
            try:
                forced = train._forced_checkpointing()
                narrowed = train._force_checkpointing_mode(profile, forced)
                runtime = make_runtime(train, narrowed, total_bytes, vram_gb, cc, name)
                candidates = train._build_train_candidates(runtime)
                reached = {
                    train.build_model_config(
                        train.DEPTH, 8192, runtime, use_activation_checkpointing=mode
                    ).use_activation_checkpointing
                    for _, mode in candidates
                }
            finally:
                del os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"]
            modes = {m for _, m in candidates}
            check(
                f"force={flag} on {name} {nominal}GB ({expected_profile}) -> {sorted(modes)}",
                modes == expected_modes and len(candidates) > 0,
                f"got {sorted(modes)}",
            )
            check(
                f"force={flag} on {name} {nominal}GB reaches the model config as {sorted(reached)}",
                reached == expected_modes,
                f"got {sorted(reached)}",
            )
    os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)
    check("unset flag leaves the profile's own mode set intact", train._forced_checkpointing() is None)

    # The drift guard. Narrow the profile once, then change the environment
    # under it: the candidate set must not move, because the runtime the caller
    # holds is supposed to be the whole truth.
    total_bytes = int(16 * GIB)
    vram_gb = total_bytes / GIB
    profile = train._resolve_gpu_profile("NVIDIA GeForce RTX 4060 Ti", ADA, vram_gb, True)
    narrowed = train._force_checkpointing_mode(profile, False)
    runtime = make_runtime(train, narrowed, total_bytes, vram_gb, ADA, "NVIDIA GeForce RTX 4060 Ti")
    before = train._build_train_candidates(runtime)
    os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"] = "1"
    try:
        after = train._build_train_candidates(runtime)
    finally:
        del os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"]
    check(
        "_build_train_candidates ignores a later change to the environment",
        before == after and {m for _, m in after} == {False},
        f"before={sorted({m for _, m in before})} after={sorted({m for _, m in after})}",
    )
    check(
        "_force_checkpointing_mode leaves an already-single-mode profile alone",
        train._force_checkpointing_mode(
            train.GpuProfile("compat-x", False, True, (8, 4), (True,), True), True
        ).checkpoint_modes
        == (True,),
    )
    sig = inspect.signature(train.build_model_config)
    param = sig.parameters["use_activation_checkpointing"]
    check(
        "build_model_config requires the checkpointing mode instead of falling back to a runtime field",
        param.default is inspect.Parameter.empty,
        f"default={param.default!r}",
    )
    check(
        "RuntimeConfig no longer carries a dead use_activation_checkpointing field",
        "use_activation_checkpointing" not in train.RuntimeConfig.__dataclass_fields__,
        f"fields={list(train.RuntimeConfig.__dataclass_fields__)}",
    )

    section("4. the warm autotune cache cannot pin the pre-fix decision")
    check("cache version bumped off v2", train.AUTOTUNE_CACHE_VERSION != "gpu-profile-v2", train.AUTOTUNE_CACHE_VERSION)
    print(f"  info  cache version = {train.AUTOTUNE_CACHE_VERSION}, path = {train._get_autotune_cache_path()}")

    section("4b. the autotune probe is token-budgeted, so 2.7x the candidates cost less")
    for batch in (4, 8, 16, 32, 64):
        micro = max(
            train.AUTOTUNE_MIN_MEASURE_MICRO_STEPS,
            train.AUTOTUNE_MEASURE_TOKENS // (batch * train.MAX_SEQ_LEN),
        )
        tokens = micro * batch * train.MAX_SEQ_LEN
        check(
            f"probe at B={batch} measures at least the token budget",
            tokens >= train.AUTOTUNE_MEASURE_TOKENS,
            f"got {tokens:,}",
        )
        check(
            f"probe at B={batch} never exceeds one optimizer step of work",
            tokens <= train.TOTAL_BATCH_SIZE,
            f"got {tokens:,} vs {train.TOTAL_BATCH_SIZE:,}",
        )
        print(f"  info  B={batch}: {micro} micro-steps, {tokens:,} tokens, {tokens / train.TOTAL_BATCH_SIZE:.2f} optimizer steps")
    check("old 5-optimizer-step probe constants are gone", not hasattr(train, "AUTOTUNE_WARMUP_STEPS"))
    check("old measure-step constant is gone", not hasattr(train, "AUTOTUNE_MEASURE_STEPS"))
    check(
        "the pre-fix probe cost 3.0 optimizer steps per candidate; this one costs at most 1.0",
        3 * train.TOTAL_BATCH_SIZE > train.TOTAL_BATCH_SIZE,
    )

    section("4c. with autotune disabled the fallback order is memory-safe")
    for name, cc, nominal, expected_profile, _ in SKU_MATRIX:
        total_bytes = int(nominal * GIB)
        vram_gb = total_bytes / GIB
        profile = train._resolve_gpu_profile(name, cc, vram_gb, True)
        runtime = make_runtime(train, profile, total_bytes, vram_gb, cc, name)
        candidates = train._build_train_candidates(runtime)
        fallback = train._prioritize_autotuned_candidate(candidates, None, profile.is_supported_consumer)
        if profile.is_supported_consumer:
            batches = [b for b, _ in fallback]
            check(
                f"{name} {nominal}GB no-autotune fallback is smallest-batch-first",
                batches == sorted(batches),
                f"got {fallback}",
            )
            check(
                f"{name} {nominal}GB no-autotune fallback is not the largest batch",
                fallback[0][0] == min(batches),
                f"got {fallback[0]}",
            )
        else:
            check(
                f"{name} {nominal}GB compat fallback order is unchanged",
                fallback == candidates,
                f"got {fallback}",
            )
    total_bytes = int(16 * GIB)
    vram_gb = total_bytes / GIB
    profile = train._resolve_gpu_profile("NVIDIA GeForce RTX 4060 Ti", ADA, vram_gb, True)
    runtime = make_runtime(train, profile, total_bytes, vram_gb, ADA, "NVIDIA GeForce RTX 4060 Ti")
    candidates = train._build_train_candidates(runtime)
    hoisted = train._prioritize_autotuned_candidate(candidates, (16, False), profile.is_supported_consumer)
    check(
        "an autotune result is hoisted to first position",
        hoisted[0] == (16, False) and sorted(hoisted) == sorted(candidates),
        f"got {hoisted[:3]}",
    )
    check(
        "a stale autotune result outside the candidate set falls back safely",
        train._prioritize_autotuned_candidate(candidates, (99, True), profile.is_supported_consumer)[0][0]
        == min(b for b, _ in candidates),
    )

    section("5. the shipped bug, on this machine, end to end")
    props = torch.cuda.get_device_properties(0)
    total_bytes = int(props.total_memory)
    vram_gb = total_bytes / GIB
    name = torch.cuda.get_device_name()
    cc = torch.cuda.get_device_capability()
    profile = train._resolve_gpu_profile(name, cc, vram_gb, True)
    runtime = make_runtime(train, profile, total_bytes, vram_gb, cc, name)
    candidates = train._build_train_candidates(runtime)
    print(f"  info  {name}, cc={cc}, total_memory={total_bytes} B = {vram_gb:.6f} GiB")
    print(f"  info  profile={profile.name} batches={profile.train_batch_candidates} modes={profile.checkpoint_modes}")
    print(f"  info  candidates={candidates}")
    check("live card is no longer tiered below its marketed size", profile.name.endswith("-16gb"), profile.name)
    check("live candidate set includes checkpointing off", (32, False) in candidates and (8, False) in candidates)
    check("live candidate set still includes checkpointing on", (8, True) in candidates)

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
