r"""Regression check for the autotune cache-key fix in autoresearch-win-rtx.

Run from the fork worktree:

    uv run --no-sync python tests/verify_cache_key.py

No training, no tokenizer, no data. It calls the fork's own key builder and its
own invalidation helper against a stub runtime, so it proves the shipped
behaviour rather than a restatement of it. Every assertion prints a PASS/FAIL
line; the script exits non-zero on any failure.

What it pins down:

1. The bug, restated and proved: the pre-fix key was byte-identical across
   `DEPTH` 8/12/16/24 while the model changed from n_layer=8 n_embd=512 to
   n_layer=24 n_embd=1536. If that stops being true, section 1 has regressed
   into not testing anything.
2. Every field the old key was missing now changes it: DEPTH, ASPECT_RATIO,
   HEAD_DIM, WINDOW_PATTERN, vocab_size, TOTAL_BATCH_SIZE, MUON_COMPUTE_DTYPE,
   and the candidate list itself.
3. The fields the old key did have still change it, so the fix is additive
   rather than a replacement that lost the hardware fingerprint.
4. The key is deterministic, and it names the model it was tuned for, so an
   operator reading a cache hit can tell what it is looking at.
5. Nothing regresses into re-probing every run: identical inputs give an
   identical key, so a warm cache is still warm.
6. A cache entry the real run contradicted is deleted, an entry the run used is
   kept, another run's newer entry is never clobbered, and unrelated entries in
   the same file survive.
7. End to end on live hardware: the entry written at DEPTH=8 is simply not
   found at DEPTH=12, so a model-size change re-probes instead of inheriting.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.getcwd())

FAILURES = []
CHECKS = 0

GIB = 1024 ** 3
VOCAB = 50257
DEPTHS = (8, 12, 16, 24)


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


def make_runtime(train, torch, profile=None, name="NVIDIA GeForce RTX 4060 Ti", cc=(8, 9), total_bytes=16 * GIB):
    if profile is None:
        profile = train._resolve_gpu_profile(name, cc, total_bytes / GIB, True)
    return train.RuntimeConfig(
        device=torch.device("cuda"),
        device_type="cuda",
        amp_dtype=torch.bfloat16,
        use_compile=False,
        use_activation_checkpointing=profile.default_checkpointing,
        attention_backend="sdpa",
        gpu_name=name,
        gpu_vram_gb=total_bytes / GIB,
        gpu_peak_flops=None,
        gpu_cc=cc,
        gpu_total_memory_bytes=total_bytes,
        tf32_enabled=True,
        gpu_profile=profile,
    )


def legacy_key(train, torch, runtime):
    """The key as it shipped before this fix, rebuilt from the same runtime.

    Kept here so section 1 is a real comparison against the broken behaviour
    rather than a claim about it.
    """
    cc = f"{runtime.gpu_cc[0]}.{runtime.gpu_cc[1]}"
    return "|".join(
        [
            runtime.gpu_name,
            cc,
            str(runtime.gpu_total_memory_bytes),
            torch.__version__,
            train.platform.system(),
            str(train.MAX_SEQ_LEN),
        ]
    )


def key_at_depth(train, runtime, candidates, depth):
    original = train.DEPTH
    train.DEPTH = depth
    try:
        return train._make_autotune_cache_key(runtime, VOCAB, candidates)
    finally:
        train.DEPTH = original


def shape_at_depth(train, runtime, depth):
    original = train.DEPTH
    train.DEPTH = depth
    try:
        cfg = train.build_model_config(depth, VOCAB, runtime)
        return f"n_layer={cfg.n_layer} n_embd={cfg.n_embd} n_head={cfg.n_head}"
    finally:
        train.DEPTH = original


def with_module_globals(train, **overrides):
    """Set module-level knobs, yield, restore. Proves the key reads them."""
    saved = {name: getattr(train, name) for name in overrides}
    for name, value in overrides.items():
        setattr(train, name, value)
    return saved


def restore_module_globals(train, saved):
    for name, value in saved.items():
        setattr(train, name, value)


def main():
    import torch

    import train

    runtime = make_runtime(train, torch)
    candidates = train._build_train_candidates(runtime)
    print(f"  info  profile={runtime.gpu_profile.name} candidates={candidates}")
    if not candidates:
        print("  FAIL  no candidates from the shipped builder")
        return 1

    section("1. the pre-fix key was shape-invariant, so the bug was real")
    legacy = {d: legacy_key(train, torch, runtime) for d in DEPTHS}
    shapes = {d: shape_at_depth(train, runtime, d) for d in DEPTHS}
    for d in DEPTHS:
        print(f"  info  DEPTH={d:<3} {shapes[d]}")
    check(
        "one legacy key covers every depth (the reported defect)",
        len(set(legacy.values())) == 1,
        f"{len(set(legacy.values()))} distinct keys",
    )
    check(
        "but the models really do differ across those depths",
        len(set(shapes.values())) == len(DEPTHS),
        str(shapes),
    )

    section("2. every unkeyed field from the report now moves the key")
    fresh = {d: key_at_depth(train, runtime, candidates, d) for d in DEPTHS}
    check("DEPTH", len(set(fresh.values())) == len(DEPTHS), str(sorted(set(fresh.values()))))

    base = fresh[8]
    print(f"  info  key = {base}")

    saved = with_module_globals(train, ASPECT_RATIO=train.ASPECT_RATIO + 1)
    try:
        check("ASPECT_RATIO", train._make_autotune_cache_key(runtime, VOCAB, candidates) != base)
    finally:
        restore_module_globals(train, saved)

    saved = with_module_globals(train, HEAD_DIM=train.HEAD_DIM * 2)
    try:
        check("HEAD_DIM", train._make_autotune_cache_key(runtime, VOCAB, candidates) != base)
    finally:
        restore_module_globals(train, saved)

    saved = with_module_globals(train, WINDOW_PATTERN="LSSL")
    try:
        check("WINDOW_PATTERN", train._make_autotune_cache_key(runtime, VOCAB, candidates) != base)
    finally:
        restore_module_globals(train, saved)

    saved = with_module_globals(train, TOTAL_BATCH_SIZE=train.TOTAL_BATCH_SIZE // 2)
    try:
        check("TOTAL_BATCH_SIZE", train._make_autotune_cache_key(runtime, VOCAB, candidates) != base)
    finally:
        restore_module_globals(train, saved)

    saved = with_module_globals(train, MUON_COMPUTE_DTYPE=torch.float32)
    try:
        check("MUON_COMPUTE_DTYPE", train._make_autotune_cache_key(runtime, VOCAB, candidates) != base)
    finally:
        restore_module_globals(train, saved)

    check("vocab_size", train._make_autotune_cache_key(runtime, VOCAB + 1, candidates) != base)
    check("vocab_size (down, not only up)", train._make_autotune_cache_key(runtime, VOCAB - 1, candidates) != base)

    section("3. the candidate list is keyed, so the entry cannot outlive the resolver")
    check("reordered candidate set", train._make_autotune_cache_key(runtime, VOCAB, list(reversed(candidates))) != base)
    check("dropped candidate", train._make_autotune_cache_key(runtime, VOCAB, candidates[:-1]) != base)
    check("appended candidate", train._make_autotune_cache_key(runtime, VOCAB, candidates + [(64, True)]) != base)
    check(
        "flipped checkpointing mode",
        train._make_autotune_cache_key(runtime, VOCAB, [(b, not c) for b, c in candidates]) != base,
    )
    check("same candidate set is stable", train._make_autotune_cache_key(runtime, VOCAB, list(candidates)) == base)
    digests = {train._candidate_set_digest(c) for c in (candidates, list(reversed(candidates)), candidates[:-1])}
    check("digest distinguishes the three sets", len(digests) == 3, str(digests))
    check("digest is short enough to read in a log", all(len(d) == 12 for d in digests), str(digests))

    section("4. the hardware and runtime fields the old key had are still there")
    other_card = make_runtime(train, torch, name="NVIDIA GeForce RTX 3080", cc=(8, 6), total_bytes=10 * GIB)
    check("gpu_name + cc + total memory", train._make_autotune_cache_key(other_card, VOCAB, candidates) != base)
    same_card_more_vram = make_runtime(train, torch, total_bytes=24 * GIB)
    check("total memory alone", train._make_autotune_cache_key(same_card_more_vram, VOCAB, candidates) != base)
    check("torch version is in the key", torch.__version__ in base)
    check("platform is in the key", train.platform.system() in base)
    check("MAX_SEQ_LEN is in the key", f"seq={train.MAX_SEQ_LEN}" in base or str(train.MAX_SEQ_LEN) in base)

    section("5. the key is deterministic and self-describing")
    check("same inputs give the same key", train._make_autotune_cache_key(runtime, VOCAB, candidates) == base)
    check("key is stable across repeated builds", len({key_at_depth(train, runtime, candidates, 8) for _ in range(5)}) == 1)
    for token in ("depth=8", "n_embd=", "n_head=", "n_kv_head=", "vocab=", "window=", "total_batch=", "candidates="):
        check(f"key carries {token}", token in base, base)

    section("6. main's OOM fallback invalidates the entry the run contradicted")
    with tempfile.TemporaryDirectory() as tmp:
        cache_path = Path(tmp) / "autotune.json"
        real_path_fn = train._get_autotune_cache_path
        train._get_autotune_cache_path = lambda: cache_path
        try:
            probed = candidates[0]
            other = candidates[-1] if candidates[-1] != probed else candidates[0]
            neighbour_key = "some-other-gpu"
            cache_key = "k-shape-8"
            train._save_autotune_entries(
                cache_path,
                {
                    cache_key: {
                        "train_batch_size": probed[0],
                        "use_activation_checkpointing": probed[1],
                        "peak_reserved_bytes": 1,
                    },
                    neighbour_key: {"train_batch_size": 1, "use_activation_checkpointing": True, "peak_reserved_bytes": 2},
                },
            )

            autotuned = {"candidate": probed, "cache_key": cache_key}

            check("no-op when there is no autotune result", train._invalidate_stale_autotune_entry(None, other) is False)
            check(
                "entry survives a run that used the probed candidate",
                train._invalidate_stale_autotune_entry(autotuned, probed) is False
                and cache_key in train._load_autotune_entries(cache_path),
            )
            check(
                "a run that fell through deletes the probed entry",
                train._invalidate_stale_autotune_entry(autotuned, other) is True
                and cache_key not in train._load_autotune_entries(cache_path),
            )
            check(
                "the other GPU's entry in the same file survives",
                neighbour_key in train._load_autotune_entries(cache_path),
            )
            check(
                "no-op when the probed candidate is absent",
                train._invalidate_stale_autotune_entry({"candidate": None, "cache_key": cache_key}, other) is False,
            )
            check(
                "no-op when the autotune result carries no cache key",
                train._invalidate_stale_autotune_entry({"candidate": probed}, other) is False,
            )

            # A run where every candidate was abandoned: `chosen` is None.
            train._save_autotune_entries(
                cache_path,
                {cache_key: {"train_batch_size": probed[0], "use_activation_checkpointing": probed[1], "peak_reserved_bytes": 1}},
            )
            check(
                "a run where nothing completed also deletes the entry",
                train._invalidate_stale_autotune_entry(autotuned, None) is True
                and cache_key not in train._load_autotune_entries(cache_path),
            )

            # Someone else re-tuned the same shape after this run started.
            train._save_autotune_entries(
                cache_path,
                {cache_key: {"train_batch_size": other[0], "use_activation_checkpointing": other[1], "peak_reserved_bytes": 3}},
            )
            check(
                "a newer entry for the same shape is not clobbered",
                train._invalidate_stale_autotune_entry(autotuned, None) is False
                and cache_key in train._load_autotune_entries(cache_path),
            )

            # The file must stay valid JSON with the documented shape.
            payload = json.loads(cache_path.read_text())
            check("cache file is still an entries object", isinstance(payload.get("entries"), dict))
        finally:
            train._get_autotune_cache_path = real_path_fn

    section("7. end to end on this card: a depth change cannot inherit a batch size")
    if not torch.cuda.is_available():
        print("  skip  no CUDA device on this machine")
    else:
        props = torch.cuda.get_device_properties(0)
        live_total = int(props.total_memory)
        live_name = torch.cuda.get_device_name()
        live_cc = torch.cuda.get_device_capability()
        live_profile = train._resolve_gpu_profile(live_name, live_cc, live_total / GIB, True)
        live_runtime = make_runtime(train, torch, live_profile, live_name, live_cc, live_total)
        live_candidates = train._build_train_candidates(live_runtime)
        train._configure_step_kernels(live_runtime)
        print(f"  info  {live_name}, cc={live_cc}, {live_total} B, profile={live_profile.name}")
        print(f"  info  candidates={live_candidates}")
        print(f"  info  muon dtype={train.MUON_COMPUTE_DTYPE}")

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "autotune.json"
            real_path_fn = train._get_autotune_cache_path
            train._get_autotune_cache_path = lambda: cache_path
            try:
                k8 = key_at_depth(train, live_runtime, live_candidates, 8)
                k12 = key_at_depth(train, live_runtime, live_candidates, 12)
                k24 = key_at_depth(train, live_runtime, live_candidates, 24)
                check("live card: depth 8/12/24 produce three keys", len({k8, k12, k24}) == 3)

                # Exactly the phase-1 report: a legal decision cached at depth 8.
                stale = (live_candidates[0][0], live_candidates[0][1])
                train._save_autotune_entries(
                    cache_path,
                    {k8: {"train_batch_size": stale[0], "use_activation_checkpointing": stale[1], "peak_reserved_bytes": 1}},
                )
                entries = train._load_autotune_entries(cache_path)
                check("live card: the depth-8 entry is still legal at depth 8", k8 in entries)
                check("live card: it is NOT reachable at depth 12", k12 not in entries)
                check("live card: it is NOT reachable at depth 24", k24 not in entries)
                check(
                    "so the depth-12 run must probe, not inherit",
                    train._cached_candidate(live_runtime, entries.get(k12), live_candidates, 16 * GIB, "test")[0] is None,
                )
                check(
                    "and the depth-8 run does inherit it",
                    train._cached_candidate(live_runtime, entries.get(k8), live_candidates, 16 * GIB, "test")[0] == stale,
                )
            finally:
                train._get_autotune_cache_path = real_path_fn

    section("8. main() actually calls it, and calls it in the right place")
    source = Path("train.py").read_text(encoding="utf-8")
    main_body = source.split("def main():", 1)[1]
    check("main() calls the invalidation helper", "_invalidate_stale_autotune_entry(" in main_body)
    check(
        "and passes it the chosen candidate, not the autotuned one",
        "chosen_candidate = (chosen_train_batch, chosen_checkpointing)" in main_body
        and "_invalidate_stale_autotune_entry(autotuned, chosen_candidate)" in main_body,
    )
    invalidate_at = main_body.index("_invalidate_stale_autotune_entry(autotuned, chosen_candidate)")
    bailout_at = main_body.index("if result is None:")
    check("it runs before the failed-run bail-out, so a total loss also invalidates", invalidate_at < bailout_at)
    check("it runs after the candidate loop, so the outcome is known", invalidate_at > main_body.index("for train_batch_size, use_checkpointing in train_candidates:"))
    check(
        "the autotune result dict still carries the key it used",
        '"cache_key": cache_key' in source.split("def _autotune_train_candidate", 1)[1].split("def _invalidate_stale_autotune_entry", 1)[0],
    )
    check(
        "the key builder is called with the shape and the candidate set",
        "_make_autotune_cache_key(runtime, vocab_size, train_candidates)" in source,
    )
    check("the no-argument key builder signature is gone", "def _make_autotune_cache_key(runtime):" not in source)
    check(
        "the cache version was not bumped to paper over the key",
        source.count("AUTOTUNE_CACHE_VERSION = ") == 1,
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
