r"""Regression check for AUTORESEARCH_FORCE_CHECKPOINTING: the hatch is not a lie.

Run from the fork worktree:

    uv run --no-sync python tests/verify_force_checkpointing.py

No GPU training, no tokenizer, no data, no network. It calls the fork's own
parser, profile resolver, candidate builder and model-config builder, so it
proves shipped behaviour rather than restating it. Exits non-zero if any
property is missing.

Why this file exists separately from tests/verify_tier_fix.py: that harness
answers "did the tier fix hold", which is a different question. This one answers
"if I set the environment variable while a run is OOMing, does the run actually
do what I asked". It is also written to run against the *pre-fix* tree without
erroring out, so you can point it at any checkout and see which properties that
checkout is missing:

    git stash && uv run --no-sync python tests/verify_force_checkpointing.py

Read that output carefully, because one of its failures is easy to
misinterpret. The override was never fully inert. The AUT-5 tier fix had
already made it narrow the candidate list, by reading the environment a second
time inside _build_train_candidates. So section 1 passes on the pre-fix tree
too, and it should: the flag did work. What the pre-fix tree fails is section 2
(a typo reads as unset), section 3 (the decision lives in the shell rather than
in the runtime, so the candidate set moves when the environment does), and
section 4 (the dead RuntimeConfig field and the optional-argument fallback in
build_model_config that no call site ever took). Those are the real remainder
of AUT-30.
"""

import inspect
import os
import sys

sys.path.insert(0, os.getcwd())

FAILURES = []
CHECKS = 0

GIB = 1024 ** 3
ADA = (8, 9)
CARD = "NVIDIA GeForce RTX 4060 Ti"
CARD_GIB = 16.0


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


def make_runtime(train, profile):
    """A RuntimeConfig built the way this tree expects one.

    The dead `use_activation_checkpointing` field is passed only when the tree
    under test still has it, so the same file runs against pre-fix and post-fix
    checkouts and reports the difference rather than crashing on it.
    """
    import torch

    kwargs = dict(
        device=torch.device("cuda"),
        device_type="cuda",
        amp_dtype=torch.bfloat16,
        use_compile=False,
        attention_backend="sdpa",
        gpu_name=CARD,
        gpu_vram_gb=CARD_GIB,
        gpu_peak_flops=None,
        gpu_cc=ADA,
        gpu_total_memory_bytes=int(CARD_GIB * GIB),
        tf32_enabled=True,
        gpu_profile=profile,
    )
    if "use_activation_checkpointing" in train.RuntimeConfig.__dataclass_fields__:
        kwargs["use_activation_checkpointing"] = profile.default_checkpointing
    return train.RuntimeConfig(**kwargs)


def with_flag(value):
    """Context-manager-free env setter: returns the restore callable."""
    if value is None:
        os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)
    else:
        os.environ["AUTORESEARCH_FORCE_CHECKPOINTING"] = value

    def restore():
        os.environ.pop("AUTORESEARCH_FORCE_CHECKPOINTING", None)

    return restore


def narrow(train, profile, flag):
    """Apply the override the way detect_runtime does, then read it back.

    On a tree with no _force_checkpointing_mode the profile is returned
    unchanged, because on such a tree the override is applied later and
    elsewhere, by _build_train_candidates reading the environment itself.
    """
    set_env = with_flag(flag)
    try:
        forced = train._forced_checkpointing()
        if hasattr(train, "_force_checkpointing_mode"):
            return train._force_checkpointing_mode(profile, forced)
        return profile
    finally:
        set_env()


def main():
    import train

    profile = train._resolve_gpu_profile(CARD, ADA, CARD_GIB, True)
    print(f"  info  card={CARD} cc={ADA} profile={profile.name} modes={profile.checkpoint_modes}")

    section("1. the override reaches GPTConfig.use_activation_checkpointing")
    # Measured through the real production path: set the variable, call
    # detect_runtime, and ask what the candidate builder and the model config
    # make of the result. This is the check that would have caught a hatch that
    # printed a confirmation and changed nothing.
    #
    # It passes on the pre-fix tree as well, and it should. The override was
    # never inert: the AUT-5 tier fix had already made it narrow the candidate
    # list, by reading the environment a second time inside
    # _build_train_candidates. The difference between the trees is section 3,
    # not this one.
    import torch

    if not torch.cuda.is_available():
        print("  SKIP  no CUDA device; section 1 needs detect_runtime()")
    else:
        for flag, want in (("1", True), ("0", False)):
            set_env = with_flag(flag)
            try:
                runtime = train.detect_runtime()
                candidates = train._build_train_candidates(runtime)
                reached = {
                    train.build_model_config(
                        train.DEPTH, 8192, runtime, use_activation_checkpointing=mode
                    ).use_activation_checkpointing
                    for _, mode in candidates
                }
            finally:
                set_env()
            check(f"force={flag} offers {len(candidates)} candidates", len(candidates) > 0)
            check(
                f"with AUTORESEARCH_FORCE_CHECKPOINTING={flag}, detect_runtime yields "
                f"modes {runtime.gpu_profile.checkpoint_modes} and every GPTConfig "
                f"carries {want}",
                runtime.gpu_profile.checkpoint_modes == (want,) and reached == {want},
                f"modes={runtime.gpu_profile.checkpoint_modes} reached={sorted(reached)}",
            )
        set_env = with_flag(None)
        try:
            runtime = train.detect_runtime()
            unforced = {m for _, m in train._build_train_candidates(runtime)}
        finally:
            set_env()
        check(
            "unset still offers both modes, because the tier chooses",
            unforced == {False, True},
            f"got {sorted(unforced)}",
        )

    section("2. the parser is honest about what it accepts")
    for raw, want in (("1", True), ("0", False), (None, None), ("", None), ("  ", None)):
        set_env = with_flag(raw)
        try:
            got = train._forced_checkpointing()
        finally:
            set_env()
        check(f"{raw!r} parses to {want}", got is want, f"got {got!r}")
    set_env = with_flag("true")
    try:
        train._forced_checkpointing()
        check("'true' raises instead of silently reading as unset", False, "returned None")
    except ValueError:
        check("'true' raises instead of silently reading as unset", True)
    finally:
        set_env()

    section("3. one read: the candidate set is a function of the runtime, not the shell")
    check(
        "detect_runtime narrows the profile, so the runtime carries the decision",
        hasattr(train, "_force_checkpointing_mode"),
        "no _force_checkpointing_mode, so the override is applied by a second env read "
        "inside _build_train_candidates and the shell is part of the candidate set's input",
    )
    narrowed = narrow(train, profile, "0")
    runtime = make_runtime(train, narrowed)
    before = train._build_train_candidates(runtime)
    set_env = with_flag("1")
    try:
        after = train._build_train_candidates(runtime)
    finally:
        set_env()
    check(
        "changing the environment after the runtime was built does not move the candidate set",
        before == after and {m for _, m in after} == {False},
        f"before={sorted({m for _, m in before})} after={sorted({m for _, m in after})}",
    )
    if hasattr(train, "_force_checkpointing_mode"):
        single = train.GpuProfile("compat-x", False, True, (8, 4), (True,), True)
        check(
            "an already-single-mode profile is returned unchanged",
            train._force_checkpointing_mode(single, True) is single,
        )
        for value, want in ((True, (True,)), (False, (False,))):
            check(
                f"forcing {value} collapses a two-mode profile to {want}",
                train._force_checkpointing_mode(profile, value).checkpoint_modes == want,
            )

    section("4. no dead field and no silent fallback left behind")
    check(
        "RuntimeConfig carries no unread use_activation_checkpointing field",
        "use_activation_checkpointing" not in train.RuntimeConfig.__dataclass_fields__,
        f"fields={list(train.RuntimeConfig.__dataclass_fields__)}",
    )
    param = inspect.signature(train.build_model_config).parameters["use_activation_checkpointing"]
    check(
        "build_model_config requires the mode rather than defaulting it",
        param.default is inspect.Parameter.empty,
        f"default={param.default!r}",
    )
    check(
        "both build_model_config call sites pass the mode explicitly",
        _call_sites_pass_the_mode(train),
    )

    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED:")
        for item in FAILURES:
            print(f"  - {item}")
        return 1
    return 0


def _call_sites_pass_the_mode(train):
    """True when no caller relies on build_model_config's optional argument.

    Checked structurally rather than by source grep so that a caller added on
    another branch without the keyword is caught by name resolution instead of
    silently inheriting a default.
    """
    import ast
    import inspect as _inspect

    try:
        tree = ast.parse(_inspect.getsource(train))
    except (OSError, TypeError):
        return True  # source unavailable (frozen import); signature check above stands
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Name) and func.id == "build_model_config"):
            continue
        if not any(kw.arg == "use_activation_checkpointing" for kw in node.keywords):
            if len(node.args) < 4:
                return False
    return True


if __name__ == "__main__":
    sys.exit(main())
