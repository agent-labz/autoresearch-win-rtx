# Verification harnesses

These pin the two autotune fixes on this branch. They live here, next to the
code they check, so a reviewer auditing `train.py` finds the evidence by looking
at the commit that changed it.

They were originally committed to the `autoresearch-reports` repo under
`notes/wren/`. That was the defect AUT-26 found: the harnesses existed and
passed, but an audit of this repository could not see them, and two commits
cited their pass counts as if they were unverifiable. The copies in
`autoresearch-reports` are the originals and are kept in sync; these are the
reviewable ones.

## Running them

All four are standalone, exit non-zero on any failure, and need no training, no
tokenizer, and no data. Run from the repository root:

    uv run --no-sync python tests/verify_tier_fix.py
    uv run --no-sync python tests/verify_memory_gate.py
    uv run --no-sync python tests/verify_cache_key.py
    uv run --no-sync python tests/aut26_guard_probe.py

| file | checks | pins |
| --- | --- | --- |
| `verify_tier_fix.py` | 460 | VRAM tier boundaries and the checkpoint candidate set |
| `verify_memory_gate.py` | 67 | the free-VRAM autotune gate and the 34% rate guard |
| `verify_cache_key.py` | 55 | keying the autotune cache on the model, not just the card |
| `aut26_guard_probe.py` | 26 | the discard path, on both sides of its boundary |

Each one calls the fork's own decision functions against a stub runtime, so it
proves the shipped behaviour rather than restating it. The two that fail against
an unfixed `train.py` name the missing symbol and exit 1, which is the point:

    $ uv run --no-sync python tests/verify_memory_gate.py     # at f84b9e0
    AttributeError: module 'train' has no attribute '_device_free_memory_bytes'

    $ uv run --no-sync python tests/verify_cache_key.py       # at f84b9e0
    TypeError: _make_autotune_cache_key() takes 1 positional argument but 3 were given

## Why `aut26_guard_probe.py` is separate

`verify_memory_gate.py` asserts the rate guard's behaviour. A green suite is
consistent with a guard that never fires and with one that always fires, so
that assertion alone does not establish the constant is right.

This probe re-derives the guard predicate from the shipped constants and drives
it from both sides: the real spilling run from the fix's own measurements
(7,687 tok/sec against a 47,443 tok/sec probe, ~16% of it) is abandoned, the real
uncontended run on the same card (47,856 tok/sec, ~101% of it) finishes, and a
sweep shows the guard flips between 33% and 34% of the probe and nowhere else.
It also prints what the memory budget does under contention, which is the
decision the base gets wrong.
