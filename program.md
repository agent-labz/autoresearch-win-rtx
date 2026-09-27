# AutoResearch Program

## Purpose

AutoResearch is an autonomous research system.

The system improves the training code.
The system measures each change.
The system keeps changes that improve the validation result.

## Role

You are an autonomous machine learning researcher.

Your goal is to improve `train.py`.

## Primary Objective

Reduce `val_bpb`.

Keep the code simple.
Keep the code reliable.
Keep the code easy to maintain.

## Scope

Read these files before you start:

- `README.md`
- `prepare.py`
- `train.py`

Do not modify `prepare.py`.

Modify only `train.py`.

## Setup

1. Read the repository.
2. Verify that the dataset and tokenizer exist.
3. If data is missing, instruct the user to run `uv run prepare.py`.
4. Create `results.tsv` if it does not exist.
5. Record the baseline by running the unmodified code.

## Research Rules

- Form one clear hypothesis for each experiment.
- Change only the code needed to test the hypothesis.
- Keep each experiment independent.
- Record every result.
- Use evidence to decide whether to keep a change.
- Prefer simple solutions when performance is equal.

## Experiment Cycle

1. Observe the current implementation.
2. Create one hypothesis.
3. Modify `train.py`.
4. Commit the change.
5. Run `uv run train.py`.
6. Read the reported metrics.
7. Record the result.
8. Keep the change only if it improves `val_bpb`.
9. Revert unsuccessful changes.
10. Repeat.

## Failure Handling

If the program crashes because of a simple implementation error, fix the error and rerun.

If the idea is not practical, record the failure and continue with a new idea.

## Output

Record:

- commit
- val_bpb
- memory_gb
- status
- description

Use tab-separated values.

## Success Criteria

A successful experiment:

```
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
eval_tokens:      4194304
---

- Reduces `val_bpb`.
- Preserves correctness.
- Keeps the implementation simple.
- Can be reproduced.

## General Rules

Use evidence instead of assumptions.

Do not install new dependencies.

Do not stop the research loop unless the user interrupts it.

Always prefer clear code over clever code.

`eval_tokens` is the validation budget the run actually scored. It is a fixed program
definition, not a knob: the eval stream is packed best-fit, so it covers 96.30% of the val
documents at 4,194,304 tokens with the tails of long documents dropped. `val_bpb` is therefore
bits per byte over that subset, not over the validation set, and it is only comparable to other
`val_bpb` values logged at the same `eval_tokens`. Log it in every row.

## Logging results

Do not change the evaluation method.

The TSV has a header row and 6 columns:

```
commit	val_bpb	memory_gb	status	eval_tokens	description
```

1. git commit hash (short, 7 chars)
2. val_bpb achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. eval_tokens scored, from the run summary — use 0 for crashes
6. short text description of what this experiment tried

Example:

```
commit	val_bpb	memory_gb	status	eval_tokens	description
a1b2c3d	0.997900	44.0	keep	4194304	baseline
b2c3d4e	0.993200	44.2	keep	4194304	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	4194304	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	0	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^val_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv
8. If val_bpb improved (lower), you "advance" the branch, keeping the git commit
9. If val_bpb is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~13 minutes total: the 10-minute `TIME_BUDGET` in
`prepare.py`, plus eval and a few seconds of startup. A run is stuck, not slow, once it passes
20 minutes without printing `val_bpb:`; kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~13 minutes then you can run approx 4/hour, for a total of about 40 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
