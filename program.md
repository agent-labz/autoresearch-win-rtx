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

- Reduces `val_bpb`.
- Preserves correctness.
- Keeps the implementation simple.
- Can be reproduced.

## General Rules

Use evidence instead of assumptions.

Do not change the evaluation method.

Do not install new dependencies.

Do not stop the research loop unless the user interrupts it.

Always prefer clear code over clever code.
