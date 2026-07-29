# AutoResearch

> Run autonomous machine learning experiments on a single desktop NVIDIA
> GPU.

## Overview

AutoResearch is an autonomous research framework. The system improves
`train.py` through repeated experiments. Each experiment has one
hypothesis. Each experiment uses the same evaluation method. Keep a
change only if it improves `val_bpb`.

This fork targets native Windows execution on supported desktop NVIDIA
GPUs. It uses official PyTorch CUDA builds and a single runtime path.

## Design Principles

-   Modify only `train.py`.
-   Do not modify `prepare.py`.
-   Test one hypothesis in each experiment.
-   Use evidence to make decisions.
-   Keep the code simple.
-   Make experiments reproducible.

## Repository Layout

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `prepare.py`                        Download data, train the tokenizer,
                                      and provide runtime utilities.

  `train.py`                          Model, optimizer, and training
                                      loop. This is the file that the
                                      agent edits.

  `program.md`                        Instructions for the autonomous
                                      research agent.

  `README.md`                         User documentation.
  -----------------------------------------------------------------------

## Requirements

-   Windows 10 or later
-   Python 3.10 or later
-   `uv`
-   One supported NVIDIA desktop GPU

## Quick Start

``` powershell
uv sync
uv run prepare.py
uv run train.py
```

Run `prepare.py` one time for each dataset. The script downloads the
dataset and trains the tokenizer.

## Experiment Workflow

1.  Read `README.md`.
2.  Read `program.md`.
3.  Verify that the dataset exists.
4.  Record a baseline result.
5.  Form one hypothesis.
6.  Modify only `train.py`.
7.  Run the experiment.
8.  Compare `val_bpb`.
9.  Keep the change only if the metric improves.
10. Record the result.

## Prompting Guidelines

Use these rules when you instruct an AI coding agent.

-   Give one clear objective.
-   State the success metric.
-   Define the files that the agent can modify.
-   Define the files that the agent must not modify.
-   Require evidence for each decision.
-   Require one hypothesis per experiment.
-   Require rollback of unsuccessful changes.
-   Keep experiments independent.
-   Prefer simple implementations when results are equal.

## Example Prompt

    Read program.md and README.md.

    Modify only train.py.

    Create one hypothesis.

    Run one experiment.

    Measure val_bpb.

    Keep the change only if val_bpb improves.

    Record the result.

    Repeat until I stop the process.

## Platform Support

This fork supports native Windows execution on supported desktop NVIDIA
RTX GPUs. Consumer GPU support follows project VRAM policies.

## Troubleshooting

If the dataset or tokenizer is missing, run:

``` powershell
uv run prepare.py
```

If an experiment fails because of an implementation error, fix the error
and repeat the experiment.

## Contributing

Submit changes that improve correctness, readability, reliability, or
validation performance. Keep documentation in ASD-STE100 style.
