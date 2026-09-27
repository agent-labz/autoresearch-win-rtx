# autoresearch

> Convert your gaming PC into an autonomous AI researcher.

> This repository is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The purpose of this fork is native support for desktop consumer NVIDIA GPUs on Windows, with tiered VRAM floors by architecture.

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 10 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## Fork scope

- Upstream source: [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- Primary objective: run natively on Windows with desktop consumer NVIDIA GPUs (Turing with >=8 GB VRAM, Ampere/Ada/Blackwell with >=10 GB VRAM), without unofficial Triton-on-Windows stacks.
- Scope of changes: compatibility and stability updates required for that target platform.
- The original Linux/H100-oriented path from upstream is removed in this fork and is not supported here.
- If you need the upstream Linux/H100 path, use [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## How it works

The repo is deliberately kept small and only really has a three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads TinyStories data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation).
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 10-minute time budget** (`TIME_BUDGET` in `prepare.py`, wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start (PowerShell)

**Requirements:** A single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

- Single runtime path uses PyTorch SDPA attention and eager execution (no FA3/`torch.compile` fast path).
- Native Windows support targets desktop consumer GPUs with a tiered VRAM policy (Turing >=8 GB, Ampere/Ada/Blackwell >=10 GB), official PyTorch CUDA wheels, and SDPA attention.
- Default dataset is now TinyStories GPT-4 clean for practical consumer-GPU setup.

```powershell

# 1. Install uv project manager (if you don't already have it)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time)
#    Default dataset: TinyStories GPT-4 clean
uv run prepare.py

# 4. Manually run a single training experiment (~13 min)
uv run train.py
```

Quick validation run (recommended after setup):

```powershell
uv run train.py --smoke-test
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Experiment workflow

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

## Prompting guidelines

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

## Example prompt

    Read program.md and README.md.

    Modify only train.py.

    Create one hypothesis.

    Run one experiment.

    Measure val_bpb.

    Keep the change only if val_bpb improves.

    Record the result.

    Repeat until I stop the process.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 10 minutes, regardless of your specific platform. This means you can expect approx 4 experiments/hour and approx 40 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This fork's platform policy is explicit and tiered.

| Architecture | Minimum VRAM floor | Supported desktop consumer GPUs |
| --- | --- | --- |
| Turing | `>=8 GB` | `RTX 2060 12GB`, `RTX 2060 SUPER 8GB`, `RTX 2070 8GB`, `RTX 2070 SUPER 8GB`, `RTX 2080 8GB`, `RTX 2080 SUPER 8GB`, `RTX 2080 Ti 11GB` |
| Ampere | `>=10 GB` | `RTX 3060 12GB`, `RTX 3080 10GB`, `RTX 3080 12GB`, `RTX 3080 Ti 12GB`, `RTX 3090 24GB`, `RTX 3090 Ti 24GB` |
| Ada | `>=10 GB` | `RTX 4060 Ti 16GB`, `RTX 4070 12GB`, `RTX 4070 SUPER 12GB`, `RTX 4070 Ti 12GB`, `RTX 4070 Ti SUPER 16GB`, `RTX 4080 16GB`, `RTX 4080 SUPER 16GB`, `RTX 4090 24GB` |
| Blackwell | `>=10 GB` | `RTX 5060 Ti 16GB`, `RTX 5070 12GB`, `RTX 5070 Ti 16GB`, `RTX 5080 16GB`, `RTX 5090 32GB` |
- Desktop only: laptop GPUs are not officially supported due to wide power and thermal variance.
- Floor policy: Turing desktop GPUs are supported at >=8 GB VRAM; Ampere/Ada/Blackwell desktop GPUs require >=10 GB VRAM.
- `RTX 2060 6GB` remains out of matrix support due to VRAM floor.
- VRAM tier boundaries are compared with a 0.25 GB reporting tolerance, because `props.total_memory` excludes a small NVIDIA-reserved region and so a card sold as N GB always reports slightly under N GiB (an RTX 4060 Ti 16 GB reports 15.9956 GiB). The tolerance is far below the 2 GB+ gap between adjacent retail VRAM sizes, so it corrects the boundary without promoting a card into a tier it cannot hold.
- Tier boundaries select *which candidates are offered*; they do not force activation checkpointing. Every supported consumer profile offers both checkpointing modes and lets the autotune measure which one wins, so a mis-tiered card costs throughput headroom rather than correctness.
- `AUTORESEARCH_FORCE_CHECKPOINTING=0|1` is the hard override for a run that OOMs or a card that misbehaves. It collapses the profile's checkpointing mode set to the single named mode before any candidate is built, so the run never gets a choice. It is read exactly once, in `detect_runtime`, and the run log says so: the startup line reads `Activation checkpointing forced by ...` when the hatch fires and `Activation checkpointing: not forced; offering modes [...]` when it does not, and `main()` prints `Activation checkpointing candidates:` before the candidate loop. Unset it to let the VRAM tier choose. Any value other than `0`, `1` or unset raises at startup rather than training at the tier default while you believe the override is live.
- The override narrows *train* candidates only. It does not change the eval batch ladder, which already walks down from `min(train_batch, eval_batch_cap)` on OOM.
- Runtime path is intentionally unified across platforms: PyTorch SDPA attention + eager optimizer steps.
- Runtime adaptation is profile-driven: compute capability, BF16/TF32 support, OS, and VRAM tier determine candidate batch sizes and checkpointing strategy.
- Supported consumer profiles run a short eager-mode autotune pass and cache the selected candidate per GPU/runtime fingerprint. The cache is versioned; bump `AUTOTUNE_CACHE_VERSION` in `train.py` when the candidate set or the resolver changes, or stale decisions will be accepted from cache.
- An autotune probe measures device-batch throughput over a fixed 131,072-token window rather than simulating whole optimizer steps, so autotune wall clock does not scale with `TOTAL_BATCH_SIZE`. On an RTX 4060 Ti the full 8-candidate search costs about 107 s; the pre-fix 3-candidate search cost about 212 s.
- Without an autotune result, supported consumer profiles fall back to smallest-batch-first. On Windows an oversized candidate can spill to shared system memory instead of raising OOM, so the unmeasured fallback must not lead with the largest batch.
- Autotune env controls: `AUTORESEARCH_DISABLE_AUTOTUNE=1` skips probing; `AUTORESEARCH_AUTOTUNE_REFRESH=1` refreshes the cached decision; `AUTORESEARCH_SAVE_PRE_EVAL_CHECKPOINT=1` opts into writing the 169 MB `checkpoint_pre_eval.pt` debug artifact (off by default; nothing reads it).
- Tested hardware in this repo remains RTX 3080 10 GB on Windows. Other listed SKUs are matrix-supported but may be less field-tested here.
- Non-goals for this fork include FA3/H100-specialized paths, unofficial Triton-for-Windows stacks, AMD/ROCm, Apple Metal, and multi-GPU training.
- Default dataset is `karpathy/tinystories_gpt4_clean` for consumer-GPU practicality.

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

## License

MIT
