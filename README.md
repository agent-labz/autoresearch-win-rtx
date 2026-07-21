# autoresearch

> Convert your gaming PC into an autonomous AI researcher.

> This repository is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). The purpose of this fork is native support for desktop consumer NVIDIA GPUs on Windows, with tiered VRAM floors by architecture.

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

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

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

# 3. Download data and train the 16,384-token nanochat-compatible tokenizer (one-time)
#    Default dataset: TinyStories GPT-4 clean
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

Quick validation run (recommended after setup):

```powershell
uv run train.py --smoke-test
```

## Optional experiment tracking and publishing

The project works entirely locally by default. Set credentials only when you want
the corresponding integration enabled; keep API keys in environment variables so
they are not placed in shell history or source control.

```powershell
# Log each experiment to Weights & Biases (optional)
$env:WANDB_API_KEY = "..."
$env:WANDB_PROJECT = "autoresearch"  # optional; this is the default

# Upload every successfully evaluated model and its tokenizer to this HF repo (optional)
$env:HF_TOKEN = "..."
$env:HF_REPO_ID = "your-account/autoresearch-experiments"
# Or pass the repository explicitly: uv run train.py --hf-repo-id your-account/autoresearch-experiments

uv run train.py
```

If you prepared data with an older checkout, run `uv run prepare.py` once after
updating: it detects the old tokenizer and rebuilds the required tokenizer.

Use `--hf-private` when the target repository should be created as private. Each
upload includes `model.pt`, `config.json`, metrics, and the nanochat-compatible
tokenizer files. The tokenizer reserves nanochat's conversation special tokens
(`user`, `assistant`, and tool delimiters) for a later SFT stage.

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Focused experiment surface.** Autonomous experiments normally modify only `train.py`; setup integrations and tokenizer changes are maintained in `prepare.py`.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
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
- Runtime path is intentionally unified across platforms: PyTorch SDPA attention + eager optimizer steps.
- Runtime adaptation is profile-driven: compute capability, BF16/TF32 support, OS, and VRAM tier determine candidate batch sizes and checkpointing strategy.
- Supported consumer profiles run a short eager-mode autotune pass and cache the selected candidate per GPU/runtime fingerprint.
- Autotune env controls: `AUTORESEARCH_DISABLE_AUTOTUNE=1` skips probing; `AUTORESEARCH_AUTOTUNE_REFRESH=1` refreshes the cached decision.
- Tested hardware in this repo remains RTX 3080 10 GB on Windows. Other listed SKUs are matrix-supported but may be less field-tested here.
- Non-goals for this fork include FA3/H100-specialized paths, unofficial Triton-for-Windows stacks, AMD/ROCm, Apple Metal, and multi-GPU training.
- Default dataset is `karpathy/tinystories_gpt4_clean` for consumer-GPU practicality.

## License

MIT
