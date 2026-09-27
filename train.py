"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import argparse
import gc
import json
import math
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from prepare import (
    DATASET_CHOICES,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    Tokenizer,
    evaluate_bpb,
    make_dataloader,
)

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


@dataclass
class RuntimeConfig:
    device: torch.device
    device_type: str
    amp_dtype: torch.dtype
    use_compile: bool
    use_activation_checkpointing: bool
    attention_backend: str
    gpu_name: str
    gpu_vram_gb: float
    gpu_peak_flops: float | None
    gpu_cc: tuple[int, int]
    gpu_total_memory_bytes: int
    tf32_enabled: bool
    gpu_profile: "GpuProfile"


@dataclass(frozen=True)
class GpuProfile:
    name: str
    is_supported_consumer: bool
    is_compatibility_only: bool
    train_batch_candidates: tuple[int, ...]
    checkpoint_modes: tuple[bool, ...]
    default_checkpointing: bool
    eval_batch_cap: int = 16


SUPPORTED_CONSUMER_CAPABILITIES = {
    (7, 5): "turing",
    (8, 6): "ampere",
    (8, 9): "ada",
    (12, 0): "blackwell",
}
MIN_SUPPORTED_VRAM_GB_BY_ARCH = {
    "turing": 8.0,
    "ampere": 10.0,
    "ada": 10.0,
    "blackwell": 10.0,
}
VRAM_REPORTING_TOLERANCE_GB = 0.25
AUTOTUNE_MEASURE_TOKENS = 131072
AUTOTUNE_MIN_MEASURE_MICRO_STEPS = 4
AUTOTUNE_MAX_MEMORY_FRACTION = 0.90
# v4 records per-candidate probe error signatures in the autotune cache, so a
# candidate chosen while probes were erroring cannot be reused silently.
AUTOTUNE_CACHE_VERSION = "gpu-profile-v4"


def _get_gpu_peak_flops(gpu_name):
    name = gpu_name.lower()
    lookup = (
        ("5090", 360.0e12),
        ("4090 d", 280.0e12),
        ("4090d", 280.0e12),
        ("4090", 330.3e12),
        ("5080", 280.0e12),
        ("4080 super", 260.0e12),
        ("4070 ti super", 176.4e12),
        ("4070 ti", 160.4e12),
        ("4070 super", 142.2e12),
        ("4070", 116.8e12),
        ("4080", 242.5e12),
        ("5070 ti", 190.0e12),
        ("5070", 150.0e12),
        ("5060 ti", 120.0e12),
        ("4060 ti", 88.4e12),
        ("2080 ti", 107.5e12),
        ("2080 super", 89.6e12),
        ("2080", 80.3e12),
        ("2070 super", 72.6e12),
        ("2070", 59.7e12),
        ("2060 super", 57.4e12),
        ("2060", 52.4e12),
        ("3090 ti", 160.0e12),
        ("3090", 142.6e12),
        ("3080 ti", 136.0e12),
        ("3080", 119.5e12),
        ("3060", 51.0e12),
        ("3070", 81.1e12),
    )
    for key, flops in lookup:
        if key in name:
            return flops
    return None


def _vram_reaches(gpu_vram_gb, threshold_gb):
    return gpu_vram_gb >= threshold_gb - VRAM_REPORTING_TOLERANCE_GB


def _forced_checkpointing():
    raw = os.environ.get("AUTORESEARCH_FORCE_CHECKPOINTING")
    if raw == "1":
        return True
    if raw == "0":
        return False
    return None


def _checkpoint_modes(default_checkpointing):
    if default_checkpointing:
        return (True, False)
    return (False, True)


def _resolve_gpu_profile(gpu_name, capability, gpu_vram_gb, is_windows):
    name = gpu_name.lower()
    arch = SUPPORTED_CONSUMER_CAPABILITIES.get(capability)
    min_vram_gb = MIN_SUPPORTED_VRAM_GB_BY_ARCH.get(arch, float("inf"))
    is_rtx = "rtx" in name
    is_laptop = "laptop" in name
    supported_consumer = (
        is_rtx
        and not is_laptop
        and arch is not None
        and _vram_reaches(gpu_vram_gb, min_vram_gb)
    )

    if supported_consumer:
        if arch == "turing" and not _vram_reaches(gpu_vram_gb, 12.0):
            return GpuProfile(
                name=f"{arch}-8-11gb",
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(8, 4, 2, 1),
                checkpoint_modes=_checkpoint_modes(True),
                default_checkpointing=True,
                eval_batch_cap=4,
            )
        if not _vram_reaches(gpu_vram_gb, 16.0):
            mid_tier_name = f"{arch}-12-15gb" if arch == "turing" else f"{arch}-10-15gb"
            return GpuProfile(
                name=mid_tier_name,
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(16, 8, 4),
                checkpoint_modes=_checkpoint_modes(True),
                default_checkpointing=True,
            )
        if not _vram_reaches(gpu_vram_gb, 24.0):
            return GpuProfile(
                name=f"{arch}-16gb",
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(32, 16, 8, 4),
                checkpoint_modes=_checkpoint_modes(False),
                default_checkpointing=False,
            )
        return GpuProfile(
            name=f"{arch}-24gb-plus",
            is_supported_consumer=True,
            is_compatibility_only=False,
            train_batch_candidates=(64, 32, 16, 8, 4),
            checkpoint_modes=_checkpoint_modes(False),
            default_checkpointing=False,
        )

    default_checkpointing = is_windows or not _vram_reaches(gpu_vram_gb, 16.0)
    return GpuProfile(
        name="compatibility",
        is_supported_consumer=False,
        is_compatibility_only=True,
        train_batch_candidates=(DEVICE_BATCH_SIZE, 16, 8, 4),
        checkpoint_modes=(default_checkpointing,),
        default_checkpointing=default_checkpointing,
    )


def _compatibility_warning(gpu_name, capability, gpu_vram_gb):
    name = gpu_name.lower()
    arch = SUPPORTED_CONSUMER_CAPABILITIES.get(capability)
    if "rtx" not in name:
        return None
    if "laptop" in name:
        return "laptop GPUs are outside the supported desktop matrix"
    if arch is None:
        return f"compute capability {capability[0]}.{capability[1]} is outside supported consumer tiers"
    min_vram_gb = MIN_SUPPORTED_VRAM_GB_BY_ARCH.get(arch, float("inf"))
    if not _vram_reaches(gpu_vram_gb, min_vram_gb):
        return f"{gpu_vram_gb:.1f} GB VRAM is below the {min_vram_gb:g} GB floor for {arch}"
    return None


def _get_autotune_cache_path():
    if platform.system().lower().startswith("win"):
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else (Path.home() / "AppData" / "Local")
    else:
        base = Path.home() / ".cache"
    return base / "autoresearch" / f"{AUTOTUNE_CACHE_VERSION}.json"


def _load_autotune_entries(path):
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"Warning: could not read autotune cache ({exc}); ignoring cache.")
        return {}
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("entries", {})
    return entries if isinstance(entries, dict) else {}


def _save_autotune_entries(path, entries):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        payload = {"entries": entries}
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp_path.replace(path)
    except Exception as exc:
        print(f"Warning: could not write autotune cache ({exc}).")


def _make_autotune_cache_key(runtime):
    cc = f"{runtime.gpu_cc[0]}.{runtime.gpu_cc[1]}"
    return "|".join(
        [
            runtime.gpu_name,
            cc,
            str(runtime.gpu_total_memory_bytes),
            torch.__version__,
            platform.system(),
            str(MAX_SEQ_LEN),
        ]
    )


def _select_amp_dtype(gpu_cc):
    if gpu_cc >= (8, 0) and torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16
    return torch.float16


def detect_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. No CUDA device detected.")

    is_windows = platform.system().lower().startswith("win")
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name()
    gpu_total_memory_bytes = int(props.total_memory)
    gpu_vram_gb = gpu_total_memory_bytes / (1024 ** 3)
    gpu_cc = torch.cuda.get_device_capability()
    gpu_profile = _resolve_gpu_profile(gpu_name, gpu_cc, gpu_vram_gb, is_windows)
    warning = _compatibility_warning(gpu_name, gpu_cc, gpu_vram_gb)
    if warning is not None:
        print(f"Warning: {warning}; running compatibility runtime path.")

    amp_dtype = _select_amp_dtype(gpu_cc)
    tf32_enabled = bool(getattr(torch.cuda, "is_tf32_supported", lambda: False)())
    torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = tf32_enabled

    use_compile = False
    print("torch.compile disabled in this fork runtime path.")
    attention_backend = "sdpa"
    print("Using PyTorch SDPA attention backend.")
    forced_checkpointing = _forced_checkpointing()
    if forced_checkpointing is None:
        use_activation_checkpointing = gpu_profile.default_checkpointing
    else:
        use_activation_checkpointing = forced_checkpointing
        print(
            "Activation checkpointing forced by AUTORESEARCH_FORCE_CHECKPOINTING="
            f"{'1' if forced_checkpointing else '0'}"
        )

    return RuntimeConfig(
        device=device,
        device_type=device.type,
        amp_dtype=amp_dtype,
        use_compile=use_compile,
        use_activation_checkpointing=use_activation_checkpointing,
        attention_backend=attention_backend,
        gpu_name=gpu_name,
        gpu_vram_gb=gpu_vram_gb,
        gpu_peak_flops=_get_gpu_peak_flops(gpu_name),
        gpu_cc=gpu_cc,
        gpu_total_memory_bytes=gpu_total_memory_bytes,
        tf32_enabled=tf32_enabled,
        gpu_profile=gpu_profile,
    )


USE_COMPILE = False
MUON_COMPUTE_DTYPE = torch.bfloat16


def _maybe_compile(obj, **kwargs):
    return obj


# ---------------------------------------------------------------------------
# GPT Model
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"
    attention_backend: str = "sdpa"
    use_activation_checkpointing: bool = False
    compute_dtype: torch.dtype = torch.bfloat16


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2


# ---------------------------------------------------------------------------
# Init-scale census and Muon's small-init precondition (AUT-37)
# ---------------------------------------------------------------------------
# Two literature lines disagree about where a matrix should be initialised, and
# both state their case as a property of *the matrices*:
#
#   * arXiv:2601.13474 (Ma, Huang, Chi, Chen) proves Muon's convergence rate is
#     independent of the condition number, and states that all guarantees
#     "require small initialization". It verifies alpha = 0.1 and names
#     robustness to a broader init range as future work.
#   * arXiv:2606.17945 and arXiv:2608.30315 write std = d**-gamma as a *single*
#     gamma shared by all matrices, and reason about gradient flow from it.
#
# Neither can notice us. init_weights sets four different scales and nothing
# downstream can tell: no assert, no warning, no log line, and the run looks
# completely normal. So the scales are measured here, once, from the tensors
# themselves, and the two preconditions are asserted rather than assumed.
#
# What the tensors actually are at n_embd = 512 (see the census this prints):
#
#   wte           normal(0, 1.0)      gamma 0.000   AdamW
#   hidden        uniform(-s, s)      gamma 0.500   Muon   (s/sqrt(3) = d**-0.5)
#   c_proj, gate  zeros               gamma inf     Muon
#   lm_head       normal(0, 0.001)    gamma 1.107   AdamW
#   value_embeds  uniform(-s, s)      gamma 0.500   AdamW
#
# The 22.6x embedding/hidden gap is real and is not Muon's problem: Muon is
# given only the hidden matrices, and those are already uniform at one gamma.
# The ceiling below is what stops "uniform at one gamma" from being satisfied
# degenerately - every matrix at std 1.0 would pass a uniformity check and
# break every guarantee in 2601.13474.
MUON_MAX_INIT_RMS = 0.1
MUON_INIT_UNIFORMITY_RTOL = 0.05

# Expected init scale per role, as gamma in the std = d**-gamma convention.
# Informational, not enforced: changing an init scale is a legitimate
# experiment, and the census this guard prints is the record of the new value.
# Re-baseline here when one is made, so drift stops being silent.
INIT_GAMMA_REFERENCE = {
    "wte": 0.0,
    "value_embeds": 0.5,
    "lm_head": 1.107,
    "muon": 0.5,
    "muon_zero": math.inf,
}
INIT_GAMMA_RTOL = 0.02

# value_embeds is the one tensor whose init scale is *not* discarded, and it is
# the only init knob left that is live.
#
# wte gets its scale thrown away twice over: `x = norm(wte(idx))` removes it on
# the forward pass, and AdamW removes it from the update (|g| falls as
# 1/init_scale, and the update divides the scale straight back out). So wte's
# init scale reaches neither the forward pass nor the optimizer. value_embeds
# gets neither treatment: v = c_v(x) + gate * ve is a plain sum with no norm on
# it, so the table's own entry RMS is a direct multiplier on the value vector.
#
# At the shipped gamma 0.500 the table lands at entry RMS 0.0442 while
# c_v(norm(x)) lands at ~1.0, so the lookup starts 22.6x weaker than the context
# path it is added to. Two independently derived arguments say to start it
# stronger: arXiv:2606.02780 reports the lookup and c_v(x) alone at identical
# BPB, and eps = 1e-10 (AUT-34) prefers the larger gradient that larger entries
# produce. Neither is a muP argument and neither is in the muP literature.
#
# Arm selection is by environment rather than by a code edit per run, because
# the arms are one constant apart and the census below already prints the value
# that was used:
#
#   AUTORESEARCH_VE_INIT_GAMMA=0.000   parity with c_v(x); std 1.0, what wte uses
#   AUTORESEARCH_VE_INIT_GAMMA=0.250   midpoint, in case parity overshoots
#   (unset)                            0.500, the shipped value
#
# Not claimed: that a larger ve is better. That needs a run, and the effect may
# sit under the unmeasured val_bpb noise floor like the rest of the backlog.
#
# INIT_GAMMA_REFERENCE["value_embeds"] stays at 0.500 on purpose: both arms trip
# the drift warning, and that warning is the "this init change was deliberate"
# line in the run log. Re-baseline it when an arm is adopted, not when it is
# proposed.
VALUE_EMBED_INIT_GAMMA = 0.500


def _ve_init_gamma():
    """The value-embedding init gamma this run uses, and where it came from.

    Read once per process so that init_weights, the census and the arm table
    cannot disagree, and rejected loudly rather than silently falling back:
    a typo'd arm that quietly trained at 0.500 would be the worst outcome,
    because it would produce a run that looks fine and answers nothing.
    """
    raw = os.environ.get("AUTORESEARCH_VE_INIT_GAMMA")
    if raw is None or raw.strip() == "":
        return VALUE_EMBED_INIT_GAMMA, "default"
    try:
        gamma = float(raw)
    except ValueError:
        raise ValueError(
            f"AUTORESEARCH_VE_INIT_GAMMA={raw!r} is not a number; expected a gamma in "
            f"[0, 1] (0.000 = parity with c_v(x), 0.500 = the shipped value)"
        ) from None
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"AUTORESEARCH_VE_INIT_GAMMA={raw!r} is outside [0, 1]; the arms under test "
            f"are 0.000 and 0.250 and the shipped value is 0.500"
        )
    return gamma, "AUTORESEARCH_VE_INIT_GAMMA"


def _entry_rms(tensor):
    """Root-mean-square of a parameter, accumulated in fp32 without casting the whole tensor.

    init_weights runs inside the timed region and the autotune path is VRAM-bound, so
    this deliberately avoids materialising an fp32 copy of a 2048x8192 table.
    """
    return float(
        torch.linalg.vector_norm(tensor.detach(), dtype=torch.float32)
        / (tensor.numel() ** 0.5)
    )


def _gamma_of(entry_rms, n_embd):
    """gamma in the std = d**-gamma; math.inf for an exactly-zero tensor."""
    if entry_rms <= 0.0:
        return math.inf
    return -math.log(entry_rms) / math.log(n_embd)


def _assert_muon_init_precondition(model, muon_params):
    """Assert the two init properties Muon is entitled to assume, and print the census.

    Called from setup_optimizer, where the Muon/AdamW partition is built, so the
    set of tensors checked is the set Muon actually owns and cannot drift away
    from it. Raises RuntimeError naming the offending tensor; a legitimate init
    experiment is a deliberate edit to the constants above, not a way to make
    this stop complaining.
    """
    n_embd = model.config.n_embd
    failures = []

    # (1) The two zero-initialised residual branches are what make the network an
    # identity map at step 0, and therefore what makes WARMUP_RATIO = 0 safe.
    # Checked per block by name rather than as a set of tensors, so that changing
    # which layers carry value embeddings does not turn this into a false alarm.
    for block in model.transformer.h:
        for sub in (block.attn, block.mlp):
            max_abs = float(sub.c_proj.weight.detach().abs().max())
            if max_abs != 0.0:
                failures.append(
                    f"{type(sub).__name__}.c_proj is not zero-init (max abs {max_abs:.6g}); "
                    "the residual branches must start as the identity map or "
                    "WARMUP_RATIO = 0 is unsafe"
                )

    # (2) and (3): one shared gamma across every non-degenerate Muon matrix, and
    # that gamma inside the range the guarantees were verified over.
    muon_rms = {id(p): _entry_rms(p) for p in muon_params}
    nonzero = sorted(v for v in muon_rms.values() if v > 0.0)
    zero_count = len(muon_rms) - len(nonzero)
    if not nonzero:
        failures.append("every Muon tensor is zero-init; there is nothing to precondition")
    else:
        median = nonzero[len(nonzero) // 2]
        shared_gamma = _gamma_of(median, n_embd)
        if median > MUON_MAX_INIT_RMS:
            failures.append(
                f"entry RMS {median:.6g} exceeds MUON_MAX_INIT_RMS {MUON_MAX_INIT_RMS} "
                f"(gamma {shared_gamma:.4f}); 2601.13474 verifies its guarantees only "
                f"at alpha=0.1, so this run would be outside its theory"
            )
        for p in muon_params:
            rms = muon_rms[id(p)]
            if rms > 0.0 and abs(rms - median) > MUON_INIT_UNIFORMITY_RTOL * median:
                failures.append(
                    f"{_muon_param_name(model, p)} entry RMS {rms:.6g} differs from the "
                    f"other Muon tensors' {median:.6g} (gamma "
                    f"{_gamma_of(rms, n_embd):.4f} vs {shared_gamma:.4f}); the small-init "
                    f"papers assume one gamma across all matrices and do not cover this"
                )

    # Census. Printed on every run so that any future init change is a diff in
    # the log rather than something nobody can find afterwards.
    census = {
        "wte": _entry_rms(model.transformer.wte.weight),
        "value_embeds": max(
            (_entry_rms(ve.weight) for ve in model.value_embeds.values()), default=0.0
        ),
        "lm_head": _entry_rms(model.lm_head.weight),
        "muon": nonzero[len(nonzero) // 2] if nonzero else 0.0,
        "muon_zero": 0.0,
    }
    owner = {
        "wte": "adamw",
        "value_embeds": "adamw",
        "lm_head": "adamw",
        "muon": "muon",
        "muon_zero": "muon",
    }
    print(f"Init scale census at n_embd={n_embd} (std = d**-gamma, {len(muon_params)} Muon tensors):")
    for role, rms in census.items():
        gamma = _gamma_of(rms, n_embd)
        reference = INIT_GAMMA_REFERENCE[role]
        print(f"  {role:14s} owner={owner[role]:5s} entry_rms={rms:.6g}  gamma={gamma:.4f}")
        if gamma != reference and abs(gamma - reference) > INIT_GAMMA_RTOL:
            print(
                f"  WARNING: init scale for {role} is gamma {gamma:.4f}, not the recorded "
                f"{reference:.4f}. Intentional? Re-baseline INIT_GAMMA_REFERENCE."
            )
    print(f"  {zero_count} of {len(muon_params)} Muon tensors are zero-init (c_proj, ve_gate)")

    # Which value-embedding arm produced this census. The gamma above is measured
    # from the tensors, so it is already in the log; this records the *source*, so
    # a run can be reproduced from its own log without knowing the shell it ran in.
    ve_gamma, ve_source = _ve_init_gamma()
    print(
        f"  value_embeds arm: gamma {ve_gamma:.4f} (std {n_embd ** -ve_gamma:.4f}) "
        f"from {ve_source}"
    )

    if failures:
        raise RuntimeError(
            "Muon init precondition violated (AUT-37):\n  - "
            + "\n  - ".join(failures)
            + "\n  The scales above are printed so the offending one is identifiable."
        )


def _muon_param_name(model, param):
    for name, tensor in model.named_parameters():
        if tensor is param:
            return name
    return "<unnamed Muon tensor>"


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.attention_backend = config.attention_backend
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        self._mask_cache = {}

    def _get_sdpa_mask(self, seq_len, window_size, device):
        window = window_size[0] if isinstance(window_size, tuple) else window_size
        cache_key = (seq_len, int(window), device.type, device.index)
        mask = self._mask_cache.get(cache_key)
        if mask is not None:
            return mask

        row = torch.arange(seq_len, device=device).unsqueeze(1)
        col = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = col <= row  # causal
        if window is not None and window >= 0 and window < seq_len:
            mask = mask & (col >= (row - window))
        self._mask_cache[cache_key] = mask
        return mask

    def forward(self, x, ve, cos_sin, window_size):
        B, T, _ = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        q = q.transpose(1, 2)  # (B, H, T, D)
        k = k.transpose(1, 2)  # (B, KVH, T, D)
        v = v.transpose(1, 2)  # (B, KVH, T, D)
        attn_mask = self._get_sdpa_mask(T, window_size, q.device)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            enable_gqa=self.n_kv_head < self.n_head,
        )
        y = y.transpose(1, 2)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        self.rotary_seq_len = config.sequence_len
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim, dtype=config.compute_dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self, embed_dtype=torch.bfloat16):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        n_embd = self.config.n_embd
        s = 3 ** 0.5 * n_embd ** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        # Same form as the hidden matrices, with value_embeds' own gamma: the
        # shipped 0.500 reproduces s exactly (3**0.5 * n_embd**-0.5), so this is
        # a no-op at the default and a one-constant change per arm.
        ve_gamma, _ = _ve_init_gamma()
        ve_s = 3 ** 0.5 * n_embd ** -ve_gamma
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -ve_s, ve_s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len,
            head_dim,
            dtype=self.config.compute_dtype,
        )
        self.cos, self.sin = cos, sin
        self.transformer.wte.to(dtype=embed_dtype)
        for ve in self.value_embeds.values():
            ve.to(dtype=embed_dtype)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None, dtype=torch.bfloat16):
        if device is None:
            device = self.transformer.wte.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern)
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel()
            + value_embeds_numel
            + self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
        )
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        t = self.config.sequence_len
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        return 6 * (nparams - nparams_exclude) + attn_flops

    def num_scaling_params(self):
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        model_dim = self.config.n_embd
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        assert len(list(self.parameters())) == (
            len(matrix_params)
            + len(embedding_params)
            + len(lm_head_params)
            + len(value_embeds_params)
            + len(resid_params)
            + len(x0_params)
        )
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        param_groups = [
            dict(kind="adamw", params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            dict(kind="adamw", params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        muon_group_chunk = 8
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            for ci in range(0, len(group_params), muon_group_chunk):
                chunk = group_params[ci:ci + muon_group_chunk]
                param_groups.append(
                    dict(
                        kind="muon",
                        params=chunk,
                        lr=matrix_lr,
                        momentum=0.95,
                        ns_steps=5,
                        beta2=0.95,
                        weight_decay=weight_decay,
                    )
                )
        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        muon_params = [p for group in param_groups if group["kind"] == "muon" for p in group["params"]]
        _assert_muon_init_precondition(self, muon_params)
        return optimizer

    def forward(self, idx, targets=None, reduction="mean"):
        B, T = idx.size()
        assert T <= self.cos.size(1)
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            window_size = self.window_sizes[i]
            if self.config.use_activation_checkpointing:
                x = torch_checkpoint(block, x, ve, cos_sin, window_size, use_reentrant=False)
            else:
                x = block(x, ve, cos_sin, window_size)
        x = norm(x)

        softcap = 15
        logits = self.lm_head(x).float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits


# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)
# ---------------------------------------------------------------------------

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]


def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    p.mul_(1 - lr_t * wd_t)
    # Keep moments in their own dtype (float32 for fp16 params) to avoid grad^2 underflow.
    g = grad.to(exp_avg.dtype)
    exp_avg.lerp_(g, 1 - beta1_t)
    exp_avg_sq.lerp_(g.square(), 1 - beta2_t)
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_((exp_avg / denom * (-step_size)).to(p.dtype))


def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    X = g.to(dtype=MUON_COMPUTE_DTYPE)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


ADAMW_STEP_IMPL = adamw_step_fused
MUON_STEP_IMPL = muon_step_fused


class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group):
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if not state:
                state["step"] = 0
                moment_dtype = torch.float32 if p.dtype == torch.float16 else p.dtype
                state["exp_avg"] = torch.zeros_like(p, dtype=moment_dtype)
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=moment_dtype)
            state["step"] += 1
            self._adamw_step_t.fill_(state["step"])
            self._adamw_lr_t.fill_(group["lr"])
            self._adamw_beta1_t.fill_(group["betas"][0])
            self._adamw_beta2_t.fill_(group["betas"][1])
            self._adamw_eps_t.fill_(group["eps"])
            self._adamw_wd_t.fill_(group["weight_decay"])
            ADAMW_STEP_IMPL(
                p,
                grad,
                state["exp_avg"],
                state["exp_avg_sq"],
                self._adamw_step_t,
                self._adamw_lr_t,
                self._adamw_beta1_t,
                self._adamw_beta2_t,
                self._adamw_eps_t,
                self._adamw_wd_t,
            )

    def _step_muon(self, group):
        params = group["params"]
        if not params:
            return
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1]) ** 0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        MUON_STEP_IMPL(
            stacked_grads,
            stacked_params,
            state["momentum_buffer"],
            state["second_momentum_buffer"],
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group["kind"] == "adamw":
                self._step_adamw(group)
            elif group["kind"] == "muon":
                self._step_muon(group)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
ASPECT_RATIO = 64         # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128            # target head dimension for attention
WINDOW_PATTERN = "SSSL"   # sliding window pattern: L=full, S=half context

# Optimization
TOTAL_BATCH_SIZE = 2 ** 19
EMBEDDING_LR = 0.6
UNEMBEDDING_LR = 0.004
MATRIX_LR = 0.04
SCALAR_LR = 0.5
WEIGHT_DECAY = 0.2
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.5
FINAL_LR_FRAC = 0.0

# Model size + memory defaults
DEPTH = 8
DEVICE_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 8
# AUT-32: 2^22 = 2,048 rows, 96.30% of the val documents
EVAL_BUDGET_TOKENS = 8 * 524288


def build_model_config(depth, vocab_size, runtime, use_activation_checkpointing=None):
    if use_activation_checkpointing is None:
        use_activation_checkpointing = runtime.use_activation_checkpointing
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    num_heads = model_dim // HEAD_DIM
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
        attention_backend=runtime.attention_backend,
        use_activation_checkpointing=use_activation_checkpointing,
        compute_dtype=runtime.amp_dtype,
    )


def _filter_train_batch_sizes(candidates):
    deduped = []
    for batch_size in list(candidates):
        if batch_size <= 0:
            continue
        tokens_per_fwdbwd = batch_size * MAX_SEQ_LEN
        if TOTAL_BATCH_SIZE % tokens_per_fwdbwd != 0:
            continue
        if batch_size not in deduped:
            deduped.append(batch_size)
    if not deduped:
        raise RuntimeError("No valid device batch sizes satisfy TOTAL_BATCH_SIZE divisibility.")
    return deduped


def _build_train_candidates(runtime):
    batch_sizes = _filter_train_batch_sizes(runtime.gpu_profile.train_batch_candidates)
    checkpoint_modes = runtime.gpu_profile.checkpoint_modes
    forced_checkpointing = _forced_checkpointing()
    if forced_checkpointing is not None:
        narrowed = tuple(m for m in checkpoint_modes if m is forced_checkpointing)
        checkpoint_modes = narrowed or (forced_checkpointing,)
    candidates = []
    for checkpointing in checkpoint_modes:
        for batch_size in batch_sizes:
            candidate = (batch_size, checkpointing)
            if candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No train candidates available for this runtime profile.")
    return candidates


def _build_eval_batch_candidates(train_batch_size, initial_eval_batch):
    candidates = [min(initial_eval_batch, train_batch_size), 8, 4, 2, 1]
    deduped = []
    for batch_size in candidates:
        if batch_size > 0 and batch_size not in deduped:
            deduped.append(batch_size)
    return deduped


# ---------------------------------------------------------------------------
# Autotune probe outcome classification
# ---------------------------------------------------------------------------
# A rejected autotune candidate has two very different causes and they must not
# be reported the same way. A capacity rejection (the GPU genuinely could not
# fit the shape) is expected and quiet. A probe error (a shape mismatch, a NaN,
# a device-side assert, a bug in the model) is a code defect that used to be
# printed as though it were a hardware limit, silently narrowing the candidate
# set and caching a worse train_batch_size for every later run.

PROBE_ACCEPTED = "accepted"
PROBE_OUT_OF_MEMORY = "out_of_memory"
PROBE_VRAM_LIMIT = "vram_limit"
PROBE_ERROR = "error"

# Narrow on purpose. "failed to allocate" and "alloc failed" are excluded: the
# torch messages "Failed to allocate tensor with shape ..." describe a shape
# error, which is a code bug and must not be filed as a capacity rejection.
_OOM_MESSAGE_MARKERS = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudaerrormemoryallocation",
)


@dataclass
class _TrainProbeOutcome:
    kind: str
    train_batch_size: int
    use_checkpointing: bool
    tok_per_sec: float = 0.0
    peak_memory_bytes: int = 0
    detail: str = ""

    @property
    def is_accepted(self):
        return self.kind == PROBE_ACCEPTED

    @property
    def is_capacity_rejection(self):
        return self.kind in (PROBE_OUT_OF_MEMORY, PROBE_VRAM_LIMIT)

    @property
    def is_error(self):
        return self.kind == PROBE_ERROR


def _is_out_of_memory_exception(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MESSAGE_MARKERS)


def _classify_probe_exception(exc):
    """Bucket a probe failure by inspecting the exception, not by catching broadly."""
    if _is_out_of_memory_exception(exc):
        return PROBE_OUT_OF_MEMORY
    return PROBE_ERROR


def _format_probe_exception(exc):
    return f"{type(exc).__name__}: {exc}".strip()


def _probe_candidate_label(train_batch_size, use_checkpointing):
    return f"batch_size={train_batch_size}, checkpointing={'on' if use_checkpointing else 'off'}"


def _print_probe_error(outcome, traceback_text):
    print(
        "\n".join(
            [
                "",
                "!" * 78,
                "AUTOTUNE PROBE FAILED - this is a code error, not a VRAM limit.",
                f"  candidate: {_probe_candidate_label(outcome.train_batch_size, outcome.use_checkpointing)}",
                f"  exception: {outcome.detail}",
                f"  traceback: {traceback_text}",
                "  This candidate is NOT recorded as a capacity rejection. The autotune",
                "  summary reports it separately and the cache entry records it so the",
                "  failure resurfaces on later runs instead of staying silent.",
                "!" * 78,
                "",
            ]
        ),
        flush=True,
    )


def _format_autotune_probe_summary(capacity_rejections, probe_errors, best_outcome):
    accepted_count = 1 if best_outcome is not None else 0
    lines = [
        "Autotune probe summary: "
        f"{accepted_count} accepted, "
        f"{len(capacity_rejections)} capacity rejection(s), "
        f"{len(probe_errors)} probe error(s)."
    ]
    if not probe_errors:
        return "\n".join(lines)
    lines.append("")
    lines.append("PROBE ERRORS (not VRAM limits - investigate these as code bugs):")
    seen = []
    for outcome in probe_errors:
        if outcome.detail not in seen:
            seen.append(outcome.detail)
    for detail in seen:
        occurrences = sum(1 for outcome in probe_errors if outcome.detail == detail)
        suffix = f" (x{occurrences} candidate(s))" if occurrences > 1 else ""
        lines.append(f"  - {detail}{suffix}")
    lines.append("")
    lines.append("The selected candidate is the fastest probe that completed. It is not proof")
    lines.append("that the erroring candidates were too large for this GPU.")
    return "\n".join(lines)


def _warn_if_cached_probe_errors(cached_entry):
    """Re-surface probe errors recorded when a cached candidate was chosen."""
    details = cached_entry.get("probe_errors")
    if not isinstance(details, list):
        return
    details = [detail for detail in details if isinstance(detail, str) and detail.strip()]
    if not details:
        return
    recorded_count = cached_entry.get("probe_error_count")
    total = recorded_count if isinstance(recorded_count, int) else len(details)
    print(
        "WARNING: this cached autotune candidate was recorded while "
        f"{total} candidate probe(s) failed with non-capacity errors. "
        "The candidate is the fastest probe that completed, not a hardware limit."
    )
    print("  Re-run with AUTORESEARCH_AUTOTUNE_REFRESH=1 after fixing the errors below.")
    for detail in details:
        print(f"  - {detail}")


def _benchmark_train_candidate(runtime, tokenizer, vocab_size, train_batch_size, use_checkpointing):
    config = build_model_config(
        DEPTH,
        vocab_size,
        runtime,
        use_activation_checkpointing=use_checkpointing,
    )
    tokens_per_micro_step = train_batch_size * MAX_SEQ_LEN
    measure_micro_steps = max(
        AUTOTUNE_MIN_MEASURE_MICRO_STEPS,
        AUTOTUNE_MEASURE_TOKENS // tokens_per_micro_step,
    )
    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)

    model = None
    optimizer = None
    train_loader = None
    x = y = None
    try:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=runtime.device)
        model.init_weights(embed_dtype=runtime.amp_dtype)
        optimizer = model.setup_optimizer(
            unembedding_lr=UNEMBEDDING_LR,
            embedding_lr=EMBEDDING_LR,
            scalar_lr=SCALAR_LR,
            adam_betas=ADAM_BETAS,
            matrix_lr=MATRIX_LR,
            weight_decay=WEIGHT_DECAY,
        )
        train_loader = make_dataloader(
            tokenizer,
            train_batch_size,
            MAX_SEQ_LEN,
            "train",
            device=runtime.device,
            dataset=tokenizer.dataset,
        )
        x, y, _ = next(train_loader)
        torch.cuda.empty_cache()

        def run_micro_steps(count):
            nonlocal x, y
            for _ in range(count):
                model.zero_grad(set_to_none=True)
                with autocast_ctx:
                    loss = model(x, y)
                loss.backward()
                x, y, _ = next(train_loader)

        run_micro_steps(measure_micro_steps)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        run_micro_steps(measure_micro_steps)
        torch.cuda.synchronize()
        measured_time = time.time() - t0
        optimizer.step()
        model.zero_grad(set_to_none=True)

        peak_memory = torch.cuda.max_memory_allocated()
        peak_limit = runtime.gpu_total_memory_bytes * AUTOTUNE_MAX_MEMORY_FRACTION
        if peak_memory > peak_limit:
            return _TrainProbeOutcome(
                PROBE_VRAM_LIMIT,
                train_batch_size,
                use_checkpointing,
                peak_memory_bytes=peak_memory,
                detail=(
                    f"peak_vram_mb={peak_memory / 1024 / 1024:.1f} exceeds "
                    f"{AUTOTUNE_MAX_MEMORY_FRACTION:.0%} of "
                    f"{runtime.gpu_total_memory_bytes / 1024 / 1024:.1f} MB"
                ),
            )
        tokens_measured = measure_micro_steps * tokens_per_micro_step
        tok_per_sec = tokens_measured / max(measured_time, 1e-6)
        return _TrainProbeOutcome(
            PROBE_ACCEPTED,
            train_batch_size,
            use_checkpointing,
            tok_per_sec=tok_per_sec,
            peak_memory_bytes=peak_memory,
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _TrainProbeOutcome(
            PROBE_OUT_OF_MEMORY,
            train_batch_size,
            use_checkpointing,
            detail=_format_probe_exception(exc),
        )
    except Exception as exc:
        kind = _classify_probe_exception(exc)
        detail = _format_probe_exception(exc)
        if kind == PROBE_OUT_OF_MEMORY:
            return _TrainProbeOutcome(
                PROBE_OUT_OF_MEMORY,
                train_batch_size,
                use_checkpointing,
                detail=detail,
            )
        outcome = _TrainProbeOutcome(
            PROBE_ERROR,
            train_batch_size,
            use_checkpointing,
            detail=detail,
        )
        _print_probe_error(outcome, traceback.format_exc())
        return outcome
    finally:
        del x, y, train_loader, optimizer, model
        torch.cuda.empty_cache()
        _restore_gc_after_attempt()


def _autotune_train_candidate(runtime, tokenizer, vocab_size, train_candidates):
    if not runtime.gpu_profile.is_supported_consumer:
        return None
    if os.environ.get("AUTORESEARCH_DISABLE_AUTOTUNE", "0") == "1":
        print("Autotune disabled by AUTORESEARCH_DISABLE_AUTOTUNE=1.")
        return None

    cache_path = _get_autotune_cache_path()
    cache_key = _make_autotune_cache_key(runtime)
    refresh_cache = os.environ.get("AUTORESEARCH_AUTOTUNE_REFRESH", "0") == "1"
    cache_entries = _load_autotune_entries(cache_path)
    if refresh_cache:
        print("Autotune cache refresh requested by AUTORESEARCH_AUTOTUNE_REFRESH=1.")
    else:
        cached = cache_entries.get(cache_key)
        if isinstance(cached, dict):
            cached_batch_size = cached.get("train_batch_size")
            cached_checkpointing = cached.get("use_activation_checkpointing")
            if isinstance(cached_batch_size, int) and isinstance(cached_checkpointing, bool):
                cached_candidate = (cached_batch_size, cached_checkpointing)
                if cached_candidate in train_candidates:
                    print(
                        "Using cached autotune candidate: "
                        f"batch_size={cached_batch_size}, checkpointing={'on' if cached_checkpointing else 'off'}."
                    )
                    _warn_if_cached_probe_errors(cached)
                    return cached_candidate

    print("Running consumer GPU autotune in eager mode...")
    best_outcome = None
    capacity_rejections = []
    probe_errors = []
    for train_batch_size, use_checkpointing in train_candidates:
        ckpt_label = "on" if use_checkpointing else "off"
        print(f"Autotune probe: train_batch_size={train_batch_size}, checkpointing={ckpt_label}")
        outcome = _benchmark_train_candidate(
            runtime=runtime,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            train_batch_size=train_batch_size,
            use_checkpointing=use_checkpointing,
        )
        if outcome.is_accepted:
            print(
                f"  accepted: tok/sec={outcome.tok_per_sec:,.0f}, "
                f"peak_vram_mb={outcome.peak_memory_bytes / 1024 / 1024:.1f}"
            )
            if best_outcome is None or outcome.tok_per_sec > best_outcome.tok_per_sec:
                best_outcome = outcome
            continue
        if outcome.is_error:
            probe_errors.append(outcome)
            print("  rejected (probe error - see the banner above and the summary below)")
            continue
        capacity_rejections.append(outcome)
        reason = (
            "out of memory"
            if outcome.kind == PROBE_OUT_OF_MEMORY
            else f"peak VRAM above {AUTOTUNE_MAX_MEMORY_FRACTION:.0%} of total"
        )
        print(f"  rejected (capacity: {reason})")

    print(_format_autotune_probe_summary(capacity_rejections, probe_errors, best_outcome))

    if best_outcome is None:
        print("Autotune could not find a viable candidate; using default fallback ordering.")
        if probe_errors:
            print("Autotune found no viable candidate because probes errored, not because of VRAM.")
        return None

    cache_entries[cache_key] = {
        "train_batch_size": best_outcome.train_batch_size,
        "use_activation_checkpointing": best_outcome.use_checkpointing,
        "tok_per_sec": round(best_outcome.tok_per_sec, 3),
        "peak_memory_bytes": int(best_outcome.peak_memory_bytes),
        "probe_error_count": len(probe_errors),
        "probe_errors": sorted({outcome.detail for outcome in probe_errors}),
        "updated_unix": int(time.time()),
    }
    _save_autotune_entries(cache_path, cache_entries)
    print(
        "Autotune selected candidate: "
        f"batch_size={best_outcome.train_batch_size}, "
        f"checkpointing={'on' if best_outcome.use_checkpointing else 'off'}."
    )
    return (best_outcome.train_batch_size, best_outcome.use_checkpointing)


def _prioritize_autotuned_candidate(train_candidates, autotuned_candidate, prefer_small_batch=False):
    if autotuned_candidate is None or autotuned_candidate not in train_candidates:
        if prefer_small_batch:
            return sorted(train_candidates, key=lambda candidate: candidate[0])
        return train_candidates
    return [autotuned_candidate] + [c for c in train_candidates if c != autotuned_candidate]


def _configure_step_kernels(runtime):
    global ADAMW_STEP_IMPL, MUON_STEP_IMPL, USE_COMPILE, MUON_COMPUTE_DTYPE
    ADAMW_STEP_IMPL = adamw_step_fused
    MUON_STEP_IMPL = muon_step_fused
    if runtime.amp_dtype != torch.float16:
        MUON_COMPUTE_DTYPE = runtime.amp_dtype
        muon_reason = "matching AMP dtype"
    elif torch.cuda.is_bf16_supported(including_emulation=True):
        # Use bf16 for Muon orthogonalization when training runs in fp16 for better numeric headroom.
        MUON_COMPUTE_DTYPE = torch.bfloat16
        muon_reason = "fp16 AMP with bf16 support (native or emulated)"
    else:
        # Safety fallback when fp16 AMP is selected but bf16 isn't available in this runtime.
        MUON_COMPUTE_DTYPE = torch.float32
        muon_reason = "fp16 AMP without bf16 support; using fp32 fallback"
    print(f"Muon compute dtype: {MUON_COMPUTE_DTYPE} ({muon_reason})")
    USE_COMPILE = False


def _run_training_once(runtime, tokenizer, config, device_batch_size, smoke_test):
    t_start = time.time()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.set_float32_matmul_precision("high")

    autocast_ctx = torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype)

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=runtime.device)
    model.init_weights(embed_dtype=runtime.amp_dtype)

    param_counts = model.num_scaling_params()
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()

    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key:24s}: {value:,}")
    print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

    tokens_per_fwdbwd = device_batch_size * MAX_SEQ_LEN
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    optimizer = model.setup_optimizer(
        unembedding_lr=UNEMBEDDING_LR,
        embedding_lr=EMBEDDING_LR,
        scalar_lr=SCALAR_LR,
        adam_betas=ADAM_BETAS,
        matrix_lr=MATRIX_LR,
        weight_decay=WEIGHT_DECAY,
    )
    model = _maybe_compile(model, dynamic=False)

    train_loader = make_dataloader(
        tokenizer,
        device_batch_size,
        MAX_SEQ_LEN,
        "train",
        device=runtime.device,
        dataset=tokenizer.dataset,
    )
    x, y, epoch = next(train_loader)
    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    def get_lr_multiplier(progress):
        if progress < WARMUP_RATIO:
            return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
        if progress < 1.0 - WARMDOWN_RATIO:
            return 1.0
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

    def get_muon_momentum(step):
        frac = min(step / 300, 1)
        return (1 - frac) * 0.85 + frac * 0.95

    def get_weight_decay(progress):
        return WEIGHT_DECAY * (1 - progress)

    target_training_seconds = 10 if smoke_test else TIME_BUDGET
    max_steps = 3 if smoke_test else None

    t_start_training = time.time()
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0

    while True:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(grad_accum_steps):
            with autocast_ctx:
                loss = model(x, y)
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / max(target_training_seconds, 1e-6), 1.0)
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(progress)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay
        optimizer.step()
        model.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        if train_loss_f > 100:
            raise RuntimeError("FAIL: training loss exploded")

        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        if step > 10:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
        if runtime.gpu_peak_flops:
            mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / runtime.gpu_peak_flops
            mfu_text = f"{mfu:.1f}%"
        else:
            mfu_text = "n/a"
        remaining = max(0, target_training_seconds - total_training_time)
        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | "
            f"lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | "
            f"mfu: {mfu_text} | epoch: {epoch} | remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1
        if max_steps is not None and step >= max_steps:
            break
        if step > 10 and total_training_time >= target_training_seconds:
            break
        if smoke_test and total_training_time >= target_training_seconds:
            break

    print()
    return {
        "model": model,
        "num_params": num_params,
        "num_flops_per_token": num_flops_per_token,
        "total_training_time": total_training_time,
        "step": step,
        "t_start": t_start,
        "t_start_training": t_start_training,
    }


def _save_pre_eval_checkpoint(model):
    if os.environ.get("AUTORESEARCH_SAVE_PRE_EVAL_CHECKPOINT", "0") != "1":
        return
    try:
        state_dict = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        torch.save(state_dict, "checkpoint_pre_eval.pt")
        print("Saved checkpoint_pre_eval.pt")
    except Exception as exc:  # pragma: no cover
        print(f"Warning: could not save pre-eval checkpoint: {exc}")


def _restore_gc_after_attempt():
    if hasattr(gc, "unfreeze"):
        gc.unfreeze()
    gc.enable()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Autoresearch training script")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short train/eval pass for validation.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default=None, help="Optional dataset override.")
    args = parser.parse_args()

    runtime = detect_runtime()
    print(f"GPU: {runtime.gpu_name}")
    print(f"GPU VRAM: {runtime.gpu_vram_gb:.1f} GB")
    print(f"GPU CC: {runtime.gpu_cc[0]}.{runtime.gpu_cc[1]}")
    print(f"GPU profile: {runtime.gpu_profile.name}")
    print(f"Consumer matrix support: {'yes' if runtime.gpu_profile.is_supported_consumer else 'compatibility path'}")
    print(f"TF32: {'enabled' if runtime.tf32_enabled else 'disabled'}")
    print(f"AMP dtype: {runtime.amp_dtype}")

    tokenizer = Tokenizer.from_directory(dataset=args.dataset)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocab size: {vocab_size:,}")
    print(f"Dataset: {tokenizer.dataset}")

    # Configure optimizer kernels/dtypes before autotune so probes match real training runtime.
    _configure_step_kernels(runtime)

    train_candidates = _build_train_candidates(runtime)
    autotuned_candidate = _autotune_train_candidate(runtime, tokenizer, vocab_size, train_candidates)
    train_candidates = _prioritize_autotuned_candidate(
        train_candidates,
        autotuned_candidate,
        prefer_small_batch=runtime.gpu_profile.is_supported_consumer,
    )

    print(f"Attention backend: {runtime.attention_backend}")
    print(f"torch.compile: {'enabled' if USE_COMPILE else 'disabled'}")

    result = None
    chosen_train_batch = None
    chosen_checkpointing = None
    for train_batch_size, use_checkpointing in train_candidates:
        config = build_model_config(
            DEPTH,
            vocab_size,
            runtime,
            use_activation_checkpointing=use_checkpointing,
        )
        print(
            "Trying train candidate: "
            f"batch_size={train_batch_size}, "
            f"activation_checkpointing={'enabled' if use_checkpointing else 'disabled'}"
        )
        print(f"Model config: {asdict(config)}")
        try:
            result = _run_training_once(
                runtime=runtime,
                tokenizer=tokenizer,
                config=config,
                device_batch_size=train_batch_size,
                smoke_test=args.smoke_test,
            )
            chosen_train_batch = train_batch_size
            chosen_checkpointing = use_checkpointing
            break
        except torch.cuda.OutOfMemoryError:
            print(
                "Train OOM at "
                f"batch_size={train_batch_size}, checkpointing={'on' if use_checkpointing else 'off'}; "
                "trying next candidate."
            )
            torch.cuda.empty_cache()
            _restore_gc_after_attempt()
        except RuntimeError as exc:
            _restore_gc_after_attempt()
            print(str(exc))
            return 1

    if result is None:
        print("FAIL: training failed for all batch size candidates.")
        return 1

    model = result["model"]
    _save_pre_eval_checkpoint(model)
    model.eval()

    eval_tokens = (max(MAX_SEQ_LEN * chosen_train_batch * 2, 8192)
                   if args.smoke_test else EVAL_BUDGET_TOKENS)
    val_bpb = None
    chosen_eval_batch = None
    initial_eval_batch = min(chosen_train_batch, runtime.gpu_profile.eval_batch_cap)
    eval_candidates = _build_eval_batch_candidates(chosen_train_batch, initial_eval_batch)
    for eval_batch_size in eval_candidates:
        try:
            torch.cuda.empty_cache()
            with torch.amp.autocast(device_type=runtime.device_type, dtype=runtime.amp_dtype):
                val_bpb = evaluate_bpb(
                    model,
                    tokenizer,
                    eval_batch_size,
                    device=runtime.device,
                    dataset=tokenizer.dataset,
                    eval_tokens=eval_tokens,
                )
            chosen_eval_batch = eval_batch_size
            print(f"Eval completed with batch_size={eval_batch_size}")
            break
        except torch.cuda.OutOfMemoryError:
            print(f"Eval OOM at batch_size={eval_batch_size}; trying smaller batch.")
            torch.cuda.empty_cache()

    if val_bpb is None:
        print("FAIL: eval failed for all batch sizes.")
        return 1

    t_end = time.time()
    step = result["step"]
    total_training_time = result["total_training_time"]
    num_flops_per_token = result["num_flops_per_token"]
    num_params = result["num_params"]
    steady_state_steps = max(step - 10, 0)
    if runtime.gpu_peak_flops and total_training_time > 0 and steady_state_steps > 0:
        steady_state_mfu = (
            100
            * num_flops_per_token
            * TOTAL_BATCH_SIZE
            * steady_state_steps
            / total_training_time
            / runtime.gpu_peak_flops
        )
    else:
        steady_state_mfu = None
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    total_tokens = step * TOTAL_BATCH_SIZE

    print("---")
    print(f"val_bpb:          {val_bpb:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - result['t_start']:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    if steady_state_mfu is None:
        print("mfu_percent:      n/a")
    else:
        print(f"mfu_percent:      {steady_state_mfu:.2f}")
    print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")
    print(f"dataset:          {tokenizer.dataset}")
    print(f"train_batch_size: {chosen_train_batch}")
    print(f"eval_batch_size:  {chosen_eval_batch}")
    print(f"eval_tokens:      {eval_tokens}")
    print(f"activation_checkpointing: {'enabled' if chosen_checkpointing else 'disabled'}")
    if args.smoke_test:
        print("smoke_test:       true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
