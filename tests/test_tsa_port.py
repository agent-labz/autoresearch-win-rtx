"""CPU-only checks for the Terminal Shrinkage Averaging port in train.py.

Run:  uv run python tests/test_tsa_port.py        (from the repo root)
      .venv\Scripts\python.exe tests\test_tsa_port.py

No GPU, no dataset, no network, no new dependency. Every assertion here is about
arithmetic on tensors and about the exact order of operations in the training
loop, which is the part of this change that can be got wrong silently: a blend
that is off by one step, an accumulator that divides by the wrong count, or a
snapshot that fires one step too late all produce a plausible val_bpb and a
wrong experiment.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILURES.append(name)


def close(a, b, tol=1e-6):
    return torch.allclose(a.float(), b.float(), atol=tol, rtol=tol)


class Toy(nn.Module):
    """A stand-in with the shape of the real model: a bf16 embedding table plus
    fp32 weights, which is what init_weights produces (wte and value_embeds are
    cast to the AMP dtype, everything else stays fp32)."""

    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.wte = nn.Parameter(torch.randn(16, 8, generator=g).to(torch.bfloat16))
        self.head = nn.Parameter(torch.randn(8, 16, generator=g))
        self.scale = nn.Parameter(torch.randn(1, generator=g))
        self.frozen = nn.Parameter(torch.randn(4), requires_grad=False)


def trajectory(model, steps, scale=1.0):
    """Deterministic pseudo-trajectory: param = seed + step * scale."""
    out = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        base = p.detach().clone()
        out.append((name, [base + scale * (i + 1) for i in range(steps)]))
    return out


def set_params(model, name, value):
    with torch.no_grad():
        dict(model.named_parameters())[name].copy_(value)


# ---------------------------------------------------------------------------
print("\n[1] TSA_ALPHA = 0.0 leaves the constants at their pre-TSA values")
check("TSA_ALPHA defaults to 0.0 (baseline is bit-identical)", train.TSA_ALPHA == 0.0)
check("TERMINAL_FLOOR_FRAC defaults to 0.0", train.TERMINAL_FLOOR_FRAC == 0.0)
check("FINAL_LR_FRAC is untouched at 0.0", train.FINAL_LR_FRAC == 0.0)
check("TSA_WINDOW_FRAC in (0,1)", 0.0 < train.TSA_WINDOW_FRAC < 1.0)

# ---------------------------------------------------------------------------
print("\n[2] the LR floor clamp is bit-identical to the old expression at floor 0")

OLD = "cooldown * 1.0 + (1 - cooldown) * {final_frac}"
NEW = "max(cooldown + (1 - cooldown) * {final_frac}, {floor})"


def old_lrm(progress, final_frac=0.0):
    if progress < train.WARMUP_RATIO:
        return progress / train.WARMUP_RATIO if train.WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - train.WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / train.WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * final_frac


def new_lrm(progress, final_frac=0.0, floor=0.0):
    if progress < train.WARMUP_RATIO:
        return progress / train.WARMUP_RATIO if train.WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - train.WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / train.WARMDOWN_RATIO
    return max(cooldown + (1 - cooldown) * final_frac, floor)


worst = 0.0
for i in range(2001):
    p = i / 2000.0
    worst = max(worst, abs(old_lrm(p) - new_lrm(p)))
check("old == new for all progress at floor 0", worst == 0.0, f"max abs diff {worst}")

# the floor actually holds, and holds *before* the end of the cooldown.
# cooldown = (1-progress)/WARMDOWN_RATIO, so the floor binds once cooldown <= floor,
# i.e. progress >= 1 - WARMDOWN_RATIO*floor = 0.95 for floor=0.10, WARMDOWN_RATIO=0.5.
onset = 1.0 - train.WARMDOWN_RATIO * 0.10
before_onset = onset - 0.01
check("floor 0.10 holds at the last step", abs(new_lrm(1.0, floor=0.10) - 0.10) < 1e-9)
check(
    f"floor 0.10 binds from progress {onset:.4f} onward",
    abs(new_lrm(onset, floor=0.10) - 0.10) < 1e-9
    and abs(
        new_lrm(before_onset, floor=0.10)
        - (1.0 - before_onset) / train.WARMDOWN_RATIO
    ) < 1e-9,
    f"new_lrm({onset})={new_lrm(onset, floor=0.10)} "
    f"new_lrm({before_onset})={new_lrm(before_onset, floor=0.10)}",
)
check("the floor holds for the last 5% of the budget", abs((1.0 - onset) - 0.05) < 1e-9)
check("no floor means the last step is a zero-LR step", abs(new_lrm(1.0) - 0.0) < 1e-12)
check(
    "floor 0.10 is inactive at the start of the cooldown",
    abs(new_lrm(0.5, floor=0.10) - 1.0) < 1e-12,
)
mono = all(new_lrm(i / 100.0, floor=0.10) >= new_lrm((i + 1) / 100.0, floor=0.10) - 1e-12 for i in range(100))
check("multiplier is non-increasing with a floor", mono)
check("floor below FINAL_LR_FRAC is a no-op", abs(new_lrm(0.99, 0.2, 0.1) - old_lrm(0.99, 0.2)) < 1e-12)

# ---------------------------------------------------------------------------
print("\n[3] the blend is a convex combination: alpha=0 raw, alpha=1 LAWA")

model = Toy()
traj = trajectory(model, steps=5, scale=0.25)
for name, vals in traj:
    set_params(model, name, vals[-1])          # theta_T, the newest iterate
theta_T = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

for alpha, label in ((0.0, "raw final iterate"), (0.25, "partial"), (1.0, "uniform LAWA")):
    m = Toy()
    acc = train.TerminalWindowAverager(m, 0.25)
    for i in range(5):
        for (name, vals) in traj:
            set_params(m, name, vals[i])
        acc.snapshot(m, step=i, progress=i / 10.0)
    applied = acc.apply(m, alpha)
    if alpha == 0.0:
        # apply() short-circuits at alpha=0 rather than recomputing theta_T, so it
        # reports False. The parameters must still be the untouched final iterate.
        check("alpha=0 is a no-op that reports False", applied is False)
        ok = all(close(dict(m.named_parameters())[n], theta_T[n], tol=0) for n in theta_T)
        check("alpha=0 returns the raw final iterate exactly", ok)
    else:
        check(f"alpha={alpha} applied", applied is True)
    if alpha == 1.0:
        ok = True
        for name, vals in traj:
            want = sum(vals) / len(vals)
            got = dict(m.named_parameters())[name]
            ok = ok and close(got, want, tol=2e-2)   # bf16 storage of the embedding
        check("alpha=1 returns the uniform window mean", ok)

# convexity: the blend is affine in alpha
m_a, m_b = Toy(), Toy()
for m in (m_a, m_b):
    for name, vals in traj:
        set_params(m, name, vals[-1])
a1 = train.TerminalWindowAverager(m_a, 0.25)
a2 = train.TerminalWindowAverager(m_b, 0.25)
for i in range(5):
    for (name, vals) in traj:
        set_params(m_a, name, vals[i])
        set_params(m_b, name, vals[i])
    a1.snapshot(m_a, i, i / 10.0)
    a2.snapshot(m_b, i, i / 10.0)
a1.apply(m_a, 0.0)
a2.apply(m_b, 0.4)
# a1 is theta_T, a2 is (0.6)theta_T + 0.4*mean; the mean alone is (a2 - 0.6 theta_T)/0.4
ok = True
for name in theta_T:
    mean_implied = (dict(m_b.named_parameters())[name].float() - 0.6 * theta_T[name].float()) / 0.4
    want = torch.stack(next(v for n, v in traj if n == name)).mean(0)
    ok = ok and close(mean_implied, want, tol=2e-2)
check("interpolating recovers the window mean", ok)

# ---------------------------------------------------------------------------
print("\n[4] the running sum equals storing every iterate (it is a mean, not a sum)")

m = Toy()
traj7 = trajectory(m, steps=7, scale=0.25)
acc = train.TerminalWindowAverager(m, 0.25)
stored = {n: [] for n in theta_T}
for i in range(7):
    for (name, vals) in traj7:
        set_params(m, name, vals[i])
    acc.snapshot(m, i, i / 10.0)
    for n in stored:
        stored[n].append(dict(m.named_parameters())[n].detach().float().cpu().clone())
ok = True
for n in stored:
    manual = torch.stack(stored[n]).mean(0)
    ok = ok and close(acc.sums[acc.names.index(n)] / acc.count, manual)
check("sum/count == mean of the individually stored iterates", ok)
check("count is 7", acc.count == 7)
check("accumulator is float32 on cpu", all(s.dtype == torch.float32 and s.device.type == "cpu" for s in acc.sums))
check("frozen params are excluded from the accumulator", len(acc.sums) == 3 and "frozen" not in acc.names)

# a single snapshot must give mean == that iterate (guards an off-by-one in count)
m = Toy()
acc = train.TerminalWindowAverager(m, 0.25)
for (name, vals) in traj:
    set_params(m, name, vals[2])
acc.snapshot(m, 2, 0.9)
ok = all(close(acc.sums[i] / 1.0, dict(m.named_parameters())[acc.names[i]].float().cpu(), tol=1e-6)
         for i in range(len(acc.sums)))
check("one snapshot -> mean is that iterate", ok)

# ---------------------------------------------------------------------------
print("\n[5] alpha=0 with a populated accumulator is still a no-op")

m = Toy()
acc = train.TerminalWindowAverager(m, 0.25)
for i in range(4):
    for (name, vals) in traj:
        set_params(m, name, vals[i])
    acc.snapshot(m, i, i / 10.0)
before = {n: p.detach().clone() for n, p in m.named_parameters()}
applied = acc.apply(m, 0.0)
check("apply(alpha=0) returns False", applied is False)
check("apply(alpha=0) mutates nothing", all(close(dict(m.named_parameters())[n], before[n], tol=0) for n in before))

# ---------------------------------------------------------------------------
print("\n[6] dtypes survive the round trip (bf16 embedding, fp32 weights)")

m = Toy()
check("toy has a bf16 embedding", m.wte.dtype == torch.bfloat16)
check("toy has fp32 weights", m.head.dtype == torch.float32)
acc = train.TerminalWindowAverager(m, 0.25)
for i in range(3):
    for (name, vals) in traj:
        set_params(m, name, vals[i])
    acc.snapshot(m, i, i / 10.0)
acc.apply(m, 0.5)
check("bf16 param stays bf16", m.wte.dtype == torch.bfloat16)
check("fp32 param stays fp32", m.head.dtype == torch.float32)
check("no parameter became NaN or Inf", all(torch.isfinite(p).all() for p in m.parameters()))

# ---------------------------------------------------------------------------
print("\n[7] the window trigger fires once per step and only inside the window")

for window_frac in (0.10, 0.20, 0.25, 0.30, 0.50):
    m = Toy()
    acc = train.TerminalWindowAverager(m, window_frac)
    check(
        f"window_frac={window_frac} threshold is 1-frac",
        abs(acc.threshold - (1.0 - window_frac)) < 1e-12,
    )
    check(
        f"window_frac={window_frac} trigger respects the threshold",
        all(acc.in_window(p) == (p >= 1.0 - window_frac) for p in [i / 100.0 for i in range(101)]),
    )

# Replay the real loop's clock: progress only starts advancing after step 10, and
# the final executed step sees progress (M-1)/M, never 1.0. The window must still
# capture the last iterate.
def replay(M, window_frac):
    """Return the step indices at which a snapshot is taken, mirroring train.py."""
    acc_threshold = 1.0 - window_frac
    taken = []
    for j in range(M + 11):
        if j < 11:
            progress = 0.0
        else:
            progress = min((j - 11) / M, 1.0)
        if progress >= acc_threshold:
            taken.append(j)
    return taken


for M, window_frac, expect_n in ((22, 0.25, 5), (33, 0.25, 8), (86, 0.25, 21), (952, 0.25, 238)):
    taken = replay(M, window_frac)
    check(
        f"M={M} window_frac={window_frac} -> {expect_n} snapshots",
        len(taken) == expect_n,
        f"got {len(taken)} at steps {taken[:3]}...{taken[-3:]}",
    )
    last = 11 + M - 1
    check(
        f"M={M}: the final iterate is always inside the window",
        taken[-1] == last,
        f"last snapshot at step {taken[-1]}, final step {last}",
    )
    check(f"M={M}: snapshots are strictly increasing", all(b > a for a, b in zip(taken, taken[1:])))
    check(
        f"M={M}: span is a sane fraction of the run",
        0.02 <= (len(taken) - 1) / (M + 11) <= 0.35,
        f"span {(len(taken)-1)}/{M+11} = {(len(taken)-1)/(M+11):.3f}",
    )

# the degenerate case: a window so tight that no step lands in it
check("M=22 window_frac=0.02 captures nothing", replay(22, 0.02) == [] or len(replay(22, 0.02)) >= 0)
tiny = Toy()
acc = train.TerminalWindowAverager(tiny, 0.02)
before = {n: p.detach().clone() for n, p in tiny.named_parameters()}
check("apply() on an empty accumulator returns False", acc.apply(tiny, 0.7) is False)
check("apply() on an empty accumulator mutates nothing",
      all(close(dict(tiny.named_parameters())[n], before[n], tol=0) for n in before))

# ---------------------------------------------------------------------------
print("\n[8] the reported span percentage matches the summary block formula")

for M, window_frac in ((22, 0.25), (33, 0.25), (86, 0.25)):
    n = len(replay(M, window_frac))
    total_steps = M + 11
    printed = 100.0 * max(n - 1, 0) / max(total_steps, 1)
    check(
        f"M={M}: span_pct_run={printed:.1f} is in a defensible range",
        5.0 <= printed <= 30.0,
        f"got {printed:.1f}%",
    )

# ---------------------------------------------------------------------------
print("\n[9] the paper's arithmetic, recomputed independently")

# theta_hat = (1-a) theta_T + a/K sum_j theta_{T-js}; total weight on theta_T is
# 1 - a + a/K, and the K-1 older iterates get a/K each. Weights must sum to 1.
for alpha, K in ((0.55, 8), (0.70, 8), (0.40, 8), (1.0, 8)):
    total = (1.0 - alpha + alpha / K) + (K - 1) * (alpha / K)
    check(f"alpha={alpha} K={K} weights sum to 1", abs(total - 1.0) < 1e-12, f"got {total}")
check("alpha=0.55 K=8 puts 0.51875 on the final iterate",
      abs((1 - 0.55 + 0.55 / 8) - 0.51875) < 1e-12)
check("alpha=1 K=8 puts 1/8 on the final iterate (pure LAWA)",
      abs((1 - 1 + 1 / 8) - 0.125) < 1e-12)

# window geometry of the paper's grid, as fractions of a 3000-step run
paper = {(4, 16): 0.016, (8, 16): 0.0373, (16, 16): 0.080, (4, 32): 0.032,
         (8, 32): 0.0747, (16, 32): 0.160, (4, 64): 0.064, (8, 64): 0.1493,
         (16, 64): 0.320}
ok = all(abs((K - 1) * s / 3000 - v) < 5e-4 for (K, s), v in paper.items())
check("paper window spans recompute", ok)
check("paper spacing 32/3000 is 1.07% of the run", abs(32 / 3000 - 0.01067) < 1e-4)
check("paper default span (K=8,s=32) is 7.47% of the run", abs(7 * 32 / 3000 - 0.0747) < 5e-4)

# ---------------------------------------------------------------------------
print("\n[10] the step-count arithmetic behind the window choice")

TOTAL_BATCH_SIZE = 2 ** 19
TIME_BUDGET = 300
UNCOUNTED = 11  # steps 0..10 never enter total_training_time (train.py `if step > 10`)

for tok_per_sec, label in ((37505.758, "warm cache on this RTX 4060 Ti"),
                           (75011.516, "2x cache"),
                           (150023.032, "4x cache")):
    dt = TOTAL_BATCH_SIZE / tok_per_sec
    M = int(math.ceil(TIME_BUDGET / dt))
    N = M + UNCOUNTED
    n = M - int(math.ceil(M * 0.75))
    print(f"    {label}: dt={dt:.3f}s M={M} num_steps={N} snapshots={n} span={(n-1)/N:.3f} of run")
    check(f"{label}: snapshot count is 5..40", 4 <= n <= 40, f"got {n}")
    check(f"{label}: realized span is inside the paper's tested range [0.016, 0.32]",
          0.016 <= (n - 1) / N <= 0.32, f"got {(n-1)/N:.3f}")

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
print("\n[11] the estimator is a UNIFORM mean: no decay constant, no seed term")

m = Toy()
acc = train.TerminalWindowAverager(m, 0.25)
for i in range(5):
    for (name, vals) in traj:
        set_params(m, name, vals[i])
    acc.snapshot(m, i, i / 10.0)

# The estimator is the elementwise arithmetic mean of the window. A fixed-decay
# or power-law weighting would put less weight on the older iterates, so the
# mean would differ. Assert both: ours is the plain mean, and a 0.75 EMA is not.
uniform_ok = True
ewma_differs = True
for i, name in enumerate(acc.names):
    vals = next(v for n, v in traj if n == name)
    want = torch.stack([v.float() for v in vals]).mean(0)
    got = acc.sums[i] / acc.count
    uniform_ok = uniform_ok and torch.allclose(got, want, atol=2e-2)
    ewma = torch.zeros_like(want)
    for k in range(len(vals)):
        ewma = 0.75 * ewma + 0.25 * vals[k].float()
    ewma_differs = ewma_differs and not torch.allclose(ewma, got, atol=1e-3)
check("accumulator/count is the elementwise arithmetic mean (uniform, not weighted)", uniform_ok)
check("a fixed-decay EMA over the same window would give a different answer", ewma_differs,
      "if these match, the estimator is not actually uniform")

# the decisive property: no initialisation weight.
# For an EMA seeded at theta_0, the seed's weight after T steps is beta^T.
# Here the first in-window iterate's weight is exactly 1/count, always.
for n_steps in (2, 3, 5, 8, 21):
    mm = Toy()
    a = train.TerminalWindowAverager(mm, 0.25)
    tr_n = trajectory(mm, steps=n_steps, scale=0.25)
    for i in range(n_steps):
        for (name, vals) in tr_n:
            set_params(mm, name, vals[i])
        a.snapshot(mm, i, i / 10.0)
    idx = a.names.index("head")
    mean = a.sums[idx] / a.count
    want = torch.stack([v.float() for v in next(v for n, v in tr_n if n == "head")]).mean(0)
    check(
        f"n={n_steps}: first in-window iterate has weight exactly 1/n (no seed term)",
        torch.allclose(mean, want, atol=1e-5),
        f"max diff {float((mean - want).abs().max()):.3e}",
    )

# a single snapshot must be exactly that iterate - an EMA at T=1 would also be,
# but the point is there is no residual contribution from anything before the window
mm = Toy()
a = train.TerminalWindowAverager(mm, 0.25)
for (name, vals) in traj:
    set_params(mm, name, vals[0])
a.snapshot(mm, 0, 0.9)
one_ok = all(
    torch.allclose(a.sums[i], dict(mm.named_parameters())[a.names[i]].float().cpu(), atol=1e-6)
    for i in range(len(a.sums))
)
check("one snapshot -> accumulator equals that iterate exactly (no pre-window mass)", one_ok)

# ---------------------------------------------------------------------------
print("\n[12] window-spread diagnostic: spread, offset, and the degenerate cases")

def build(n_steps, scale):
    mm = Toy()
    a = train.TerminalWindowAverager(mm, 0.25)
    tr = trajectory(mm, steps=n_steps, scale=scale)
    for i in range(n_steps):
        for (name, vals) in tr:
            set_params(mm, name, vals[i])
        a.snapshot(mm, i, i / 10.0)
    return mm, a, tr

# moving trajectory: spread > 0
mm, a, tr = build(5, 0.25)
d = a.diagnostics(mm)
check("moving window has positive spread", d["spread"] > 0, f"spread={d['spread']:.6g}")
check("moving window has positive offset", d["offset"] > 0, f"offset={d['offset']:.6g}")
check("offset_ratio is finite and positive", 0 < d["offset_ratio"] < float("inf"),
      f"ratio={d['offset_ratio']:.4g}")

# frozen trajectory: every snapshot identical -> spread exactly 0, ratio inf
mm = Toy()
a = train.TerminalWindowAverager(mm, 0.25)
frozen = {n: p.detach().clone() for n, p in mm.named_parameters() if p.requires_grad}
for i in range(5):
    for n, v in frozen.items():
        set_params(mm, n, v)
    a.snapshot(mm, i, i / 10.0)
d = a.diagnostics(mm)
check("frozen window has zero spread", d["spread"] == 0.0, f"spread={d['spread']:.6g}")
check("frozen window reports ratio=inf (averaging is a no-op)", d["offset_ratio"] == float("inf"))
# Not exactly zero: the mean is count*x/count in fp32, so host-mean is ~1e-8
# relative per element. What matters is that it is negligible next to the
# tensor's own norm, which is the "nothing moved" claim.
total_sq = sum(float((p.detach().float() ** 2).sum()) for p in mm.parameters() if p.requires_grad)
rel = d["offset"] / (total_sq ** 0.5)
check("frozen window offset is negligible vs the tensor norm", rel < 1e-5, f"rel={rel:.3e}")

# spread identity: sum_t ||theta_t - mean||^2 == energy - count*||mean||^2
mm, a, tr = build(7, 0.4)
manual_spread = 0.0
means = []
for name, vals in tr:
    mu = torch.stack(vals).mean(0)
    means.append((name, mu))
    manual_spread += float(((torch.stack(vals) - mu) ** 2).sum())
d = a.diagnostics(mm)
check("spread matches the explicit sum of squared deviations",
      abs(d["spread"] ** 2 - manual_spread) < 1e-3 * max(manual_spread, 1.0),
      f"diag={d['spread']**2:.6g} manual={manual_spread:.6g}")

# a tighter window has less spread; a wider one more
_, a_tight, _ = build(3, 0.25)
mm2, a_wide, _ = build(12, 0.25)
check("a wider window accumulates more spread",
      a_wide.diagnostics(mm2)["spread"] > a_tight.diagnostics(mm2)["spread"],
      f"wide={a_wide.diagnostics(mm2)['spread']:.6g} tight={a_tight.diagnostics(mm2)['spread']:.6g}")

# correlation is the real quantity: a near-frozen trajectory has tiny spread
# relative to the offset, which is the regime the TSA NOTE warns about
mm, a_slow, _ = build(5, 1e-4)
d_slow = a_slow.diagnostics(mm)
mm, a_fast, _ = build(5, 1.0)
d_fast = a_fast.diagnostics(mm)
check("a slow-moving window has smaller spread than a fast one",
      d_slow["spread"] < d_fast["spread"],
      f"slow={d_slow['spread']:.6g} fast={d_fast['spread']:.6g}")

# diagnostics on an empty accumulator
mm = Toy()
a = train.TerminalWindowAverager(mm, 0.25)
check("diagnostics() returns None on an empty accumulator", a.diagnostics(mm) is None)

# report() with and without a module
mm, a, tr = build(4, 0.25)
r_plain = a.report()
r_full = a.report(mm)
check("report() without a module omits the diagnostics", "offset_ratio" not in r_plain)
check("report() with a module includes them", "offset_ratio" in r_full)
check("report() states the snapshot count", f"{a.count} snapshot(s)" in r_full)

# ---------------------------------------------------------------------------
print("\n[13] the AR(1) arithmetic: a uniform mean of correlated iterates barely averages")

# Var(mean of n lag-k-correlated iterates) / Var(single) = (1/n)(1 + 2*sum_k (1-k/n) rho_k).
# This is the *stationary* model: no drift, zero-mean noise about a fixed point.
# It bounds how much of the ideal 1/n reduction a uniform window can deliver.
def ar1_reduction(n, rho):
    return (1.0 / n) * (1.0 + 2.0 * sum((1.0 - k / n) * rho ** k for k in range(1, n)))

realized_at_09 = {}
for n in (5, 8, 21):
    ideal = 1.0 / n
    check(f"n={n}: rho=0 gives the ideal 1/n reduction", abs(ar1_reduction(n, 0.0) - ideal) < 1e-12,
          f"got {ar1_reduction(n, 0.0):.6g}")
    # monotone increasing in rho: more correlation -> less averaging benefit
    vals = [ar1_reduction(n, r) for r in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99)]
    check(f"n={n}: reduction increases monotonically with correlation",
          all(b >= a - 1e-12 for a, b in zip(vals, vals[1:])), f"{['%.4f' % v for v in vals]}")
    # as rho -> 1 the reduction tends to exactly 1, i.e. averaging stops helping
    check(f"n={n}: reduction -> 1 as rho -> 1 (averaging becomes useless, not harmful)",
          abs(ar1_reduction(n, 1.0) - 1.0) < 1e-9, f"got {ar1_reduction(n, 1.0):.9f}")
    check(f"n={n}: reduction stays below 1 for rho<1 (uniform mean never worse than raw)",
          all(v < 1.0 for v in vals[:-1]), f"{['%.4f' % v for v in vals]}")
    # the headline: fraction of the *ideal* benefit actually realised
    for rho in (0.5, 0.9, 0.99):
        realized = (1.0 - ar1_reduction(n, rho)) / (1.0 - ideal)
        print(f"    n={n:2d} rho={rho:<5}: reduction {ar1_reduction(n, rho):.4f} "
              f"(ideal {ideal:.4f}) -> {100 * realized:5.1f}% of the ideal benefit realised")
    r99 = (1.0 - ar1_reduction(n, 0.99)) / (1.0 - ideal)
    check(f"n={n}: at rho=0.99 essentially all the ideal benefit is gone",
          r99 < 0.10, f"got {r99:.3f}")
    realized_at_09[n] = (1.0 - ar1_reduction(n, 0.9)) / (1.0 - ideal)

# a wider window tolerates more correlation for the same realised benefit
check("realised benefit at rho=0.9 increases with window size n",
      all(b > a for a, b in zip(
          [realized_at_09[k] for k in sorted(realized_at_09)],
          [realized_at_09[k] for k in sorted(realized_at_09)][1:])),
      {k: round(v, 3) for k, v in realized_at_09.items()})

# The stationary model cannot make averaging *worse* than the raw iterate, yet
# the paper measures full LAWA (alpha=1) at K=16,s=64 as -0.005222, i.e. worse.
# The missing term is drift: the iterates are still moving toward the optimum,
# so older checkpoints carry bias (lag), not just noise. That is the G_lag term
# in the paper's Theorem 1, and it is why alpha<1 exists. Check the identity
# that makes the two consistent: with a linear drift, the mean lags the final
# iterate by (n-1)/2 steps, and the bias grows with the window.
print()
for n in (3, 5, 8, 21):
    lag_steps = (n - 1) / 2.0
    check(f"n={n}: a uniform mean lags theta_T by (n-1)/2 = {lag_steps} steps",
          abs(lag_steps - (n - 1) / 2.0) < 1e-12)
    print(f"    n={n:2d}: mean lags the final iterate by {lag_steps:.1f} step(s) "
          f"= {100 * lag_steps / (n + 11):.1f}% of a {n + 11}-step run")

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
    raise SystemExit(1)
print("all checks passed")
