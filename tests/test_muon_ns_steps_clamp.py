"""AUT-80: the Newton-Schulz iteration count must not be a silent no-op.

`polar_express_coeffs` is a fixed five-triple schedule, and `muon_step_fused` slices
it as `polar_express_coeffs[:ns_steps]`. Slicing is unbounded above, so any
`ns_steps > 5` used to run five iterations and return a bit-identical update, with
no assert, warning or log line. A results file could then only record it as
"no change", which is a false negative: the experiment never ran.

This drives the real `muon_step_fused` from train.py at the real group shapes, in
both the shipped MUON_COMPUTE_DTYPE and fp32, and asserts:

  1. ns_steps at or below the cap is accepted and the iteration count is honoured
  2. ns_steps above the cap raises instead of silently truncating
  3. with the guard neutralised, ns_steps=8 is bit-identical to ns_steps=5 on the
     shipped function itself -- the regression this guard exists to make impossible
     to record as an experiment
  4. the shipped default equals len(polar_express_coeffs), so the default is the cap
     rather than an arbitrary number that happens to fit
  5. setup_optimizer routes the value through the guard and still owns every
     parameter exactly once

Point 3 is deliberately done with the guard patched out rather than against a
hand-copied copy of the iteration, so "bit-identical" is a statement about the code
that ships and not about a re-implementation of it.

CPU only, no dataset, no CUDA, no optimizer state: the iteration is a pure function
of (grad, ns_steps).
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import train  # noqa: E402
from train import (  # noqa: E402
    MUON_COMPUTE_DTYPE,
    POLAR_EXPRESS_MAX_STEPS,
    assert_ns_steps_available,
    muon_step_fused,
    polar_express_coeffs,
)

# Real setup_optimizer group shapes at DEPTH=8, plus the wide-thin (4, 32) that AUT-33
# flagged. Both sides of the rows > cols branch are covered, because the truncation
# bug is in the shared coefficient list, not in either branch.
SHAPES = [(512, 512), (2048, 512), (512, 2048), (4, 32)]
STACK = 2


def _drive(grads, ns_steps, red_dim):
    """Call the real muon_step_fused on freshly cloned state; return the update."""
    grads = grads.clone()
    params = torch.zeros_like(grads)
    params += 0.01  # a non-zero parameter so the sign-masked weight decay is exercised
    mom = torch.zeros_like(grads)
    mom2 = torch.zeros(grads.size(), dtype=grads.dtype)
    before = params.clone()
    muon_step_fused(
        grads, params, mom, mom2,
        torch.tensor(0.95), torch.tensor(0.04), torch.tensor(0.0),
        torch.tensor(0.95), ns_steps, red_dim,
    )
    return params - before


def _raw_ns(X, ns_steps):
    """The iteration with no guard at all, for a second opinion on the truncation.

    Kept as a cross-check rather than the primary evidence: the primary evidence is
    test_unpatched_iteration_would_have_be_a_no_op, which neutralises the guard on the
    real muon_step_fused instead of re-implementing it.
    """
    if X.size(-2) > X.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    return X


class TestPolarExpressSchedule(unittest.TestCase):
    def test_default_is_the_cap(self):
        self.assertEqual(POLAR_EXPRESS_MAX_STEPS, len(polar_express_coeffs))
        sig = train.GPT.setup_optimizer.__defaults__
        default_ns_steps = sig[-1]
        self.assertEqual(default_ns_steps, POLAR_EXPRESS_MAX_STEPS,
                         "setup_optimizer's default ns_steps should be the cap, not a "
                         "number that merely happens to fit")

    def test_shipped_default_is_honoured(self):
        # The default the code actually runs must not trip its own guard.
        assert_ns_steps_available(POLAR_EXPRESS_MAX_STEPS)

    def test_cap_and_below_accepted(self):
        for k in range(1, POLAR_EXPRESS_MAX_STEPS + 1):
            assert_ns_steps_available(k)

    def test_above_cap_raises(self):
        for k in (POLAR_EXPRESS_MAX_STEPS + 1, 8, 12):
            with self.assertRaises(ValueError) as ctx:
                assert_ns_steps_available(k)
            self.assertIn("coeffs_list", str(ctx.exception),
                          "the error should name where the missing coefficients come from")

    def test_non_positive_raises(self):
        for k in (0, -1):
            with self.assertRaises(ValueError):
                assert_ns_steps_available(k)

    def test_more_iterations_changes_the_update(self):
        # A real guard has to be able to distinguish K from K+1, or it is guarding nothing.
        for shape in SHAPES:
            red_dim = -1 if shape[0] >= shape[1] else -2
            torch.manual_seed(0)
            grads = torch.randn((STACK,) + shape)
            k = POLAR_EXPRESS_MAX_STEPS
            self.assertFalse(
                torch.equal(_drive(grads, k, red_dim), _drive(grads, k - 1, red_dim)),
                f"ns_steps={k} and {k - 1} produced identical updates at {shape}",
            )

    def test_muon_step_fused_raises_above_the_cap(self):
        for shape in SHAPES:
            red_dim = -1 if shape[0] >= shape[1] else -2
            torch.manual_seed(0)
            grads = torch.randn((STACK,) + shape)
            with self.assertRaises(ValueError):
                _drive(grads, POLAR_EXPRESS_MAX_STEPS + 1, red_dim)

    def test_unpatched_iteration_would_have_been_a_no_op(self):
        # The regression itself, stated as a fact about the shipped coefficient list:
        # with the guard neutralised, ns_steps=8 cannot differ from ns_steps=5. This is the
        # assertion that makes the fix load-bearing -- without it, "bit-identical" would be
        # a claim about a re-implementation of the loop rather than about the code that runs.
        import unittest.mock as mock
        with mock.patch.object(train, "assert_ns_steps_available", lambda n: None):
            for dtype in (torch.float32, MUON_COMPUTE_DTYPE):
                for shape in SHAPES:
                    torch.manual_seed(0)
                    grads = torch.randn((STACK,) + shape, dtype=dtype)
                    red_dim = -1 if shape[0] >= shape[1] else -2
                    k = POLAR_EXPRESS_MAX_STEPS
                    self.assertTrue(
                        torch.equal(_drive(grads, k + 3, red_dim), _drive(grads, k, red_dim)),
                        f"ns_steps={k + 3} was expected to truncate to {k} "
                        f"at {shape}/{dtype}",
                    )

    def test_coefficient_schedule_itself_truncates(self):
        # The mechanism, stated directly on the list: five triples, so the slice cannot
        # yield more than five. Independent of the guard and of muon_step_fused.
        self.assertEqual(len(polar_express_coeffs), 5)
        self.assertEqual(len(polar_express_coeffs[:8]), len(polar_express_coeffs))

    def test_update_is_finite_and_moves(self):
        for dtype in (torch.float32, MUON_COMPUTE_DTYPE):
            for shape in SHAPES:
                red_dim = -1 if shape[0] >= shape[1] else -2
                torch.manual_seed(0)
                grads = torch.randn((STACK,) + shape, dtype=dtype)
                update = _drive(grads, POLAR_EXPRESS_MAX_STEPS, red_dim)
                self.assertTrue(torch.isfinite(update).all().item(), f"{shape}/{dtype}")
                self.assertGreater(update.abs().max().item(), 0.0, f"{shape}/{dtype}")


class TestSetupOptimizerWiring(unittest.TestCase):
    """The guard is only load-bearing if setup_optimizer routes the shipped value through it."""

    @classmethod
    def setUpClass(cls):
        cls.model = train.GPT(train.GPTConfig())

    def test_default_setup_optimizer_builds_and_every_muon_group_is_honourable(self):
        optimizer = self.model.setup_optimizer()
        muon_groups = [g for g in optimizer.param_groups if g.get("kind") == "muon"]
        self.assertGreater(len(muon_groups), 0, "no Muon groups were built")
        for group in muon_groups:
            assert_ns_steps_available(group["ns_steps"])
        self.assertEqual(
            {g["ns_steps"] for g in muon_groups}, {POLAR_EXPRESS_MAX_STEPS},
            "every Muon group should carry the same, honoured iteration count",
        )

    def test_setup_optimizer_rejects_an_unhonourable_ns_steps(self):
        with self.assertRaises(ValueError):
            self.model.setup_optimizer(ns_steps=POLAR_EXPRESS_MAX_STEPS + 1)

    def test_every_parameter_is_owned_exactly_once(self):
        optimizer = self.model.setup_optimizer()
        seen = [id(p) for g in optimizer.param_groups for p in g["params"]]
        self.assertEqual(len(seen), len(set(seen)), "a parameter is in two optimizer groups")
        self.assertEqual(len(seen), len(list(self.model.parameters())),
                         "an optimizer group lost a parameter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
