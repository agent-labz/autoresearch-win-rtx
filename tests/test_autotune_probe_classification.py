"""Autotune probe failure classification.

These tests need no GPU. They pin the contract that a non-OOM probe failure is
never reported as a hardware/capacity limit.

Run with:
    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

import train  # noqa: E402


class ClassifyProbeExceptionTest(unittest.TestCase):
    """A capacity rejection and a code bug must land in different buckets."""

    def test_cuda_oom_type_is_capacity(self):
        exc = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertEqual(train.PROBE_OUT_OF_MEMORY, train._classify_probe_exception(exc))

    def test_oom_wrapped_in_a_plain_runtime_error_is_capacity(self):
        # Allocator failures are not always raised as the OOM type; cuBLAS reports
        # them as a bare RuntimeError, and those are still capacity rejections.
        exc = RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED")
        self.assertEqual(train.PROBE_OUT_OF_MEMORY, train._classify_probe_exception(exc))

    def test_shape_error_is_not_capacity(self):
        # The exact failure this fix exists for: a shape bug that used to print as
        # "rejected (OOM, runtime error, or >90% VRAM use)".
        exc = RuntimeError("Failed to allocate tensor with shape [1, 1, 4096, 4096]")
        self.assertEqual(train.PROBE_ERROR, train._classify_probe_exception(exc))

    def test_device_side_assert_is_not_capacity(self):
        exc = RuntimeError("CUDA error: device-side assert triggered")
        self.assertEqual(train.PROBE_ERROR, train._classify_probe_exception(exc))

    def test_non_runtime_errors_are_not_capacity(self):
        for exc in (
            ValueError("expected scalar type Float but found Half"),
            AssertionError("Muon precondition failed: gamma=0"),
            TypeError("init_weights() got an unexpected keyword argument"),
        ):
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(train.PROBE_ERROR, train._classify_probe_exception(exc))

    def test_detail_carries_exception_type_and_message(self):
        exc = RuntimeError("nan detected in loss")
        detail = train._format_probe_exception(exc)
        self.assertIn("RuntimeError", detail)
        self.assertIn("nan detected in loss", detail)


class _TrainProbeOutcomeTest(unittest.TestCase):
    def test_bucket_predicates(self):
        accepted = train._TrainProbeOutcome(train.PROBE_ACCEPTED, 8, True)
        oom = train._TrainProbeOutcome(train.PROBE_OUT_OF_MEMORY, 8, True)
        vram = train._TrainProbeOutcome(train.PROBE_VRAM_LIMIT, 8, True)
        error = train._TrainProbeOutcome(train.PROBE_ERROR, 8, True, detail="RuntimeError: boom")
        self.assertTrue(accepted.is_accepted)
        self.assertTrue(oom.is_capacity_rejection)
        self.assertTrue(vram.is_capacity_rejection)
        self.assertTrue(error.is_error)
        for outcome in (accepted, oom, vram, error):
            self.assertFalse(outcome.is_accepted and outcome.is_capacity_rejection)
            self.assertFalse(outcome.is_accepted and outcome.is_error)
            self.assertFalse(outcome.is_capacity_rejection and outcome.is_error)


class ProbeSummaryTest(unittest.TestCase):
    def test_clean_run_reports_no_error_section(self):
        summary = train._format_autotune_probe_summary(
            capacity_rejections=[
                train._TrainProbeOutcome(train.PROBE_OUT_OF_MEMORY, 64, False),
                train._TrainProbeOutcome(train.PROBE_VRAM_LIMIT, 32, False),
            ],
            probe_errors=[],
            best_outcome=train._TrainProbeOutcome(train.PROBE_ACCEPTED, 8, True),
        )
        self.assertIn("1 accepted", summary)
        self.assertIn("2 capacity rejection(s)", summary)
        self.assertIn("0 probe error(s)", summary)
        self.assertNotIn("PROBE ERRORS", summary)

    def test_probe_errors_are_listed_with_type_and_message(self):
        summary = train._format_autotune_probe_summary(
            capacity_rejections=[],
            probe_errors=[
                train._TrainProbeOutcome(
                    train.PROBE_ERROR, 64, False, detail="RuntimeError: nan detected in loss"
                ),
            ],
            best_outcome=train._TrainProbeOutcome(train.PROBE_ACCEPTED, 8, True),
        )
        self.assertIn("1 probe error(s)", summary)
        self.assertIn("PROBE ERRORS", summary)
        self.assertIn("RuntimeError: nan detected in loss", summary)
        self.assertIn("not proof", summary)

    def test_repeated_identical_errors_are_deduplicated_with_a_count(self):
        detail = "RuntimeError: nan detected in loss"
        summary = train._format_autotune_probe_summary(
            capacity_rejections=[],
            probe_errors=[
                train._TrainProbeOutcome(train.PROBE_ERROR, 64, False, detail=detail),
                train._TrainProbeOutcome(train.PROBE_ERROR, 32, False, detail=detail),
            ],
            best_outcome=train._TrainProbeOutcome(train.PROBE_ACCEPTED, 8, True),
        )
        self.assertIn("2 probe error(s)", summary)
        self.assertIn("x2 candidate(s)", summary)
        self.assertEqual(1, summary.count(detail))

    def test_no_accepted_candidate_with_probe_errors_is_stated(self):
        summary = train._format_autotune_probe_summary(
            capacity_rejections=[],
            probe_errors=[
                train._TrainProbeOutcome(train.PROBE_ERROR, 64, False, detail="ValueError: bad shape")
            ],
            best_outcome=None,
        )
        self.assertIn("0 accepted", summary)
        self.assertIn("1 probe error(s)", summary)


class ProbeErrorBannerTest(unittest.TestCase):
    def test_banner_is_loud_and_names_the_exception(self):
        outcome = train._TrainProbeOutcome(
            train.PROBE_ERROR, 64, False, detail="RuntimeError: device-side assert triggered"
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train._print_probe_error(outcome, "Traceback (most recent call last): ...")
        printed = buffer.getvalue()
        self.assertIn("AUTOTUNE PROBE FAILED", printed)
        self.assertIn("this is a code error, not a VRAM limit", printed)
        self.assertIn("RuntimeError: device-side assert triggered", printed)
        self.assertIn("batch_size=64, checkpointing=off", printed)
        self.assertIn("Traceback (most recent call last): ...", printed)
        self.assertIn("NOT recorded as a capacity rejection", printed)


class CachedProbeErrorWarningTest(unittest.TestCase):
    def test_warns_when_cached_candidate_carries_probe_errors(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train._warn_if_cached_probe_errors(
                {
                    "train_batch_size": 8,
                    "use_activation_checkpointing": True,
                    "probe_error_count": 2,
                    "probe_errors": ["RuntimeError: nan detected in loss"],
                }
            )
        printed = buffer.getvalue()
        self.assertIn("WARNING", printed)
        self.assertIn("2 candidate probe(s) failed with non-capacity errors", printed)
        self.assertIn("RuntimeError: nan detected in loss", printed)
        self.assertIn("AUTORESEARCH_AUTOTUNE_REFRESH=1", printed)

    def test_silent_for_a_clean_cache_entry(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train._warn_if_cached_probe_errors(
                {
                    "train_batch_size": 8,
                    "use_activation_checkpointing": True,
                    "probe_error_count": 0,
                    "probe_errors": [],
                }
            )
        self.assertEqual("", buffer.getvalue())

    def test_tolerates_legacy_cache_entries_without_the_new_fields(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            train._warn_if_cached_probe_errors(
                {"train_batch_size": 8, "use_activation_checkpointing": True}
            )
        self.assertEqual("", buffer.getvalue())


class AutotuneCacheVersionTest(unittest.TestCase):
    def test_cache_version_records_probe_errors(self):
        # A v3 entry has no probe_error_fields, so a warm cache written by the old
        # code would keep re-selecting a degraded candidate in silence. Bumping the
        # version forces one honest re-probe.
        self.assertEqual("gpu-profile-v4", train.AUTOTUNE_CACHE_VERSION)


class AutotuneProbeIntegrationTest(unittest.TestCase):
    """Drives _autotune_train_candidate with a stubbed probe.

    No GPU required: _benchmark_train_candidate is replaced, so only the
    classification, bucketing, summary, and cache-write logic runs.
    """

    def setUp(self):
        self._scratch = tempfile.TemporaryDirectory()
        self._scratch_path = Path(self._scratch.name) / "cache.json"
        self._added = []
        self._real_get_path = train._get_autotune_cache_path
        self._real_benchmark = train._benchmark_train_candidate
        train._get_autotune_cache_path = lambda: self._scratch_path
        self._added.append((train, "_get_autotune_cache_path"))

        class FakeRuntime:
            gpu_name = "fake-rtx-0000"
            gpu_cc = (8, 6)
            gpu_total_memory_bytes = 16 * 1024**3
            gpu_profile = train.GpuProfile(
                name="fake",
                is_supported_consumer=True,
                is_compatibility_only=False,
                train_batch_candidates=(8,),
                checkpoint_modes=(True,),
                default_checkpointing=True,
            )

        self.runtime = FakeRuntime()

    def tearDown(self):
        train._get_autotune_cache_path = self._real_get_path
        train._benchmark_train_candidate = self._real_benchmark
        self._scratch.cleanup()

    def _stub_probe(self, outcomes):
        queue = list(outcomes)

        def fake_probe(runtime, tokenizer, vocab_size, train_batch_size, use_checkpointing):
            return queue.pop(0)

        train._benchmark_train_candidate = fake_probe

    def _run(self, outcomes):
        self._stub_probe(outcomes)
        os.environ["AUTORESEARCH_AUTOTUNE_REFRESH"] = "1"
        os.environ.pop("AUTORESEARCH_DISABLE_AUTOTUNE", None)
        buffer = io.StringIO()
        try:
            with redirect_stdout(buffer):
                selected = train._autotune_train_candidate(
                    self.runtime, tokenizer=None, vocab_size=1024, train_candidates=[(8, True), (4, True)]
                )
        finally:
            os.environ.pop("AUTORESEARCH_AUTOTUNE_REFRESH", None)
        return selected, buffer.getvalue()

    def test_error_bucket_is_not_reported_as_a_capacity_rejection(self):
        selected, printed = self._run(
            [
                train._TrainProbeOutcome(
                    train.PROBE_ERROR, 8, True, detail="RuntimeError: nan detected in loss"
                ),
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 4, True, tok_per_sec=1000.0),
            ]
        )
        self.assertEqual((4, True), selected)
        self.assertIn("rejected (probe error", printed)
        self.assertNotIn("rejected (OOM, runtime error, or >90% VRAM use)", printed)
        self.assertIn("1 probe error(s)", printed)
        self.assertIn("RuntimeError: nan detected in loss", printed)

    def test_oom_rejection_stays_quiet(self):
        _, printed = self._run(
            [
                train._TrainProbeOutcome(train.PROBE_OUT_OF_MEMORY, 8, True),
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 4, True, tok_per_sec=1000.0),
            ]
        )
        self.assertIn("rejected (capacity: out of memory)", printed)
        self.assertIn("0 probe error(s)", printed)
        self.assertNotIn("PROBE ERRORS", printed)

    def test_vram_limit_rejection_names_the_limit(self):
        _, printed = self._run(
            [
                train._TrainProbeOutcome(
                    train.PROBE_VRAM_LIMIT, 8, True, detail="peak_vram_mb=1.0 exceeds 90%"
                ),
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 4, True, tok_per_sec=1000.0),
            ]
        )
        self.assertIn("rejected (capacity: peak VRAM above 90% of total)", printed)
        self.assertIn("0 probe error(s)", printed)

    def test_probe_errors_are_persisted_in_the_cache_entry(self):
        selected, _ = self._run(
            [
                train._TrainProbeOutcome(
                    train.PROBE_ERROR, 8, True, detail="RuntimeError: device-side assert triggered"
                ),
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 4, True, tok_per_sec=1000.0),
            ]
        )
        self.assertEqual((4, True), selected)
        entries = train._load_autotune_entries(self._scratch_path)
        self.assertEqual(1, len(entries))
        entry = next(iter(entries.values()))
        self.assertEqual(4, entry["train_batch_size"])
        self.assertEqual(1, entry["probe_error_count"])
        self.assertEqual(["RuntimeError: device-side assert triggered"], entry["probe_errors"])

    def test_clean_run_records_no_probe_errors(self):
        self._run(
            [
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 8, True, tok_per_sec=1000.0),
                train._TrainProbeOutcome(train.PROBE_ACCEPTED, 4, True, tok_per_sec=900.0),
            ]
        )
        entry = next(iter(train._load_autotune_entries(self._scratch_path).values()))
        self.assertEqual(8, entry["train_batch_size"])
        self.assertEqual(0, entry["probe_error_count"])
        self.assertEqual([], entry["probe_errors"])

    def test_all_candidates_erroring_falls_back_and_says_so(self):
        selected, printed = self._run(
            [
                train._TrainProbeOutcome(
                    train.PROBE_ERROR, 8, True, detail="RuntimeError: nan detected in loss"
                ),
                train._TrainProbeOutcome(
                    train.PROBE_ERROR, 4, True, detail="RuntimeError: nan detected in loss"
                ),
            ]
        )
        self.assertIsNone(selected)
        self.assertIn("2 probe error(s)", printed)
        self.assertIn("found no viable candidate because probes errored, not because of VRAM", printed)
        self.assertEqual(0, len(train._load_autotune_entries(self._scratch_path)))

    def test_cached_entry_with_probe_errors_warns_on_reuse(self):
        key = train._make_autotune_cache_key(self.runtime)
        train._save_autotune_entries(
            self._scratch_path,
            {
                key: {
                    "train_batch_size": 4,
                    "use_activation_checkpointing": True,
                    "probe_error_count": 1,
                    "probe_errors": ["RuntimeError: nan detected in loss"],
                }
            },
        )
        buffer = io.StringIO()
        os.environ.pop("AUTORESEARCH_AUTOTUNE_REFRESH", None)
        os.environ.pop("AUTORESEARCH_DISABLE_AUTOTUNE", None)
        try:
            with redirect_stdout(buffer):
                selected = train._autotune_train_candidate(
                    self.runtime, tokenizer=None, vocab_size=1024, train_candidates=[(8, True), (4, True)]
                )
        finally:
            os.environ.pop("AUTORESEARCH_DISABLE_AUTOTUNE", None)
        self.assertEqual((4, True), selected)
        printed = buffer.getvalue()
        self.assertIn("Using cached autotune candidate", printed)
        self.assertIn("WARNING", printed)
        self.assertIn("RuntimeError: nan detected in loss", printed)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
class RealProbeErrorPathTest(unittest.TestCase):
    """Drives the real _benchmark_train_candidate with a bug injected into the probe.

    This is the regression the card describes: a non-OOM failure raised from inside
    a probe must come back as PROBE_ERROR with a loud banner, not as a quiet
    capacity rejection. The dataloader is stubbed so no dataset is needed.
    """

    def setUp(self):
        self._real_dataloader = train.make_dataloader
        self._real_gpt = train.GPT
        self.runtime = train.detect_runtime()
        self.vocab_size = 2048
        self._fake_tokenizer = type(
            "FakeTokenizer", (), {"dataset": "fake", "get_vocab_size": lambda self: self.vocab_size}
        )()

    def tearDown(self):
        train.make_dataloader = self._real_dataloader
        train.GPT = self._real_gpt

    def _stub_dataloader(self, raise_in_forward):
        def fake_make_dataloader(tokenizer, batch_size, seq_len, split, device, dataset=None):
            def forever():
                x = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
                while True:
                    yield (x, x, 0)

            return forever()

        train.make_dataloader = fake_make_dataloader

        if raise_in_forward is None:
            return

        real_gpt = self._real_gpt

        class BuggyGPT(real_gpt):
            def forward(self, *args, **kwargs):
                raise_in_forward()
                return super().forward(*args, **kwargs)

        train.GPT = BuggyGPT

    def test_shape_error_inside_the_probe_is_reported_as_a_probe_error(self):
        def raise_shape_error():
            raise RuntimeError("Failed to allocate tensor with shape [1, 1, 8192, 8192]")

        self._stub_dataloader(raise_shape_error)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            outcome = train._benchmark_train_candidate(
                runtime=self.runtime,
                tokenizer=self._fake_tokenizer,
                vocab_size=self.vocab_size,
                train_batch_size=1,
                use_checkpointing=False,
            )
        self.assertEqual(train.PROBE_ERROR, outcome.kind)
        self.assertTrue(outcome.is_error)
        self.assertFalse(outcome.is_capacity_rejection)
        self.assertIn("RuntimeError", outcome.detail)
        self.assertIn("8192", outcome.detail)
        printed = buffer.getvalue()
        self.assertIn("AUTOTUNE PROBE FAILED", printed)
        self.assertIn("this is a code error, not a VRAM limit", printed)

    def test_cuda_oom_inside_the_probe_is_a_quiet_capacity_rejection(self):
        def raise_oom():
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 900.00 GiB")

        self._stub_dataloader(raise_oom)
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            outcome = train._benchmark_train_candidate(
                runtime=self.runtime,
                tokenizer=self._fake_tokenizer,
                vocab_size=self.vocab_size,
                train_batch_size=1,
                use_checkpointing=False,
            )
        self.assertEqual(train.PROBE_OUT_OF_MEMORY, outcome.kind)
        self.assertTrue(outcome.is_capacity_rejection)
        self.assertNotIn("AUTOTUNE PROBE FAILED", buffer.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)

