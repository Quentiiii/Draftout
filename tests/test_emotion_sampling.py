"""CPU regressions for sampling, generated-token penalties and safe resume."""

import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from emotion_eval.data import LABELS, Sample
from emotion_eval.infer import argument_parser, data_fingerprint, main, prepare_run
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT
from emotion_eval.sampling import (
    PresencePenaltyLogitsProcessor,
    batch_seed,
    build_decoding_config,
    validate_decoding_config,
)


SAMPLING_ARGS = [
    "--do-sample", "--temperature", "0.7", "--top-p", "0.8", "--top-k", "20",
    "--min-p", "0", "--presence-penalty", "1.5", "--repetition-penalty", "1", "--seed", "42",
]


class DecodingConfigTests(unittest.TestCase):
    def sampling_config(self):
        args = argument_parser().parse_args([*SAMPLING_ARGS, "--batch-size", "8"])
        return build_decoding_config(args)

    def test_greedy_defaults_preserve_existing_config_shape(self):
        config = build_decoding_config(argument_parser().parse_args([]))
        self.assertEqual(config, {"do_sample": False})
        validate_decoding_config(config)

    def test_sampling_records_all_effective_parameters_and_seed_policy(self):
        config = self.sampling_config()
        self.assertEqual(config, {
            "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
            "seed": 42, "seed_policy": "sha256_batch_sample_ids_v1",
            "presence_penalty_scope": "generated_tokens_only", "seed_batch_size": 8,
        })
        validate_decoding_config(config)

    def test_invalid_sampling_settings_rejected(self):
        invalid = {
            "do_sample": [1, "true", None],
            "temperature": [0, -0.1, float("inf"), float("nan")],
            "top_p": [0, -0.1, 1.1, float("nan")],
            "top_k": [-1, 2.5, True],
            "min_p": [-0.1, 1.1, float("inf")],
            "presence_penalty": [-2.1, 2.1, float("nan")],
            "repetition_penalty": [0, -1, float("inf")],
            "seed": [-1, 2**63, 1.5, True],
        }
        valid = self.sampling_config()
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate_decoding_config({**valid, key: value})

    def test_valid_sampling_boundaries(self):
        config = self.sampling_config()
        for changes in (
            {"top_k": 0, "top_p": 1.0, "min_p": 1.0, "presence_penalty": -2.0, "seed": 0},
            {"presence_penalty": 2.0, "seed": 2**63 - 1},
        ):
            with self.subTest(changes=changes):
                validate_decoding_config({**config, **changes})


class PresencePenaltyTests(unittest.TestCase):
    def test_penalizes_generated_presence_once_per_row_and_ignores_prompt(self):
        # Token 1 appears in both prompts, but only row 1 generated it.
        token_ids = torch.tensor([[1, 3, 3, 5, 5], [1, 3, 1, 2, 2]])
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
        result = PresencePenaltyLogitsProcessor(1.5, prompt_length=2)(token_ids, scores.clone())
        self.assertTrue(torch.equal(result, torch.tensor([
            [0.0, 1.0, 2.0, 1.5, 4.0, 3.5],
            [5.0, 2.5, 1.5, 2.0, 1.0, 0.0],
        ])))

    def test_first_generation_step_and_zero_penalty_preserve_logits(self):
        token_ids = torch.tensor([[1, 2], [3, 0]])
        scores = torch.tensor([[float("-inf"), 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
        for processor in (
            PresencePenaltyLogitsProcessor(1.5, prompt_length=2),
            PresencePenaltyLogitsProcessor(0.0, prompt_length=0),
        ):
            with self.subTest(processor=processor):
                self.assertTrue(torch.equal(processor(token_ids, scores.clone()), scores))

    def test_negative_presence_penalty_boosts_seen_tokens_once(self):
        result = PresencePenaltyLogitsProcessor(-0.5, prompt_length=1)(
            torch.tensor([[3, 2, 2]]), torch.zeros((1, 4)),
        )
        self.assertTrue(torch.equal(result, torch.tensor([[0.0, 0.0, 0.5, 0.0]])))


class BatchSeedTests(unittest.TestCase):
    def test_seed_is_repeatable_and_depends_on_ordered_sample_identity(self):
        samples = [Sample("emotion6", f"image/{i}.jpg", Path(f"/images/{i}.jpg"), "Joy") for i in range(2)]
        original = batch_seed(42, samples)
        self.assertIsInstance(original, int)
        self.assertGreaterEqual(original, 0)
        self.assertLess(original, 2**63)
        self.assertEqual(batch_seed(42, list(samples)), original)
        # Moving files or changing ground truth cannot affect model randomness.
        moved = [replace(sample, image_path=Path("/other") / sample.image_path.name, label="Fear")
                 for sample in samples]
        self.assertEqual(batch_seed(42, moved), original)
        for seed, batch in (
            (43, samples), (42, list(reversed(samples))), (42, samples[:1]),
            (42, [replace(samples[0], dataset="emo8"), samples[1]]),
            (42, [replace(samples[0], sample_id="image/new.jpg"), samples[1]]),
        ):
            with self.subTest(seed=seed, batch=batch):
                self.assertNotEqual(batch_seed(seed, batch), original)


class SamplingResumeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.samples = [Sample("emotion6", f"joy/{i}.jpg", self.root / f"joy/{i}.jpg", "Joy") for i in range(7)]
        self.prompt = DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(LABELS["emotion6"]))
        self.split = "all_images_no_official_test_split"

    def test_sampling_resume_rejects_changed_temperature_seed_or_batch_size(self):
        args = argument_parser().parse_args([*SAMPLING_ARGS, "--output-dir", str(self.root), "--batch-size", "3"])
        run = prepare_run(args, "emotion6", self.samples, self.split, self.prompt)
        original = (run["directory"] / "config.json").read_bytes()
        args.resume = True
        for key, value in (("temperature", 0.9), ("seed", 43), ("batch_size", 4)):
            previous = getattr(args, key)
            setattr(args, key, value)
            try:
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    prepare_run(args, "emotion6", self.samples, self.split, self.prompt)
            finally:
                setattr(args, key, previous)
            self.assertEqual((run["directory"] / "config.json").read_bytes(), original)

    def test_existing_greedy_run_remains_resumable_without_sampling_metadata(self):
        args = argument_parser().parse_args(["--output-dir", str(self.root), "--resume"])
        directory = self.root / "emotion6"
        directory.mkdir()
        old_config = {
            "format_version": 2, "dataset": "emotion6", "split": self.split,
            "num_samples": len(self.samples), "samples_sha256": data_fingerprint(self.samples),
            "model": str(args.model), "prompt": self.prompt, "labels": list(LABELS["emotion6"]),
            "enable_thinking": False, "response_format": "draft_then_boxed_label", "do_sample": False,
            "max_new_tokens": 512, "min_pixels": 65536, "max_pixels": 1048576,
            "dtype": "bfloat16", "device": "auto", "batch_size": 1,
        }
        path = directory / "config.json"
        path.write_text(json.dumps(old_config), encoding="utf-8")
        original = path.read_bytes()
        run = prepare_run(args, "emotion6", self.samples, self.split, self.prompt)
        self.assertEqual(run["pending"], self.samples)
        self.assertEqual(path.read_bytes(), original)

    def test_partial_write_resume_regenerates_original_batch_without_duplicate_records(self):
        argv = [*SAMPLING_ARGS, "--datasets", "emotion6", "--model", str(self.root),
                "--output-dir", str(self.root), "--batch-size", "3"]

        def predict(batch, prompt):
            seed = batch_seed(42, batch)
            return [f"<draft>Batch {seed}, position {index}.</draft>\\boxed{{Joy}}"
                    for index in range(len(batch))]

        with (
            patch("emotion_eval.infer.load_emotion6", return_value=self.samples),
            patch("emotion_eval.infer.QwenPredictor") as predictor,
            patch("sys.stdout", new_callable=io.StringIO),
            patch("sys.stderr", new_callable=io.StringIO),
        ):
            predictor.return_value.predict.side_effect = predict
            self.assertEqual(main(argv), 0)
            path = self.root / "emotion6/predictions.jsonl"
            original = path.read_bytes()
            lines = original.splitlines(keepends=True)
            summary = (self.root / "summary.json").read_bytes()
            # The first batch completed; a kill interrupted the second batch.
            path.write_bytes(b"".join(lines[:4]) + b'{"dataset": "emotion6", "sample_id": "jo')
            predictor.reset_mock()
            self.assertEqual(main([*argv, "--resume"]), 0)
            batches = [call.args[0] for call in predictor.return_value.predict.call_args_list]
            self.assertEqual(batches, [self.samples[3:6], self.samples[6:7]])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual((self.root / "summary.json").read_bytes(), summary)
            predictor.reset_mock()
            self.assertEqual(main([*argv, "--resume"]), 0)
            predictor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
