import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emotion_eval.data import LABELS, Sample
from emotion_eval.infer import argument_parser, main, prepare_run, read_predictions
from emotion_eval.metrics import classification_metrics, parse_prediction
from emotion_eval.prompts import EMOTION_PROMPT


class ScoringTests(unittest.TestCase):
    def test_parser_does_not_score_explanations_or_alternatives(self):
        for output in ("joy or sadness", "not joy", "I think joy", "<think>joy", "", "happy"):
            with self.subTest(output=output):
                self.assertIsNone(parse_prediction(output, LABELS["emotion6"]))
        for output in ("joy", "Joy", "JOY", " Joy.\n", "<answer>joy</answer>",
                       "<think>sadness?</think>joy", "**joy**"):
            with self.subTest(output=output):
                self.assertEqual(parse_prediction(output, LABELS["emotion6"]), "Joy")
        self.assertIsNone(parse_prediction("joy", LABELS["emoset"]))

    def test_invalid_is_counted_as_wrong_and_macro_f1_uses_all_classes(self):
        metrics = classification_metrics([
            {"label": "anger", "prediction": "anger"},
            {"label": "anger", "prediction": None},
            {"label": "joy", "prediction": "anger"},
            {"label": "joy", "prediction": "joy"},
        ], ("anger", "joy", "fear"))
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["num_invalid"], 1)
        self.assertAlmostEqual(metrics["macro_f1"], (0.5 + 2 / 3) / 3)
        self.assertEqual(metrics["confusion_matrix"]["values"], [
            [1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 0],
        ])

    def test_metrics_match_mixed_case_labels_and_keep_invalid_in_denominator(self):
        metrics = classification_metrics([
            {"label": "anger", "prediction": "ANGER"},
            {"label": "JOY", "prediction": "joy"},
            {"label": "Joy", "prediction": "happy"},
            {"label": " Anger ", "prediction": None},
        ], ("Anger", "Joy", "Fear"))
        self.assertEqual(metrics["num_samples"], 4)
        self.assertEqual(metrics["num_correct"], 2)
        self.assertEqual(metrics["num_invalid"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 4 / 9)
        self.assertEqual(metrics["confusion_matrix"], {
            "rows": ["Anger", "Joy", "Fear"],
            "columns": ["Anger", "Joy", "Fear", "invalid"],
            "values": [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 0, 0]],
        })


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples = [Sample("emotion6", "joy/1.jpg", self.root / "joy/1.jpg", "Joy")]
        self.record = {"dataset": "emotion6", "sample_id": "joy/1.jpg", "image_path": str(self.samples[0].image_path),
                       "label": "Joy", "prediction": "Joy", "raw_output": "joy", "correct": True}
        self.path = self.root / "predictions.jsonl"

    def save(self, records, tail=""):
        self.path.write_text("".join(json.dumps(row) + "\n" for row in records) + tail, encoding="utf-8")

    def read(self):
        return read_predictions(self.path, self.samples, LABELS["emotion6"])

    def test_interrupted_final_write_is_recovered(self):
        self.save([self.record], '{"sample_id": "incomplete')
        self.assertEqual(self.read(), [self.record])
        self.assertEqual(self.path.read_text(), json.dumps(self.record) + "\n")

    def test_complete_final_record_without_newline_is_preserved(self):
        self.path.write_text(json.dumps(self.record), encoding="utf-8")
        self.assertEqual(self.read(), [self.record])
        self.assertTrue(self.path.read_bytes().endswith(b"\n"))

    def test_duplicate_stale_and_tampered_results_fail(self):
        for records in ([self.record, self.record], [{**self.record, "label": "Anger"}],
                        [{**self.record, "prediction": "Anger"}], [{**self.record, "sample_id": "joy/99.jpg"}]):
            with self.subTest(records=records):
                self.save(records)
                with self.assertRaises(ValueError):
                    self.read()

    def saved_run(self):
        args = argument_parser().parse_args(["--output-dir", str(self.root)])
        prompt = EMOTION_PROMPT.format(labels=", ".join(LABELS["emotion6"]))
        run = prepare_run(args, "emotion6", self.samples, "all_images_no_official_test_split", prompt)
        self.path = run["directory"] / "predictions.jsonl"
        self.save([self.record])
        (run["directory"] / "metrics.json").write_text('{"old": true}\n', encoding="utf-8")
        return args, prompt, run["directory"]

    def test_fresh_run_overwrites_results_and_settings_but_preserves_unrelated_files(self):
        args, prompt, directory = self.saved_run()
        notes = directory / "notes.txt"
        notes.write_text("keep me", encoding="utf-8")
        other = self.root / "emoset"
        other.mkdir()
        other_predictions = other / "predictions.jsonl"
        other_predictions.write_text("keep other dataset", encoding="utf-8")
        args.batch_size = 8
        prompt += " Changed."

        run = prepare_run(args, "emotion6", self.samples, "all_images_no_official_test_split", prompt)

        self.assertEqual(run["records"], [])
        self.assertEqual(run["pending"], self.samples)
        self.assertEqual(self.path.read_text(encoding="utf-8"), "")
        self.assertFalse((directory / "metrics.json").exists())
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["batch_size"], 8)
        self.assertEqual(config["prompt"], prompt)
        self.assertEqual(notes.read_text(encoding="utf-8"), "keep me")
        self.assertEqual(other_predictions.read_text(encoding="utf-8"), "keep other dataset")

    def test_resume_preserves_saved_records_and_rejects_changed_prompt_settings_or_dataset(self):
        args, prompt, directory = self.saved_run()
        original = self.path.read_bytes()
        args.resume = True
        run = prepare_run(args, "emotion6", self.samples, "all_images_no_official_test_split", prompt)
        self.assertEqual(run["records"], [self.record])
        self.assertEqual(run["pending"], [])
        self.assertEqual(self.path.read_bytes(), original)
        self.assertTrue((directory / "metrics.json").exists())
        with self.assertRaisesRegex(ValueError, "prompt"):
            prepare_run(args, "emotion6", self.samples, "all_images_no_official_test_split", prompt + " Changed.")
        with self.assertRaisesRegex(ValueError, "samples_sha256"):
            prepare_run(args, "emotion6", [], "all_images_no_official_test_split", prompt)
        args.max_pixels //= 2
        with self.assertRaisesRegex(ValueError, "max_pixels"):
            prepare_run(args, "emotion6", self.samples, "all_images_no_official_test_split", prompt)
        self.assertEqual(self.path.read_bytes(), original)

    def test_fresh_main_replaces_predictions_without_appending_old_results(self):
        self.saved_run()
        summary_path = self.root / "summary.json"
        summary_path.write_text('{"old": true}', encoding="utf-8")
        with (
            patch("emotion_eval.infer.load_emotion6", return_value=self.samples),
            patch("emotion_eval.infer.QwenPredictor") as predictor,
        ):
            predictor.return_value.predict.return_value = ["Anger"]
            self.assertEqual(main([
                "--datasets", "emotion6", "--model", str(self.root), "--output-dir", str(self.root),
            ]), 0)
        predictions = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["prediction"], "Anger")
        predictor.return_value.predict.assert_called_once()
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(list(summary), ["emotion6"])
        self.assertEqual(summary["emotion6"]["num_samples"], 1)
        self.assertEqual(summary["emotion6"]["accuracy"], 0)

    def test_fresh_main_clears_stale_summary_before_model_load(self):
        self.saved_run()
        summary_path = self.root / "summary.json"
        summary_path.write_text('{"old": true}', encoding="utf-8")
        with (
            patch("emotion_eval.infer.load_emotion6", return_value=self.samples),
            patch("emotion_eval.infer.QwenPredictor", side_effect=RuntimeError("model unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                main(["--datasets", "emotion6", "--model", str(self.root), "--output-dir", str(self.root)])
        self.assertFalse(summary_path.exists())

    def test_dry_run_preserves_existing_results(self):
        self.saved_run()
        (self.root / "summary.json").write_text('{"old": true}', encoding="utf-8")
        original = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        with (
            patch("emotion_eval.infer.load_emotion6", return_value=self.samples),
            patch("emotion_eval.infer.QwenPredictor") as predictor,
        ):
            self.assertEqual(main([
                "--datasets", "emotion6", "--model", str(self.root), "--output-dir", str(self.root), "--dry-run",
            ]), 0)
        predictor.assert_not_called()
        self.assertEqual(
            {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, original
        )


class Emo8IntegrationTests(unittest.TestCase):
    def test_cli_evaluates_all_three_splits_with_emo8_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected_ids = ["train/sad/shared.jpg", "val/amusement/shared.jpg", "test/awe/shared.jpg"]
            for sample_id in expected_ids:
                path = root / "EMO8" / sample_id
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            output = root / "outputs"
            with patch("emotion_eval.infer.QwenPredictor") as predictor:
                predictor.return_value.predict.side_effect = [["sadness", "AMUSEMENT"], ["Awe"]]
                self.assertEqual(main([
                    "--datasets", "emo8", "--data-root", str(root), "--model", str(root),
                    "--batch-size", "2", "--output-dir", str(output),
                ]), 0)
            records = [json.loads(line) for line in (output / "emo8/predictions.jsonl").read_text().splitlines()]
            self.assertEqual([record["sample_id"] for record in records], expected_ids)
            self.assertEqual([record["label"] for record in records], ["Sadness", "Amusement", "Awe"])
            self.assertTrue(all(record["correct"] for record in records))
            config = json.loads((output / "emo8/config.json").read_text())
            self.assertEqual(config["split"], "all_train_val_test_as_test")
            self.assertEqual(config["labels"], list(LABELS["emo8"]))
            for call in predictor.return_value.predict.call_args_list:
                self.assertEqual(call.kwargs["prompt"], config["prompt"])
            metrics = json.loads((output / "emo8/metrics.json").read_text())
            self.assertEqual(metrics["num_samples"], 3)
            self.assertEqual(metrics["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
