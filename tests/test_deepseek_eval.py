"""Offline coverage of the API evaluator's data, persistence, and HTTP boundaries."""

import base64
import io
import json
import os
import tempfile
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image, PngImagePlugin

from emotion_eval import deepseek
from emotion_eval.data import LABELS
from emotion_eval.metrics import parse_draft_prediction, parse_draft_response
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT


API_KEY = "dummy-private-api-key"


def formatted_answer(label, draft="The visible expression supports this emotion."):
    return f"<draft>{draft}</draft>\n\\boxed{{{label}}}"


def completion(text, *, finish_reason="stop"):
    return {
        "id": "mock-completion",
        "model": "deepseek-flash",
        "choices": [{"message": {"content": text}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 2, "total_tokens": 13},
    }


def json_response(value):
    return io.BytesIO(json.dumps(value).encode("utf-8"))


class DeepSeekDraftParserTests(unittest.TestCase):
    def test_final_box_controls_prediction_even_when_draft_mentions_other_emotions(self):
        draft = "Anger and Sadness are alternatives; the example \\boxed{Fear} is not the answer."
        response = " \n" + formatted_answer(" aWe ", draft) + "\n "
        self.assertEqual(parse_draft_response(response, LABELS["emoset"]), (draft, "Awe"))
        self.assertEqual(parse_draft_prediction(response, LABELS["emoset"]), "Awe")

    def test_multiline_explanation_is_preserved_and_outer_whitespace_trimmed(self):
        draft = "A broad landscape fills the frame.\nIts scale suggests awe."
        self.assertEqual(
            parse_draft_response(formatted_answer("Awe", "\n " + draft + " \n"), LABELS["emoset"]),
            (draft, "Awe"),
        )

    def test_invalid_boxed_label_keeps_explanation_without_guessing_an_emotion(self):
        draft = "The face could convey Anger or Sadness."
        for label in ("Anger or Sadness", "I think Anger", "happy", "", "Anger."):
            with self.subTest(label=label):
                self.assertEqual(
                    parse_draft_response(formatted_answer(label, draft), LABELS["emoset"]),
                    (draft, None),
                )

    def test_missing_truncated_duplicated_or_misplaced_delimiters_are_invalid(self):
        invalid = (
            "Anger",
            r"\boxed{Anger}",
            "<draft>Anger is visible.</draft>",
            "<draft>Anger is visible.",
            r"<draft>Anger is visible.</draft>\boxed{Anger",
            r"<draft></draft>\boxed{Anger}",
            "<draft> \n </draft>\\boxed{Anger}",
            r"<draft><draft>Anger is visible.</draft></draft>\boxed{Anger}",
            r"<draft>First.</draft><draft>Second.</draft>\boxed{Anger}",
            r"<draft>Visible anger.</draft>\boxed{Anger}\boxed{Sadness}",
            r"\boxed{Anger}<draft>Visible anger.</draft>",
            "Here is the answer:\n" + formatted_answer("Anger"),
            formatted_answer("Anger") + "\nThat is my answer.",
        )
        for response in invalid:
            with self.subTest(response=response):
                self.assertEqual(parse_draft_response(response, LABELS["emoset"]), (None, None))


class DeepSeekEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "EMOSET"
        self.data_root.mkdir()
        self.test_list = self.root / "subset.json"
        self.output_dir = self.root / "results"
        self.rows = []
        for index, (label, color) in enumerate(zip(
            ("Anger", "Awe", "Sadness"), ("red", "green", "blue")
        )):
            image_name = f"images/{label.lower()}_truth_{index}.png"
            annotation_name = f"annotations/{label.lower()}_truth_{index}.json"
            image_path = self.data_root / image_name
            annotation_path = self.data_root / annotation_name
            image_path.parent.mkdir(exist_ok=True)
            annotation_path.parent.mkdir(exist_ok=True)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("label", f"hidden-metadata-{label}")
            with Image.new("RGB", (24, 12), color=color) as image:
                image.save(image_path, pnginfo=metadata)
            annotation_path.write_text(json.dumps({
                "emotion": label.lower(), "caption": "private-annotation-content",
            }), encoding="utf-8")
            self.rows.append([label.lower(), image_name, annotation_name])
        self.write_rows(self.data_root / "test.json", self.rows)
        self.write_rows(self.test_list, self.rows)
        self.argv = [
            "--data-root", str(self.data_root), "--test-list", str(self.test_list),
            "--output-dir", str(self.output_dir), "--workers", "1",
        ]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"DEEPSEEK_API_KEY": API_KEY}).start()

    @staticmethod
    def write_rows(path, rows):
        path.write_text(json.dumps(rows), encoding="utf-8")

    def run_main(self, *extra):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return deepseek.main([*self.argv, *extra])

    def read_json(self, filename):
        return json.loads((self.output_dir / filename).read_text(encoding="utf-8"))

    def read_records(self):
        return [json.loads(line) for line in
                (self.output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]

    def seed_completed_run(self):
        with patch.object(deepseek, "request_completion", side_effect=[
            completion(formatted_answer("Anger")), completion(formatted_answer("Awe")),
            completion(formatted_answer("Sadness")),
        ]) as request:
            self.assertEqual(self.run_main(), 0)
            self.assertEqual(request.call_count, 3)

    def test_main_sends_only_pixels_and_prompt_and_scores_invalid_answers(self):
        draft = "A tense face and clenched fists suggest Anger; this is unlike Awe."
        first_response = formatted_answer("Anger", draft)
        responses = [
            json_response(completion(first_response)),
            json_response(completion("I think Awe", finish_reason="length")),
            json_response(completion(formatted_answer("Anger"))),
        ]
        with patch.object(deepseek.urllib.request, "urlopen", side_effect=responses) as urlopen:
            self.assertEqual(self.run_main("--max-image-side", "16"), 0)

        self.assertEqual(urlopen.call_count, 3)
        for call in urlopen.call_args_list:
            request = call.args[0]
            self.assertEqual(request.full_url, "https://api.deepseek.com/chat/completions")
            self.assertEqual(request.get_header("Authorization"), f"Bearer {API_KEY}")
            payload = json.loads(request.data)
            self.assertEqual(payload["model"], "deepseek-flash")
            self.assertEqual(payload["thinking"], {"type": "disabled"})
            messages = payload["messages"]
            self.assertEqual(len(messages), 1)
            content = messages[0]["content"]
            self.assertEqual(len(content), 2)
            self.assertEqual(content[1], {"type": "text", "text": deepseek.PROMPT})
            url = content[0]["image_url"]["url"]
            self.assertTrue(url.startswith("data:image/jpeg;base64,"))
            pixels = base64.b64decode(url.split(",", 1)[1])
            with Image.open(io.BytesIO(pixels)) as image:
                self.assertEqual(image.size, (16, 8))
                self.assertFalse(image.getexif())
            self.assertNotIn(b"hidden-metadata", pixels)
            for forbidden in (str(self.data_root), "private-annotation-content", API_KEY):
                self.assertNotIn(forbidden.encode(), request.data)
            for row in self.rows:
                for name in row[1:]:
                    self.assertNotIn(name.encode(), request.data)
        metrics = self.read_json("metrics.json")
        self.assertTrue(metrics["complete"])
        self.assertEqual(metrics["num_pending"], 0)
        self.assertEqual(metrics["num_samples"], 3)
        self.assertEqual(metrics["num_correct"], 1)
        self.assertEqual(metrics["num_invalid"], 1)
        self.assertEqual(metrics["num_truncated"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        self.assertEqual(metrics["usage"], {
            "prompt_tokens": 33, "completion_tokens": 6, "total_tokens": 39,
        })
        records = self.read_records()
        self.assertEqual([record["prediction"] for record in records], ["Anger", None, "Anger"])
        self.assertEqual(records[0]["draft"], draft)
        self.assertEqual(records[0]["raw_output"], first_response)
        self.assertIsNone(records[1]["draft"])
        for path in self.output_dir.iterdir():
            self.assertNotIn(API_KEY, path.read_text(encoding="utf-8"))

    def test_dry_run_requires_no_key_and_creates_no_output_or_network_requests(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch.object(deepseek.urllib.request, "urlopen") as urlopen:
            self.assertEqual(self.run_main("--dry-run"), 0)
        urlopen.assert_not_called()
        self.assertFalse(self.output_dir.exists())
        self.assertIn("3 official test images", self.stdout.getvalue())

    def test_subset_must_match_official_test_rows_before_any_api_call(self):
        variants = [
            [self.rows[0], self.rows[0]],
            [["awe", self.rows[0][1], self.rows[0][2]]],
            [[self.rows[0][0], self.rows[0][1], self.rows[1][2]]],
            [["anger", "images/unknown.png", self.rows[0][2]]],
        ]
        with patch.object(deepseek.urllib.request, "urlopen") as urlopen:
            for rows in variants:
                with self.subTest(rows=rows):
                    self.write_rows(self.test_list, rows)
                    with self.assertRaises(ValueError):
                        self.run_main()
                    self.assertFalse(self.output_dir.exists())
        urlopen.assert_not_called()

    def test_existing_run_is_not_overwritten_and_resume_skips_saved_responses(self):
        self.seed_completed_run()
        original = {path.name: path.read_bytes() for path in self.output_dir.iterdir()}
        with patch.object(deepseek, "request_completion") as request:
            with self.assertRaisesRegex(ValueError, "Results already exist"):
                self.run_main()
            self.assertEqual(original, {path.name: path.read_bytes() for path in self.output_dir.iterdir()})
            self.assertEqual(self.run_main("--resume", "--workers", "2"), 0)
        request.assert_not_called()
        self.assertEqual((self.output_dir / "predictions.jsonl").read_bytes(), original["predictions.jsonl"])
        self.assertEqual(self.read_json("metrics.json")["usage"]["total_tokens"], 39)
        self.assertEqual(self.read_json("metrics.json")["accuracy"], 1)
        self.assertEqual([row["prediction"] for row in self.read_records()], ["Anger", "Awe", "Sadness"])

    def test_resume_rejects_changed_settings_or_image_content_without_modifying_results(self):
        self.seed_completed_run()
        original = {path.name: path.read_bytes() for path in self.output_dir.iterdir()}
        with patch.object(deepseek, "request_completion") as request:
            for option, value in (("--model", "another-model"), ("--max-tokens", "32"),
                                  ("--max-image-side", "512")):
                with self.subTest(option=option):
                    with self.assertRaisesRegex(ValueError, "changed settings/data"):
                        self.run_main("--resume", option, value)
            with Image.new("RGB", (24, 12), color="yellow") as image:
                image.save(self.data_root / self.rows[0][1])
            with self.assertRaisesRegex(ValueError, "changed settings/data"):
                self.run_main("--resume")
        request.assert_not_called()
        self.assertEqual(original, {path.name: path.read_bytes() for path in self.output_dir.iterdir()})

    def test_failed_batch_saves_successful_peer_and_resume_requests_only_pending_images(self):
        urls = [deepseek.encode_image(self.data_root / row[1], 1024) for row in self.rows]

        def first_batch(args, api_key, payload):
            url = payload["messages"][0]["content"][0]["image_url"]["url"]
            if url == urls[0]:
                raise deepseek.APIError(f"Simulated request failure: {api_key}")
            self.assertEqual(url, urls[1], "A later batch must not be submitted after failure")
            return completion(formatted_answer("Awe"))

        with patch.object(deepseek, "request_completion", side_effect=first_batch) as request:
            self.assertEqual(self.run_main("--workers", "2"), 1)
            self.assertEqual(request.call_count, 2)
        self.assertEqual([row["sample_id"] for row in self.read_records()], [self.rows[1][1]])
        metrics = self.read_json("metrics.json")
        self.assertFalse(metrics["complete"])
        self.assertEqual(metrics["num_pending"], 2)
        self.assertEqual(metrics["num_samples"], 1)
        error = self.read_json("errors.jsonl")
        self.assertEqual(error["sample_id"], self.rows[0][1])
        self.assertIn("[REDACTED]", error["error"])
        self.assertNotIn(API_KEY, self.stderr.getvalue())

        with patch.object(deepseek, "request_completion", side_effect=[
            completion(formatted_answer("Anger")), completion(formatted_answer("Sadness")),
        ]) as request:
            self.assertEqual(self.run_main("--resume"), 0)
        resumed_urls = [call.args[2]["messages"][0]["content"][0]["image_url"]["url"]
                        for call in request.call_args_list]
        self.assertEqual(resumed_urls, [urls[0], urls[2]])
        self.assertEqual(len(self.read_records()), 3)
        self.assertEqual(self.read_json("metrics.json")["accuracy"], 1)
        self.assertEqual(self.read_json("metrics.json")["usage"]["total_tokens"], 39)

    def test_resume_recovers_interrupted_final_record_and_runs_only_that_sample(self):
        self.seed_completed_run()
        predictions = self.output_dir / "predictions.jsonl"
        lines = predictions.read_bytes().splitlines(keepends=True)
        predictions.write_bytes(b"".join(lines[:2]) + b'{"sample_id": "interrupted')
        with patch.object(deepseek, "request_completion", return_value=completion(formatted_answer("Sadness"))) as request:
            self.assertEqual(self.run_main("--resume"), 0)
        request.assert_called_once()
        self.assertEqual([row["sample_id"] for row in self.read_records()], [row[1] for row in self.rows])
        self.assertTrue(predictions.read_bytes().startswith(b"".join(lines[:2])))
        self.assertIn("Recovering incomplete final JSONL record", self.stderr.getvalue())
        self.assertTrue(self.read_json("metrics.json")["complete"])

    def test_truncated_and_duplicate_answers_remain_invalid_on_resume(self):
        responses = [
            completion(r"<draft>Anger is visible.</draft>\boxed{Anger", finish_reason="length"),
            completion(r"<draft>Awe is visible.</draft>\boxed{Awe}\boxed{Anger}"),
            completion(formatted_answer("Sadness", "The person is tearful.")),
        ]
        with patch.object(deepseek, "request_completion", side_effect=responses):
            self.assertEqual(self.run_main(), 0)
        records = self.read_records()
        self.assertEqual([record["prediction"] for record in records], [None, None, "Sadness"])
        metrics = self.read_json("metrics.json")
        self.assertEqual(metrics["num_invalid"], 2)
        self.assertEqual(metrics["num_truncated"], 1)
        self.assertAlmostEqual(metrics["accuracy"], 1 / 3)
        with patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--resume"), 0)
        request.assert_not_called()
        self.assertEqual(self.read_records(), records)
        self.assertEqual(self.read_json("metrics.json")["num_invalid"], 2)


class DeepSeekMultiDatasetTests(unittest.TestCase):
    """Exercise dataset selection and persistence without credentials or API access."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "dataset"
        self.output_dir = self.root / "results"
        self.samples = {name: [] for name in ("emoset", "emotion6", "emo8")}
        self.responses = {}
        self.emoset_rows = []
        for index, label in enumerate(("Anger", "Awe", "Sadness")):
            sample_id = f"images/{index}.png"
            annotation = f"annotation/{index}.json"
            self.add_image("emoset", sample_id, label, self.data_root / "EMOSET" / sample_id)
            path = self.data_root / "EMOSET" / annotation
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"emotion": label.lower()}), encoding="utf-8")
            self.emoset_rows.append([label.lower(), sample_id, annotation])
        self.write_emoset_rows()
        for label in LABELS["emotion6"]:
            sample_id = f"{label.lower()}/same.png"
            self.add_image(
                "emotion6", sample_id, label,
                self.data_root / "Emotion6" / "Emotion6" / "images" / sample_id,
            )
        # This evaluation intentionally follows Qwen's directory labels, not distributions.
        (self.data_root / "Emotion6" / "Emotion6" / "ground_truth.txt").write_text(
            "Not used by the directory-label evaluation.\n", encoding="utf-8"
        )
        for split in ("train", "val", "test"):
            for label in LABELS["emo8"]:
                directory_label = "sad" if label == "Sadness" else label.lower()
                sample_id = f"{split}/{directory_label}/same.png"
                self.add_image("emo8", sample_id, label, self.data_root / "EMO8" / sample_id)
        self.argv = [
            "--data-root", str(self.data_root), "--output-dir", str(self.output_dir),
            "--workers", "1",
        ]
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"DEEPSEEK_API_KEY": API_KEY}).start()
        # Fail closed if any test unexpectedly escapes its mocked completion boundary.
        self.network = patch.object(
            deepseek.urllib.request, "urlopen", side_effect=AssertionError("Unexpected network request")
        ).start()

    def add_image(self, dataset, sample_id, label, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        index = sum(len(samples) for samples in self.samples.values()) + 1
        with Image.new("RGB", (12, 8), color=(index * 7 % 256, index * 19 % 256, index * 31 % 256)) as image:
            image.save(path)
        self.samples[dataset].append((sample_id, label, path))
        url = deepseek.encode_image(path, 1024)
        self.assertNotIn(url, self.responses)
        self.responses[url] = (dataset, sample_id, label)

    def write_emoset_rows(self):
        (self.data_root / "EMOSET" / "test.json").write_text(
            json.dumps(self.emoset_rows), encoding="utf-8"
        )

    def run_main(self, *extra):
        with redirect_stdout(self.stdout), redirect_stderr(self.stderr):
            return deepseek.main([*self.argv, *extra])

    def request_sample(self, payload):
        content = payload["messages"][0]["content"]
        dataset, sample_id, label = self.responses[content[0]["image_url"]["url"]]
        self.assertEqual(content[1]["text"], DEEPSEEK_EMOTION_PROMPT.format(
            labels=", ".join(LABELS[dataset])
        ))
        return dataset, sample_id, label

    def reply(self, args, api_key, payload):
        _, _, label = self.request_sample(payload)
        return completion(formatted_answer(label))

    def read_json(self, filename, dataset=None):
        directory = self.output_dir / dataset if dataset else self.output_dir
        return json.loads((directory / filename).read_text(encoding="utf-8"))

    def read_records(self, dataset):
        path = self.output_dir / dataset / "predictions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def snapshot(self):
        return {str(path.relative_to(self.output_dir)): path.read_bytes()
                for path in self.output_dir.rglob("*") if path.is_file()}

    def test_default_runs_all_datasets_with_their_labels_splits_and_separate_artifacts(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply) as request:
            self.assertEqual(self.run_main(), 0)
        self.assertEqual(request.call_count, 33)
        summary = self.read_json("summary.json")
        self.assertEqual(set(summary), set(self.samples))
        splits = {
            "emoset": "official_test", "emotion6": "all_images_no_official_test_split",
            "emo8": "all_train_val_test_as_test",
        }
        for dataset, samples in self.samples.items():
            with self.subTest(dataset=dataset):
                self.assertEqual(
                    {path.name for path in (self.output_dir / dataset).iterdir()}, set(deepseek.ARTIFACTS)
                )
                config = self.read_json("config.json", dataset)
                metrics = self.read_json("metrics.json", dataset)
                records = self.read_records(dataset)
                self.assertEqual(config["dataset"], dataset)
                self.assertEqual(config["labels"], list(LABELS[dataset]))
                self.assertEqual(config["split"], splits[dataset])
                self.assertEqual(metrics, summary[dataset])
                self.assertEqual(metrics["num_selected"], len(samples))
                self.assertEqual(metrics["num_correct"], len(samples))
                self.assertEqual(metrics["num_pending"], 0)
                self.assertTrue(metrics["complete"])
                self.assertEqual(set(metrics["per_class"]), set(LABELS[dataset]))
                self.assertEqual({row["sample_id"] for row in records}, {row[0] for row in samples})
                self.assertEqual({row["dataset"] for row in records}, {dataset})
        emotion6 = {row["label"]: row for row in self.read_records("emotion6")}
        self.assertEqual(emotion6["Joy"]["prediction"], "Joy")
        self.assertEqual(emotion6["Surprise"]["prediction"], "Surprise")
        self.assertEqual(len({row["sample_id"] for row in self.read_records("emo8")}), 24)
        self.network.assert_not_called()

    def test_emotion6_rejects_an_emoset_only_label_and_preserves_it_on_resume(self):
        def reply(args, api_key, payload):
            _, _, label = self.request_sample(payload)
            return completion(formatted_answer("Amusement" if label == "Joy" else label))

        with patch.object(deepseek, "request_completion", side_effect=reply):
            self.assertEqual(self.run_main("--datasets", "emotion6"), 0)
        records = self.read_records("emotion6")
        self.assertEqual(next(row for row in records if row["label"] == "Joy")["prediction"], None)
        metrics = self.read_json("metrics.json", "emotion6")
        self.assertEqual(metrics["num_invalid"], 1)
        self.assertEqual(metrics["num_correct"], 5)
        with patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--datasets", "emotion6", "--resume"), 0)
        request.assert_not_called()
        self.assertEqual(self.read_records("emotion6"), records)

    def test_resume_skips_every_completed_dataset_without_rewriting_predictions(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply):
            self.assertEqual(self.run_main(), 0)
        saved = {dataset: (self.output_dir / dataset / "predictions.jsonl").read_bytes()
                 for dataset in self.samples}
        with patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--resume", "--workers", "2"), 0)
        request.assert_not_called()
        for dataset in self.samples:
            self.assertEqual((self.output_dir / dataset / "predictions.jsonl").read_bytes(), saved[dataset])
        self.assertTrue(all(metrics["complete"] for metrics in self.read_json("summary.json").values()))

    def test_legacy_subset_cannot_write_flat_artifacts_into_an_existing_multi_dataset_run(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply):
            self.assertEqual(self.run_main(), 0)
        test_list = self.root / "legacy_subset.json"
        test_list.write_text(json.dumps(self.emoset_rows), encoding="utf-8")
        original = self.snapshot()
        with patch.object(deepseek, "request_completion") as request:
            for extra in ((), ("--resume",)):
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    self.run_main("--data-root", str(self.data_root / "EMOSET"),
                                  "--test-list", str(test_list), *extra)
        request.assert_not_called()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.output_dir / "config.json").exists())

    def test_failure_keeps_later_dataset_pending_and_resume_requests_only_remaining_images(self):
        first_call = True

        def reply(args, api_key, payload):
            nonlocal first_call
            dataset, _, label = self.request_sample(payload)
            if first_call:
                first_call = False
                summary = self.read_json("summary.json")
                self.assertEqual(set(summary), set(self.samples))
                self.assertEqual({key: value["num_pending"] for key, value in summary.items()},
                                 {key: len(samples) for key, samples in self.samples.items()})
            if dataset == "emotion6" and label == "Fear":
                raise deepseek.APIError("Simulated failure")
            self.assertNotEqual(dataset, "emo8", "Stop before submitting a later dataset after failure")
            return completion(formatted_answer(label))

        with patch.object(deepseek, "request_completion", side_effect=reply) as request:
            self.assertEqual(self.run_main(), 1)
        self.assertEqual(request.call_count, 6)
        summary = self.read_json("summary.json")
        self.assertEqual(set(summary), set(self.samples))
        self.assertTrue(summary["emoset"]["complete"])
        self.assertEqual(summary["emotion6"]["num_samples"], 2)
        self.assertEqual(summary["emotion6"]["num_pending"], 4)
        self.assertEqual(summary["emo8"]["num_samples"], 0)
        self.assertEqual(summary["emo8"]["num_pending"], 24)
        self.assertFalse(summary["emo8"]["complete"])
        saved = (self.output_dir / "emoset" / "predictions.jsonl").read_bytes()
        with patch.object(deepseek, "request_completion", side_effect=self.reply) as request:
            self.assertEqual(self.run_main("--resume"), 0)
        requested = [self.request_sample(call.args[2]) for call in request.call_args_list]
        self.assertEqual(len(requested), 28)
        self.assertFalse(any(dataset == "emoset" for dataset, _, _ in requested))
        self.assertEqual((self.output_dir / "emoset" / "predictions.jsonl").read_bytes(), saved)
        self.assertTrue(all(metrics["complete"] for metrics in self.read_json("summary.json").values()))

    def test_dry_run_checks_all_datasets_without_a_key_writes_or_network(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}), \
                patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--dry-run"), 0)
        request.assert_not_called()
        self.network.assert_not_called()
        self.assertFalse(self.output_dir.exists())
        self.assertIn("33 images", self.stdout.getvalue())
        for dataset in self.samples:
            self.assertIn(dataset, self.stdout.getvalue())

    def test_all_alias_selects_every_dataset_and_cannot_be_mixed_with_a_dataset_name(self):
        with patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--datasets", "all", "--dry-run"), 0)
            with self.assertRaises(SystemExit) as caught:
                self.run_main("--datasets", "all", "emoset", "--dry-run")
        self.assertEqual(caught.exception.code, 2)
        request.assert_not_called()
        self.assertIn("33 images", self.stdout.getvalue())
        self.assertFalse(self.output_dir.exists())

    def test_corrupt_image_in_last_dataset_prevents_all_requests_and_output_writes(self):
        self.samples["emo8"][-1][2].write_bytes(b"not an image")
        with patch.object(deepseek, "request_completion") as request:
            with self.assertRaises((OSError, ValueError)):
                self.run_main()
        request.assert_not_called()
        self.assertFalse(self.output_dir.exists())

    def test_last_dataset_existing_results_are_rejected_before_initializing_earlier_datasets(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply):
            self.assertEqual(self.run_main("--datasets", "emo8"), 0)
        original = self.snapshot()
        with patch.object(deepseek, "request_completion") as request:
            with self.assertRaisesRegex(ValueError, "Results already exist"):
                self.run_main()
        request.assert_not_called()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.output_dir / "emoset").exists())
        self.assertFalse((self.output_dir / "emotion6").exists())

    def test_last_dataset_resume_config_conflict_is_rejected_before_first_api_request(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply):
            self.assertEqual(self.run_main("--datasets", "emo8"), 0)
        path = self.output_dir / "emo8" / "config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["model"] = "incompatible-saved-model"
        path.write_text(json.dumps(config), encoding="utf-8")
        original = self.snapshot()
        with patch.object(deepseek, "request_completion") as request:
            with self.assertRaisesRegex(ValueError, "changed settings/data"):
                self.run_main("--resume")
        request.assert_not_called()
        self.assertEqual(self.snapshot(), original)
        self.assertFalse((self.output_dir / "emoset").exists())
        self.assertFalse((self.output_dir / "emotion6").exists())

    def test_limit_is_per_dataset_and_oversized_later_selection_fails_before_requests(self):
        with patch.object(deepseek, "request_completion", side_effect=self.reply) as request:
            self.assertEqual(self.run_main("--limit", "2"), 0)
        self.assertEqual(request.call_count, 6)
        self.assertEqual({dataset: metrics["num_selected"]
                          for dataset, metrics in self.read_json("summary.json").items()},
                         {dataset: 2 for dataset in self.samples})
        other_output = self.root / "oversized"
        with patch.object(deepseek, "request_completion") as request:
            with self.assertRaisesRegex(ValueError, "exceeds"):
                self.run_main("--datasets", "emotion6", "emoset", "--limit", "4",
                              "--output-dir", str(other_output))
        request.assert_not_called()
        self.assertFalse(other_output.exists())

    def test_explicit_emoset_subset_uses_new_dataset_subdirectory_layout(self):
        test_list = self.root / "emoset_subset.json"
        test_list.write_text(json.dumps(self.emoset_rows[1:2]), encoding="utf-8")
        with patch.object(deepseek, "request_completion", side_effect=self.reply) as request:
            self.assertEqual(self.run_main("--datasets", "emoset", "--test-list", str(test_list)), 0)
        request.assert_called_once()
        self.assertEqual(set(self.read_json("summary.json")), {"emoset"})
        self.assertEqual(self.read_json("config.json", "emoset")["split"], "official_test_subset")
        self.assertEqual([row["sample_id"] for row in self.read_records("emoset")], [self.emoset_rows[1][1]])
        self.assertFalse((self.output_dir / "predictions.jsonl").exists())

    def test_emotion6_optional_list_keeps_directory_labels_and_subset_scope(self):
        test_list = self.root / "emotion6_subset.txt"
        test_list.write_text("joy/same.png\nsurprise/same.png\n", encoding="utf-8")
        with patch.object(deepseek, "request_completion", side_effect=self.reply) as request:
            self.assertEqual(self.run_main("--datasets", "emotion6", "--emotion6-test-list", str(test_list)), 0)
        self.assertEqual(request.call_count, 2)
        self.assertEqual([row["label"] for row in self.read_records("emotion6")], ["Joy", "Surprise"])
        self.assertEqual(self.read_json("config.json", "emotion6")["split"], "custom_test_list")

    def test_full_emoset_selection_has_no_implicit_2000_image_cap(self):
        # Distinct paths with identical tiny pixels keep the >2k regression fixture inexpensive.
        source = self.samples["emoset"][0][2]
        for index in range(3, 2001):
            sample_id = f"images/{index}.png"
            annotation = f"annotation/{index}.json"
            (self.data_root / "EMOSET" / sample_id).hardlink_to(source)
            (self.data_root / "EMOSET" / annotation).write_text(
                json.dumps({"emotion": "anger"}), encoding="utf-8"
            )
            self.emoset_rows.append(["anger", sample_id, annotation])
        self.write_emoset_rows()
        with patch.object(deepseek, "request_completion") as request:
            self.assertEqual(self.run_main("--datasets", "emoset", "--dry-run"), 0)
        request.assert_not_called()
        self.assertIn('"selected": 2001', self.stdout.getvalue())
        self.assertFalse(self.output_dir.exists())


class DeepSeekHTTPTests(unittest.TestCase):
    def setUp(self):
        self.args = deepseek.argument_parser().parse_args(["--max-retries", "2"])
        self.payload = deepseek.build_payload(self.args, "data:image/jpeg;base64,cGl4ZWxz")

    @staticmethod
    def http_error(status, headers=None):
        return urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions", status, f"private-error-{API_KEY}",
            headers or {}, io.BytesIO(f"private-response-{API_KEY}".encode()),
        )

    def test_unauthorized_is_not_retried_and_error_excludes_response_and_key(self):
        with patch.object(deepseek.urllib.request, "urlopen", side_effect=self.http_error(401)) as urlopen, \
                patch.object(deepseek.time, "sleep") as sleep:
            with self.assertRaises(deepseek.APIError) as caught:
                deepseek.request_completion(self.args, API_KEY, self.payload)
        self.assertIn("HTTP 401", str(caught.exception))
        self.assertNotIn(API_KEY, str(caught.exception))
        self.assertNotIn("private-response", str(caught.exception))
        urlopen.assert_called_once()
        sleep.assert_not_called()

    def test_rate_limit_and_server_failure_retry_then_return_success(self):
        expected = completion(formatted_answer("Anger"))
        with patch.object(deepseek.urllib.request, "urlopen", side_effect=[
            self.http_error(429, {"Retry-After": "5.5"}), self.http_error(503), json_response(expected),
        ]) as urlopen, patch.object(deepseek.time, "sleep") as sleep, redirect_stderr(io.StringIO()):
            actual = deepseek.request_completion(self.args, API_KEY, self.payload)
        self.assertEqual(actual, expected)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5.5, 2])

    def test_incomplete_response_body_is_retried_without_logging_partial_content(self):
        expected = completion(formatted_answer("Anger"))
        interrupted = MagicMock()
        interrupted.__enter__.return_value = interrupted
        interrupted.read.side_effect = IncompleteRead(f"private-response-{API_KEY}".encode(), 100)
        stderr = io.StringIO()
        with patch.object(deepseek.urllib.request, "urlopen", side_effect=[
            interrupted, json_response(expected),
        ]) as urlopen, patch.object(deepseek.time, "sleep") as sleep, redirect_stderr(stderr):
            actual = deepseek.request_completion(self.args, API_KEY, self.payload)
        self.assertEqual(actual, expected)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(1)
        self.assertIn("IncompleteRead", stderr.getvalue())
        self.assertNotIn(API_KEY, stderr.getvalue())
        self.assertNotIn("private-response", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
