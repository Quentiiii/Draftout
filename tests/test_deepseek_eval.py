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
