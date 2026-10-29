"""Offline regressions for the planning, analysis, and boxed-answer prompt."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from emotion_eval.data import LABELS, Sample
from emotion_eval.infer import argument_parser, main
from emotion_eval.metrics import parse_angles_prediction, parse_angles_response
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT, EMOTION_ANALYSIS_ANGLES_PROMPT


def angles_answer(label="Awe", draft="Consider the scene's scale and lighting.",
                  analysis="The vast mountain and dramatic light convey awe."):
    return f"<draft>\n{draft}\n</draft>\n{analysis}\n\\boxed{{{label}}}"


class AnglesParserTests(unittest.TestCase):
    def test_plan_and_external_analysis_are_preserved_and_only_final_label_is_scored(self):
        draft = "Consider scale.\nCompare lighting and facial expressions."
        analysis = "Dark shadows might suggest Fear.\nThe immense mountain instead conveys Awe."
        response = " \n" + angles_answer(" aWe ", draft, analysis) + "\n "
        self.assertEqual(parse_angles_response(response, LABELS["emoset"]), (draft, analysis, "Awe"))
        self.assertEqual(parse_angles_prediction(response, LABELS["emoset"]), "Awe")

    def test_invalid_label_does_not_inherit_a_label_from_plan_or_analysis(self):
        draft, analysis = "Check visual cues for Awe.", "The image conveys Awe."
        for label in ("Joy", "Awe or Fear", "I think Awe", ""):
            with self.subTest(label=label):
                response = angles_answer(label, draft, analysis)
                self.assertEqual(parse_angles_response(response, LABELS["emoset"]), (draft, analysis, None))
                self.assertIsNone(parse_angles_prediction(response, LABELS["emoset"]))

    def test_missing_empty_nested_or_extra_format_sections_are_rejected(self):
        invalid = (
            "Awe", r"\boxed{Awe}", r"A mountain conveys Awe.\boxed{Awe}",
            angles_answer(draft=""), angles_answer(draft=" \n "),
            angles_answer(analysis=""), angles_answer(analysis=" \n "),
            r"<draft>Consider scale.</draft>\boxed{Awe}",
            r"<draft>Consider scale. The mountain conveys Awe.\boxed{Awe}",
            angles_answer(draft="<draft>Consider scale.</draft>"),
            angles_answer(analysis="<draft>Another plan.</draft>The mountain conveys Awe."),
            angles_answer(draft=r"Consider scale and the example \boxed{Fear}."),
            angles_answer(analysis=r"Example \boxed{Fear}; the mountain conveys Awe."),
            angles_answer() + r"\boxed{Fear}",
            angles_answer()[:-1],
            "Here is my answer.\n" + angles_answer(),
            angles_answer() + "\nThat is my answer.",
        )
        for response in invalid:
            with self.subTest(response=response):
                self.assertEqual(parse_angles_response(response, LABELS["emoset"]), (None, None, None))
                self.assertIsNone(parse_angles_prediction(response, LABELS["emoset"]))


class AnglesInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "results"
        self.samples = [Sample("emoset", "awe/1.jpg", self.root / "awe/1.jpg", "Awe")]
        self.argv = ["--datasets", "emoset", "--model", str(self.root), "--output-dir", str(self.output)]
        self.predictions = self.output / "emoset/predictions.jsonl"

    def run_main(self, *extra):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return main([*self.argv, *extra])

    def test_cli_keeps_draft_default_and_accepts_angles(self):
        self.assertEqual(argument_parser().parse_args([]).prompt_style, "draft")
        self.assertEqual(argument_parser().parse_args(["--prompt-style", "angles"]).prompt_style, "angles")

    def test_default_inference_still_uses_original_draft_prompt_and_format(self):
        response = "<draft>The immense mountain conveys awe.</draft>\n\\boxed{Awe}"
        with patch("emotion_eval.infer.load_emoset", return_value=self.samples), \
                patch("emotion_eval.infer.QwenPredictor") as predictor:
            predictor.return_value.predict.return_value = [response]
            self.assertEqual(self.run_main(), 0)
        predictor.return_value.predict.assert_called_once_with(
            self.samples, prompt=DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(LABELS["emoset"]))
        )
        record = json.loads(self.predictions.read_text())
        self.assertEqual(record["prediction"], "Awe")
        self.assertEqual(record["draft"], "The immense mountain conveys awe.")
        self.assertNotIn("analysis", record)
        config = json.loads((self.output / "emoset/config.json").read_text())
        self.assertEqual(config["response_format"], "draft_then_boxed_label")
        self.assertIn(config["format_version"], (2, 3))
        self.assertNotIn("prompt_style", config)

    def test_angles_inference_saves_analysis_resumes_and_rejects_prompt_change(self):
        draft = "Consider the scene's scale and lighting."
        analysis = "Fear is less apt than awe because of the mountain's immense scale."
        response = angles_answer("Awe", draft, analysis)
        expected_prompt = EMOTION_ANALYSIS_ANGLES_PROMPT.format(labels=", ".join(LABELS["emoset"]))
        with patch("emotion_eval.infer.load_emoset", return_value=self.samples), \
                patch("emotion_eval.infer.QwenPredictor") as predictor:
            predictor.return_value.predict.return_value = [response]
            self.assertEqual(self.run_main("--prompt-style", "angles"), 0)
            predictor.return_value.predict.assert_called_once_with(self.samples, prompt=expected_prompt)
            record = json.loads(self.predictions.read_text())
            self.assertEqual((record["draft"], record["analysis"], record["prediction"]),
                             (draft, analysis, "Awe"))
            self.assertEqual(record["raw_output"], response)
            self.assertTrue(record["correct"])
            config = json.loads((self.output / "emoset/config.json").read_text())
            self.assertEqual(config["prompt"], expected_prompt)
            self.assertEqual(config["prompt_style"], "angles")
            self.assertEqual(config["format_version"], 4)
            self.assertEqual(config["response_format"], "angles_then_analysis_then_boxed_label")

            original = self.predictions.read_bytes()
            predictor.reset_mock()
            self.assertEqual(self.run_main("--prompt-style", "angles", "--resume"), 0)
            predictor.assert_not_called()
            self.assertEqual(self.predictions.read_bytes(), original)
            self.assertEqual(json.loads((self.output / "emoset/metrics.json").read_text())["accuracy"], 1.0)
            before_change = {path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()}
            with self.assertRaisesRegex(ValueError, "changed settings/data"):
                self.run_main("--prompt-style", "draft", "--resume")
            predictor.assert_not_called()
            self.assertEqual({path: path.read_bytes() for path in self.output.rglob("*") if path.is_file()},
                             before_change)


if __name__ == "__main__":
    unittest.main()
