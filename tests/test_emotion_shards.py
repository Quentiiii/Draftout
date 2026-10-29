import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from emotion_eval.data import LABELS, Sample
from emotion_eval.infer import argument_parser, data_fingerprint, prepare_run
from emotion_eval.metrics import parse_angles_response, parse_draft_response
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT, EMOTION_ANALYSIS_ANGLES_PROMPT
from scripts import aggregate_emotion_shards as merger


class AggregateShardTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.datasets = {}
        for dataset, labels in LABELS.items():
            samples = [Sample(dataset, f"image/{i}.jpg", self.root / dataset / f"{i}.jpg", labels[i % 2])
                       for i in range(5)]
            self.datasets[dataset] = samples
            for index in range(2):
                args = argument_parser().parse_args([
                    "--output-dir", str(self.root / f"shard_{index}"),
                    "--num-shards", "2", "--shard-index", str(index), "--device", f"cuda:{index}",
                ])
                selected = samples[index::2]
                prompt = DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(labels))
                run = prepare_run(args, dataset, selected, merger.SPLITS[dataset], prompt)
                records = []
                for sample in reversed(selected):
                    raw = ("unparseable" if sample.sample_id == "image/1.jpg" else
                           f"<draft>Visible evidence.</draft>\\boxed{{{sample.label}}}")
                    draft, prediction = parse_draft_response(raw, labels)
                    records.append({"dataset": dataset, "sample_id": sample.sample_id,
                                    "image_path": str(sample.image_path), "label": sample.label,
                                    "raw_output": raw, "draft": draft, "prediction": prediction,
                                    "correct": prediction == sample.label})
                self.write_records(run["directory"] / "predictions.jsonl", records)
        for loader, dataset in (("load_emoset", "emoset"), ("load_emotion6", "emotion6"), ("load_emo8", "emo8")):
            mocked = patch.object(merger, loader, return_value=self.datasets[dataset])
            mocked.start()
            self.addCleanup(mocked.stop)

    @staticmethod
    def write_records(path, records):
        path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    def assert_no_aggregates(self):
        self.assertFalse((self.root / "summary.json").exists())
        for dataset in LABELS:
            self.assertFalse((self.root / dataset).exists())

    def use_sampling_configs(self):
        """Create version-3 fixture configs through the real inference entry point."""
        for dataset, samples in self.datasets.items():
            prompt = DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(LABELS[dataset]))
            for index in range(2):
                directory = self.root / f"shard_{index}" / dataset
                predictions = (directory / "predictions.jsonl").read_bytes()
                args = argument_parser().parse_args([
                    "--output-dir", str(directory.parent), "--num-shards", "2",
                    "--shard-index", str(index), "--device", f"cuda:{index}",
                    "--do-sample", "--temperature", "0.7", "--top-p", "0.8",
                    "--top-k", "20", "--min-p", "0", "--presence-penalty", "1.5",
                    "--repetition-penalty", "1", "--seed", "42", "--batch-size", "2",
                ])
                prepare_run(args, dataset, samples[index::2], merger.SPLITS[dataset], prompt)
                (directory / "predictions.jsonl").write_bytes(predictions)

    def use_angles_configs(self, sampled=True):
        """Produce actual version-4 configs with valid and missing-analysis outputs."""
        for dataset, samples in self.datasets.items():
            labels = LABELS[dataset]
            prompt = EMOTION_ANALYSIS_ANGLES_PROMPT.format(labels=", ".join(labels))
            for index in range(2):
                arguments = [
                    "--output-dir", str(self.root / f"shard_{index}"),
                    "--num-shards", "2", "--shard-index", str(index),
                    "--device", f"cuda:{index}", "--prompt-style", "angles",
                    "--batch-size", "32", "--max-new-tokens", "1024",
                ]
                if sampled:
                    arguments += ["--do-sample", "--presence-penalty", "1.5", "--seed", "42"]
                args = argument_parser().parse_args(arguments)
                selected = samples[index::2]
                run = prepare_run(args, dataset, selected, merger.SPLITS[dataset], prompt)
                records = []
                for sample in reversed(selected):
                    raw = "<draft>1. Inspect expressions.\n2. Examine the setting.</draft>\n"
                    if sample.sample_id != "image/1.jpg":
                        raw += "Visible cues support the selected emotion.\n"
                    raw += f"\\boxed{{{sample.label}}}"
                    draft, analysis, prediction = parse_angles_response(raw, labels)
                    records.append({
                        "dataset": dataset, "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path), "label": sample.label,
                        "raw_output": raw, "draft": draft, "analysis": analysis,
                        "prediction": prediction, "correct": prediction == sample.label,
                    })
                self.write_records(run["directory"] / "predictions.jsonl", records)

    def test_complete_uneven_shards_restore_order_and_keep_invalid_denominator(self):
        originals = {path: path.read_bytes() for path in self.root.glob("shard_*/*/*")}
        summary = merger.aggregate(self.root)
        for dataset, samples in self.datasets.items():
            records = [json.loads(line) for line in (self.root / dataset / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([record["sample_id"] for record in records], [sample.sample_id for sample in samples])
            self.assertEqual(summary[dataset]["num_samples"], 5)
            self.assertEqual(summary[dataset]["num_invalid"], 1)
            self.assertEqual(summary[dataset]["accuracy"], 0.8)
            self.assertTrue(summary[dataset]["complete"])
            config = json.loads((self.root / dataset / "config.json").read_text())
            self.assertEqual(config["samples_sha256"], data_fingerprint(samples))
            self.assertEqual(config["devices"], ["cuda:0", "cuda:1"])
            self.assertEqual(len(config["shards_provenance"]), 2)
        for path, payload in originals.items():
            self.assertEqual(path.read_bytes(), payload)

    def test_sampling_shards_preserve_decoding_settings_and_provenance(self):
        self.use_sampling_configs()
        originals = {path: path.read_bytes() for path in self.root.glob("shard_*/*/*")}
        expected = {
            "format_version": 3, "do_sample": True, "temperature": 0.7, "top_p": 0.8,
            "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "seed": 42, "batch_size": 2,
            "seed_batch_size": 2, "seed_policy": "sha256_batch_sample_ids_v1",
            "presence_penalty_scope": "generated_tokens_only",
        }

        summary = merger.aggregate(self.root)

        for dataset, samples in self.datasets.items():
            config = json.loads((self.root / dataset / "config.json").read_text())
            self.assertTrue(config["aggregation_complete"])
            self.assertEqual(config["samples_sha256"], data_fingerprint(samples))
            self.assertEqual(config["devices"], ["cuda:0", "cuda:1"])
            self.assertEqual({key: config[key] for key in expected}, expected)
            self.assertEqual(len(config["shards_provenance"]), 2)
            for shard in config["shards_provenance"]:
                self.assertEqual({key: shard["config"][key] for key in expected}, expected)
                self.assertEqual(shard["config"]["shard_index"], shard["shard_index"])
            records = [json.loads(line) for line in
                       (self.root / dataset / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in records], [s.sample_id for s in samples])
            self.assertEqual(summary[dataset]["num_samples"], 5)
            self.assertEqual(summary[dataset]["num_invalid"], 1)
            self.assertEqual(summary[dataset]["accuracy"], 0.8)
        for path, payload in originals.items():
            self.assertEqual(path.read_bytes(), payload)

    def test_sampling_shards_reject_different_decoding_settings_or_seed(self):
        self.use_sampling_configs()
        path = self.root / "shard_1/emoset/config.json"
        original = json.loads(path.read_text())
        changes = (
            {"temperature": 0.9}, {"top_p": 0.9}, {"top_k": 40}, {"min_p": 0.05},
            {"presence_penalty": 0.5}, {"repetition_penalty": 1.1}, {"seed": 43},
            {"do_sample": False}, {"batch_size": 4, "seed_batch_size": 4},
        )
        for change in changes:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaisesRegex(ValueError, "Incompatible shard settings"):
                    merger.aggregate(self.root)
                self.assert_no_aggregates()
        path.write_text(json.dumps(original))

    def test_sampling_shards_reject_incomplete_or_invalid_metadata(self):
        self.use_sampling_configs()
        path = self.root / "shard_1/emoset/config.json"
        original = json.loads(path.read_text())
        malformed = []
        for key in ("temperature", "presence_penalty", "seed", "seed_policy", "presence_penalty_scope"):
            missing = dict(original)
            del missing[key]
            malformed.append((f"missing {key}", missing, "Missing sampling configuration"))
        malformed.extend((
            ("seed batch", {**original, "seed_batch_size": 4}, "Sampling batch size mismatch"),
            ("seed policy", {**original, "seed_policy": "global_rng"}, "Unsupported sampling seed policy"),
            ("penalty scope", {**original, "presence_penalty_scope": "prompt_and_generated"},
             "Unsupported sampling seed policy or presence penalty scope"),
            ("legacy version", {**original, "format_version": 2}, "Invalid legacy greedy configuration"),
        ))
        for description, config, error in malformed:
            with self.subTest(case=description):
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, error):
                    merger.aggregate(self.root)
                self.assert_no_aggregates()
        path.write_text(json.dumps(original))

    def test_angles_sampling_shards_preserve_outline_analysis_and_metadata(self):
        self.use_angles_configs()
        originals = {path: path.read_bytes() for path in self.root.glob("shard_*/*/*")}

        summary = merger.aggregate(self.root)

        for dataset, samples in self.datasets.items():
            config = json.loads((self.root / dataset / "config.json").read_text())
            self.assertEqual(config["format_version"], 4)
            self.assertEqual(config["prompt_style"], "angles")
            self.assertEqual(config["response_format"], "angles_then_analysis_then_boxed_label")
            self.assertEqual(config["prompt"], EMOTION_ANALYSIS_ANGLES_PROMPT.format(
                labels=", ".join(LABELS[dataset]),
            ))
            self.assertTrue(config["do_sample"])
            self.assertEqual(config["presence_penalty"], 1.5)
            self.assertEqual(config["seed"], 42)
            self.assertEqual(config["seed_batch_size"], 32)
            self.assertEqual(config["max_new_tokens"], 1024)
            self.assertEqual(config["samples_sha256"], data_fingerprint(samples))
            self.assertEqual(len(config["shards_provenance"]), 2)
            self.assertTrue(all(shard["config"]["prompt_style"] == "angles"
                                for shard in config["shards_provenance"]))
            records = [json.loads(line) for line in
                       (self.root / dataset / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual([row["sample_id"] for row in records], [sample.sample_id for sample in samples])
            for row in records:
                self.assertEqual((row["draft"], row["analysis"], row["prediction"]),
                                 parse_angles_response(row["raw_output"], LABELS[dataset]))
                if row["prediction"] is not None:
                    # The former draft-only parser must not score these valid answers.
                    self.assertIsNone(parse_draft_response(row["raw_output"], LABELS[dataset])[1])
                    self.assertEqual(row["analysis"], "Visible cues support the selected emotion.")
            self.assertEqual(summary[dataset]["num_samples"], 5)
            self.assertEqual(summary[dataset]["num_invalid"], 1)
            self.assertEqual(summary[dataset]["accuracy"], 0.8)
        for path, payload in originals.items():
            self.assertEqual(path.read_bytes(), payload)

    def test_angles_greedy_shards_do_not_require_sampling_metadata(self):
        self.use_angles_configs(sampled=False)

        summary = merger.aggregate(self.root)

        for dataset in self.datasets:
            config = json.loads((self.root / dataset / "config.json").read_text())
            self.assertEqual(config["format_version"], 4)
            self.assertFalse(config["do_sample"])
            self.assertNotIn("seed_policy", config)
            self.assertNotIn("presence_penalty", config)
            self.assertEqual(summary[dataset]["num_correct"], 4)

    def test_angles_shards_reject_wrong_style_template_or_sampling_settings(self):
        self.use_angles_configs()
        path = self.root / "shard_1/emoset/config.json"
        original = json.loads(path.read_text())
        malformed = [
            ({**original, "prompt_style": "draft"}, "configuration"),
            ({**original, "response_format": "draft_then_boxed_label"}, "configuration"),
            ({**original, "prompt": DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(LABELS["emoset"]))},
             "configuration"),
            ({**original, "format_version": 3}, "legacy prompt style configuration"),
            ({**original, "seed": 43}, "Incompatible shard settings"),
            ({**original, "seed_batch_size": 16}, "Sampling batch size mismatch"),
        ]
        for key, message in (("prompt_style", "configuration"), ("presence_penalty", "Missing decoding settings")):
            missing = dict(original)
            del missing[key]
            malformed.append((missing, message))
        for config, message in malformed:
            with self.subTest(config=config):
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, message):
                    merger.aggregate(self.root)
                self.assert_no_aggregates()
        path.write_text(json.dumps(original))

    def test_angles_shards_reject_missing_or_tampered_parsed_fields(self):
        self.use_angles_configs()
        path = self.root / "shard_0/emoset/predictions.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        malformed = [
            ({**original[0], "draft": "Changed outline"}, "explanation"),
            ({**original[0], "analysis": "Changed reasoning"}, "analysis"),
            ({**original[0], "prediction": None}, "current data/parser"),
            ({**original[0], "raw_output": "<draft>Outline only.</draft>\\boxed{Amusement}"},
             "current data/parser"),
        ]
        for field, message in (("draft", "explanation"), ("analysis", "analysis")):
            missing = dict(original[0])
            del missing[field]
            malformed.append((missing, message))
        for record, message in malformed:
            with self.subTest(record=record):
                self.write_records(path, [record, *original[1:]])
                payload = path.read_bytes()
                with self.assertRaisesRegex(ValueError, message):
                    merger.aggregate(self.root)
                self.assertEqual(path.read_bytes(), payload)
                self.assert_no_aggregates()
        self.write_records(path, original)

    def test_incomplete_last_dataset_cannot_publish_partial_aggregates(self):
        path = self.root / "shard_1/emo8/predictions.jsonl"
        path.write_text(path.read_text().splitlines()[0] + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete shard"):
            merger.aggregate(self.root)
        self.assert_no_aggregates()

    def test_bad_core_settings_prompt_index_and_fingerprint_rejected(self):
        path = self.root / "shard_1/emoset/config.json"
        original = json.loads(path.read_text())
        for key, value in (("max_new_tokens", 16), ("prompt", "Changed"), ("shard_index", 0),
                           ("samples_sha256", "incorrect")):
            with self.subTest(key=key):
                path.write_text(json.dumps({**original, key: value}))
                with self.assertRaisesRegex(ValueError, "configuration|settings"):
                    merger.aggregate(self.root)
                self.assert_no_aggregates()
        path.write_text(json.dumps(original))

    def test_duplicate_and_tampered_explanation_rejected(self):
        path = self.root / "shard_0/emoset/predictions.jsonl"
        original = [json.loads(line) for line in path.read_text().splitlines()]
        for records in ([*original, original[0]], [{**original[0], "draft": "Changed"}, *original[1:]]):
            with self.subTest(records=records):
                self.write_records(path, records)
                with self.assertRaisesRegex(ValueError, "duplicate|explanation"):
                    merger.aggregate(self.root)
                self.assert_no_aggregates()

    def test_partial_final_record_rejected_without_repairing_source(self):
        path = self.root / "shard_0/emoset/predictions.jsonl"
        payload = path.read_bytes() + b'{"sample_id": "unfinished'
        path.write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "incomplete JSONL"):
            merger.aggregate(self.root)
        self.assertEqual(path.read_bytes(), payload)
        self.assert_no_aggregates()


if __name__ == "__main__":
    unittest.main()
