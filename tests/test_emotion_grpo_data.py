import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from emotion_grpo.prepare_data import (
    ARTIFACT_NAMES,
    EMOSET_LABELS,
    PROMPT,
    SplitEntry,
    check_disjoint,
    make_rows,
    prepare_dataset,
    read_split,
    select_entries,
    sha256_file,
)


class GrpoDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "dataset"
        self.output = Path(self.temp.name) / "prepared"
        self.root.mkdir()

    def write_split(self, split, per_class=3):
        rows = []
        for label in EMOSET_LABELS:
            for index in range(per_class):
                # Filenames intentionally imply a different answer.
                image = f"image/sadness/{split}_{label.lower()}_{index}.jpg"
                annotation = f"annotation/{split}_{label.lower()}_{index}.json"
                for relative, content in (
                    (image, f"unique image bytes: {split} {label} {index}"),
                    (annotation, json.dumps({"emotion": label.lower()})),
                ):
                    path = self.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                rows.append([label.lower(), image, annotation])
        (self.root / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        return rows

    def test_balanced_sampling_is_deterministic_and_source_order_independent(self):
        self.write_split("train", per_class=8)
        entries = read_split(self.root, "train")
        first = select_entries(entries, 16, 42)
        self.assertEqual(Counter(entry.label for entry in first), dict.fromkeys(EMOSET_LABELS, 2))
        self.assertEqual(first, select_entries(list(reversed(entries)), 16, 42))
        self.assertNotEqual(first, select_entries(entries, 16, 43))
        self.assertEqual(len({entry.image for entry in first}), 16)

    def test_invalid_subset_sizes_and_insufficient_classes(self):
        entries = [SplitEntry("Anger", "image/x.jpg", "annotation/x.json", 0)]
        for size in (0, -1, 7, 16):
            with self.subTest(size=size), self.assertRaises(ValueError):
                select_entries(entries, size, 42)
        self.assertEqual(len(select_entries(entries, 1, 42, "random")), 1)

    def test_overlap_with_test_is_rejected_before_sampling(self):
        self.write_split("train")
        self.write_split("val")
        train = read_split(self.root, "train")
        with self.assertRaisesRegex(ValueError, "Split overlap"):
            check_disjoint({"train": train, "val": read_split(self.root, "val"), "test": [train[-1]]})

    def test_schema_and_prompt_do_not_leak_per_example_information(self):
        self.write_split("train")
        selected = select_entries(read_split(self.root, "train"), 8, 42)
        rows, manifest = make_rows(self.root, "train", selected)
        for row, provenance in zip(rows, manifest, strict=True):
            self.assertEqual(row["prompt"], [{"role": "user", "content": PROMPT}])
            self.assertEqual(row["reward_model"], {"style": "rule", "ground_truth": provenance["ground_truth"]})
            self.assertEqual(row["images"], [{
                "image": provenance["image_path"], "min_pixels": 65536, "max_pixels": 262144,
            }])
            self.assertNotIn(provenance["image"], PROMPT)
            self.assertNotIn(provenance["image_path"], PROMPT)
            self.assertNotIn(Path(provenance["image"]).name, PROMPT)
            self.assertEqual(PROMPT.count("<image>"), 1)
            self.assertEqual(len(provenance["image_sha256"]), 64)

    def test_annotation_mismatch_missing_files_and_escaping_symlinks_rejected(self):
        self.write_split("train")
        entry = read_split(self.root, "train")[0]
        annotation = self.root / entry.annotation
        original = annotation.read_text()
        annotation.write_text(json.dumps({"emotion": "Fear"}))
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            make_rows(self.root, "train", [entry])
        annotation.write_text(original)
        image = self.root / entry.image
        image.unlink()
        with self.assertRaises(FileNotFoundError):
            make_rows(self.root, "train", [entry])
        outside = self.root.parent / "outside.jpg"
        outside.write_bytes(b"outside")
        image.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes"):
            make_rows(self.root, "train", [entry])

    def test_parquet_roundtrip_idempotence_and_overwrite_guard(self):
        import pyarrow.parquet as pq

        for split in ("train", "val", "test"):
            self.write_split(split)
        config = {"dataset_root": self.root, "output_dir": self.output, "train_size": 8, "val_size": 8}
        metadata = prepare_dataset(**config)
        self.assertEqual(metadata["subsets"]["train"]["class_counts"], dict.fromkeys(EMOSET_LABELS, 1))
        table = pq.read_table(self.output / "train.parquet").to_pylist()
        self.assertEqual(len(table), 8)
        self.assertEqual(table[0]["prompt"], [{"role": "user", "content": PROMPT}])
        hashes = {name: sha256_file(self.output / name) for name in ARTIFACT_NAMES}
        mtimes = {name: (self.output / name).stat().st_mtime_ns for name in ARTIFACT_NAMES}
        self.assertEqual(metadata, prepare_dataset(**config))
        self.assertEqual(mtimes, {name: (self.output / name).stat().st_mtime_ns for name in ARTIFACT_NAMES})
        with self.assertRaises(FileExistsError):
            prepare_dataset(**config, seed=43)
        self.assertEqual(hashes, {name: sha256_file(self.output / name) for name in ARTIFACT_NAMES})
        metadata = prepare_dataset(**config, seed=43, overwrite=True)
        self.assertEqual(metadata["config"]["seed"], 43)

    def test_identical_train_val_image_bytes_are_rejected(self):
        for split in ("train", "val", "test"):
            self.write_split(split, per_class=1)
        train = read_split(self.root, "train")[0]
        val = read_split(self.root, "val")[0]
        (self.root / val.image).write_bytes((self.root / train.image).read_bytes())
        with self.assertRaisesRegex(ValueError, "identical image contents"):
            prepare_dataset(self.root, self.output, train_size=8, val_size=8)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
