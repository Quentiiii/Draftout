import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from emotion_eval.data import LABELS, load_emo8, load_emotion6, load_emoset


class DatasetLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, relative, content=""):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def emoset(self, split_label="anger", annotation_label="anger"):
        # Deliberately misleading filename: only the annotation is authoritative.
        self.write("image/sadness/sadness_1.jpg")
        self.write("annotation/sample.json", json.dumps({"emotion": annotation_label}))
        row = [split_label, "image/sadness/sadness_1.jpg", "annotation/sample.json"]
        self.write("test.json", json.dumps([row]))
        return row

    def test_emoset_uses_annotation_not_filename_and_only_test(self):
        self.emoset()
        self.write("train.json", "invalid json that must not be read")
        sample, = load_emoset(self.root)
        self.assertEqual(sample.label, "Anger")
        self.assertEqual(sample.sample_id, "image/sadness/sadness_1.jpg")
        self.assertEqual(sample.image_path, self.root / "image/sadness/sadness_1.jpg")
        self.assertEqual(sample.dataset, "emoset")
        with self.assertRaises(FrozenInstanceError):
            sample.label = "sadness"

    def test_emoset_matches_annotation_and_split_regardless_of_case(self):
        for split_label, annotation_label in (("awe", "awe"), ("awe", "Awe"), (" AWE ", "aWe")):
            with self.subTest(split_label=split_label, annotation_label=annotation_label):
                self.emoset(split_label=split_label, annotation_label=annotation_label)
                sample, = load_emoset(self.root)
                self.assertEqual(sample.label, "Awe")
                self.assertEqual(sample.sample_id, "image/sadness/sadness_1.jpg")
                self.assertEqual(sample.image_path, self.root / "image/sadness/sadness_1.jpg")

    def test_emoset_rejects_annotation_mismatch(self):
        self.emoset(annotation_label="fear")
        with self.assertRaisesRegex(ValueError, "label mismatch"):
            load_emoset(self.root)

    def test_emoset_rejects_duplicate_image(self):
        row = self.emoset()
        self.write("test.json", json.dumps([row, row]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_emoset(self.root)

    def test_emoset_rejects_unknown_or_missing_annotation_label(self):
        for label in ("joy", None, ["anger"]):
            with self.subTest(label=label):
                self.emoset(annotation_label=label)
                with self.assertRaisesRegex(ValueError, "unknown or missing"):
                    load_emoset(self.root)

    def test_emoset_rejects_missing_annotation(self):
        self.emoset()
        (self.root / "annotation/sample.json").unlink()
        with self.assertRaises(FileNotFoundError):
            load_emoset(self.root)

    def test_emoset_rejects_malformed_rows(self):
        for rows in ({}, [], [["anger"]], [[None, "x.jpg", "x.json"]]):
            with self.subTest(rows=rows):
                self.write("test.json", json.dumps(rows))
                with self.assertRaises(ValueError):
                    load_emoset(self.root)

    def test_emotion6_uses_parent_directory_and_ignores_ground_truth(self):
        self.write("Emotion6/images/joy/1.jpg")
        self.write("Emotion6/images/anger/1.jpg")
        self.write("Emotion6/ground_truth.txt", "this must not be parsed")
        samples = load_emotion6(self.root)
        self.assertEqual([(s.sample_id, s.label) for s in samples], [
            ("anger/1.jpg", "Anger"), ("joy/1.jpg", "Joy"),
        ])
        self.assertEqual([s.image_path for s in samples], [
            self.root / "Emotion6/images/anger/1.jpg", self.root / "Emotion6/images/joy/1.jpg",
        ])
        self.assertEqual(len(LABELS["emotion6"]), 6)

    def test_emotion6_list_filters_and_preserves_order(self):
        self.write("images/anger/1.jpg")
        self.write("images/joy/2.jpg")
        self.write("images/fear/3.jpg")
        test_list = self.write("test.txt", "joy/2.jpg\n\nanger/1.jpg\n")
        samples = load_emotion6(self.root, test_list)
        self.assertEqual([s.sample_id for s in samples], ["joy/2.jpg", "anger/1.jpg"])
        self.assertEqual([s.label for s in samples], ["Joy", "Anger"])
        self.assertEqual([s.image_path for s in samples], [
            self.root / "images/joy/2.jpg", self.root / "images/anger/1.jpg",
        ])

    def test_emotion6_list_rejects_duplicates(self):
        self.write("images/joy/1.jpg")
        test_list = self.write("test.txt", "joy/1.jpg\njoy/1.jpg\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_emotion6(self.root, test_list)

    def test_emotion6_list_rejects_invalid_entries(self):
        self.write("images/joy/1.jpg")
        for entry in ("neutral/1.jpg", "../1.jpg", "/joy/1.jpg", "joy/../1.jpg", "joy\\1.jpg"):
            with self.subTest(entry=entry):
                test_list = self.write("test.txt", entry)
                with self.assertRaises(ValueError):
                    load_emotion6(self.root, test_list)

    def test_emotion6_list_rejects_missing_image(self):
        self.write("images/joy/1.jpg")
        test_list = self.write("test.txt", "joy/99.jpg\n")
        with self.assertRaises(FileNotFoundError):
            load_emotion6(self.root, test_list)

    def test_path_cannot_escape_dataset_through_symlink(self):
        self.write("images/joy/1.jpg")
        outside = self.write("outside.jpg")
        (self.root / "images/joy/2.jpg").symlink_to(outside)
        test_list = self.write("test.txt", "joy/2.jpg")
        with self.assertRaisesRegex(ValueError, "escapes"):
            load_emotion6(self.root, test_list)

    def test_emo8_loads_all_splits_and_normalizes_sad_without_changing_paths(self):
        self.write("train/AWE/anger.JPG")
        self.write("val/anger/awe.jpeg")
        self.write("test/sAd/fear.webp")
        samples = load_emo8(self.root)
        self.assertEqual([(s.sample_id, s.label) for s in samples], [
            ("train/AWE/anger.JPG", "Awe"),
            ("val/anger/awe.jpeg", "Anger"),
            ("test/sAd/fear.webp", "Sadness"),
        ])
        self.assertTrue(all(s.dataset == "emo8" for s in samples))
        self.assertEqual([s.image_path for s in samples], [
            self.root / "train/AWE/anger.JPG",
            self.root / "val/anger/awe.jpeg",
            self.root / "test/sAd/fear.webp",
        ])
        self.assertEqual(set(LABELS["emo8"]), set(LABELS["emoset"]))

    def test_emo8_keeps_same_filename_in_different_splits(self):
        paths = [f"{split}/anger/shared.jpg" for split in ("train", "val", "test")]
        for path in paths:
            self.write(path)
        samples = load_emo8(self.root)
        self.assertEqual([s.sample_id for s in samples], paths)
        self.assertEqual(len({s.image_path for s in samples}), 3)

    def test_emo8_rejects_unknown_class(self):
        for split in ("train", "val", "test"):
            self.write(f"{split}/anger/1.jpg")
        self.write("val/neutral/2.jpg")
        with self.assertRaisesRegex(ValueError, "Unknown EMO8 class"):
            load_emo8(self.root)

    def test_emo8_requires_all_three_splits(self):
        self.write("train/anger/1.jpg")
        self.write("test/anger/1.jpg")
        with self.assertRaisesRegex(FileNotFoundError, "Missing EMO8 split directory.*val"):
            load_emo8(self.root)

    def test_emo8_rejects_empty_split(self):
        self.write("train/anger/1.jpg")
        self.write("test/anger/1.jpg")
        (self.root / "val").mkdir()
        with self.assertRaisesRegex(ValueError, "No EMO8 evaluation images.*val"):
            load_emo8(self.root)

    def test_emo8_rejects_duplicate_resolved_image(self):
        original = self.write("train/anger/1.jpg")
        duplicate = self.write("val/anger/2.jpg")
        duplicate.unlink()
        duplicate.symlink_to(original)
        self.write("test/anger/3.jpg")
        with self.assertRaisesRegex(ValueError, "duplicate image"):
            load_emo8(self.root)

    def test_emo8_rejects_image_symlink_outside_dataset(self):
        for split in ("train", "val", "test"):
            self.write(f"EMO8/{split}/anger/1.jpg")
        outside = self.write("outside.jpg")
        (self.root / "EMO8/train/anger/2.jpg").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "escapes dataset root"):
            load_emo8(self.root / "EMO8")


if __name__ == "__main__":
    unittest.main()
