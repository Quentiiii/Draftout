import unittest

from emotion_eval.data import LABELS
from emotion_grpo.reward import compute_score


class EmotionGRPORewardTests(unittest.TestCase):
    def test_all_eight_classes_and_case_insensitive_ground_truth(self):
        for label in LABELS["emoset"]:
            with self.subTest(label=label):
                self.assertEqual(compute_score("emoset", label.upper(), f" {label.lower()} "), 1.0)
                wrong = "Anger" if label != "Anger" else "Fear"
                self.assertEqual(compute_score("emoset", wrong, label), 0.0)

    def test_existing_answer_formats_receive_identical_correctness_reward(self):
        for answer in (
            "Amusement", " aMuSeMeNt.\n", "**Amusement**", "`Amusement`",
            "<answer> Amusement </answer>",
            "<ANSWER>Amusement</ANSWER>",
            "<think>Maybe awe, but the scene is funny.</think>\nAmusement",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(compute_score("emoset", answer, "Amusement"), 1.0)
                self.assertEqual(compute_score("emoset", answer, "Awe"), 0.0)

    def test_invalid_empty_and_multi_class_answers_receive_zero(self):
        for answer in (
            "", " \n", None, 1, "Joy", "happy", "not Amusement", "I think Amusement",
            "Amusement or Awe", "Amusement, Awe", "Amusement\nAwe",
            "<answer>Amusement or Awe</answer>",
            "<answer>Amusement</answer><answer>Awe</answer>",
            "<think>Amusement", "<think>Amusement</think>",
            ", ".join(LABELS["emoset"]),
        ):
            with self.subTest(answer=answer):
                self.assertEqual(compute_score("emoset", answer, "Amusement"), 0.0)

    def test_bad_ground_truth_raises_instead_of_silently_training(self):
        for truth in ("", None, 0, "Joy", "<answer>Anger</answer>", "Anger or Fear"):
            with self.subTest(truth=truth):
                with self.assertRaisesRegex(ValueError, "ground-truth"):
                    compute_score("emoset", "Anger", truth)

    def test_verl_additional_context_does_not_change_the_reward(self):
        self.assertEqual(
            compute_score(
                data_source="emoset",
                solution_str="Fear",
                ground_truth="fear",
                extra_info={"split": "train", "label": "Anger"},
                reward_router_address=None,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
