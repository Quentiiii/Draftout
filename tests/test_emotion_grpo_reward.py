import unittest

from emotion_eval.data import LABELS
from emotion_grpo.reward import compute_score


def answer(label, draft="Examine the expression and scene.", analysis="The visible expression supports this emotion."):
    return f"<draft>{draft}</draft>\n{analysis}\n\\boxed{{{label}}}"


class EmotionGRPORewardTests(unittest.TestCase):
    def test_all_eight_classes_and_case_insensitive_ground_truth(self):
        for label in LABELS["emoset"]:
            with self.subTest(label=label):
                self.assertEqual(compute_score("emoset", answer(label.upper()), f" {label.lower()} "), 1.0)
                wrong = "Anger" if label != "Anger" else "Fear"
                self.assertEqual(compute_score("emoset", answer(wrong), label), 0.0)

    def test_only_final_box_controls_reward_without_explanation_bonus(self):
        for analysis in (
            "A smiling face supports amusement.",
            "Awe is another candidate, but the playful scene supports amusement.",
        ):
            with self.subTest(analysis=analysis):
                response = " \n" + answer(" aMuSeMeNt ", analysis=analysis) + "\n "
                self.assertEqual(compute_score("emoset", response, "Amusement"), 1.0)
                self.assertEqual(compute_score("emoset", response, "Awe"), 0.0)

    def test_angle_count_and_prose_quality_do_not_add_reward_terms(self):
        for draft in ("Examine the expression.", "1. Faces\n2. Light\n3. Colors\n4. Scene"):
            with self.subTest(draft=draft):
                self.assertEqual(compute_score("emoset", answer("Amusement", draft), "Amusement"), 1.0)
                self.assertEqual(compute_score("emoset", answer("Awe", draft), "Amusement"), 0.0)

    def test_invalid_empty_and_multi_class_answers_receive_zero(self):
        for response in (
            "", " \n", None, 1, "Joy", "happy", "not Amusement", "I think Amusement",
            "Amusement or Awe", "Amusement, Awe", "Amusement\nAwe",
            "Amusement", "**Amusement**", "<answer>Amusement</answer>",
            "<think>Maybe awe, but the scene is funny.</think>\nAmusement",
            "<answer>Amusement or Awe</answer>",
            "<answer>Amusement</answer><answer>Awe</answer>",
            "<think>Amusement", "<think>Amusement</think>",
            ", ".join(LABELS["emoset"]),
            answer("Amusement or Awe"), answer("happy"), answer("Amusement", ""),
            answer("Amusement", " \n "), r"\boxed{Amusement}",
            answer("Amusement", analysis=""), answer("Amusement", analysis=" \n "),
            r"<draft>The scene conveys amusement.</draft>\boxed{Amusement}",
            answer("Amusement", draft=r"Consider \boxed{Awe}."),
            answer("Amusement", analysis=r"Example \boxed{Awe}; the scene conveys amusement."),
            r"<draft>Amusement is visible.</draft>\boxed{Amusement",
            r"<draft>Amusement is visible.</draft>\boxed{Amusement}\boxed{Awe}",
            r"<draft><draft>Amusement.</draft></draft>\boxed{Amusement}",
            answer("Amusement") + " Extra text.",
        ):
            with self.subTest(response=response):
                self.assertEqual(compute_score("emoset", response, "Amusement"), 0.0)

    def test_bad_ground_truth_raises_instead_of_silently_training(self):
        for truth in ("", None, 0, "Joy", "<answer>Anger</answer>", "Anger or Fear"):
            with self.subTest(truth=truth):
                with self.assertRaisesRegex(ValueError, "ground-truth"):
                    compute_score("emoset", answer("Anger"), truth)

    def test_verl_additional_context_does_not_change_the_reward(self):
        self.assertEqual(
            compute_score(
                data_source="emoset",
                solution_str=answer("Fear"),
                ground_truth="fear",
                extra_info={"split": "train", "label": "Anger"},
                reward_router_address=None,
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
