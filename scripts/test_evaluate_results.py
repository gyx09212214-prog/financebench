import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from evaluate_results import deterministic_match, is_numeric_answer, looks_like_refusal


class EvaluateResultsTest(unittest.TestCase):
    def test_numeric_answer_matches_number_in_model_answer(self):
        self.assertTrue(
            deterministic_match(
                "$1,577.00",
                "The FY2018 capital expenditure amount was $1,577 million.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_refusal_does_not_match_even_when_it_mentions_numbers(self):
        self.assertFalse(
            deterministic_match(
                "$1,577.00",
                "As an AI, I don't have access to the 2018 filing.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_explanatory_answer_is_not_treated_as_numeric_only(self):
        answer = "Operating margin decreased by 1.7% due to one-off charges."
        self.assertFalse(is_numeric_answer(answer))
        self.assertFalse(
            deterministic_match(
                answer,
                "Operating margin changed in FY2022.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_refusal_detector_is_phrase_based(self):
        self.assertTrue(looks_like_refusal("I do not have access to that filing."))
        self.assertFalse(looks_like_refusal("No, revenue did not increase."))


if __name__ == "__main__":
    unittest.main()
