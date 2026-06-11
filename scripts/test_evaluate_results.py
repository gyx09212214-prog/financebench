import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))

from evaluate_results import (
    deterministic_match,
    is_numeric_answer,
    looks_like_refusal,
    parse_args,
)


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

    def test_numeric_answer_normalizes_billion_unit_to_millions(self):
        self.assertTrue(
            deterministic_match(
                5818,
                (
                    "Lockheed Martin's FY2021 net working capital was "
                    "$19.815 billion - $13.997 billion = $5.818 billion."
                ),
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
            )
        )
        self.assertFalse(
            deterministic_match(
                5818,
                "Final answer: $5,818 billion.",
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
            )
        )

    def test_numeric_answer_preserves_billion_question_scale(self):
        self.assertTrue(
            deterministic_match(
                8.7,
                (
                    "The year end FY2018 net PP&E (Property, Plant, and "
                    "Equipment) for 3M is $8.738 billion."
                ),
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
                question=(
                    "What is the year end FY2018 net PPNE for 3M? "
                    "Answer in USD billions."
                ),
            )
        )
        self.assertTrue(
            deterministic_match(
                4.6,
                "The FY2021 capital expenditure amount for PepsiCo is $4.625 billion.",
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
                question=(
                    "What is the FY2021 capital expenditure amount "
                    "(in USD billions) for PepsiCo?"
                ),
            )
        )
        self.assertTrue(
            deterministic_match(
                4.6,
                "The FY2021 capital expenditure amount for PepsiCo is $4,625 million.",
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
                question=(
                    "What is the FY2021 capital expenditure amount "
                    "(in USD billions) for PepsiCo?"
                ),
            )
        )

    def test_numeric_answer_ignores_alphanumeric_identifiers(self):
        self.assertTrue(
            deterministic_match(
                1577,
                "For Q2 of FY2023, the capital expenditure amount for 3M is $1,577 million.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_numeric_answer_does_not_match_copied_filing_context(self):
        self.assertFalse(
            deterministic_match(
                59268,
                "[START OF FILING]\nTOTAL ASSETS\n$59,268\n[END OF FILING]",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_numeric_answer_uses_explicit_answer_span(self):
        self.assertFalse(
            deterministic_match(
                59268,
                (
                    "[START OF FILING]\nTOTAL ASSETS\n$59,268\n[END OF FILING]\n"
                    "Final answer: $55,556"
                ),
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )
        self.assertTrue(
            deterministic_match(
                59268,
                "The filing lists $55,556 for 2020. Final answer: $59,268",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_numeric_answer_uses_final_calculation_span_without_marker(self):
        self.assertTrue(
            deterministic_match(
                24.26,
                (
                    "The FY2019 revenue for Activision Blizzard is $6,489 million. "
                    "The Property, Plant, and Equipment (PP&E) for FY2018 is "
                    "$282 million and for FY2019 is $253 million. The average PP&E "
                    "between FY2018 and FY2019 is ($282 million + $253 million) / "
                    "2 = $267.5 million.\n\nTherefore, the fixed asset turnover "
                    "ratio for FY2019 is $6,489 million / $267.5 million = 24.26."
                ),
                relative_tolerance=0.001,
                absolute_tolerance=1e-6,
            )
        )

    def test_numeric_answer_requires_unambiguous_span(self):
        self.assertFalse(
            deterministic_match(
                59268,
                "The filing lists $59,268 in 2021 and $55,556 in 2020.",
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

    def test_percent_answer_preserves_percent_scale(self):
        self.assertTrue(
            deterministic_match(
                1.015,
                "The final answer is 101.4%.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )
        self.assertFalse(
            deterministic_match(
                1.015,
                "The final answer is 1.014%.",
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_default_tolerance_rejects_distinct_rounded_percentages(self):
        args = parse_args(["results.jsonl"])

        self.assertEqual(args.relative_tolerance, 0.001)
        self.assertFalse(
            deterministic_match(
                0.654,
                "The final answer is 65.2%.",
                relative_tolerance=args.relative_tolerance,
                absolute_tolerance=args.absolute_tolerance,
            )
        )

    def test_no_evidence_zero_answer_can_match_explicit_zero_gold(self):
        answer = "There is no explicit mention of this in the filing. Final answer: 0"
        self.assertTrue(looks_like_refusal(answer))
        self.assertTrue(
            deterministic_match(
                0,
                answer,
                relative_tolerance=0.01,
                absolute_tolerance=1e-6,
            )
        )

    def test_no_evidence_refusal_does_not_override_nonzero_gold(self):
        self.assertFalse(
            deterministic_match(
                1577,
                "There is no explicit mention of this in the filing. Final answer: 0",
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
