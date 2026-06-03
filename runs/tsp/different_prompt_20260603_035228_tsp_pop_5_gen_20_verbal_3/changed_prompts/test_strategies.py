import unittest

from dynagen.evolution.strategies import (
    STRATEGIES,
    STRATEGIES_METADATA,
    Strategy,
    parent_count,
)


class StrategyMetadataTests(unittest.TestCase):
    def test_parent_counts_are_fixed(self) -> None:
        expected = {
            Strategy.E1_DIVERGENT_EXPLORATION: 3,
            Strategy.E2_PATTERN_GUIDED_RECOMBINATION: 3,
            Strategy.M1_CORE_LOGIC_MUTATION: 1,
            Strategy.M2_PARAMETER_TUNING: 1,
            Strategy.M3_SIMPLIFICATION: 1,
        }

        for strategy, count in expected.items():
            with self.subTest(strategy=strategy):
                self.assertEqual(STRATEGIES_METADATA[strategy]["parent_count"], count)
                self.assertEqual(parent_count(strategy), count)

    def test_strategy_descriptions_are_present(self) -> None:
        
        expected_descriptions = {
            "e1_divergent_exploration": (
                "Generate a heuristic that is as different as possible from every provided "
                "example, deliberately avoiding their structural and conceptual patterns. "
                "Explore entirely new algorithmic ideas, alternative decision criteria, or "
                "unconventional control flow rather than refining what already exists. Treat the "
                "provided parents only as a list of approaches to avoid duplicating, anchoring on "
                "none of them. Prioritize maximal diversity even at the risk of lower immediate "
                "performance, so the search can escape local optima."
            ),
            "e2_pattern_guided_recombination": (
                "Examine the provided top-performing heuristics and identify the common "
                "structural motifs, design patterns, and decision rules they share. Build a new "
                "heuristic that retains these proven components as a foundation while recombining "
                "them coherently. Then introduce at least one genuinely new mechanism or "
                "sub-strategy that none of the parents contain. The result should be recognizably "
                "grounded in the elite solutions yet extended beyond a simple blend of them."
            ),
            "m1_core_logic_mutation": (
                "Take the assigned parent and modify its core decision logic to improve "
                "performance. Change how the central choice is made — the scoring function, the "
                "selection rule, or the main control structure — rather than merely adjusting "
                "numeric values. Preserve the parent's interface and intent so it remains a valid "
                "drop-in replacement. Aim for a meaningful behavioral change that could plausibly "
                "outperform the parent on the target objective."
            ),
            "m2_parameter_tuning": (
                "Take the assigned parent and adjust its numeric parameters, thresholds, or "
                "weights without altering the underlying logic. Keep the algorithmic structure, "
                "control flow, and decision rules exactly as they appear in the parent. Reason "
                "about which constants most influence performance and shift them in a direction "
                "likely to yield gains. The output must be structurally identical to the parent "
                "and differ only in its tuned values."
            ),
            "m3_simplification": (
                "Take the assigned parent and remove components that are redundant, rarely "
                "triggered, or expensive relative to their benefit. Streamline the logic so the "
                "heuristic is shorter and faster while preserving — or improving — its solution "
                "quality. Eliminate dead branches, unnecessary state, and over-engineered steps "
                "that do not contribute to the objective. The result should be a leaner version "
                "of the parent that is cheaper to evaluate and less prone to overfitting."
            ),
        }


        for name, metadata in STRATEGIES.items():
            with self.subTest(strategy=name):
                self.assertEqual(metadata["description"], expected_descriptions[name])


if __name__ == "__main__":
    unittest.main()
