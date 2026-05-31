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
            Strategy.E1_RADICAL_EXPLORATION: 3,
            Strategy.E2_BACKBONE_EXPLORATION: 3,
            Strategy.M1_COMPONENT_REPLACEMENT: 1,
            Strategy.M2_PARAMETER_SCHEDULE_MUTATION: 1,
            Strategy.M3_SIMPLIFY_GENERALIZE: 1,
        }

        for strategy, count in expected.items():
            with self.subTest(strategy=strategy):
                self.assertEqual(STRATEGIES_METADATA[strategy]["parent_count"], count)
                self.assertEqual(parent_count(strategy), count)

    def test_strategy_descriptions_are_present(self) -> None:
        expected_descriptions = {
            "e1_radical_exploration": (
                "Create a new algorithm from a substantially different algorithmic family than "
                "the provided parents. Avoid reusing the parents' dominant representation, "
                "search pattern, update rule, and control flow. The goal is broad exploration "
                "of the design space, not incremental improvement."
            ),
            "e2_backbone_exploration": (
                "Identify the useful shared backbone among the provided parents, then create "
                "a new algorithm that preserves that core principle while changing the surrounding "
                "mechanism, structure, or execution strategy. The result should be motivated by "
                "the parents but clearly not a direct copy."
            ),
            "m1_component_replacement": (
                "Create a modified version of one parent by replacing exactly one major component "
                "while preserving the parent algorithm's overall identity. Possible components "
                "include initialization, candidate generation, scoring, selection, update, "
                "acceptance, refinement, restart logic, or budget allocation."
            ),
            "m2_parameter_schedule_mutation": (
                "Create a variant of one parent by changing its parameters, thresholds, rates, "
                "limits, or scheduling rules. Prefer meaningful adaptive or budget-aware schedules "
                "over arbitrary constant changes, while keeping the parent algorithm's structure."
            ),
            "m3_simplify_generalize": (
                "Create a simpler and more robust version of one parent by removing brittle, "
                "over-specialized, redundant, or overly complex components. Preserve the essential "
                "idea and required behavior while improving generalization and maintainability."
            ),
        }

        for name, metadata in STRATEGIES.items():
            with self.subTest(strategy=name):
                self.assertEqual(metadata["description"], expected_descriptions[name])


if __name__ == "__main__":
    unittest.main()
