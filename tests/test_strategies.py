import unittest

from dynagen.evolution.strategies import STRATEGIES, STRATEGIES_METADATA, Strategy, parent_count


class StrategyMetadataTests(unittest.TestCase):
    def test_parent_counts_are_fixed(self) -> None:
        expected = {
            Strategy.E1_RADICAL_EXPLORATION: 3,
            Strategy.E2_BACKBONE_EXPLORATION: 3,
            Strategy.E3_HYBRID_RECOMBINATION: 2,
            Strategy.M1_COMPONENT_REPLACEMENT: 1,
            Strategy.M2_PARAMETER_SCHEDULE_MUTATION: 1,
            Strategy.M3_SIMPLIFY_GENERALIZE: 1,
            Strategy.M4_CONTRACT_REPAIR: 1,
            Strategy.M5_INTENSIFY_SEARCH: 1,
            Strategy.M6_DIVERSIFY_SEARCH: 1,
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
            "e3_hybrid_recombination": (
                "Combine compatible strengths from multiple parents into one coherent algorithm. "
                "Select only components that work well together, resolve conflicts between their "
                "design choices, and avoid merely concatenating all parent mechanisms."
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
            "m4_contract_repair": (
                "Revise one parent to better satisfy the required interface, constraints, safety "
                "rules, and budget limits. Preserve the main algorithmic idea, but fix invalid "
                "return paths, unsafe assumptions, missing fallback behavior, and fragile edge cases."
            ),
            "m5_intensify_search": (
                "Create a stronger exploitation-focused version of one parent. Improve how it "
                "refines promising candidates, allocates effort to high-quality regions, reduces "
                "wasted work, or makes more precise local decisions while preserving correctness."
            ),
            "m6_diversify_search": (
                "Create a more exploration-focused version of one parent. Improve how it generates "
                "diverse candidates, escapes stagnation, varies search trajectories, uses restarts "
                "or perturbations, or avoids premature convergence while respecting the contract."
            ),
        }

        for name, metadata in STRATEGIES.items():
            with self.subTest(strategy=name):
                self.assertEqual(metadata["description"], expected_descriptions[name])


if __name__ == "__main__":
    unittest.main()
