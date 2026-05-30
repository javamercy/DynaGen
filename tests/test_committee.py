import math

import pytest

from dynagen.candidates.candidate import Candidate
from dynagen.config import EvolutionConfig, NicheConfig
from dynagen.evolution.committee import (
    assign_instances,
    compute_vbs,
    niche_probabilities,
    plateau_detect,
    select_committee,
)


def _make_candidate(cid: str, name: str, scores: dict[str, float]) -> Candidate:
    return Candidate(
        id=cid,
        generation=0,
        strategy="initial",
        name=name,
        metrics={"problem": "bbob", "aocc_by_function": scores, "score_name": "mean_aocc"},
    )


def _per_instance_scores(candidate: Candidate) -> dict[str, float]:
    metrics = candidate.metrics if isinstance(candidate.metrics, dict) else {}
    by_function = metrics.get("aocc_by_function")
    if isinstance(by_function, dict):
        return {str(k): float(v) for k, v in by_function.items()}
    return {}


class TestGreedyCover:
    def test_two_specialists_cover_all_functions(self):
        c1 = _make_candidate("c1", "DE", {"1": 0.9, "2": 0.8, "3": 0.1, "4": 0.15})
        c2 = _make_candidate("c2", "CMA", {"1": 0.1, "2": 0.15, "3": 0.85, "4": 0.9})
        specialists, assignments = select_committee(
            [c1, c2], per_instance_scores_fn=_per_instance_scores, committee_size=2
        )
        assert len(specialists) == 2
        all_assigned = set()
        for cid, inst_list in assignments.items():
            all_assigned.update(inst_list)
        assert all_assigned == {"1", "2", "3", "4"}

    def test_single_candidate_returns_one(self):
        c1 = _make_candidate("c1", "DE", {"1": 0.9})
        specialists, assignments = select_committee(
            [c1], per_instance_scores_fn=_per_instance_scores, committee_size=3
        )
        assert len(specialists) == 1
        assert assignments["c1"] == ["1"]

    def test_empty_candidates_returns_empty(self):
        specialists, assignments = select_committee(
            [], per_instance_scores_fn=_per_instance_scores, committee_size=3
        )
        assert specialists == []
        assert assignments == {}


class TestNicheProbabilities:
    def test_uniform_when_improvement_disabled(self):
        c1 = _make_candidate("c1", "A", {"1": 0.9, "2": 0.8})
        c2 = _make_candidate("c2", "B", {"1": 0.1, "2": 0.15})
        probs = niche_probabilities(
            [c1, c2],
            instance_assignments={"c1": ["1", "2"], "c2": ["1", "2"]},
            per_instance_scores_fn=_per_instance_scores,
            improvement_weight=False,
        )
        assert probs == {"c1": 0.5, "c2": 0.5}

    def test_low_improvement_room_gets_lower_weight(self):
        c1 = _make_candidate("c1", "good", {"1": 0.95, "2": 0.98})
        c2 = _make_candidate("c2", "bad", {"1": 0.3, "2": 0.4})
        probs = niche_probabilities(
            [c1, c2],
            instance_assignments={"c1": ["1", "2"], "c2": ["1", "2"]},
            per_instance_scores_fn=_per_instance_scores,
            improvement_weight=True,
            improvement_power=1.0,
        )
        assert probs["c1"] < probs["c2"]


class TestComputeVBS:
    def test_vbs_picks_best_per_instance(self):
        c1 = _make_candidate("c1", "A", {"1": 0.9, "2": 0.2})
        c2 = _make_candidate("c2", "B", {"1": 0.5, "2": 0.8})
        vbs = compute_vbs([c1, c2], per_instance_scores_fn=_per_instance_scores)
        assert vbs["1"] == 0.9
        assert vbs["2"] == 0.8

    def test_vbs_empty_candidates(self):
        vbs = compute_vbs([], per_instance_scores_fn=_per_instance_scores)
        assert vbs == {}


class TestAssignInstances:
    def test_each_instance_to_best_specialist(self):
        c1 = _make_candidate("c1", "A", {"1": 0.9, "2": 0.1})
        c2 = _make_candidate("c2", "B", {"1": 0.2, "2": 0.85})
        assignments = assign_instances(
            [c1, c2], all_instances=["1", "2"], per_instance_scores_fn=_per_instance_scores
        )
        assert "1" in assignments["c1"]
        assert "2" in assignments["c2"]


class TestPlateauDetect:
    def test_not_enough_history_returns_false(self):
        assert plateau_detect([0.0001], threshold=0.001, patience=3) is False
        assert plateau_detect([0.0001, 0.0002], threshold=0.001, patience=3) is False

    def test_plateau_detected(self):
        assert plateau_detect([0.0001, 0.0002, 0.0001], threshold=0.001, patience=3) is True

    def test_not_plateaued_when_improving(self):
        assert plateau_detect([0.01, 0.05, 0.0001], threshold=0.001, patience=2) is False


class TestConfig:
    def test_default_output_mode_is_single(self):
        cfg = EvolutionConfig(
            population_size=5,
            generations=10,
            offspring_per_strategy=1,
        )
        assert cfg.output_mode == "single"
        assert cfg.committee_size == 3
        assert cfg.committee_test_budget == 1000

    def test_committee_mode_with_niche_config(self):
        cfg = EvolutionConfig(
            population_size=5,
            generations=10,
            offspring_per_strategy=1,
            output_mode="committee_specialist",
            committee_size=5,
            niche={"cadence_generations": 15, "improvement_power": 0.5},
        )
        assert cfg.output_mode == "committee_specialist"
        assert cfg.committee_size == 5
        assert cfg.niche.cadence_generations == 15
        assert cfg.niche.improvement_power == 0.5

    def test_invalid_output_mode_raises(self):
        with pytest.raises(ValueError, match="output_mode"):
            EvolutionConfig(
                population_size=5,
                generations=10,
                offspring_per_strategy=1,
                output_mode="invalid_mode",
            )

    def test_committee_size_must_be_positive(self):
        with pytest.raises(ValueError, match="committee_size"):
            EvolutionConfig(
                population_size=5,
                generations=10,
                offspring_per_strategy=1,
                committee_size=0,
            )

    def test_niche_cadence_must_be_positive(self):
        with pytest.raises(ValueError, match="cadence_generations"):
            NicheConfig(cadence_generations=0)

    def test_niche_improvement_power_must_be_positive(self):
        with pytest.raises(ValueError, match="improvement_power"):
            NicheConfig(improvement_power=0)

    def test_committee_test_budget_must_be_positive(self):
        with pytest.raises(ValueError, match="committee_test_budget"):
            EvolutionConfig(
                population_size=5,
                generations=10,
                offspring_per_strategy=1,
                committee_test_budget=0,
            )
