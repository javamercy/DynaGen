import unittest
from pathlib import Path
from unittest.mock import patch

from dynagen.cli import _provider_from_config
from dynagen.config import RunConfig, load_config


class OllamaProviderSelectionTests(unittest.TestCase):
    def test_load_config_preserves_ollama_model_suffix(self) -> None:
        config = load_config(Path("configs/tsp/tsp_calls_20.yaml"))

        self.assertEqual(config.llm.provider, "ollama")
        self.assertEqual(config.llm.model, "gemma4:31-cloud")
        self.assertEqual(config.evolution.verbal_gradients.llm_model, "gemma4:31-cloud")

    def test_provider_from_config_supports_ollama(self) -> None:
        config = RunConfig.from_dict({
            "run": {"name": "test", "output_dir": "runs/test", "seed": 1},
            "llm": {
                "provider": "ollama",
                "model": "llama3.1-cloud",
                "temperature": 0.2,
            },
            "evolution": {
                "population_size": 1,
                "generations": 1,
                "offspring_per_strategy": 1,
                "strategies": ["S1"],
            },
            "evaluation": {
                "budget": 10,
                "timeout_seconds": 1,
                "seeds": [1],
                "metric": "mean_gap",
            },
            "data": {
                "search_instances": "unused",
                "test_instances": "unused",
            },
        })

        with patch("dynagen.llm.ollama_provider.OllamaProvider") as mock_provider, patch(
            "dynagen.cli.CountingLLMProvider"
        ) as mock_counting:
            provider_instance = object()
            mock_provider.return_value = provider_instance
            mock_counting.side_effect = lambda provider, configured_budget=None: {
                "provider": provider,
                "configured_budget": configured_budget,
            }

            result = _provider_from_config(config)

        mock_provider.assert_called_once_with(model="llama3.1-cloud")
        self.assertEqual(result["provider"], provider_instance)
        self.assertIsNotNone(result["configured_budget"])


if __name__ == "__main__":
    unittest.main()
