import os
import sys
from datetime import datetime
from pathlib import Path

EOH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(EOH_ROOT / "eoh" / "src"))

from eoh import eoh
from eoh.utils.getParas import Paras


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _csv_env(name: str, default: list[str]) -> list[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    use_local = os.environ.get("EOH_LLM_USE_LOCAL", "0") == "1"
    api_endpoint = os.environ.get("EOH_LLM_API_ENDPOINT", "api.openai.com")
    api_key = os.environ.get("OPENAI_API_KEY")
    local_url = os.environ.get("EOH_LLM_LOCAL_URL")
    model = os.environ.get("EOH_LLM_MODEL", "gpt-5.4-nano")
    eva_timeout = _float_env("EOH_EVA_TIMEOUT", 60.0)
    ec_pop_size = _int_env("EOH_EC_POP_SIZE", 1)
    ec_n_pop = _int_env("EOH_EC_N_POP", 3)
    ec_operators = _csv_env("EOH_EC_OPERATORS", ["e1", "e2", "m1", "m2"])
    ec_operator_weights = [float(item) for item in _csv_env("EOH_EC_OPERATOR_WEIGHTS", ["1", "1", "1", "1"])]
    run_name = os.environ.get(
        "EOH_RUN_NAME",
        datetime.now().strftime(
            f"%Y%m%d_%H%M%S_eoh_tsp_{2 * ec_pop_size + ec_n_pop * len(ec_operators) * ec_pop_size}calls"),
    )

    if use_local:
        if not local_url:
            raise RuntimeError("Set EOH_LLM_LOCAL_URL when EOH_LLM_USE_LOCAL=1")
    elif not api_endpoint or not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for remote EoH comparison runs")

    output_path = PROJECT_ROOT / "runs" / "tsp" / run_name

    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem="tsp_construct",
        llm_use_local=use_local,
        llm_local_url=local_url,
        llm_api_endpoint=api_endpoint,
        llm_api_key=api_key,
        llm_model=model,
        ec_pop_size=ec_pop_size,
        ec_n_pop=ec_n_pop,
        ec_operators=ec_operators,
        ec_operator_weights=ec_operator_weights,
        exp_n_proc=1,
        exp_debug_mode=False,
        exp_output_path=str(output_path),
        eva_timeout=eva_timeout,
        llm_enable_health_check=False,
        llm_api_max_attempts=1,
        llm_parse_retries=0,
        tsp_search_instances=str(PROJECT_ROOT / "data" / "tsp" / "search_instances"),
        tsp_test_instances=str(PROJECT_ROOT / "data" / "tsp" / "test_instances"),
    )

    expected_calls = 2 * ec_pop_size + ec_n_pop * len(ec_operators) * ec_pop_size
    print(f"Expected candidate-generation LLM calls: {expected_calls}")
    print(f"Output path: {output_path}")
    evolution = eoh.EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
