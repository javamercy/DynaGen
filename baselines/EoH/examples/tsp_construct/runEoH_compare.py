import os
import sys
from datetime import datetime
from pathlib import Path

EOH_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(EOH_ROOT / "eoh" / "src"))

from eoh import eoh
from eoh.utils.getParas import Paras


def main():
    use_local = False
    api_endpoint = "api.openai.com"
    api_key = os.environ.get("OPENAI_API_KEY")
    local_url = os.environ.get("EOH_LLM_LOCAL_URL")
    model = "gpt-5.4-mini"
    eva_timeout = float(os.environ.get("EOH_EVA_TIMEOUT", "60"))

    if use_local:
        if not local_url:
            raise RuntimeError("Set EOH_LLM_LOCAL_URL when EOH_LLM_USE_LOCAL=1")
    elif not api_endpoint or not api_key:
        raise RuntimeError("Set OPENAI_API_KEY for remote EoH comparison runs")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_eoh_tsp_14calls")
    output_path = PROJECT_ROOT / "runs" / run_name

    paras = Paras()
    paras.set_paras(
        method="eoh",
        problem="tsp_construct",
        llm_use_local=use_local,
        llm_local_url=local_url,
        llm_api_endpoint=api_endpoint,
        llm_api_key=api_key,
        llm_model=model,
        ec_pop_size=1,
        ec_n_pop=3,
        ec_operators=["e1", "e2", "m1", "m2"],
        ec_operator_weights=[1, 1, 1, 1],
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

    print(f"Expected candidate-generation LLM calls: 14")
    print(f"Output path: {output_path}")
    evolution = eoh.EVOL(paras)
    evolution.run()


if __name__ == "__main__":
    main()
