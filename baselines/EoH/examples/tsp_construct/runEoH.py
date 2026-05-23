import os
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'eoh', 'src'))

from eoh import EoH, LLMConfig
from prob import TSPCONST

if __name__ == "__main__":
    llm = LLMConfig(
        api_endpoint='api.deepseek.com',
        api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-xxx"),
        model='deepseek-v4-flash',
        timeout=150,
    )

    pop_size = 5
    n_pop = 10

    task = TSPCONST(problem_size=500, n_instance=20, timeout=60, n_processes=4)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(__file__), f"results_{timestamp}_pops{pop_size}_gens{n_pop}")

    eoh = EoH(
        llm=llm,
        problem=task,
        pop_size=pop_size,
        n_pop=n_pop,
        operators=['e1', 'e2', 'm1', 'm2', 'm3'],
        output_dir=output_dir,
    )

    eoh.run()
