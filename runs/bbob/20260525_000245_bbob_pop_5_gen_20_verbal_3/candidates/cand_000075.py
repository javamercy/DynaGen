import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        NP = max(4, min(20, budget // (dim + 1)))
        if NP < 4:
            NP = 4

        pop = rng.uniform(lb, ub, size=(NP, dim))
        pop_fitness = np.full(NP, np.inf)
        calls = 0
        best_x = None
        best_val = np.inf

        for i in range(NP):
            if calls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            calls += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        F_start = 0.9
        F_end = 0.5
        CR_base = 0.7
        CR_amp = 0.2
        generation = 0
        max_generations = (budget - calls) // NP if NP > 0 else 0

        while calls < budget:
            if max_generations > 0:
                frac = generation / max_generations
            else:
                frac = 0.0
            F = F_start - (F_start - F_end) * min(frac, 1.0)
            CR = CR_base + CR_amp * np.sin(2 * np.pi * generation / max_generations) if max_generations > 0 else CR_base
            generation += 1

            for i in range(NP):
                if calls >= budget:
                    break
                r1, r2 = rng.choice([j for j in range(NP) if j != i], size=2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.uniform() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                calls += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Diversity-triggered restart
            remaining = budget - calls
            if remaining >= NP // 3 + 2 and NP > 1:
                var = np.var(pop_fitness)
                threshold = 1e-8 * (best_val + 1e-10) if best_val < 1e10 else 1e-8
                if var < threshold:
                    indices = np.argsort(pop_fitness)
                    num_restart = max(1, NP // 3)
                    for idx in indices[-num_restart:]:
                        if calls >= budget:
                            break
                        pop[idx] = rng.uniform(lb, ub, size=dim)
                        x = np.clip(pop[idx], lb, ub)
                        val = func(x)
                        calls += 1
                        pop_fitness[idx] = val
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)

            remaining = budget - calls
            if NP > 0:
                max_generations = max(0, remaining // NP)

        return best_val, best_x