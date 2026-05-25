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
        CR_base = 0.5
        CR_amp = 0.4
        max_gen = max(1, (budget - calls) // NP) if NP > 0 else 1
        period = max(1, max_gen // 4)
        generation = 0

        while calls < budget:
            remaining = budget - calls
            if remaining <= 0:
                break
            max_gen = max(1, remaining // NP) if NP > 0 else 1
            CR = CR_base + CR_amp * np.sin(2 * np.pi * generation / period)
            CR = np.clip(CR, 0.1, 0.9)
            if max_gen > 1:
                F = F_start - (F_start - F_end) * (generation / (max_gen - 1))
            else:
                F = F_start
            generation += 1

            # Diversity-triggered restart (every few generations)
            if generation % 5 == 0 and calls < budget - NP // 2:
                if NP >= 2:
                    finite_mask = np.isfinite(pop_fitness)
                    if np.sum(finite_mask) >= 2:
                        f_min = np.min(pop_fitness[finite_mask])
                        f_max = np.max(pop_fitness[finite_mask])
                        var = np.var(pop_fitness[finite_mask])
                        threshold = 1e-8 * (f_max - f_min + 1e-12)
                        if var < threshold or np.all(pop_fitness[finite_mask] == pop_fitness[finite_mask][0]):
                            n_replace = max(1, NP // 2)
                            sorted_idx = np.argsort(pop_fitness)
                            replace_idx = sorted_idx[1:1+n_replace]
                            for idx in replace_idx:
                                if calls >= budget:
                                    break
                                x = rng.uniform(lb, ub, size=dim)
                                x = np.clip(x, lb, ub)
                                val = func(x)
                                calls += 1
                                pop[idx] = x
                                pop_fitness[idx] = val
                                if val < best_val:
                                    best_val = val
                                    best_x = x.copy()
                                    report_best(best_val, best_x)

            for i in range(NP):
                if calls >= budget:
                    break
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 2:
                    continue
                r1, r2 = rng.choice(indices, size=2, replace=False)
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

        return best_val, best_x