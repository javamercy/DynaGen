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
        CR_period = 4
        variance_threshold = 1e-10
        gen = 0
        max_generations = (budget - calls) // NP if NP > 0 else 0
        no_improve = 0

        while calls < budget and max_generations > 0:
            if max_generations > 1:
                F = F_start - (F_start - F_end) * (gen / (max_generations - 1))
            else:
                F = F_start
            gen += 1

            # Sinusoidal CR
            CR = 0.6 + 0.3 * np.sin(2 * np.pi * gen / CR_period)

            # Diversity-triggered restart
            if gen > 5:  # Wait a few generations
                variance = np.var(pop_fitness)
                if variance < variance_threshold and calls < budget:
                    # Reinitialize worst half (keep best)
                    idx_sorted = np.argsort(pop_fitness)
                    keep = idx_sorted[0]  # best
                    n_restart = NP // 2
                    restart_idx = idx_sorted[-n_restart:]
                    new_pop = rng.uniform(lb, ub, size=(n_restart, dim))
                    for j, idx in enumerate(restart_idx):
                        if idx != keep:
                            pop[idx] = new_pop[j]
                            # Evaluate new points? No, will evaluate in loop below
                            # But we need to assign fitness? We'll evaluate in next generation loop
                            # To avoid asymmetry, we can set fitness to inf and evaluate in mutation loop
                            # Actually we'll handle in the loop: before mutation, evaluate if fitness is inf?
                            # Simpler: evaluate new points immediately
                            if calls < budget:
                                x = np.clip(pop[idx], lb, ub)
                                val = func(x)
                                calls += 1
                                pop_fitness[idx] = val
                                if val < best_val:
                                    best_val = val
                                    best_x = x.copy()
                                    report_best(best_val, best_x)

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

            remaining = budget - calls
            max_generations = max(0, remaining // NP)

        return best_val, best_x