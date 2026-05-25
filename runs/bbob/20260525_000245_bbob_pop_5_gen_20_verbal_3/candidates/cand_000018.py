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
        CR_cycle = 10.0
        generation = 0
        restart_min_gen = 5

        while calls < budget:
            remaining = budget - calls
            if NP > 0:
                max_generations = max(0, remaining // NP)
            else:
                break
            if max_generations == 0:
                break

            F = F_start - (F_start - F_end) * min(generation / 100.0, 1.0)
            CR = CR_base + CR_amp * np.sin(2 * np.pi * generation / CR_cycle)
            CR = np.clip(CR, 0.0, 1.0)
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

            # Diversity check
            if calls < budget and generation >= restart_min_gen:
                var = np.var(pop_fitness)
                if var < 1e-8 and calls < budget:
                    # Restart half the population, keep best
                    n_restart = NP // 2
                    if n_restart > 0:
                        idx = np.argsort(pop_fitness)
                        # keep best (index 0) and worst NP//2 - 1? Actually keep best, reinitialize others
                        # Choose indices to restart: those not best
                        restart_idx = [j for j in range(NP) if j != idx[0]]
                        if len(restart_idx) > n_restart:
                            restart_idx = rng.choice(restart_idx, size=n_restart, replace=False)
                        for j in restart_idx:
                            pop[j] = rng.uniform(lb, ub, size=dim)
                            # No evaluation yet, will evaluate next generation
                            # But we need to set fitness to inf so they get replaced
                            pop_fitness[j] = np.inf
                        # Reset generation counter? Not necessary
                        generation = 0  # reset generation to restart F and CR cycle
                        # Re-evaluate only if we want to use them immediately, but we'll let them be evaluated in next generation loop

        return best_val, best_x