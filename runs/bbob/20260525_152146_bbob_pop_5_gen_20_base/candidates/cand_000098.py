import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        pop_size = max(4, min(10, budget // 2))
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        evals = 0

        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1
            evals += 1

        best_idx = np.argmin(pop_f[:evals])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.7
        CR = 0.9

        while budget > 0:
            for i in range(evals):
                if budget <= 0:
                    break
                # select two distinct indices different from i
                candidates = [j for j in range(evals) if j != i]
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == rng.randint(dim):
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)

            # local refinement of best
            if budget > 0:
                sigma = 0.1 * (ub - lb) * (1 - evals / self.budget)
                trial = best_x + sigma * rng.randn(dim)
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < best_f:
                    best_x = trial.copy()
                    best_f = trial_f
                    report_best(best_f, best_x)

        return best_f, best_x