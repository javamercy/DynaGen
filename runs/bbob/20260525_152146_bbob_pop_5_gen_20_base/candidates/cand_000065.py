import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = np.random.RandomState(self.seed)
        budget = self.budget

        pop_size = min(10, max(4, budget // 10))
        pop = rng.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f[:min(pop_size, max(1, self.budget))])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.5
        CR = 0.9

        while budget > 0:
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(indices, 2, replace=False)
                mut = best_x + F * (pop[a] - pop[b])
                trial = pop[i].copy()
                j0 = rng.randint(dim)
                j = j0
                L = 0
                while True:
                    trial[j] = mut[j]
                    j = (j + 1) % dim
                    L += 1
                    if L == dim or rng.rand() > CR:
                        break
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

            if budget > 0:
                sigma = 0.01 * (ub - lb)
                direction = rng.randn(dim)
                candidate = best_x + sigma * direction
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                budget -= 1
                if val < best_f:
                    best_x = candidate.copy()
                    best_f = val
                    report_best(best_f, best_x)

        return best_f, best_x