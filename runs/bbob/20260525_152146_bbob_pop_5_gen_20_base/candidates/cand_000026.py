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

        def clip(x):
            return np.clip(x, lb, ub)

        if budget == 0:
            raise ValueError("Budget must be at least 1")
        if budget == 1:
            x = rng.uniform(lb, ub, size=dim)
            f = func(x)
            report_best(f, x)
            return f, x

        # Population size: at least 3, at most min(budget//2, 10)
        pop_size = max(3, min(budget // 2, 10))
        pop_size = min(pop_size, budget)

        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.8
        CR = 0.9

        while budget > 0:
            for i in range(pop_size):
                if budget <= 0:
                    break
                # random indices distinct from i
                a, b = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mut = pop[i] + F * (pop[a] - pop[b])
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)

        return best_f, best_x