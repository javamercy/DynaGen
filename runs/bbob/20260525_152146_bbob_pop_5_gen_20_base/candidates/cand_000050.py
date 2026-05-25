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

        # Population size: at least 4, at most 20, not exceeding budget-1
        pop_size = min(20, budget // 2)
        if pop_size < 4:
            pop_size = min(4, budget - 1)
        if pop_size < 1:
            pop_size = 1

        # Special case: very low budget -> random sampling
        if pop_size < 4:
            best_x = rng.uniform(lb, ub, size=dim)
            best_f = func(best_x)
            budget -= 1
            report_best(best_f, best_x)
            while budget > 0:
                x = rng.uniform(lb, ub, size=dim)
                f = func(x)
                budget -= 1
                if f < best_f:
                    best_f = f
                    best_x = x
                    report_best(best_f, best_x)
            return best_f, best_x

        # Initialize population uniformly
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.5
        CR = 0.5

        while budget > 0:
            success = 0
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select three distinct individuals different from i
                indices = [j for j in range(pop_size) if j != i]
                if len(indices) < 3:
                    a, b, c = rng.choice(pop_size, 3, replace=False)
                else:
                    a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
                # Binomial crossover
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    success += 1
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
            # Adapt parameters based on success ratio
            if pop_size > 0:
                success_ratio = success / pop_size
                if success_ratio > 0.5:
                    F = min(1.0, F + 0.05)
                    CR = min(1.0, CR + 0.05)
                else:
                    F = max(0.2, F - 0.1)
                    CR = max(0.0, CR - 0.1)

        return best_f, best_x