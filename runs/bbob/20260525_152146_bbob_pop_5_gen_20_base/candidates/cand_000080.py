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

        # Adaptive population size
        if budget >= 6:
            pop_size = min(20, budget // 2)
        else:
            pop_size = budget
        pop_size = max(4, pop_size)
        if pop_size > budget:
            pop_size = budget

        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        num_eval = 0

        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1
            num_eval += 1

        best_idx = np.argmin(pop_f[:num_eval])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        if budget <= 0:
            return best_f, best_x

        F = 0.8
        CR = 0.9

        while budget > 0:
            for i in range(num_eval):
                if budget <= 0:
                    break
                # Mutation: current-to-best/1 with robust fallback
                indices = [j for j in range(num_eval) if j != i]
                if len(indices) >= 2:
                    r1, r2 = rng.choice(indices, 2, replace=False)
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                else:
                    mutant = pop[i] + F * rng.uniform(-1, 1, dim)
                # Binomial crossover
                trial = pop[i].copy()
                j0 = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j0:
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

        return best_f, best_x