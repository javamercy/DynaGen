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

        pop_size = min(20, max(4, budget // 10))
        pop_size = max(2, pop_size)

        if budget < pop_size:
            best_x = rng.uniform(lb, ub, size=dim)
            best_f = func(best_x)
            budget -= 1
            report_best(best_f, best_x)
            while budget > 0:
                x = rng.uniform(lb, ub, size=dim)
                f = func(x)
                budget -= 1
                if f < best_f:
                    best_x = x.copy()
                    best_f = f
                    report_best(best_f, best_x)
            return best_f, best_x

        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        total_budget = self.budget
        F_start = 0.8
        F_end = 0.5
        CR_start = 0.9
        CR_end = 0.5
        stagnation_limit = max(1, (budget // pop_size) // 2)
        stagnation_counter = 0

        while budget > 0:
            frac = budget / total_budget
            F = F_start + (F_end - F_start) * (1 - frac)
            CR = CR_start + (CR_end - CR_start) * (1 - frac)
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
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
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        improved = True
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit and budget >= pop_size:
                    new_size = pop_size - 1
                    new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                    new_pop_f = np.full(new_size, np.inf)
                    for i in range(new_size):
                        if budget <= 0:
                            break
                        new_pop_f[i] = func(new_pop[i])
                        budget -= 1
                    pop = np.vstack((best_x.reshape(1, -1), new_pop))
                    pop_f = np.concatenate(([best_f], new_pop_f))
                    stagnation_counter = 0
        return best_f, best_x