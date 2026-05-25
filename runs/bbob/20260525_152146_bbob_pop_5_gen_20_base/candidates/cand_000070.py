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
        if pop_size < 2:
            pop_size = 2

        if budget < pop_size:
            # random search
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

        # initialize population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.8
        CR = 0.9
        stagnation_limit = max(1, (budget // pop_size) // 2) if pop_size > 0 else 1
        stagnation_counter = 0

        while budget > 0:
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # select three distinct random indices different from i
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
                # exponential crossover
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
                        improved = True
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit and budget >= pop_size:
                    # restart: keep best, reinitialize rest
                    new_size = pop_size - 1
                    new_pop = rng.uniform(lb, ub, size=(new_size, dim))
                    new_pop_f = np.full(new_size, np.inf)
                    for ii in range(new_size):
                        if budget <= 0:
                            break
                        new_pop_f[ii] = func(new_pop[ii])
                        budget -= 1
                    pop = np.vstack((best_x.reshape(1, -1), new_pop))
                    pop_f = np.concatenate(([best_f], new_pop_f))
                    stagnation_counter = 0
        return best_f, best_x