import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = min(4 * dim, 20, budget // 3)
        pop_size = max(pop_size, 3)
        if pop_size > budget:
            pop_size = budget

        pop_x = rng.uniform(lb, ub, (pop_size, dim))
        pop_y = np.full(pop_size, np.inf)
        best_x = None
        best_y = np.inf

        for i in range(pop_size):
            pop_y[i] = func(pop_x[i])
            if pop_y[i] < best_y:
                best_y = pop_y[i]
                best_x = pop_x[i].copy()
                report_best(best_y, best_x)

        evals = pop_size
        F = 0.8
        CR = 0.9

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                target_idx = i
                indices = list(range(pop_size))
                indices.remove(target_idx)
                if len(indices) < 2:
                    continue
                b_idx, c_idx = rng.choice(indices, 2, replace=False)
                b = pop_x[b_idx]
                c = pop_x[c_idx]
                # best/1 mutation
                mutant = best_x + F * (b - c)
                trial = pop_x[target_idx].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_y = func(trial)
                evals += 1
                if trial_y < pop_y[target_idx]:
                    pop_x[target_idx] = trial
                    pop_y[target_idx] = trial_y
                    if trial_y < best_y:
                        best_y = trial_y
                        best_x = trial.copy()
                        report_best(best_y, best_x)

        return best_y, best_x