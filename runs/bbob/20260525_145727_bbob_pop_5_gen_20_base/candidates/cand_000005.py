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

        pop_size = max(3, min(20, budget//2, 2*dim))
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
        remaining = budget - evals
        if remaining <= 0:
            return best_y, best_x

        if pop_size >= 3:
            DE_frac = 0.7
            N_DE = int(DE_frac * remaining)
        else:
            N_DE = 0
        N_local = remaining - N_DE

        F = 0.8
        CR = 0.9

        for _ in range(N_DE):
            target_idx = rng.randint(pop_size)
            indices = list(range(pop_size))
            indices.remove(target_idx)
            if len(indices) < 3:
                break
            a_idx, b_idx, c_idx = rng.choice(indices, 3, replace=False)
            a = pop_x[a_idx]
            b = pop_x[b_idx]
            c = pop_x[c_idx]
            mutant = a + F * (b - c)
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

        step_init = 0.1 * (ub - lb)
        for i in range(N_local):
            progress = i / max(1, N_local)
            step = step_init * (1 - progress)
            trial = best_x + rng.randn(dim) * step
            trial = np.clip(trial, lb, ub)
            trial_y = func(trial)
            evals += 1
            if trial_y < best_y:
                best_y = trial_y
                best_x = trial.copy()
                report_best(best_y, best_x)
                worst_idx = np.argmax(pop_y)
                if trial_y < pop_y[worst_idx]:
                    pop_x[worst_idx] = trial
                    pop_y[worst_idx] = trial_y

        return best_y, best_x