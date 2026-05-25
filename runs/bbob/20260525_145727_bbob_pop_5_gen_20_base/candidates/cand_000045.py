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
        max_gen = (budget - pop_size) // pop_size
        if max_gen <= 0:
            return best_y, best_x

        F0 = 0.8
        CR0 = 0.9
        dF = 0.4 / max_gen  # decrease from 0.8 to 0.4
        dCR = 0.8 / max_gen # decrease from 0.9 to 0.1

        gen = 0
        while evals < budget and gen < max_gen:
            F = F0 - dF * gen
            CR = CR0 - dCR * gen
            for i in range(pop_size):
                if evals >= budget:
                    break
                target_idx = i
                # Choose two distinct random indices different from i and best index
                best_idx = np.argmin(pop_y)  # note: best_y may be different but we use population best
                indices = list(range(pop_size))
                indices.remove(target_idx)
                if best_idx in indices:
                    indices.remove(best_idx)
                if len(indices) < 2:
                    continue
                a_idx, b_idx = rng.choice(indices, 2, replace=False)
                a = pop_x[a_idx]
                b = pop_x[b_idx]
                mutant = pop_x[best_idx] + F * (a - b)
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
            gen += 1

        # Local search: small Gaussian perturbations around best
        sigma = 0.001 * (ub - lb)  # small relative step
        while evals < budget:
            x_try = best_x + rng.normal(0, sigma)
            x_try = np.clip(x_try, lb, ub)
            y_try = func(x_try)
            evals += 1
            if y_try < best_y:
                best_y = y_try
                best_x = x_try.copy()
                report_best(best_y, best_x)

        return best_y, best_x