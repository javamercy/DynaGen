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
        best_val = float('inf')
        best_x = None
        evals = 0

        # initial random point
        x = lb + self.rng.random(self.dim) * (ub - lb)
        val = func(x)
        evals += 1
        best_val = val
        best_x = x.copy()
        report_best(best_val, best_x)

        # Latin hypercube sampling for initial population
        n_pop = min(20, (self.budget - evals) // 2)
        if n_pop > 0:
            samples = np.zeros((n_pop, self.dim))
            for i in range(self.dim):
                perm = self.rng.permutation(n_pop)
                samples[:, i] = (perm + self.rng.random(n_pop)) / n_pop
            samples = lb + samples * (ub - lb)
            for s in samples:
                if evals >= self.budget:
                    break
                val = func(s)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = s.copy()
                    report_best(best_val, best_x)

        # (1+1)-ES with step size adaptation and restarts
        x = best_x.copy()
        sigma = 0.2  # relative to domain size
        success_counter = 0
        eval_window = 10
        no_improve = 0
        patience = max(5, self.dim)
        scale = ub - lb

        while evals < self.budget:
            z = self.rng.normal(0, sigma, self.dim)
            candidate = x + z * scale
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                x = candidate
                success_counter += 1
                no_improve = 0
            else:
                no_improve += 1

            if evals % eval_window == 0:
                success_rate = success_counter / eval_window
                if success_rate > 0.2:
                    sigma *= 1.22
                else:
                    sigma *= 0.82
                success_counter = 0
                if sigma < 1e-8:
                    sigma = 0.2

            # restart condition
            if no_improve >= patience and evals < self.budget:
                x = lb + self.rng.random(self.dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                sigma = 0.2
                no_improve = 0
                success_counter = 0

        return best_val, best_x