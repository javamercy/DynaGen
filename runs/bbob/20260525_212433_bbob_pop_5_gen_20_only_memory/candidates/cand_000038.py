import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.T0 = 1.0
        self.T_end = 1e-3

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        x = lb + (ub - lb) * rng.rand(dim)
        f = func(x)
        evals = 1
        best_x = x.copy()
        best_val = f
        report_best(best_val, best_x)
        if evals >= budget:
            return best_val, best_x
        if budget > 1:
            cooling_rate = np.exp((np.log(self.T_end) - np.log(self.T0)) / (budget - 1))
        else:
            cooling_rate = 1.0
        sigma0 = 0.1 * (ub - lb)
        T = self.T0
        current_x = x
        current_f = f
        while evals < budget:
            sigma = sigma0 * np.sqrt(T / self.T0)
            candidate = current_x + sigma * rng.randn(dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_f = func(candidate)
            evals += 1
            delta = candidate_f - current_f
            if delta < 0:
                current_x = candidate
                current_f = candidate_f
                if candidate_f < best_val:
                    best_val = candidate_f
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
            else:
                prob = np.exp(-delta / T)
                if rng.rand() < prob:
                    current_x = candidate
                    current_f = candidate_f
            T *= cooling_rate
        return best_val, best_x