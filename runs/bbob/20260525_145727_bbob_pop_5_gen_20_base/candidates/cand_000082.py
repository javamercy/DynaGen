import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        # initial random point
        best_x = None
        best_val = np.inf
        current_x = lb + rng.rand(dim) * (ub - lb)
        current_val = func(current_x)
        evals = 1
        best_x = current_x.copy()
        best_val = current_val
        report_best(best_val, best_x)

        # parameters
        T0 = 1.0
        stagnate_counter = 0
        max_stagnate = max(1, budget // 10)

        while evals < budget:
            # fraction of budget used
            frac = evals / budget
            # temperature (linear cooling)
            T = T0 * (1.0 - frac)
            # step size (adaptive, starts large, reduces)
            sigma = (ub - lb) * (0.5 * (1.0 - frac) + 0.2)

            # generate candidate
            candidate = current_x + rng.normal(0, sigma, dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_val = func(candidate)
            evals += 1

            # accept or reject
            if candidate_val < current_val:
                current_x = candidate.copy()
                current_val = candidate_val
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                stagnate_counter = 0
            else:
                if T > 0:
                    prob = np.exp(-(candidate_val - current_val) / T)
                    if rng.rand() < prob:
                        current_x = candidate.copy()
                        current_val = candidate_val
                        # note: even if not improving, we update current
                        # but no improvement to best
                        stagnate_counter = 0
                    else:
                        stagnate_counter += 1
                else:
                    stagnate_counter += 1

            # restart if stagnation
            if stagnate_counter >= max_stagnate:
                # restart with new random point
                current_x = lb + rng.rand(dim) * (ub - lb)
                # but we need to evaluate it; check budget
                if evals < budget:
                    current_val = func(current_x)
                    evals += 1
                    # possible improvement
                    if current_val < best_val:
                        best_val = current_val
                        best_x = current_x.copy()
                        report_best(best_val, best_x)
                stagnate_counter = 0

        return best_val, best_x