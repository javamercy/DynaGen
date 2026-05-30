import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        best_val = np.inf
        best_x = None
        evals = 0

        def evaluate(x):
            nonlocal best_val, best_x, evals
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Phase 1: Adaptive coordinate pattern search with aggressive step adaptation
        x = rng.uniform(lb, ub)
        f_x = evaluate(x)
        step = 0.3 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        local_best_x = x.copy()
        local_best_val = f_x
        no_improve = 0
        max_no_improve = max(2 * dim, 10)
        phase1_budget = int(budget * 0.4)
        while evals < phase1_budget and no_improve < max_no_improve and np.any(step > min_step):
            improved = False
            for d in rng.permutation(dim):
                if evals >= phase1_budget:
                    break
                # positive step
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = evaluate(x_new)
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
                        continue
                # negative step
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = evaluate(x_new)
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                        improved = True
            if improved:
                direction = x - local_best_x
                x_new = x + direction
                x_new = np.clip(x_new, lb, ub)
                if np.any(x_new != x) and evals < phase1_budget:
                    val = evaluate(x_new)
                    if val < local_best_val:
                        local_best_val = val
                        local_best_x = x_new.copy()
                        x = x_new.copy()
                step *= 1.8  # aggressive increase on success
                no_improve = 0
            else:
                step *= 0.5  # aggressive decrease on failure
                x = local_best_x.copy()
                no_improve += 1

        # Phase 2: (1+1)-ES with cumulative step size adaptation (CSA)
        remaining = budget - evals
        if remaining > 0:
            x = best_x.copy()
            sigma = 0.1 * np.linalg.norm(ub - lb) / np.sqrt(dim)
            min_sigma = 1e-8 * np.linalg.norm(ub - lb) / np.sqrt(dim)
            p = np.zeros(dim)
            c = 2.0 / (dim + 2.0)
            d = 1.0 + dim / 2.0
            for _ in range(remaining):
                if evals >= budget:
                    break
                z = rng.randn(dim)
                candidate = x + sigma * z
                candidate = np.clip(candidate, lb, ub)
                val = evaluate(candidate)
                if val < best_val:
                    x = candidate.copy()
                    p = (1 - c) * p + np.sqrt(c * (2 - c)) * z
                else:
                    p = (1 - c) * p
                sigma = sigma * np.exp((np.dot(p, p) / dim - 1) / d)
                sigma = max(sigma, min_sigma)

        return best_val, best_x