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

        # Phase 1: Adaptive coordinate pattern search
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
                step *= 1.5
                no_improve = 0
            else:
                step *= 0.7
                x = local_best_x.copy()
                no_improve += 1

        # Phase 2: 1+1-ES with restart on stall
        remaining = budget - evals
        if remaining > 0:
            x = best_x.copy()
            sigma = 0.1 * np.linalg.norm(ub - lb) / np.sqrt(dim)
            min_sigma = 1e-8 * np.linalg.norm(ub - lb) / np.sqrt(dim)
            successes = 0
            trials = 0
            success_window = max(5, dim)
            no_improve_count = 0
            restart_count = 0
            max_restarts = min(3, remaining // (dim * 10)) if dim > 0 else 3
            restart_threshold = max(dim * 20, 50)
            while evals < budget:
                z = rng.randn(dim)
                candidate = x + sigma * z
                candidate = np.clip(candidate, lb, ub)
                val = evaluate(candidate)
                if val < best_val:
                    x = candidate.copy()
                    best_val = val
                    best_x = candidate.copy()
                    successes += 1
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                trials += 1
                if trials >= success_window:
                    success_rate = successes / trials
                    if success_rate > 0.2:
                        sigma *= 1.22
                    elif success_rate < 0.2:
                        sigma *= 0.82
                    successes = 0
                    trials = 0
                sigma = max(sigma, min_sigma)
                if (no_improve_count >= restart_threshold and restart_count < max_restarts
                        and evals < budget - 1):
                    x = rng.uniform(lb, ub)
                    successes = 0
                    trials = 0
                    no_improve_count = 0
                    sigma = 0.1 * np.linalg.norm(ub - lb) / np.sqrt(dim)
                    restart_count += 1

        return best_val, best_x