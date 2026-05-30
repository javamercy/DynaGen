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

        def evaluate(x):
            nonlocal best_val, best_x
            x = np.clip(x, lb, ub)
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Initial per-coordinate step sizes
        step = 0.1 * (ub - lb)
        min_step = 1e-10 * (ub - lb)
        max_step = 0.5 * (ub - lb)

        # Exponential moving average of success
        success_ema = np.ones(dim) * 0.5
        alpha = 0.1

        # Start from random point
        x = rng.uniform(lb, ub)
        best_val = evaluate(x)
        best_x = x.copy()
        evals = 1

        no_improve = 0
        max_no_improve = max(2 * dim, 50)

        while evals < budget:
            improved = False
            perm = rng.permutation(dim)
            for d in perm:
                if evals >= budget:
                    break
                success = False
                # Positive direction
                x_new = x.copy()
                x_new[d] = x[d] + step[d]
                if x_new[d] <= ub[d]:
                    val = evaluate(x_new)
                    evals += 1
                    if val < best_val:
                        success = True
                        x = x_new
                        step[d] = min(step[d] * 1.2, max_step[d])
                        success_ema[d] = (1 - alpha) * success_ema[d] + alpha
                        # Pattern move
                        if evals < budget and best_x is not None:
                            direction = x - best_x
                            x_pat = x + direction
                            x_pat = np.clip(x_pat, lb, ub)
                            if np.any(x_pat != x):
                                val_pat = evaluate(x_pat)
                                evals += 1
                                if val_pat < best_val:
                                    x = x_pat
                if not success:
                    # Negative direction
                    x_new = x.copy()
                    x_new[d] = x[d] - step[d]
                    if x_new[d] >= lb[d]:
                        val = evaluate(x_new)
                        evals += 1
                        if val < best_val:
                            success = True
                            x = x_new
                            step[d] = min(step[d] * 1.2, max_step[d])
                            success_ema[d] = (1 - alpha) * success_ema[d] + alpha
                            # Pattern move
                            if evals < budget and best_x is not None:
                                direction = x - best_x
                                x_pat = x + direction
                                x_pat = np.clip(x_pat, lb, ub)
                                if np.any(x_pat != x):
                                    val_pat = evaluate(x_pat)
                                    evals += 1
                                    if val_pat < best_val:
                                        x = x_pat
                if not success:
                    # No improvement in both directions
                    success_ema[d] = (1 - alpha) * success_ema[d]
                    step[d] = max(step[d] * 0.8, min_step[d])
                else:
                    improved = True
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= max_no_improve:
                # Restart from random point
                x = rng.uniform(lb, ub)
                step = 0.1 * (ub - lb)
                success_ema = np.ones(dim) * 0.5
                val = evaluate(x)
                evals += 1
                no_improve = 0

        # Leftover evaluations with random sampling
        while evals < budget:
            x_rand = rng.uniform(lb, ub)
            evaluate(x_rand)
            evals += 1

        return best_val, best_x