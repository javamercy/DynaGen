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
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Determine number of restarts
        n_restarts = max(1, budget // (10 * dim))
        n_restarts = min(n_restarts, 10)
        evals_per_restart = budget // n_restarts
        leftover = budget - n_restarts * evals_per_restart

        for restart in range(n_restarts):
            evals_used = 0
            # Starting point: best so far or random
            if best_x is not None and rng.rand() < 0.5:
                x = best_x.copy() + 0.05 * rng.uniform(-1, 1, dim) * (ub - lb)
                x = np.clip(x, lb, ub)
            else:
                x = rng.uniform(lb, ub)
            f_x = evaluate(x)
            evals_used += 1
            local_best_val = f_x
            local_best_x = x.copy()
            step = 0.1 * (ub - lb)
            min_step = 1e-10 * (ub - lb)
            no_improve = 0
            max_no_improve = max(2 * dim, 10)

            while evals_used < evals_per_restart and no_improve < max_no_improve and np.any(step > min_step):
                improved = False
                # Coordinate search with random order
                for d in rng.permutation(dim):
                    if evals_used >= evals_per_restart:
                        break
                    # Positive direction
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                    if x_new[d] != x[d]:
                        val_new = evaluate(x_new)
                        evals_used += 1
                        if val_new < local_best_val:
                            local_best_val = val_new
                            local_best_x = x_new.copy()
                            x = x_new.copy()
                            improved = True
                            continue
                    # Negative direction
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                    if x_new[d] != x[d]:
                        val_new = evaluate(x_new)
                        evals_used += 1
                        if val_new < local_best_val:
                            local_best_val = val_new
                            local_best_x = x_new.copy()
                            x = x_new.copy()
                            improved = True

                if improved:
                    # Pattern move: try to extend
                    direction = x - local_best_x
                    x_new = x + direction
                    x_new = np.clip(x_new, lb, ub)
                    if np.any(x_new != x) and evals_used < evals_per_restart:
                        val_new = evaluate(x_new)
                        evals_used += 1
                        if val_new < local_best_val:
                            local_best_val = val_new
                            local_best_x = x_new.copy()
                            x = x_new.copy()
                    step *= 1.1  # expand step slightly
                    no_improve = 0
                else:
                    step *= 0.5
                    x = local_best_x.copy()
                    no_improve += 1

            # Fill remaining evaluations in this restart with random points
            while evals_used < evals_per_restart:
                x_rand = rng.uniform(lb, ub)
                evaluate(x_rand)
                evals_used += 1

        # Use leftover evaluations
        for _ in range(leftover):
            x_rand = rng.uniform(lb, ub)
            evaluate(x_rand)

        return best_val, best_x