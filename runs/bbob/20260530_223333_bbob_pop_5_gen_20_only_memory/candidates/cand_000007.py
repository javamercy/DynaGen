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
        min_evals_per_restart = max(2 * dim, 10)
        num_restarts = max(1, budget // min_evals_per_restart)
        num_restarts = min(num_restarts, 10)  # cap to 10 restarts
        evals_per_restart = budget // num_restarts
        evals_remaining = budget

        for restart in range(num_restarts):
            if evals_remaining <= 0:
                break
            if restart < num_restarts - 1:
                restart_budget = evals_per_restart
            else:
                restart_budget = evals_remaining
            evals_remaining -= restart_budget

            # Initialize restart
            x = rng.uniform(lb, ub)
            local_best_val = evaluate(x)
            local_best_x = x.copy()
            evals_used = 1

            step = 0.1 * (ub - lb)
            min_step = 1e-10 * (ub - lb)
            prev_x = x.copy()
            improved_any = False

            while evals_used < restart_budget and np.any(step > min_step):
                improved = False
                order = rng.permutation(dim)
                for d in order:
                    if evals_used >= restart_budget:
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
                    # Pattern move: extend in direction of improvement
                    direction = x - prev_x
                    x_pattern = x + direction
                    x_pattern = np.clip(x_pattern, lb, ub)
                    if np.any(x_pattern != x):
                        val_pattern = evaluate(x_pattern)
                        evals_used += 1
                        if val_pattern < local_best_val:
                            local_best_val = val_pattern
                            local_best_x = x_pattern.copy()
                            x = x_pattern.copy()
                            improved = True
                    prev_x = x.copy()
                    improved_any = True
                else:
                    # No improvement: shrink step
                    step *= 0.5
                    # Reset to best point found in this restart (local best)
                    x = local_best_x.copy()
                    prev_x = x.copy()

            # Optionally, after restart, if any budget remains, do random sampling within restart_budget? Already accounted. But for safety, we can fill remaining evaluations with random points.
            while evals_used < restart_budget:
                x_rand = rng.uniform(lb, ub)
                evaluate(x_rand)
                evals_used += 1

        # Final random sampling for any remaining budget
        while evals_remaining > 0:
            x_rand = rng.uniform(lb, ub)
            evaluate(x_rand)
            evals_remaining -= 1

        return best_val, best_x