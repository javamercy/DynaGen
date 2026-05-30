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

        # Initial random search: 10% of budget
        init_evals = max(1, budget // 10)
        best_val = float('inf')
        best_x = None
        for _ in range(init_evals):
            x = rng.uniform(lb, ub)
            val = func(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        evals = init_evals

        # Determine number of restarts for pattern search
        remaining = budget - evals
        num_restarts = max(1, min(10, remaining // (dim + 10)))
        if remaining < 50:
            num_restarts = 1
        per_restart = remaining // num_restarts
        remainder = remaining % num_restarts

        for restart in range(num_restarts):
            alloc = per_restart + (1 if restart < remainder else 0)
            if alloc <= 0:
                continue
            restart_seed = self.seed + 1000 * (restart + 1)
            restart_rng = np.random.RandomState(restart_seed)

            # Start pattern search from a random point
            x0 = restart_rng.uniform(lb, ub)
            x = x0.copy()
            fx = func(x)
            evals += 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
                report_best(best_val, best_x)

            step = 0.1 * (ub - lb)
            min_step = 1e-10 * (ub - lb)
            prev_x = x.copy()
            prev_val = fx

            while evals < budget and np.any(step > min_step):
                improved = False
                order = restart_rng.permutation(dim)
                for d in order:
                    if evals >= budget:
                        break
                    # Positive direction
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                    if x_new[d] != x[d]:
                        val_new = func(x_new)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        if val_new < fx:
                            fx = val_new
                            x = x_new.copy()
                            improved = True
                            step[d] *= 1.5
                            continue
                    # Negative direction
                    x_new = x.copy()
                    x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                    if x_new[d] != x[d]:
                        val_new = func(x_new)
                        evals += 1
                        if val_new < best_val:
                            best_val = val_new
                            best_x = x_new.copy()
                            report_best(best_val, best_x)
                        if val_new < fx:
                            fx = val_new
                            x = x_new.copy()
                            improved = True
                            step[d] *= 1.5
                        else:
                            step[d] *= 0.5

                if improved:
                    # Pattern move
                    direction = x - prev_x
                    x_pattern = x + direction
                    x_pattern = np.clip(x_pattern, lb, ub)
                    if np.any(x_pattern != x) and evals < budget:
                        val_pattern = func(x_pattern)
                        evals += 1
                        if val_pattern < best_val:
                            best_val = val_pattern
                            best_x = x_pattern.copy()
                            report_best(best_val, best_x)
                        if val_pattern < fx:
                            fx = val_pattern
                            x = x_pattern.copy()
                    prev_x = x.copy()
                    prev_val = fx
                else:
                    # No improvement in cycle: try a random point to escape
                    if evals < budget:
                        x_rand = restart_rng.uniform(lb, ub)
                        val_rand = func(x_rand)
                        evals += 1
                        if val_rand < best_val:
                            best_val = val_rand
                            best_x = x_rand.copy()
                            report_best(best_val, best_x)
                        if val_rand < fx:
                            fx = val_rand
                            x = x_rand.copy()
                            improved = True
                    if not improved:
                        step[:] = np.clip(step * 0.5, min_step, None)
                        # Restart from best if step becomes too small
                        if np.all(step <= min_step):
                            x = best_x.copy() if best_x is not None else x0.copy()
                            step = 0.1 * (ub - lb)

            # Random fill for any remaining budget in this restart
            while evals < budget:
                x_rand = restart_rng.uniform(lb, ub)
                val_rand = func(x_rand)
                evals += 1
                if val_rand < best_val:
                    best_val = val_rand
                    best_x = x_rand.copy()
                    report_best(best_val, best_x)

        return best_val, best_x