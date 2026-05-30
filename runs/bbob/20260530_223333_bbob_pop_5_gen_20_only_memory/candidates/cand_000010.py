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

        # Number of restarts: at least 1, at most 5, based on budget
        num_restarts = max(1, min(5, budget // (2 * dim + 10)))
        if budget < 50:
            num_restarts = 1

        # Split budget among restarts
        per_restart = budget // num_restarts
        remainder = budget % num_restarts

        best_val = float('inf')
        best_x = None

        start_idx = 0
        for restart in range(num_restarts):
            # Allocate budget for this restart
            if restart < remainder:
                alloc = per_restart + 1
            else:
                alloc = per_restart
            if alloc <= 0:
                continue

            # Seed for this restart
            restart_seed = self.seed + 1000 * restart
            restart_rng = np.random.RandomState(restart_seed)

            # Initial point
            x0 = restart_rng.uniform(lb, ub)
            evals = 0
            x = x0.copy()
            fx = func(x)
            evals += 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
                report_best(best_val, best_x)

            # Step sizes
            step = 0.1 * (ub - lb)
            min_step = 1e-10 * (ub - lb)

            prev_x = x.copy()
            prev_val = fx

            while evals < alloc and np.any(step > min_step):
                improved = False
                order = restart_rng.permutation(dim)
                for d in order:
                    if evals >= alloc:
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

                if improved:
                    direction = x - prev_x
                    x_pattern = x + direction
                    x_pattern = np.clip(x_pattern, lb, ub)
                    if np.any(x_pattern != x) and evals < alloc:
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
                    step *= 0.5
                    x = best_x.copy() if best_x is not None else x0.copy()
                    prev_x = x.copy()
                    prev_val = fx if fx is not None else np.inf

            # If any budget remains, random search
            while evals < alloc:
                x_rand = restart_rng.uniform(lb, ub)
                val_rand = func(x_rand)
                evals += 1
                if val_rand < best_val:
                    best_val = val_rand
                    best_x = x_rand.copy()
                    report_best(best_val, best_x)

        return best_val, best_x