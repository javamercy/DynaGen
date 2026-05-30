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

        num_restarts = max(1, min(5, budget // (2 * dim + 10)))
        if budget < 50:
            num_restarts = 1

        per_restart = budget // num_restarts
        remainder = budget % num_restarts

        best_val = float('inf')
        best_x = None

        for restart in range(num_restarts):
            if restart < remainder:
                alloc = per_restart + 1
            else:
                alloc = per_restart
            if alloc <= 0:
                continue

            restart_seed = self.seed + 1000 * restart
            restart_rng = np.random.RandomState(restart_seed)

            x = restart_rng.uniform(lb, ub)
            evals = 0
            fx = func(x)
            evals += 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
                report_best(best_val, best_x)

            step = 0.1 * (ub - lb)
            min_step = 1e-12 * (ub - lb)
            prev_x = x.copy()
            prev_fx = fx

            while evals < alloc and np.any(step > min_step):
                improved_in_cycle = False
                order = restart_rng.permutation(dim)
                for d in order:
                    if evals >= alloc:
                        break
                    # positive direction
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
                            step[d] *= 1.2
                            improved_in_cycle = True
                            continue
                    # negative direction
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
                            step[d] *= 1.2
                            improved_in_cycle = True
                            continue
                    # no improvement
                    step[d] *= 0.5

                if improved_in_cycle:
                    # pattern move
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
                else:
                    # no improvement: reset to best and shrink steps
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_val if best_val != float('inf') else fx
                    else:
                        x = restart_rng.uniform(lb, ub)
                        fx = func(x)
                        evals += 1
                        if fx < best_val:
                            best_val = fx
                            best_x = x.copy()
                            report_best(best_val, best_x)
                    step *= 0.5
                    prev_x = x.copy()

            # random search for leftover budget
            while evals < alloc:
                x_rand = restart_rng.uniform(lb, ub)
                val_rand = func(x_rand)
                evals += 1
                if val_rand < best_val:
                    best_val = val_rand
                    best_x = x_rand.copy()
                    report_best(best_val, best_x)

        return best_val, best_x