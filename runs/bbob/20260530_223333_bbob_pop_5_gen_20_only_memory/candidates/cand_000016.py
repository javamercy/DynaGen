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

        # initial point
        x0 = rng.uniform(lb, ub)
        best_val = func(x0)
        best_x = x0.copy()
        report_best(best_val, best_x)
        evals = 1

        x = x0.copy()
        fx = best_val
        step = 0.1 * (ub - lb)
        min_step = 1e-10 * (ub - lb)
        no_improve_cycles = 0

        while evals < budget and np.any(step > min_step):
            # store state before cycle
            x_prev = x.copy()
            fx_prev = fx
            improved = False

            # coordinate pattern search
            for d in rng.permutation(dim):
                if evals >= budget:
                    break
                # positive direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    if val < fx:
                        fx = val
                        x = x_new.copy()
                        improved = True
                        step[d] *= 2.0
                        continue

                # negative direction
                x_new = x.copy()
                x_new[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                if x_new[d] != x[d]:
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                    if val < fx:
                        fx = val
                        x = x_new.copy()
                        improved = True
                        step[d] *= 2.0
                    else:
                        step[d] *= 0.5

            if improved:
                no_improve_cycles = 0
                # pattern move
                if evals < budget:
                    direction = x - x_prev
                    x_pattern = np.clip(x + direction, lb, ub)
                    if np.any(x_pattern != x):
                        val = func(x_pattern)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_pattern.copy()
                            report_best(best_val, best_x)
                        if val < fx:
                            fx = val
                            x = x_pattern.copy()

                # local random perturbation
                if evals < budget:
                    step_size = np.linalg.norm(step) / (2.0 * np.sqrt(dim))
                    dir = rng.randn(dim)
                    dir = dir / (np.linalg.norm(dir) + 1e-12)
                    x_rand = np.clip(x + step_size * dir, lb, ub)
                    if np.any(x_rand != x):
                        val = func(x_rand)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x_rand.copy()
                            report_best(best_val, best_x)
                        if val < fx:
                            fx = val
                            x = x_rand.copy()
            else:
                no_improve_cycles += 1
                step = np.clip(step * 0.5, min_step, None)
                if no_improve_cycles >= 5:
                    # reset to best point with smaller step
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_val
                        step = 0.01 * (ub - lb)
                    no_improve_cycles = 0

        # remaining budget: random search
        while evals < budget:
            x_rand = rng.uniform(lb, ub)
            val = func(x_rand)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_rand.copy()
                report_best(best_val, best_x)

        return best_val, best_x