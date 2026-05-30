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

        num_restarts = max(1, min(3, budget // (2 * dim)))
        if budget < 30:
            num_restarts = 1

        per_restart = budget // num_restarts
        remainder = budget % num_restarts

        best_val = float('inf')
        best_x = None

        for restart in range(num_restarts):
            alloc = per_restart + (1 if restart < remainder else 0)
            if alloc <= 0:
                continue

            restart_seed = self.seed + 1000 * restart
            restart_rng = np.random.RandomState(restart_seed)

            if best_x is not None and restart > 0:
                x = best_x.copy() + restart_rng.uniform(-0.05, 0.05, dim) * (ub - lb)
                x = np.clip(x, lb, ub)
            else:
                x = restart_rng.uniform(lb, ub)
            fx = func(x)
            evals = 1
            if fx < best_val:
                best_val = fx
                best_x = x.copy()
                report_best(best_val, best_x)

            step = 0.1 * (ub - lb)
            min_step = 1e-10 * (ub - lb)
            no_improve_counter = 0

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
                            step[d] *= 0.4

                if improved:
                    # Pattern move
                    direction = x - x0
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
                            # Extra pattern move
                            if evals < alloc:
                                x_pattern2 = x + direction
                                x_pattern2 = np.clip(x_pattern2, lb, ub)
                                if np.any(x_pattern2 != x):
                                    val_pattern2 = func(x_pattern2)
                                    evals += 1
                                    if val_pattern2 < best_val:
                                        best_val = val_pattern2
                                        best_x = x_pattern2.copy()
                                        report_best(best_val, best_x)
                                    if val_pattern2 < fx:
                                        fx = val_pattern2
                                        x = x_pattern2.copy()
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
                    step[:] = np.clip(step * 0.4, min_step, None)
                    if no_improve_counter >= 2:
                        perturb = restart_rng.uniform(-1, 1, dim) * step * 0.2
                        x_new = np.clip(x + perturb, lb, ub)
                        if np.any(x_new != x) and evals < alloc:
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
                                no_improve_counter = 0
                                step[:] = np.clip(step * 1.5, min_step, None)
                    if no_improve_counter >= 3:
                        x = best_x.copy() if best_x is not None else x0.copy()
                        fx = best_val
                        no_improve_counter = 0

            # Local refinement around best_x
            refine_step = 0.001 * (ub - lb)
            while evals < alloc:
                x_rand = best_x + restart_rng.uniform(-1, 1, dim) * refine_step
                x_rand = np.clip(x_rand, lb, ub)
                if np.any(x_rand != best_x):
                    val_rand = func(x_rand)
                    evals += 1
                    if val_rand < best_val:
                        best_val = val_rand
                        best_x = x_rand.copy()
                        report_best(best_val, best_x)
                else:
                    # If all within bounds and same point, just sample random to avoid waste
                    x_rand = restart_rng.uniform(lb, ub)
                    val_rand = func(x_rand)
                    evals += 1
                    if val_rand < best_val:
                        best_val = val_rand
                        best_x = x_rand.copy()
                        report_best(best_val, best_x)
                refine_step *= 0.9

        return best_val, best_x