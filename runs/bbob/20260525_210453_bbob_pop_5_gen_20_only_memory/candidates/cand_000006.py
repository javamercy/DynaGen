import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        calls = 0
        best_x = None
        best_y = np.inf

        # Phase 1: global random sampling (30% of budget)
        n_global = max(1, int(0.3 * self.budget))
        for _ in range(n_global):
            if calls >= self.budget:
                break
            x = self.rng.uniform(lb, ub)
            y = func(x)
            calls += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
                report_best(best_y, best_x)

        if calls >= self.budget:
            return best_y, best_x

        # Phase 2: adaptive coordinate pattern search
        x = best_x.copy()
        step = 0.1 * (ub - lb)
        min_step = 1e-8 * (ub - lb)
        no_improve_coord = 0
        max_no_improve_coord = max(5, self.dim)

        while calls < self.budget:
            # coordinate search cycle
            x_prev = x.copy()
            improved = False
            for i in range(self.dim):
                if calls >= self.budget:
                    break
                # try positive step
                x_new = x.copy()
                x_new[i] += step[i]
                x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                y_new = func(x_new)
                calls += 1
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                    report_best(best_y, best_x)
                    x = x_new
                    step[i] *= 1.2
                    improved = True
                    continue
                # try negative step
                x_new = x.copy()
                x_new[i] -= step[i]
                x_new[i] = np.clip(x_new[i], lb[i], ub[i])
                y_new = func(x_new)
                calls += 1
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new.copy()
                    report_best(best_y, best_x)
                    x = x_new
                    step[i] *= 1.2
                    improved = True
                    continue
                # no improvement on this coordinate
                step[i] *= 0.9
                if step[i] < min_step[i]:
                    step[i] = min_step[i]

            # pattern move if improved during cycle
            if improved and calls < self.budget:
                direction = x - x_prev
                if np.linalg.norm(direction) > 0:
                    factor = 1.0
                    x_pattern = x + factor * direction
                    x_pattern = np.clip(x_pattern, lb, ub)
                    y_pattern = func(x_pattern)
                    calls += 1
                    if y_pattern < best_y:
                        best_y = y_pattern
                        best_x = x_pattern.copy()
                        report_best(best_y, best_x)
                        x = x_pattern
            else:
                no_improve_coord += 1
                if no_improve_coord >= max_no_improve_coord:
                    # Gaussian perturbation around best
                    sigma = 0.2 * (ub - lb).mean()
                    candidate = best_x + sigma * self.rng.normal(0, 1, size=self.dim)
                    candidate = np.clip(candidate, lb, ub)
                    if calls < self.budget:
                        y_cand = func(candidate)
                        calls += 1
                        if y_cand < best_y:
                            best_y = y_cand
                            best_x = candidate.copy()
                            report_best(best_y, best_x)
                            x = candidate.copy()
                            step = 0.1 * (ub - lb)
                            no_improve_coord = 0
                        else:
                            step *= 0.5
                            no_improve_coord = 0
                    # restart if step too small
                    if np.max(step) < 1e-8 * (ub - lb).max():
                        x = self.rng.uniform(lb, ub)
                        if calls < self.budget:
                            y_restart = func(x)
                            calls += 1
                            if y_restart < best_y:
                                best_y = y_restart
                                best_x = x.copy()
                                report_best(best_y, best_x)
                        step = 0.1 * (ub - lb)
                        no_improve_coord = 0

        return best_y, best_x