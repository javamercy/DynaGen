import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_value = np.inf

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        calls = 0

        # Initial point
        x = self.rng.uniform(lb, ub, size=dim)
        fx = func(x)
        calls += 1
        self.best_value = fx
        self.best_x = x.copy()
        report_best(fx, x)

        step = (ub - lb) / 4.0
        min_step = 1e-12
        max_step = (ub - lb) / 2.0
        improved = True
        pattern = np.zeros(dim)

        while calls < budget:
            if not improved:
                step = step / 3.0
                if step.max() < min_step:
                    break
                pattern = np.zeros(dim)
            else:
                step = np.minimum(step * 1.5, max_step)
            improved = False

            # Coordinate search
            for i in range(dim):
                if calls >= budget:
                    break
                # positive step
                x_new = x.copy()
                x_new[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                fx_new = func(x_new)
                calls += 1
                if fx_new < self.best_value:
                    self.best_value = fx_new
                    self.best_x = x_new.copy()
                    report_best(fx_new, x_new)
                    x = x_new
                    fx = fx_new
                    improved = True
                    pattern = np.zeros(dim)
                    pattern[i] = step[i]
                    while calls < budget:
                        x_next = x.copy()
                        x_next[i] = np.clip(x[i] + step[i], lb[i], ub[i])
                        if x_next[i] == x[i]:
                            break
                        fx_next = func(x_next)
                        calls += 1
                        if fx_next < self.best_value:
                            self.best_value = fx_next
                            self.best_x = x_next.copy()
                            report_best(fx_next, x_next)
                            x = x_next
                            fx = fx_next
                            improved = True
                            pattern[i] += step[i]
                        else:
                            break
                    continue

                # negative step
                x_new[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                fx_new = func(x_new)
                calls += 1
                if fx_new < self.best_value:
                    self.best_value = fx_new
                    self.best_x = x_new.copy()
                    report_best(fx_new, x_new)
                    x = x_new
                    fx = fx_new
                    improved = True
                    pattern = np.zeros(dim)
                    pattern[i] = -step[i]
                    while calls < budget:
                        x_next = x.copy()
                        x_next[i] = np.clip(x[i] - step[i], lb[i], ub[i])
                        if x_next[i] == x[i]:
                            break
                        fx_next = func(x_next)
                        calls += 1
                        if fx_next < self.best_value:
                            self.best_value = fx_next
                            self.best_x = x_next.copy()
                            report_best(fx_next, x_next)
                            x = x_next
                            fx = fx_next
                            improved = True
                            pattern[i] -= step[i]
                        else:
                            break

            # Pattern move
            if improved and np.any(pattern != 0):
                x_pattern = x + 2 * pattern
                x_pattern = np.clip(x_pattern, lb, ub)
                if not np.array_equal(x_pattern, x) and calls < budget:
                    fx_pattern = func(x_pattern)
                    calls += 1
                    if fx_pattern < self.best_value:
                        self.best_value = fx_pattern
                        self.best_x = x_pattern.copy()
                        report_best(fx_pattern, x_pattern)
                        x = x_pattern
                        fx = fx_pattern
                        improved = True
                        while calls < budget:
                            x_next = x + pattern
                            x_next = np.clip(x_next, lb, ub)
                            if np.array_equal(x_next, x):
                                break
                            fx_next = func(x_next)
                            calls += 1
                            if fx_next < self.best_value:
                                self.best_value = fx_next
                                self.best_x = x_next.copy()
                                report_best(fx_next, x_next)
                                x = x_next
                                fx = fx_next
                                improved = True
                            else:
                                break
                    else:
                        improved = False

        # Final local random refinement
        if calls < budget:
            radius = (ub - lb) * 0.1
            while calls < budget:
                x_perturb = self.best_x + self.rng.uniform(-radius, radius, size=dim)
                x_perturb = np.clip(x_perturb, lb, ub)
                fx_perturb = func(x_perturb)
                calls += 1
                if fx_perturb < self.best_value:
                    self.best_value = fx_perturb
                    self.best_x = x_perturb.copy()
                    report_best(fx_perturb, x_perturb)
                else:
                    radius *= 0.99  # shrink
                if radius.max() < 1e-12:
                    break

        return self.best_value, self.best_x