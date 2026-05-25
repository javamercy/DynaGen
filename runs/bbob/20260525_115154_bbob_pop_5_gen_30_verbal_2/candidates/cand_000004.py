import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_val = np.inf
        self.step_size = 0.1
        self.contraction = 0.5
        self.expansion = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        x0 = self.rng.uniform(lb, ub)
        self.best_x = x0.copy()
        self.best_val = func(x0)
        report_best(self.best_val, self.best_x)
        evals = 1
        while evals < self.budget:
            trial_points = []
            for i in range(self.dim):
                step = self.step_size * (ub[i] - lb[i])
                xp = self.best_x.copy()
                xp[i] += step
                xp = np.clip(xp, lb, ub)
                trial_points.append(xp)
                xn = self.best_x.copy()
                xn[i] -= step
                xn = np.clip(xn, lb, ub)
                trial_points.append(xn)
            rand_dir = self.rng.normal(0, 1, self.dim)
            rand_dir = rand_dir / np.linalg.norm(rand_dir) * self.step_size * np.mean(ub - lb)
            xr = np.clip(self.best_x + rand_dir, lb, ub)
            trial_points.append(xr)
            remaining = self.budget - evals
            if len(trial_points) > remaining:
                trial_points = trial_points[:remaining]
            if not trial_points:
                break
            best_trial_val = np.inf
            best_trial_x = None
            for xt in trial_points:
                evals += 1
                val = func(xt)
                if val < best_trial_val:
                    best_trial_val = val
                    best_trial_x = xt.copy()
                if evals >= self.budget:
                    break
            if best_trial_val < self.best_val:
                self.best_val = best_trial_val
                self.best_x = best_trial_x
                report_best(self.best_val, self.best_x)
                self.step_size *= self.expansion
            else:
                self.step_size *= self.contraction
            if self.step_size < 1e-15:
                break
        return self.best_val, self.best_x