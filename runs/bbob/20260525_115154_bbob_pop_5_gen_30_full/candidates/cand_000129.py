import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_val = np.inf
        self.step_size = 0.1  # relative to bounds range
        self.contraction = 0.5
        self.expansion = 2.0
        self.max_no_improve = 2 * dim + 5
        self.no_improve_count = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # Initial random point
        x0 = self.rng.uniform(lb, ub)
        self.best_x = x0.copy()
        self.best_val = func(x0)
        report_best(self.best_val, self.best_x)
        evals = 1
        # Main loop
        while evals < self.budget:
            # Generate trial points
            trial_points = []
            # Coordinate steps
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
            # Random perturbation
            rand_dir = self.rng.normal(0, 1, self.dim)
            norm = np.linalg.norm(rand_dir)
            if norm > 0:
                rand_dir = rand_dir / norm * self.step_size * np.mean(ub - lb)
            xr = np.clip(self.best_x + rand_dir, lb, ub)
            trial_points.append(xr)
            # Limit to remaining budget
            remaining = self.budget - evals
            if len(trial_points) > remaining:
                trial_points = trial_points[:remaining]
            if not trial_points:
                break
            # Evaluate trial points
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
                self.no_improve_count = 0
            else:
                self.step_size *= self.contraction
                self.no_improve_count += 1
            # Restart if stagnation
            if self.no_improve_count >= self.max_no_improve:
                # Minimum evaluations for restart (1 point)
                if evals < self.budget - 1:
                    # Reset to random point
                    self.best_x = self.rng.uniform(lb, ub)
                    self.best_val = func(self.best_x)
                    evals += 1
                    report_best(self.best_val, self.best_x)
                    self.step_size = 0.1  # reset step size
                    self.no_improve_count = 0
                else:
                    break
            if self.step_size < 1e-15:
                break
        return self.best_val, self.best_x