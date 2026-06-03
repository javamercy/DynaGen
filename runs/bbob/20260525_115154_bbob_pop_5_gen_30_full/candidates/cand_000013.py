import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.best_x = None
        self.best_val = np.inf
        self.current_x = None
        self.current_val = np.inf
        self.step_size = 0.1
        self.contraction = 0.5
        self.expansion = 2.0
        self.no_improve_iter = 0
        self.restart_threshold = max(1, dim * 5)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial random point
        x0 = self.rng.uniform(lb, ub)
        self.current_x = x0.copy()
        self.current_val = func(x0)
        self.best_x = x0.copy()
        self.best_val = self.current_val
        report_best(self.best_val, self.best_x)
        evals = 1

        while evals < self.budget:
            # generate trial points around current_x
            trial_points = []
            for i in range(self.dim):
                step = self.step_size * (ub[i] - lb[i])
                xp = self.current_x.copy()
                xp[i] += step
                xp = np.clip(xp, lb, ub)
                trial_points.append(xp)
                xn = self.current_x.copy()
                xn[i] -= step
                xn = np.clip(xn, lb, ub)
                trial_points.append(xn)
            rand_dir = self.rng.normal(0, 1, self.dim)
            rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-15)
            rand_dir *= self.step_size * np.mean(ub - lb)
            xr = np.clip(self.current_x + rand_dir, lb, ub)
            trial_points.append(xr)

            # limit to remaining budget
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
            if best_trial_val < self.current_val:
                self.current_val = best_trial_val
                self.current_x = best_trial_x
                self.no_improve_iter = 0
                self.step_size *= self.expansion
                if best_trial_val < self.best_val:
                    self.best_val = best_trial_val
                    self.best_x = best_trial_x
                    report_best(self.best_val, self.best_x)
            else:
                self.step_size *= self.contraction
                self.no_improve_iter += 1

            # check restart condition
            if self.no_improve_iter >= self.restart_threshold and evals < self.budget - 1:
                xnew = self.rng.uniform(lb, ub)
                self.current_x = xnew.copy()
                self.current_val = func(xnew)
                evals += 1
                self.step_size = 0.2  # larger step on restart
                self.no_improve_iter = 0
                if self.current_val < self.best_val:
                    self.best_val = self.current_val
                    self.best_x = self.current_x.copy()
                    report_best(self.best_val, self.best_x)

            if self.step_size < 1e-15:
                break

        return self.best_val, self.best_x