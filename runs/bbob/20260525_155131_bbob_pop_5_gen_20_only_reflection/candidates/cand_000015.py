import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.best_x = None
        self.best_val = np.inf
        self.evals = 0
        self.step_sizes = None
        self.no_improve_count = 0
        self.stagnation_limit = max(5 * dim, 20)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial random point
        self.best_x = lb + self.rng.rand(self.dim) * (ub - lb)
        self.best_val = func(self.best_x)
        self.evals = 1
        report_best(self.best_val, self.best_x)
        # initialize per-coordinate step sizes as 10% of range
        self.step_sizes = 0.1 * (ub - lb)
        while self.evals < self.budget:
            improved_any = False
            for i in range(self.dim):
                if self.evals >= self.budget:
                    break
                step = self.step_sizes[i]
                # positive step
                x_new = self.best_x.copy()
                x_new[i] = np.clip(self.best_x[i] + step, lb[i], ub[i])
                val_new = func(x_new)
                self.evals += 1
                if val_new < self.best_val:
                    self.best_val = val_new
                    self.best_x[i] = x_new[i]
                    improved_any = True
                    self.step_sizes[i] *= 1.2
                    report_best(self.best_val, self.best_x)
                else:
                    if self.evals >= self.budget:
                        break
                    # negative step
                    x_new2 = self.best_x.copy()
                    x_new2[i] = np.clip(self.best_x[i] - step, lb[i], ub[i])
                    val_new2 = func(x_new2)
                    self.evals += 1
                    if val_new2 < self.best_val:
                        self.best_val = val_new2
                        self.best_x[i] = x_new2[i]
                        improved_any = True
                        self.step_sizes[i] *= 1.2
                        report_best(self.best_val, self.best_x)
                    else:
                        self.step_sizes[i] *= 0.5
                        # ensure minimum step size
                        self.step_sizes[i] = max(self.step_sizes[i], 1e-12 * (ub[i] - lb[i]))
            # parallel random probes for diversification
            if self.evals < self.budget:
                n_probes = min(5, self.budget - self.evals)
                candidates = []
                for _ in range(n_probes):
                    if self.evals >= self.budget:
                        break
                    # random direction scaled by current step sizes
                    perturb = self.step_sizes * self.rng.randn(self.dim)
                    x_probe = np.clip(self.best_x + perturb, lb, ub)
                    val_probe = func(x_probe)
                    self.evals += 1
                    candidates.append((val_probe, x_probe))
                # accept best if improvement
                best_candidate = min(candidates, key=lambda x: x[0])
                if best_candidate[0] < self.best_val:
                    self.best_val = best_candidate[0]
                    self.best_x = best_candidate[1].copy()
                    improved_any = True
                    report_best(self.best_val, self.best_x)
            # stagnation detection and restart
            if improved_any:
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.stagnation_limit and self.evals < self.budget:
                    # restart from new random point
                    self.best_x = lb + self.rng.rand(self.dim) * (ub - lb)
                    self.best_val = func(self.best_x)
                    self.evals += 1
                    report_best(self.best_val, self.best_x)
                    self.step_sizes = 0.1 * (ub - lb)
                    self.no_improve_count = 0
        return self.best_val, self.best_x