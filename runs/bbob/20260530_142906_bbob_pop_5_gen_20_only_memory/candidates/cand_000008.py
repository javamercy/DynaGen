import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point at center
        best_x = (lb + ub) / 2.0
        best_y = func(best_x)
        self.calls += 1
        report_best(best_y, best_x)
        
        # Phase 1: Coordinate pattern search
        phase1_budget = int(0.7 * self.budget)
        step = 0.1 * (ub - lb)
        while self.calls < phase1_budget:
            order = np.random.permutation(self.dim)
            improved = False
            for i in order:
                if self.calls >= self.budget:
                    break
                # positive step
                trial = np.copy(best_x)
                trial[i] += step[i]
                trial = np.clip(trial, lb, ub)
                y = func(trial)
                self.calls += 1
                if y < best_y:
                    best_x = trial
                    best_y = y
                    report_best(best_y, best_x)
                    step[i] *= 2.0
                    improved = True
                    continue
                # negative step
                trial = np.copy(best_x)
                trial[i] -= step[i]
                trial = np.clip(trial, lb, ub)
                y = func(trial)
                self.calls += 1
                if y < best_y:
                    best_x = trial
                    best_y = y
                    report_best(best_y, best_x)
                    step[i] *= 2.0
                    improved = True
                    continue
                step[i] *= 0.5
            if not improved:
                # Optionally restart, but keep simple
                pass
        
        # Phase 2: Local random search
        local_step = 0.05 * (ub - lb)
        while self.calls < self.budget:
            # random perturbation
            trial = best_x + np.random.randn(self.dim) * local_step
            trial = np.clip(trial, lb, ub)
            y = func(trial)
            self.calls += 1
            if y < best_y:
                best_x = trial
                best_y = y
                report_best(best_y, best_x)
                # optionally increase step? keep decreasing
            else:
                local_step *= 0.9  # shrink
        
        return best_y, best_x