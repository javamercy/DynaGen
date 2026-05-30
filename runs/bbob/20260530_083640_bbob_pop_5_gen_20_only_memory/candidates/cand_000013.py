import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.best_x = None
        self.best_value = np.inf
        self.num_calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial point
        x0 = np.random.uniform(lb, ub)
        val0 = func(x0)
        self.num_calls = 1
        self.best_x = x0.copy()
        self.best_value = val0
        report_best(val0, x0)
        
        range_max = np.max(ub - lb)
        radius = 0.5 * range_max
        min_radius = 1e-6 * range_max
        decay = 0.9
        no_improve_steps = 0
        restart_threshold = max(1, int(0.1 * self.budget))
        
        while self.num_calls < self.budget:
            if no_improve_steps >= restart_threshold:
                # restart
                candidate = np.random.uniform(lb, ub)
                radius = 0.5 * range_max
                no_improve_steps = 0
                val = func(candidate)
                self.num_calls += 1
                if val < self.best_value:
                    self.best_value = val
                    self.best_x = candidate.copy()
                    report_best(val, candidate)
                    no_improve_steps = 0
                    # local search after improvement
                    self._local_search(func, lb, ub)
                else:
                    no_improve_steps += 1
                    radius = max(min_radius, radius * decay)
            else:
                # Gaussian perturbation
                candidate = self.best_x + radius * np.random.randn(self.dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                self.num_calls += 1
                if val < self.best_value:
                    self.best_value = val
                    self.best_x = candidate.copy()
                    report_best(val, candidate)
                    no_improve_steps = 0
                    # local search after improvement
                    self._local_search(func, lb, ub)
                else:
                    no_improve_steps += 1
                    radius = max(min_radius, radius * decay)
        
        return self.best_value, self.best_x

    def _local_search(self, func, lb, ub):
        # Refine current best with a few small Gaussian steps
        n_local_iters = 3
        n_local_samples = min(2 * self.dim, 10)
        local_radius = 0.1 * (np.max(ub - lb))  # fixed small radius relative to bounds
        for _ in range(n_local_iters):
            if self.num_calls >= self.budget:
                break
            # Sample points around current best
            candidates = self.best_x + local_radius * np.random.randn(n_local_samples, self.dim)
            candidates = np.clip(candidates, lb, ub)
            for i in range(n_local_samples):
                if self.num_calls >= self.budget:
                    break
                val = func(candidates[i])
                self.num_calls += 1
                if val < self.best_value:
                    self.best_value = val
                    self.best_x = candidates[i].copy()
                    report_best(val, candidates[i])
        return
}