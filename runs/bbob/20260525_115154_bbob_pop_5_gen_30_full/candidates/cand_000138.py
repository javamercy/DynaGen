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
        range_ = ub - lb
        n_calls = 0
        best_x = None
        best_y = np.inf

        # initial LHS
        n_initial = min(self.budget, max(self.dim * 2, self.budget // 2))
        if n_initial < 1:
            n_initial = 1
        lhs_points = self._lhs(lb, ub, n_initial)
        for x in lhs_points:
            if n_calls >= self.budget:
                break
            y = func(x)
            n_calls += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
                report_best(y, best_x)

        # local search parameters
        step_size = 0.1 * range_
        adaptation_period = max(1, self.dim)
        success_count = 0
        steps_in_period = 0
        last_improvement = 0
        restart_interval = max(5 * self.dim, 50)
        max_restarts = 5
        restarts_done = 0

        current_x = best_x.copy()
        current_y = best_y

        while n_calls < self.budget:
            # generate perturbation
            dx = self.rng.normal(0, step_size, size=self.dim)
            new_x = current_x + dx
            new_x = np.clip(new_x, lb, ub)
            y = func(new_x)
            n_calls += 1
            steps_in_period += 1

            if y < current_y:
                # improvement
                current_y = y
                current_x = new_x.copy()
                success_count += 1
                if y < best_y:
                    best_y = y
                    best_x = new_x.copy()
                    report_best(y, best_x)
                last_improvement = n_calls

            # adaptation period check
            if steps_in_period >= adaptation_period:
                success_rate = success_count / adaptation_period
                if success_rate > 0.2:
                    step_size = step_size * 1.2
                elif success_rate < 0.2:
                    step_size = step_size * 0.85
                # reset counters but keep step_size bounded
                step_size = np.clip(step_size, 1e-12 * range_, 0.5 * range_)
                success_count = 0
                steps_in_period = 0

            # restart condition
            if (n_calls - last_improvement >= restart_interval or
                np.all(step_size < 1e-12 * range_)) and restarts_done < max_restarts:
                if n_calls + 2 * self.dim < self.budget:
                    # generate new random starting point
                    current_x = lb + self.rng.uniform(0, 1, size=self.dim) * range_
                    current_y = np.inf
                    # evaluate new point (could be expensive, but necessary)
                    # but we already have best overall; we just set current for local search
                    # To save calls, we could skip evaluation and just set current_y large
                    # but we need to avoid evaluating unless we have budget
                    # We'll just set current_y large and continue without evaluation
                    # This is okay because we will perturb from there
                    step_size = 0.1 * range_
                    # reset adaptation counters
                    success_count = 0
                    steps_in_period = 0
                    last_improvement = n_calls  # reset timer
                    restarts_done += 1

        return best_y, best_x

    def _lhs(self, lb, ub, n):
        points = np.zeros((n, self.dim))
        for i in range(self.dim):
            strata = np.linspace(lb[i], ub[i], n + 1)[:-1]
            order = self.rng.permutation(n)
            points[:, i] = strata[order] + self.rng.uniform(0, (ub[i] - lb[i]) / n, size=n)
        return points