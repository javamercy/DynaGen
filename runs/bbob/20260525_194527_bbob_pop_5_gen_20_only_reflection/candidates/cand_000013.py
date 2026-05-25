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
        rang = ub - lb

        best_x = self.rng.uniform(lb, ub, size=self.dim)
        best_val = func(best_x)
        report_best(best_val, best_x)
        evals = 1

        step = 0.1 * rang  # initial step size
        stagnation = 0
        stagnation_limit = 7

        while evals < self.budget:
            if stagnation >= stagnation_limit:
                # restart with diversification: 70% uniform, 30% Gaussian around best
                if self.rng.uniform() < 0.7:
                    new_x = self.rng.uniform(lb, ub, size=self.dim)
                else:
                    sigma = 0.3 * rang
                    new_x = best_x + self.rng.normal(0, sigma, size=self.dim)
                    new_x = np.clip(new_x, lb, ub)
                new_val = func(new_x)
                evals += 1
                if new_val < best_val:
                    best_val, best_x = new_val, new_x
                    report_best(best_val, best_x)
                step = 0.1 * rang
                stagnation = 0
                continue

            # random direction perturbation
            direction = self.rng.normal(size=self.dim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                direction = self.rng.uniform(-1, 1, size=self.dim)
                norm = np.linalg.norm(direction)
            direction /= norm
            candidate = best_x + step * direction
            candidate = np.clip(candidate, lb, ub)
            candidate_val = func(candidate)
            evals += 1

            if candidate_val < best_val:
                # improvement: expand step and try line search extension (1 step)
                best_val, best_x = candidate_val, candidate
                report_best(best_val, best_x)
                step *= 1.2
                stagnation = 0
                # line search extension: try to continue in same direction
                if evals < self.budget:
                    extra_step = step * 1.5
                    extra_candidate = best_x + extra_step * direction
                    extra_candidate = np.clip(extra_candidate, lb, ub)
                    extra_val = func(extra_candidate)
                    evals += 1
                    if extra_val < best_val:
                        best_val, best_x = extra_val, extra_candidate
                        report_best(best_val, best_x)
                        step = extra_step
            else:
                step *= 0.7
                stagnation += 1

            step = np.clip(step, 0.01 * rang, 0.5 * rang)

        return best_val, best_x