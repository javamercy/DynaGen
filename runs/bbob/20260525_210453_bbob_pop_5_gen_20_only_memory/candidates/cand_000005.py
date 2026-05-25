import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        calls = 0

        best_x = None
        best_y = np.inf

        # fraction of budget for global sampling
        global_frac = 0.3
        n_global = max(1, min(int(budget * global_frac), budget - 1))

        # global sampling
        for _ in range(n_global):
            x = self.rng.uniform(lb, ub, size=dim)
            y = func(x)
            calls += 1
            if y < best_y:
                best_y = y
                best_x = x.copy()
                report_best(best_y, best_x)

        # local search phase
        sigma = 0.2 * (ub - lb).mean()  # initial step
        no_improve = 0
        max_no_improve = max(5, dim)

        while calls < budget:
            # generate candidate around best
            candidate = best_x + sigma * self.rng.normal(0, 1, size=dim)
            candidate = np.clip(candidate, lb, ub)
            y = func(candidate)
            calls += 1
            if y < best_y:
                best_y = y
                best_x = candidate.copy()
                no_improve = 0
                report_best(best_y, best_x)
                # optionally increase sigma?
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    sigma *= 0.5
                    no_improve = 0
                    # restart if sigma too small?
                    if sigma < 1e-8 * (ub - lb).mean():
                        # re-sample random point
                        x = self.rng.uniform(lb, ub, size=dim)
                        y = func(x)
                        calls += 1
                        if y < best_y:
                            best_y = y
                            best_x = x.copy()
                            report_best(best_y, best_x)
                        sigma = 0.2 * (ub - lb).mean()

        return best_y, best_x