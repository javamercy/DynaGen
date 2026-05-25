import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initialize parent
        parent = np.random.uniform(lb, ub, size=dim)
        parent_f = func(parent)
        fcalls = 1
        best_x = parent.copy()
        best_f = parent_f
        report_best(best_f, best_x)
        # step size
        sigma = 0.2 * (ub - lb).mean()  # initial step size
        # success rate tracking
        window_size = min(20, max(1, budget // 4))
        successes = []
        while fcalls < budget:
            # generate offspring
            offspring = parent + sigma * np.random.randn(dim)
            offspring = np.clip(offspring, lb, ub)
            offspring_f = func(offspring)
            fcalls += 1
            if offspring_f < parent_f:
                success = 1
                parent = offspring
                parent_f = offspring_f
                if parent_f < best_f:
                    best_f = parent_f
                    best_x = parent.copy()
                    report_best(best_f, best_x)
            else:
                success = 0
            successes.append(success)
            if len(successes) > window_size:
                successes.pop(0)
            # adapt sigma
            if len(successes) >= window_size:
                success_rate = sum(successes) / window_size
                if success_rate > 0.2:
                    sigma *= 1.05
                else:
                    sigma /= 1.05
        return best_f, best_x