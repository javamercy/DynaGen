import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        # initial parent
        x = lb + (ub - lb) * self.rng.rand(self.dim)
        fx = func(x)
        evals = 1
        self.best_val = fx
        self.best_x = x.copy()
        report_best(self.best_val, self.best_x)
        # initial step size
        sigma = (ub - lb).mean() / 5.0
        if sigma == 0:
            sigma = 1.0
        while evals < self.budget:
            # generate offspring via Gaussian mutation
            z = self.rng.randn(self.dim)
            offspring = x + sigma * z
            offspring = np.clip(offspring, lb, ub)
            foff = func(offspring)
            evals += 1
            if foff < self.best_val:
                self.best_val = foff
                self.best_x = offspring.copy()
                report_best(self.best_val, self.best_x)
            # selection and step-size adaptation
            if foff <= fx:
                x = offspring
                fx = foff
                sigma *= np.exp(1.0 / self.dim)
            else:
                sigma *= np.exp(-1.0 / (2.0 * self.dim))
            # bound sigma
            sigma = max(sigma, 1e-9)
            sigma = min(sigma, (ub - lb).max() * 2.0)
        return self.best_val, self.best_x