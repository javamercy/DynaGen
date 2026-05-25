import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initialize
        current_x = rng.uniform(lb, ub, size=dim)
        current_f = func(current_x)
        budget -= 1
        best_x = current_x.copy()
        best_f = current_f
        report_best(best_f, best_x)

        if budget <= 0:
            return best_f, best_x

        # Simulated annealing parameters
        T0 = 1.0
        T = T0
        step_size = 0.1 * (ub - lb)  # scale per dimension
        cooling_rate = 1.0 - 1.0 / budget  # geometric decay to roughly T_end ~ T0*e^{-1}

        while budget > 0:
            # Generate neighbor
            perturbation = rng.normal(0, step_size, size=dim)
            new_x = current_x + perturbation
            new_x = np.clip(new_x, lb, ub)
            new_f = func(new_x)
            budget -= 1

            delta = new_f - current_f
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                current_x = new_x
                current_f = new_f
                if new_f < best_f:
                    best_x = new_x.copy()
                    best_f = new_f
                    report_best(best_f, best_x)

            # Cool temperature
            T *= cooling_rate
            if T < 1e-10:
                T = 1e-10

        return best_f, best_x