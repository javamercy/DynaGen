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

        # Handle degenerate budget
        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initial point
        current_x = rng.uniform(lb, ub, size=dim)
        current_f = func(current_x)
        budget -= 1

        best_x = current_x.copy()
        best_f = current_f
        report_best(best_f, best_x)

        # SA parameters
        T0 = 1.0
        T = T0
        cooling_rate = 1.0 / budget if budget > 0 else 0.01  # linear cooling
        step_scale = 0.2 * (ub - lb)

        stagnation_limit = max(1, budget // 10)
        no_improve = 0

        while budget > 0:
            # Generate candidate
            step = step_scale * T / T0
            step = np.maximum(step, 1e-10 * (ub - lb))
            perturbation = rng.uniform(-step, step, size=dim)
            new_x = np.clip(current_x + perturbation, lb, ub)
            new_f = func(new_x)
            budget -= 1

            # Metropolis acceptance
            delta = new_f - current_f
            if delta < 0 or rng.rand() < np.exp(-delta / max(T, 1e-10)):
                current_x = new_x
                current_f = new_f
                if new_f < best_f:
                    best_x = new_x.copy()
                    best_f = new_f
                    report_best(best_f, best_x)
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

            # Cooling
            T = T0 * (1 - (self.budget - budget) / self.budget)
            T = max(T, 1e-10)

            # Restart if stagnation
            if no_improve >= stagnation_limit and budget > 0:
                current_x = rng.uniform(lb, ub, size=dim)
                current_f = func(current_x)
                budget -= 1
                no_improve = 0
                if current_f < best_f:
                    best_x = current_x.copy()
                    best_f = current_f
                    report_best(best_f, best_x)

        return best_f, best_x