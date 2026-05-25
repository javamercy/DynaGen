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
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initialize current point uniformly
        current = rng.uniform(lb, ub, size=dim)
        current_val = func(current)
        best_val = current_val
        best_x = current.copy()
        evaluations = 1
        report_best(best_val, best_x)

        # Simulated annealing parameters
        T0 = 1.0
        T = T0
        cooling_rate = 5.0 / budget
        step0 = 0.5 * (ub - lb)  # half-domain step size
        step = step0.copy()

        while evaluations < budget:
            # Generate candidate via Cauchy perturbation
            cauchy_scale = step * T / T0
            perturbation = cauchy_scale * rng.standard_cauchy(size=dim)
            candidate = np.clip(current + perturbation, lb, ub)
            candidate_val = func(candidate)
            evaluations += 1
            delta = candidate_val - current_val

            # Acceptance criterion
            if delta < 0:
                current = candidate
                current_val = candidate_val
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
            else:
                if rng.random() < np.exp(-delta / T):
                    current = candidate
                    current_val = candidate_val

            # Update temperature
            T = T0 * np.exp(-cooling_rate * evaluations)

        return best_val, best_x