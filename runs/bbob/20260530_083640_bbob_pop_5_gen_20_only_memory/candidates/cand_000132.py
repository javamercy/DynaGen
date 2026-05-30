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
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial point
        x_curr = lb + (ub - lb) * rng.rand(dim)
        best_x = x_curr.copy()
        best_val = func(x_curr)
        calls = 1
        report_best(best_val, best_x)

        # Simulated Annealing parameters
        T_initial = 1.0  # will be scaled based on initial function value? Use fixed
        T = T_initial
        cooling_rate = 0.95
        sigma = 0.2 * (ub - lb)  # initial step size per dimension

        # Adaptation parameters
        window_size = 10
        accept_history = []

        # Reheat parameters
        no_improve_count = 0
        no_improve_threshold = 20

        while calls < budget:
            # Propose new point
            x_new = x_curr + sigma * rng.randn(dim)
            x_new = np.clip(x_new, lb, ub)
            f_new = func(x_new)
            calls += 1
            delta = f_new - best_val  # actually delta from current? Use current

            # Acceptance criterion
            if f_new < best_val:
                best_val = f_new
                best_x = x_new.copy()
                report_best(best_val, best_x)
                x_curr = x_new.copy()
                accept_history.append(1)
                no_improve_count = 0
            else:
                if rng.rand() < np.exp(-(f_new - best_val) / T):
                    x_curr = x_new.copy()
                    accept_history.append(1)
                else:
                    accept_history.append(0)
                no_improve_count += 1

            # Temperature decrease
            T *= cooling_rate

            # Adapt sigma (1/5 rule)
            if len(accept_history) >= window_size:
                recent_accept = np.mean(accept_history[-window_size:])
                target_accept = 0.5
                if recent_accept > target_accept:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                # Clamp sigma to avoid extreme values
                sigma = np.clip(sigma, 1e-6 * (ub - lb), 0.5 * (ub - lb))

            # Reheat if stuck
            if no_improve_count >= no_improve_threshold:
                T = 0.1 * T_initial
                no_improve_count = 0

            # Safety: prevent infinite loop if budget runs out inside loop
            if calls >= budget:
                break

        return best_val, best_x