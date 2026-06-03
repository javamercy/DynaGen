import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial feasible point
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)

        current_x = best_x.copy()
        current_val = best_val

        # Hyperparameters
        step_size = 0.2 * np.mean(ub - lb)
        temp = 1.0
        temp_decay = 0.99
        success_ratio_target = 0.2
        success_ratio_window = 20
        success_count = 0
        no_improve_count = 0
        restart_threshold = max(5, int(budget * 0.1))

        # Cooling schedule: linear or geometric? Use geometric.
        while calls < budget:
            # Propose new point using Cauchy distribution
            noise = np.random.standard_cauchy(dim)
            new_x = current_x + step_size * noise
            new_x = np.clip(new_x, lb, ub)
            new_val = func(new_x)
            calls += 1

            delta = new_val - current_val
            # Metropolis acceptance
            if delta < 0 or np.random.uniform() < np.exp(-delta / (temp + 1e-100)):
                current_x = new_x
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_x = new_x.copy()
                    report_best(best_val, best_x)
                    no_improve_count = 0
                else:
                    success_count += 1
            else:
                # Not accepted, count as failure for step adaptation
                pass

            # Step size adaptation via 1/5 rule (over window)
            if calls % success_ratio_window == 0:
                ratio = success_count / success_ratio_window
                if ratio > success_ratio_target:
                    step_size *= 1.2
                else:
                    step_size *= 0.85
                success_count = 0

            # Temperature decay
            temp *= temp_decay

            # Restart if no improvement for too long
            if (best_val == current_val and calls < budget):
                no_improve_count += 1
            else:
                no_improve_count = 0

            if no_improve_count >= restart_threshold and calls < budget:
                # Restart from random point
                current_x = np.random.uniform(lb, ub, dim)
                current_val = func(current_x)
                calls += 1
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
                    report_best(best_val, best_x)
                # Reset step size and temperature
                step_size = 0.2 * np.mean(ub - lb)
                temp = 1.0
                no_improve_count = 0
                success_count = 0

        return best_val, best_x