import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.default_rng(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial point
        current_x = rng.uniform(lb, ub, size=dim)
        current_val = func(current_x)
        best_x = current_x.copy()
        best_val = current_val
        calls = 1
        report_best(best_val, best_x)

        if budget <= 1:
            return best_val, best_x

        # SA parameters
        T = 1.0  # initial temperature, will be scaled later
        alpha = 0.99  # temperature decay factor
        step_size = 0.1 * (ub - lb)  # per-dimension step size
        accept_target = 0.2
        refresh_interval = max(10, min(100, budget // 10))
        num_accept = 0
        num_trials = 0
        no_improve = 0
        max_no_improve = max(5, int(budget * 0.05))

        while calls < budget:
            # Generate candidate
            candidate = current_x + rng.normal(0, step_size, size=dim)
            candidate = np.clip(candidate, lb, ub)
            candidate_val = func(candidate)
            calls += 1

            # Update best
            if candidate_val < best_val:
                best_val = candidate_val
                best_x = candidate.copy()
                report_best(best_val, best_x)
                no_improve = 0
            else:
                no_improve += 1

            # Acceptance criterion
            delta = candidate_val - current_val
            if delta < 0:
                accept = True
            else:
                # Metropolis acceptance
                try:
                    p = np.exp(-delta / T)
                except OverflowError:
                    p = 0.0
                accept = rng.uniform() < p

            if accept:
                current_x = candidate
                current_val = candidate_val
                num_accept += 1
            num_trials += 1

            # Temperature decay
            T *= alpha
            # Prevent T from getting too small
            if T < 1e-10:
                T = 1e-10

            # Step size adaptation periodically
            if num_trials >= refresh_interval:
                if num_trials > 0:
                    acceptance_rate = num_accept / num_trials
                    if acceptance_rate > accept_target:
                        step_size *= 1.1  # increase
                    else:
                        step_size *= 0.9  # decrease
                    # Clamp step size to reasonable bounds
                    step_size = np.clip(step_size, 1e-10 * (ub - lb), 10.0 * (ub - lb))
                num_accept = 0
                num_trials = 0

            # Restart if no improvement for too long
            if no_improve >= max_no_improve and calls < budget:
                # Reset to a random point
                current_x = rng.uniform(lb, ub, size=dim)
                current_val = func(current_x)
                calls += 1
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
                    report_best(best_val, best_x)
                # Reset temperature and step size
                T = 1.0
                step_size = 0.1 * (ub - lb)
                no_improve = 0
                num_accept = 0
                num_trials = 0

        return best_val, best_x