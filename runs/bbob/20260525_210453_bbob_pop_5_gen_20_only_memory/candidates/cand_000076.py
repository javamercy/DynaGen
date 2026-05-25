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
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        # Initial parent
        parent = np.random.uniform(lb, ub, size=dim)
        parent_val = func(parent)
        best_val = parent_val
        best_x = parent.copy()
        fcalls = 1
        report_best(best_val, best_x)
        # Step size initialization (approx 0.2 of range)
        sigma = 0.2 * (ub - lb).mean()
        # Target success rate for 1/5 rule
        target_sr = 0.2
        # Adaptation parameters
        c = 0.817  # damping factor
        # Recent success history (for 1/5 rule) - use window of N steps
        N = max(1, int(dim / 2))
        successes = []
        while fcalls < budget:
            # Generate offspring
            offspring = parent + sigma * np.random.randn(dim)
            offspring = np.clip(offspring, lb, ub)
            offspring_val = func(offspring)
            fcalls += 1
            if offspring_val < parent_val:
                # Accept offspring
                parent = offspring
                parent_val = offspring_val
                successes.append(1)
                if offspring_val < best_val:
                    best_val = offspring_val
                    best_x = offspring.copy()
                    report_best(best_val, best_x)
            else:
                successes.append(0)
            # Adapt sigma based on average success rate over last N trials
            if len(successes) >= N:
                recent_successes = successes[-N:]
                sr = sum(recent_successes) / N
                # Rechenberg's 1/5 rule: adjust sigma
                sigma *= np.exp((sr - target_sr) / c)
        return best_val, best_x