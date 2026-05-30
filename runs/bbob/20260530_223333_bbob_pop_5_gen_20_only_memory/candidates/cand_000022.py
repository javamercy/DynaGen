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

        # Initial mean
        mean = rng.uniform(lb, ub)
        best_x = mean.copy()
        best_val = func(mean)
        evals = 1
        report_best(best_val, best_x)

        # ES parameters
        sigma = 0.1 * np.mean(ub - lb)
        mu = 4
        lam = 8
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights = weights / np.sum(weights)

        # Budget for ES (80%)
        es_budget = int(0.8 * budget)
        es_evals = evals
        max_generations = (es_budget - es_evals) // lam

        for gen in range(max_generations):
            if evals >= es_budget:
                break
            offspring = rng.normal(loc=mean, scale=sigma, size=(lam, dim))
            offspring = np.clip(offspring, lb, ub)
            vals = np.full(lam, np.inf)
            for i in range(lam):
                if evals >= budget:
                    break
                vals[i] = func(offspring[i])
                evals += 1
                if vals[i] < best_val:
                    best_val = vals[i]
                    best_x = offspring[i].copy()
                    report_best(best_val, best_x)
            if evals >= budget:
                break

            idx = np.argsort(vals)
            selected = offspring[idx[:mu]]
            mean_new = np.dot(weights, selected)
            mean_new = np.clip(mean_new, lb, ub)
            mean = mean_new

            prev_best = best_val
            # Check if any improvement in this generation
            success = (vals[idx[0]] < prev_best)  # since best_val already updated
            if success:
                sigma *= 1.2
            else:
                sigma /= 1.2
            sigma = max(sigma, 1e-10 * np.mean(ub - lb))

        # Local search phase
        sigma_local = 0.01 * np.mean(ub - lb)
        remaining = budget - evals
        for i in range(remaining):
            # Decay sigma linearly
            current_sigma = sigma_local * (1 - i / remaining) if remaining > 0 else 0
            x = best_x + current_sigma * rng.randn(dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        return best_val, best_x