import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        budget = self.budget
        rng = self.rng
        dim = self.dim

        if budget < 4:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        # ES parameters
        mu = int(max(5, min(20, budget // 50)))
        lam = 2 * mu
        sigma = 0.5 * (ub - lb).mean() / np.sqrt(dim)

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(mu, dim)
        pop_fit = np.full(mu, np.inf)
        for i in range(mu):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        # Main generation loop
        while evals < budget:
            old_best = np.min(pop_fit)
            # Generate offspring
            lam_actual = min(lam, budget - evals)
            offspring = np.zeros((lam_actual, dim))
            offspring_fit = np.full(lam_actual, np.inf)
            for i in range(lam_actual):
                if evals >= budget:
                    break
                parent_idx = rng.randint(mu)
                parent = pop[parent_idx]
                child = parent + sigma * rng.randn(dim)
                child = np.clip(child, lb, ub)
                offspring[i] = child
                val = func(child)
                evals += 1
                offspring_fit[i] = val
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = child.copy()
                    report_best(self.best_val, self.best_x)
            # Truncation selection: best mu from combined pool
            num_off = lam_actual
            combined_pop = np.vstack([pop, offspring])
            combined_fit = np.concatenate([pop_fit, offspring_fit])
            best_idx = np.argsort(combined_fit)[:mu]
            pop = combined_pop[best_idx]
            pop_fit = combined_fit[best_idx]
            # Step-size adaptation (1/5 rule)
            if num_off > 0:
                successes = np.sum(offspring_fit < old_best)
                success_rate = successes / num_off
                if success_rate > 0.2:
                    sigma *= 1.22
                elif success_rate < 0.2:
                    sigma *= 0.82
                sigma = np.clip(sigma, 1e-8, (ub - lb).max())

        return self.best_val, self.best_x