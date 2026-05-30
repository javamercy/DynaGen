import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        # Population sizes
        mu = max(3, min(20, 2 * dim))
        lam = 5 * mu

        # Initial step size
        sigma = 0.2 * (ub - lb)

        # Restart parameters
        max_stagnation = 15

        best_val = np.inf
        best_x = None
        evals = 0

        # Initial population
        pop = np.empty((mu, dim))
        pop_vals = np.empty(mu)
        for i in range(mu):
            if evals >= budget:
                break
            x = rng.uniform(lb, ub, size=dim)
            val = func(x)
            evals += 1
            pop[i] = x
            pop_vals[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # If budget exhausted, return
        if evals >= budget:
            return best_val, best_x

        # Main generation loop
        stagnation_count = 0
        while evals + lam <= budget:
            # Generate offspring
            offspring = np.empty((lam, dim))
            offspring_vals = np.empty(lam)
            for j in range(lam):
                # Select two distinct parents
                parents = rng.choice(mu, size=2, replace=False)
                p1 = pop[parents[0]]
                p2 = pop[parents[1]]
                # Intermediate recombination
                x = (p1 + p2) / 2.0
                # Mutation
                x += sigma * rng.randn(dim)
                x = np.clip(x, lb, ub)
                offspring[j] = x

            # Evaluate offspring
            for j in range(lam):
                if evals >= budget:
                    break
                val = func(offspring[j])
                evals += 1
                offspring_vals[j] = val
                if val < best_val:
                    best_val = val
                    best_x = offspring[j].copy()
                    report_best(best_val, best_x)

            # Select best mu offspring (mu, lambda)
            idx = np.argsort(offspring_vals)[:mu]
            pop = offspring[idx]
            pop_vals = offspring_vals[idx]

            # Check for improvement in this generation
            if offspring_vals[idx[0]] < best_val:  # slightly re-check
                # improvement already caught above
                stagnation_count = 0
                sigma = np.clip(sigma * 1.1, 1e-6 * (ub - lb), 0.5 * (ub - lb))
            else:
                stagnation_count += 1
                sigma = np.clip(sigma * 0.9, 1e-6 * (ub - lb), 0.5 * (ub - lb))

            # Restart if stagnation
            if stagnation_count >= max_stagnation and evals + mu <= budget:
                # Re-initialize population
                for i in range(mu):
                    x = rng.uniform(lb, ub, size=dim)
                    val = func(x)
                    evals += 1
                    pop[i] = x
                    pop_vals[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                sigma = 0.2 * (ub - lb)
                stagnation_count = 0

        # Use remaining budget for random perturbations
        while evals < budget:
            x = best_x + sigma * rng.randn(dim)
            x = np.clip(x, lb, ub)
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        return best_val, best_x