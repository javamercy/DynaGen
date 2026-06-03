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
        n = self.dim
        budget = self.budget

        # Population size heuristic
        npop = max(4, min(20, budget // (n * 5)))
        pop = rng.uniform(lb, ub, size=(npop, n))
        fit = np.full(npop, np.inf)
        best_x = None
        best_val = np.inf
        calls = 0

        # Initial evaluations
        for i in range(npop):
            if calls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            calls += 1
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if calls >= budget:
            return best_val, best_x

        # DE parameters
        F = 0.8
        CR = 0.9
        stagnation = 0
        max_stagnation = 20

        while calls < budget:
            # Check for stagnation and restart if necessary
            if stagnation >= max_stagnation:
                # Reinitialize population around best
                sigma = 0.2 * (ub - lb)
                pop = best_x + sigma * rng.normal(size=(npop, n))
                pop = np.clip(pop, lb, ub)
                for i in range(npop):
                    if calls >= budget:
                        break
                    x = pop[i]
                    val = func(x)
                    calls += 1
                    fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                stagnation = 0
                continue

            new_pop = np.empty_like(pop)
            for i in range(npop):
                # Generate donor vector
                candidates = list(range(npop))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
                donor = pop[r1] + F * (pop[r2] - pop[r3])
                # Binomial crossover
                cross_mask = rng.random(n) < CR
                if not np.any(cross_mask):
                    cross_mask[rng.integers(n)] = True
                trial = np.where(cross_mask, donor, pop[i])
                trial = np.clip(trial, lb, ub)
                new_pop[i] = trial

            # Evaluate new population
            for i in range(npop):
                if calls >= budget:
                    break
                x = new_pop[i]
                val = func(x)
                calls += 1
                if val < fit[i]:
                    pop[i] = x
                    fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                        stagnation = 0
            stagnation += 1

        return best_val, best_x