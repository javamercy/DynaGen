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

        # population size
        pop_size = min(10 * dim, budget // 3)
        pop_size = max(pop_size, 5)

        best_val = np.inf
        best_x = None
        evals = 0

        # initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_vals = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            pop_vals[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        if evals >= budget:
            return best_val, best_x

        # DE parameters
        F = 0.5
        CR = 0.9
        max_gen = max(1, int(0.8 * budget / pop_size))
        for gen in range(max_gen):
            if evals >= budget:
                break
            for i in range(pop_size):
                if evals >= budget:
                    break
                # mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                # crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_vals[i]:
                    pop[i] = trial
                    pop_vals[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search: random perturbations
        if evals < budget:
            remaining = budget - evals
            step0 = 0.1 * (ub - lb)
            for i in range(remaining):
                if evals >= budget:
                    break
                alpha = 1.0 - i / remaining
                step = step0 * alpha
                x = best_x + rng.randn(dim) * step
                x = np.clip(x, lb, ub)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

        return best_val, best_x