import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.NP = max(4, min(10*dim, budget // 4))
        self.CR = 0.9
        self.F_start = 0.5
        self.F_end = 0.1

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        pop = rng.uniform(bounds_lb, bounds_ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], bounds_lb, bounds_ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        gen = 0
        while budget > 0 and NP > 1:
            F = self.F_start + (self.F_end - self.F_start) * gen / max(1, (self.budget // NP))
            gen += 1
            for i in range(NP):
                if budget <= 0:
                    break
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 3:
                    break
                r0, r1, r2 = rng.choice(indices, size=3, replace=False)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, bounds_lb, bounds_ub)
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        if budget > 0 and best_x is not None:
            step = 0.1 * (bounds_ub - bounds_lb)
            while budget > 0:
                perturbation = rng.uniform(-step, step, size=dim)
                candidate = best_x + perturbation
                candidate = np.clip(candidate, bounds_lb, bounds_ub)
                val = func(candidate)
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                step *= 0.95
                if np.max(step) < 1e-12:
                    break

        if best_x is None:
            x = rng.uniform(bounds_lb, bounds_ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x