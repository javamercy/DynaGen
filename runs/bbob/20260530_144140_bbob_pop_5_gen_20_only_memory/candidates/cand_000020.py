import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Moderate population size
        self.NP = max(4, min(7*dim, budget // 4))
        self.CR = 0.9
        self.F_start = 0.5
        self.F_end = 0.1

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # DE/best/1/bin loop with decreasing F
        gen = 0
        while budget > 0 and NP > 1:
            F = self.F_start + (self.F_end - self.F_start) * gen / max(1, (self.budget // NP))
            gen += 1
            for i in range(NP):
                if budget <= 0:
                    break
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 2:
                    break
                r1, r2 = rng.choice(indices, size=2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
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

        # Local search intensification
        if budget > 0 and best_x is not None:
            step = 0.1 * (ub - lb)  # initial step size
            while budget > 0:
                perturbation = rng.uniform(-step, step, size=dim)
                candidate = best_x + perturbation
                candidate = np.clip(candidate, lb, ub)
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
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x