import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.evals = 0
        self.best_value = np.inf
        self.best_x = None

    def __call__(self, func):
        bounds = func.bounds
        lb = bounds.lb
        ub = bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = min(4 * dim, budget // 2)
        if pop_size < 4:
            pop_size = max(4, min(dim, budget))
        if pop_size < 3:
            pop_size = 3
        if pop_size > budget:
            pop_size = budget

        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if self.evals >= budget:
                break
            x = pop[i]
            val = func(x)
            self.evals += 1
            pop_fitness[i] = val
            if val < self.best_value:
                self.best_value = val
                self.best_x = x.copy()
                report_best(val, x)

        best_idx = np.argmin(pop_fitness)
        best = pop[best_idx].copy()
        best_fitness = pop_fitness[best_idx]

        stagnation = 0
        stagnation_limit = max(1, int(0.1 * budget))

        while self.evals < budget:
            improved = False
            for i in range(pop_size):
                if self.evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 2:
                    break
                selected = rng.choice(candidates, 2, replace=False)
                a, b = selected
                F = 0.8
                mutant = pop[a] + F * (pop[b] - pop[c])  # actually need two distinct: we used a and b but need three? Wait: DE/rand/1: base = pop[c], difference = pop[a]-pop[b]
                # Correction: choose three distinct
                # Choose three distinct indices
                idxs = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                r1, r2, r3 = idxs
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                CR = 0.9
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                self.evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < self.best_value:
                        self.best_value = val
                        self.best_x = trial.copy()
                        report_best(val, trial)
                        best = trial.copy()
                        best_fitness = val
                        improved = True
                        stagnation = 0
            if not improved:
                stagnation += pop_size
                if stagnation >= stagnation_limit and self.evals < budget:
                    # Restart: keep best point, reinitialize others
                    stagnation = 0
                    new_pop = np.vstack([best, lb + (ub - lb) * rng.rand(pop_size - 1, dim)])
                    new_fitness = np.full(pop_size, np.inf)
                    # Evaluate new points except the first (best already evaluated)
                    for j in range(1, pop_size):
                        if self.evals >= budget:
                            break
                        val = func(new_pop[j])
                        self.evals += 1
                        new_fitness[j] = val
                        if val < self.best_value:
                            self.best_value = val
                            self.best_x = new_pop[j].copy()
                            report_best(val, new_pop[j])
                            best = new_pop[j].copy()
                    new_fitness[0] = self.best_value
                    pop = new_pop
                    pop_fitness = new_fitness
            else:
                stagnation = 0

        return self.best_value, self.best_x