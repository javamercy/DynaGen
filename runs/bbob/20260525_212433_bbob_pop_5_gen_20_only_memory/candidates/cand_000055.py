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
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = min(10, budget)
        F = 0.8
        CR = 0.9

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)
        evals = 0
        for i in range(pop_size):
            if evals >= budget: break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        while evals < budget:
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget: break
                # Mutation: rand/1
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                # Greedy selection
                if val < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = val
            pop = new_pop

        return self.best_val, self.best_x