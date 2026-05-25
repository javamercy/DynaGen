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
        # Determine population size
        pop_size = min(self.budget, max(4, min(5 * self.dim, self.budget // 3)))
        # Latin Hypercube Sampling initialization
        points = np.zeros((pop_size, self.dim))
        for i in range(self.dim):
            perm = self.rng.permutation(pop_size)
            u = self.rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])
        # Evaluate initial population
        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
        # Differential Evolution parameters
        F = 0.5
        CR = 0.9
        # Main DE loop
        while evals < self.budget:
            target_idx = self.rng.randint(pop_size)
            candidates = list(range(pop_size))
            candidates.remove(target_idx)
            if len(candidates) < 3:
                continue
            idx = self.rng.choice(candidates, 3, replace=False)
            a, b, c = idx
            # Mutation
            mutant = points[a] + F * (points[b] - points[c])
            # Crossover
            trial = points[target_idx].copy()
            j_rand = self.rng.randint(self.dim)
            for j in range(self.dim):
                if self.rng.rand() < CR or j == j_rand:
                    trial[j] = mutant[j]
            # Clip to bounds
            trial = np.clip(trial, lb, ub)
            # Evaluate
            f_trial = func(trial)
            evals += 1
            if f_trial < pop_fitness[target_idx]:
                points[target_idx] = trial
                pop_fitness[target_idx] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    report_best(best_f, best_x)
        return best_f, best_x