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
        # Population size
        pop_size = min(self.budget, max(4, min(5 * dim, self.budget // 3)))
        # Latin Hypercube Sampling initialization
        points = np.zeros((pop_size, dim))
        for i in range(dim):
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
        # DE parameters
        F = 0.5
        CR = 0.9
        # Local search parameters
        sigma = 0.1  # relative to (ub-lb)
        gen = 0
        # Main loop
        while evals < self.budget:
            # DE generation
            for target_idx in range(pop_size):
                if evals >= self.budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(target_idx)
                idx = self.rng.choice(candidates, 3, replace=False)
                a, b, c = idx
                # Mutation
                mutant = points[a] + F * (points[b] - points[c])
                # Crossover
                trial = points[target_idx].copy()
                j_rand = self.rng.randint(dim)
                for j in range(dim):
                    if self.rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # Clip
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
            if evals >= self.budget:
                break
            gen += 1
            # Local refinement every generation
            # Generate candidate near best
            scale = sigma * (ub - lb)
            candidate = best_x + self.rng.randn(dim) * scale
            candidate = np.clip(candidate, lb, ub)
            f_candidate = func(candidate)
            evals += 1
            if f_candidate < best_f:
                best_f = f_candidate
                best_x = candidate.copy()
                report_best(best_f, best_x)
                sigma *= 1.2
            else:
                sigma *= 0.9
            # Bound sigma to [1e-8, 1.0] relative to full range
            sigma = np.clip(sigma, 1e-8, 1.0)
        return best_f, best_x