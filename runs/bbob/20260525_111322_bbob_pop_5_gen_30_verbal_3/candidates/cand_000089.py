import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = max(4, min(5 * self.dim, self.budget // 2))
        pop_size = min(pop_size, self.budget)
        points = self.rng.uniform(lb, ub, size=(pop_size, self.dim))
        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        # Initialize F and CR per individual
        F = self.rng.uniform(0.1, 1.0, pop_size)
        CR = self.rng.uniform(0.0, 1.0, pop_size)

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

        tau = 0.1
        while evals < self.budget:
            for target in range(pop_size):
                if evals >= self.budget:
                    break
                # Select three distinct indices not equal to target
                indices = [i for i in range(pop_size) if i != target]
                if len(indices) < 3:
                    continue
                a, b, c = self.rng.choice(indices, 3, replace=False)
                # Mutation using target's F
                mutant = points[a] + F[target] * (points[b] - points[c])
                trial = points[target].copy()
                j_rand = self.rng.randint(self.dim)
                for j in range(self.dim):
                    if self.rng.rand() < CR[target] or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()
                    report_best(best_f, best_x)
                if f_trial < pop_fitness[target]:
                    points[target] = trial
                    pop_fitness[target] = f_trial
                    # Keep F and CR
                else:
                    # Re-initialize with probability tau
                    if self.rng.rand() < tau:
                        F[target] = self.rng.uniform(0.1, 1.0)
                        CR[target] = self.rng.uniform(0.0, 1.0)
        return best_f, best_x