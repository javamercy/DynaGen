import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = None
        self.best_x = None

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Population size: smaller for exploitation
        popsize = min(budget, max(4, min(2*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.zeros(popsize)
        evals = 0

        # Initial evaluation
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x

        # DE/best/1/bin parameters
        F = 0.5
        CR = 0.9

        while evals < budget:
            # Find current best index
            best_idx = np.argmin(pop_fitness)
            best = pop[best_idx]
            for i in range(popsize):
                if evals >= budget:
                    break
                # Ensure a, b, c are different from each other and from i
                indices = [j for j in range(popsize) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                # Mutation with best as base
                mutant = best + F * (pop[a] - pop[b])
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                trial_fitness = func(trial)
                evals += 1
                if trial_fitness <= pop_fitness[i]:
                    pop_fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                # Update best for next iterations (optional, but we recompute best_idx each loop)
            # Early termination if budget exhausted
            if evals >= budget:
                break

        # Local refinement phase: if remaining budget, perturb best
        while evals < budget:
            step = (ub - lb) * 0.01 * (1 - evals / budget)  # decreasing step size
            trial = self.best_x + rng.randn(dim) * step
            trial = np.clip(trial, lb, ub)
            trial_fitness = func(trial)
            evals += 1
            if trial_fitness < self.best_value:
                self.best_value = trial_fitness
                self.best_x = trial.copy()
                report_best(self.best_value, self.best_x)

        return self.best_value, self.best_x