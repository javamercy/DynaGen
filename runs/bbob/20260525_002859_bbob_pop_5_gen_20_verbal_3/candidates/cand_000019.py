import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.zeros(popsize)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # Initialize F and CR for each individual
        F = np.full(popsize, 0.5)
        CR = np.full(popsize, 0.9)
        tau1 = 0.1
        tau2 = 0.1
        while evals < budget:
            for i in range(popsize):
                # Update F and CR for individual i with small probabilities
                if rng.rand() < tau1:
                    F[i] = 0.1 + 0.9 * rng.rand()
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
                # Randomly select indices for mutation
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
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
                if evals >= budget:
                    break
            if evals >= budget:
                break
        return self.best_value, self.best_x