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
        # population size
        if dim == 1:
            popsize = min(budget, max(4, 5))
        else:
            popsize = min(budget, max(4, min(5*dim, 20)))
        # initialize population
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            pop_fitness[i] = func(pop[i])
            evals += 1
            if evals == 1 or pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # initialize F and CR for each individual
        F = 0.5 * np.ones(popsize)
        CR = 0.9 * np.ones(popsize)
        tau1 = 0.1
        tau2 = 0.1
        F_l = 0.1
        F_u = 0.9
        while evals < budget:
            for i in range(popsize):
                # adapt F and CR
                if rng.rand() < tau1:
                    F[i] = F_l + rng.rand() * (F_u - F_l)
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
                # select three distinct individuals different from i
                idxs = [j for j in range(popsize) if j != i]
                rng.shuffle(idxs)
                a, b, c = idxs[:3]
                # mutation
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                # exponential crossover
                trial = np.copy(pop[i])
                j0 = rng.randint(dim)
                L = 1
                while rng.rand() < CR[i] and L < dim:
                    L += 1
                for j in range(dim):
                    if (j >= j0 and j < j0 + L) or (j0 + L > dim and j < (j0 + L) % dim):
                        trial[j] = mutant[j]
                # bound clamping
                trial = np.clip(trial, lb, ub)
                # evaluation
                trial_fitness = func(trial)
                evals += 1
                # selection
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