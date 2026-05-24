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
        popsize = min(budget, max(4, min(8*dim, 25)))
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
        F = 0.8
        CR = 0.9
        stagnation_counter = 0
        best_at_start = self.best_value
        while evals < budget:
            for i in range(popsize):
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2 = candidates[0], candidates[1]
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                trial = np.copy(pop[i])
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
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if self.best_value < best_at_start:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= 5 and evals + popsize <= budget:
                new_pop = np.empty((popsize, dim))
                new_fitness = np.empty(popsize)
                new_pop[0] = self.best_x.copy()
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
                    new_pop[i] = lb + (ub - lb) * rng.rand(dim)
                    new_fitness[i] = func(new_pop[i])
                    evals += 1
                    if new_fitness[i] < self.best_value:
                        self.best_value = new_fitness[i]
                        self.best_x = new_pop[i].copy()
                        report_best(self.best_value, self.best_x)
                    if evals >= budget:
                        break
                if evals >= budget:
                    break
                pop = new_pop
                pop_fitness = new_fitness
                stagnation_counter = 0
                best_at_start = self.best_value
            else:
                best_at_start = self.best_value
        return self.best_value, self.best_x