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
        popsize = min(budget, max(4, 2*dim))
        # initial population
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
        stagnation_counter = 0
        max_stagnation = max(5, int(10 * dim / popsize))
        F = 0.8
        CR = 0.9
        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                # mutation: current-to-best/1
                indices = list(range(popsize))
                indices.remove(i)
                rng.shuffle(indices)
                r1, r2 = indices[0], indices[1]
                mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                # crossover
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
                        improved = True
            if evals >= budget:
                break
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            if stagnation_counter >= max_stagnation and evals + popsize - 1 <= budget:
                # restart, keep best
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                for i in range(1, popsize):
                    if evals >= budget:
                        break
                    x = lb + (ub - lb) * rng.rand(dim)
                    x = np.clip(x, lb, ub)
                    f = func(x)
                    evals += 1
                    new_pop[i] = x
                    new_fitness[i] = f
                    if f < self.best_value:
                        self.best_value = f
                        self.best_x = x.copy()
                        report_best(self.best_value, self.best_x)
                pop = new_pop
                pop_fitness = new_fitness
                stagnation_counter = 0
        return self.best_value, self.best_x