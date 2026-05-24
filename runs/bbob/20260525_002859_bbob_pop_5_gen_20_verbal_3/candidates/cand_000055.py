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
        popsize = min(budget, max(10, 5*dim))
        # Initialize population
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
        F = 0.9
        CR = 0.9
        stagnation = 0
        max_stagnation = max(1, int(budget / (4 * popsize)))
        while evals < budget:
            improved = False
            for i in range(popsize):
                # mutation: rand/1
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # crossover: binomial
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
                        improved = True
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if improved:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= max_stagnation and evals + popsize <= budget:
                # restart: reinitialize population, keep best
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                # keep best
                new_pop[0] = self.best_x.copy()
                # re-evaluate (just in case, but it's already evaluated)
                new_fitness[0] = self.best_value
                # generate others uniformly
                for i in range(1, popsize):
                    x = lb + (ub - lb) * rng.rand(dim)
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
                stagnation = 0
                if evals >= budget:
                    break
        return self.best_value, self.best_x