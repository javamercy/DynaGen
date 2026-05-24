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
        if budget < 4:
            for _ in range(budget):
                x = lb + (ub - lb) * rng.rand(dim)
                f = func(x)
                evals += 1
                if evals == 1 or f < self.best_value:
                    self.best_value = f
                    self.best_x = x.copy()
                    report_best(self.best_value, self.best_x)
            return self.best_value, self.best_x
        popsize = min(budget, max(10, 5*dim))
        if popsize < 4:
            popsize = 4
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
        max_stagnation = max(5, int(budget / (3 * popsize)))
        stagnation_counter = 0
        generation = 0
        while evals < budget:
            improved = False
            if generation % 5 == 0 and generation > 0:
                worst_idx = np.argmax(pop_fitness)
                new_x = lb + (ub - lb) * rng.rand(dim)
                f = func(new_x)
                evals += 1
                pop[worst_idx] = new_x
                pop_fitness[worst_idx] = f
                if f < self.best_value:
                    self.best_value = f
                    self.best_x = new_x.copy()
                    report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                trial = np.copy(pop[i])
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial <= pop_fitness[i]:
                    pop_fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < self.best_value:
                        self.best_value = f_trial
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                        improved = True
            if evals >= budget:
                break
            if not improved:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            generation += 1
            if stagnation_counter >= max_stagnation and evals + popsize <= budget:
                new_pop = np.zeros((popsize, dim))
                new_fitness = np.zeros(popsize)
                new_pop[0] = self.best_x
                new_fitness[0] = self.best_value
                step = (ub - lb) * 0.05 / np.sqrt(dim)
                for i in range(1, popsize):
                    if rng.rand() < 0.5:
                        x = self.best_x + step * rng.randn(dim)
                        x = np.clip(x, lb, ub)
                    else:
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
                stagnation_counter = 0
                if evals >= budget:
                    break
        return self.best_value, self.best_x