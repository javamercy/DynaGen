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
        popsize = min(budget // 2, max(4, 10 * dim))
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
        CR = 0.9
        no_improve_gen = 0
        while evals < budget:
            improved = False
            for i in range(popsize):
                if evals >= budget:
                    break
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                r1, r2, r3 = candidates[0], candidates[1], candidates[2]
                F = 0.5 + 0.5 * rng.rand()
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
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
            if improved:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            if no_improve_gen >= 2 * dim and evals < budget:
                # restart: reinitialize 50% of population (excluding best)
                n_restart = popsize // 2
                worst_indices = np.argsort(pop_fitness)[-n_restart:]
                for idx in worst_indices:
                    if idx == np.argmin(pop_fitness):
                        continue
                    pop[idx] = lb + (ub - lb) * rng.rand(dim)
                    pop_fitness[idx] = func(pop[idx])
                    evals += 1
                    if pop_fitness[idx] < self.best_value:
                        self.best_value = pop_fitness[idx]
                        self.best_x = pop[idx].copy()
                        report_best(self.best_value, self.best_x)
                    if evals >= budget:
                        break
                no_improve_gen = 0
        return self.best_value, self.best_x