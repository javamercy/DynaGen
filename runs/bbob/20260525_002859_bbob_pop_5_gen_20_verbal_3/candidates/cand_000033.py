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
        evals = 0
        popsize = min(budget, max(4, min(5*dim, 20)))
        pop = lb + (ub - lb) * rng.rand(popsize, dim)
        pop_fitness = np.full(popsize, np.inf)
        # initialize per-individual F and CR
        F = 0.8 * np.ones(popsize)
        CR = 0.9 * np.ones(popsize)
        # evaluate initial population
        for i in range(popsize):
            val = func(pop[i])
            evals += 1
            pop_fitness[i] = val
            if evals == 1 or val < self.best_value:
                self.best_value = val
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        # jDE parameters
        tau_F = 0.1
        tau_CR = 0.1
        while evals < budget:
            for i in range(popsize):
                # adapt F and CR
                r1 = rng.rand()
                if r1 < tau_F:
                    F[i] = 0.1 + 0.9 * rng.rand()
                r2 = rng.rand()
                if r2 < tau_CR:
                    CR[i] = rng.rand()
                # select distinct random indices a, b, c
                candidates = [j for j in range(popsize) if j != i]
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                # mutation
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                # crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                # clip
                trial = np.clip(trial, lb, ub)
                # evaluate
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