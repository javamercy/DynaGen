import numpy as np
class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.best_value = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget <= 0:
            raise ValueError("Budget must be positive")
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
            if pop_fitness[i] < self.best_value:
                self.best_value = pop_fitness[i]
                self.best_x = pop[i].copy()
                report_best(self.best_value, self.best_x)
        if evals >= budget:
            return self.best_value, self.best_x
        F = 0.8
        CR = 0.9
        c = 0.1  # learning rate for adaptation
        while evals < budget:
            success_F = []
            success_CR = []
            for i in range(popsize):
                candidates = list(range(popsize))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c_idx = candidates[:3]
                mutant = pop[a] + F * (pop[b] - pop[c_idx])
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
                    success_F.append(F)
                    success_CR.append(CR)
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if len(success_F) > 0:
                F = (1 - c) * F + c * np.mean(success_F)
                CR = (1 - c) * CR + c * np.mean(success_CR)
            else:
                # increase exploration if no success
                F = min(1.0, F * 1.1)
                CR = max(0.0, CR * 0.9)
        return self.best_value, self.best_x