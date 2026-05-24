import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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
        F_mean = 0.8
        CR_mean = 0.9
        lr = 0.1
        while evals < budget:
            successes_F = []
            successes_CR = []
            for i in range(popsize):
                F = np.clip(F_mean + 0.1 * rng.randn(), 0.1, 1.0)
                CR = np.clip(CR_mean + 0.1 * rng.randn(), 0, 1)
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
                    successes_F.append(F)
                    successes_CR.append(CR)
                    if trial_fitness < self.best_value:
                        self.best_value = trial_fitness
                        self.best_x = trial.copy()
                        report_best(self.best_value, self.best_x)
                if evals >= budget:
                    break
            if evals >= budget:
                break
            if len(successes_F) > 0:
                mean_F_success = np.mean(successes_F)
                mean_CR_success = np.mean(successes_CR)
                F_mean = (1 - lr) * F_mean + lr * mean_F_success
                CR_mean = (1 - lr) * CR_mean + lr * mean_CR_success
                F_mean = np.clip(F_mean, 0.1, 1.0)
                CR_mean = np.clip(CR_mean, 0, 1)
        return self.best_value, self.best_x