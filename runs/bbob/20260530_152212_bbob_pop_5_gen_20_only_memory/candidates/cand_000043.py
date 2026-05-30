import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(budget // 10, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F_start = 0.9
        F_end = 0.1
        CR = 0.8
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        while evals < budget:
            progress = evals / budget
            F = F_start * (1 - progress) + F_end * progress
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = [j for j in range(pop_size) if j != i and j != best_idx]
                if len(candidates) < 2:
                    break
                a, b = np.random.choice(candidates, 2, replace=False)
                mutant = best + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
        while evals < budget:
            sigma = 0.02 * (ub - lb) * (1 - evals / budget)
            perturbation = np.random.randn(dim) * sigma
            trial = best_x + perturbation
            trial = np.clip(trial, lb, ub)
            val = func(trial)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = trial.copy()
                report_best(best_val, best_x)
        return best_val, best_x