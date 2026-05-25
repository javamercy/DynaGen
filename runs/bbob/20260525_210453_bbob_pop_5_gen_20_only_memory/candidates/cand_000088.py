import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(5 * dim, max(10, budget // 4))
        # Latin Hypercube Sampling
        segments = np.linspace(0, 1, pop_size + 1)
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                offset = np.random.uniform(0, 1/pop_size)
                pop[perm[i], d] = segments[i] + offset
        pop = lb + pop * (ub - lb)
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        F = 0.8
        CR = 0.9
        de_budget = int(budget * 0.4)
        while fcalls < min(de_budget, budget):
            for i in range(pop_size):
                if fcalls >= min(de_budget, budget):
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = random.sample(candidates, 2)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                crossover = np.random.rand(dim) < CR
                trial = np.where(crossover, mutant, pop[i])
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
        # Local search phase
        initial_step = 0.1 * (ub - lb)
        while fcalls < budget:
            remaining = budget - fcalls
            denom = budget - de_budget
            if denom > 0:
                factor = remaining / denom
            else:
                factor = 1.0
            step = initial_step * factor
            candidate = best_x + np.random.normal(0, 1, dim) * step
            candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            fcalls += 1
            if val < best_f:
                best_f = val
                best_x = candidate.copy()
                report_best(best_f, best_x)
        return best_f, best_x