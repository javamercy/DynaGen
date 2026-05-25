import numpy as np
import random

class Optimizer:
    def __init__(self, budget, dim, seed):
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
        # population size
        pop_size = min(5 * dim, max(5, budget // 5))
        pop_size = max(3, min(pop_size, budget // 2))
        # initial population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
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
        # adaptive DE parameters
        F_start = 0.9
        F_end = 0.2
        CR_start = 0.9
        CR_end = 0.2
        local_budget = max(5, int(0.1 * budget))
        # main DE loop
        while fcalls < budget - local_budget:
            for i in range(pop_size):
                if fcalls >= budget - local_budget:
                    break
                progress = fcalls / budget
                F = F_start - (F_start - F_end) * progress
                CR = CR_start - (CR_start - CR_end) * progress
                # mutation
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    r0 = random.choice(candidates)
                    r1 = random.choice(candidates)
                    r2 = random.choice([c for c in candidates if c != r0 and c != r1])
                else:
                    r0, r1, r2 = random.sample(candidates, 3)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
        # local search exploitation
        while fcalls < budget:
            remaining = budget - fcalls
            step = 0.1 * (ub - lb) * (remaining / budget)
            if best_x is None:
                x = np.random.uniform(lb, ub)
            else:
                x = best_x + np.random.normal(0, step, dim)
                x = np.clip(x, lb, ub)
            val = func(x)
            fcalls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)
        return best_f, best_x