import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        random.seed(self.seed)
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Population size: smaller to allocate more to local search
        pop_size = max(4, min(10, budget // 10))
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

        # DE parameters: adaptive linear decay
        F_start = 0.9
        F_end = 0.2
        CR_start = 0.9
        CR_end = 0.2
        local_budget = max(10, int(0.2 * budget))  # Increased local budget

        # Main DE loop
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
                    continue
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

        # Intensified local search: shrinking step size
        while fcalls < budget:
            remaining = budget - fcalls
            step_scale = 0.2 * (remaining / budget)
            step = step_scale * (ub - lb)
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