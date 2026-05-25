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
        if pop_size < 4:
            pop_size = 4
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        # Individual F and CR (jDE style)
        F_vals = np.random.uniform(0.1, 0.9, pop_size)
        CR_vals = np.random.uniform(0, 1, pop_size)
        tau = 0.1
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
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Adapt F and CR
                if random.random() < tau:
                    F_vals[i] = np.random.uniform(0.1, 0.9)
                if random.random() < tau:
                    CR_vals[i] = np.random.uniform(0, 1)
                # Select three distinct indices not equal to i
                candidates = [j for j in range(pop_size) if j != i]
                r0, r1, r2 = random.sample(candidates, 3)
                mutant = pop[r0] + F_vals[i] * (best_x - pop[r0]) + F_vals[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR_vals[i] or j == j_rand:
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
                # else: keep F and CR (already unchanged)
        return best_f, best_x