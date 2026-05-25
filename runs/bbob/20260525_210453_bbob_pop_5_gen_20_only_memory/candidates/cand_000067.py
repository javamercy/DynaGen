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
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # Initial evaluation
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
        p_best = 0.2  # fraction of top individuals for pbest
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Select pbest from top 20%
                sorted_idx = np.argsort(pop_f)
                top_k = max(1, int(p_best * pop_size))
                pbest_idx = sorted_idx[random.randint(0, top_k - 1)]
                # Select two distinct random indices different from i and pbest_idx
                candidates = [j for j in range(pop_size) if j != i and j != pbest_idx]
                if len(candidates) < 2:
                    # Fallback: use any two distinct
                    candidates = [j for j in range(pop_size) if j != i]
                    if len(candidates) < 2:
                        candidates = list(range(pop_size))
                r1, r2 = random.sample(candidates, 2)
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim - 1)
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
        return best_f, best_x