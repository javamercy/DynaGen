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
        # Latin hypercube sampling
        samples = np.zeros((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                offset = np.random.uniform(0, 1)
                samples[i, j] = (perm[i] + offset) / pop_size
        pop = lb + samples * (ub - lb)
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # initial evaluation
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
        # adaptive parameters
        F_start, F_end = 0.9, 0.4
        CR_start, CR_end = 0.9, 0.5
        p_start, p_end = 0.3, 0.1
        max_generations = pop_size  # approx number of generations
        gen = 0
        while fcalls < budget:
            # compute schedule based on generation
            t = gen / max_generations if max_generations > 0 else 1
            t = min(t, 1.0)
            F = F_start + (F_end - F_start) * t
            CR = CR_start + (CR_end - CR_start) * t
            p_best = p_start + (p_end - p_start) * t
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                sorted_idx = np.argsort(pop_f)
                top_k = max(1, int(p_best * pop_size))
                pbest_idx = sorted_idx[random.randint(0, top_k - 1)]
                candidates = [j for j in range(pop_size) if j != i and j != pbest_idx]
                if len(candidates) < 2:
                    candidates = [j for j in range(pop_size) if j != i]
                    if len(candidates) < 2:
                        candidates = list(range(pop_size))
                r1, r2 = random.sample(candidates, 2)
                mutant = pop[i] + F * (pop[pbest_idx] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
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
            gen += 1
        return best_f, best_x