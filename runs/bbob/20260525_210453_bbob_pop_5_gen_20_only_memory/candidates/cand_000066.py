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
        pop_size = min(10 * dim, max(20, budget // 2))
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
        
        stagnation_counter = 0
        max_stagnation = max(10, dim)
        F_min = 0.5
        F_max = 1.0
        CR = 0.95
        
        while fcalls < budget:
            # Adjust F randomly
            F = np.random.uniform(F_min, F_max)
            best_f_old = best_f
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r0, r1, r2 = random.sample(candidates, 3) if len(candidates) >= 3 else (candidates[0], candidates[0], candidates[0])
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
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
            # Check stagnation
            if best_f < best_f_old:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            if stagnation_counter >= max_stagnation and fcalls < budget:
                # Restart worst half
                indices = np.argsort(pop_f)
                half = pop_size // 2
                for idx in indices[half:]:
                    pop[idx] = np.random.uniform(lb, ub, dim)
                    pop_f[idx] = np.inf
                stagnation_counter = 0
        return best_f, best_x