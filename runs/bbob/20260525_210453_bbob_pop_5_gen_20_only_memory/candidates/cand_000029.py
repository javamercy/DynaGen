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
        if fcalls >= budget:
            return best_f, best_x
        CR = 0.9
        stagnation_counter = 0
        generation = 0
        while fcalls < budget:
            improved_this_gen = False
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                F = np.random.uniform(0.5, 1.0)
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = random.sample(candidates, 3)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
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
                        improved_this_gen = True
            generation += 1
            if not improved_this_gen:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            if stagnation_counter >= 20:
                best_idx = np.argmin(pop_f)
                best_x_keep = pop[best_idx].copy()
                best_f_keep = pop_f[best_idx]
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    if fcalls >= budget:
                        break
                    pop[i] = np.random.uniform(lb, ub)
                    val = func(pop[i])
                    fcalls += 1
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = pop[i].copy()
                        report_best(best_f, best_x)
                stagnation_counter = 0
        return best_f, best_x