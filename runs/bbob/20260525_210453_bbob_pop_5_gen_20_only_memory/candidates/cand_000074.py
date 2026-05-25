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
        F = 0.8
        CR = 0.9
        switch_budget = int(budget * 0.7)
        while fcalls < min(budget, switch_budget):
            for i in range(pop_size):
                if fcalls >= min(budget, switch_budget):
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2 = random.sample(candidates, 2) if len(candidates) >= 2 else (candidates[0], candidates[0])
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
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
        remaining = budget - fcalls
        if remaining > 0:
            step_size = (ub - lb) * 0.1
            for _ in range(remaining):
                if fcalls >= budget:
                    break
                frac = (fcalls - switch_budget) / max(1, remaining)
                current_step = step_size * (1 - frac)
                current_step = max(current_step, 1e-6 * (ub - lb))
                candidate = best_x + np.random.uniform(-current_step, current_step, dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                fcalls += 1
                if val < best_f:
                    best_f = val
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
        return best_f, best_x