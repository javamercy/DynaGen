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
        
        # Population size smaller than parent for exploitation
        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(1, budget // 2)
        
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
        
        if best_x is None:
            best_x = np.full(dim, np.nan)
            best_f = np.inf
            return best_f, best_x
        
        # DE parameters for exploitation
        F = 0.5  # smaller F for exploitation
        CR = 0.9
        
        # Phase 1: DE until 70% budget used
        de_budget = int(0.7 * budget)
        while fcalls < de_budget and fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # Parents: best, two distinct random
                idxs = list(range(pop_size))
                idxs.remove(i)
                if len(idxs) >= 2:
                    r1, r2 = random.sample(idxs, 2)
                else:
                    r1 = r2 = idxs[0] if idxs else i
                # Mutation: DE/best/1
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
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
        
        # Phase 2: Local search around best
        # Use decreasing perturbation scale
        remaining = budget - fcalls
        if remaining > 0:
            # Number of local search steps
            local_steps = min(remaining, 100)
            step_size = 0.1 * (ub - lb)  # initial step
            for step in range(local_steps):
                if fcalls >= budget:
                    break
                # Gaussian perturbation with shrinking scale
                scale = step_size * (1 - step / local_steps)
                candidate = best_x + np.random.randn(dim) * scale
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                fcalls += 1
                if val < best_f:
                    best_f = val
                    best_x = candidate.copy()
                    report_best(best_f, best_x)
        
        return best_f, best_x