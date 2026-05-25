import numpy as np
class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        pop_size = max(5, min(20, budget // 10))
        pop = rng.uniform(lb, ub, (pop_size, dim))
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
        F = 0.7
        CR = 0.5
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                r1, r2 = rng.choice(indices, size=2, replace=False)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                trial = pop[i].copy()
                j = rng.randint(0, dim)
                L = 0
                while (rng.random() < CR or L == 0) and L < dim:
                    trial[j] = mutant[j]
                    j = (j + 1) % dim
                    L += 1
                val = func(trial)
                fcalls += 1
                if val < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
            if fcalls < budget:
                sigma = 0.1 * (ub - lb).mean()
                for _ in range(min(dim, budget - fcalls)):
                    step = sigma * rng.normal(0, 1, dim)
                    candidate = best_x + step
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    fcalls += 1
                    if val < best_f:
                        best_f = val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                    else:
                        sigma *= 0.9
                    if fcalls >= budget:
                        break
        return best_f, best_x