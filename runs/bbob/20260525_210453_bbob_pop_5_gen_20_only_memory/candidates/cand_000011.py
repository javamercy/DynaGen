import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        pop_size = min(4 * dim, budget)
        if pop_size < 4:
            best_x = None
            best_f = np.inf
            for _ in range(budget):
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                    report_best(best_f, best_x)
            return best_f, best_x

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0

        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        F = 0.9
        CR = 0.9
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                r0, r1, r2 = np.random.choice(indices, 3, replace=False)
                mutant = pop[r0] + F * (pop[r1] - pop[r2])
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                j_rand = np.random.randint(dim)
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)
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