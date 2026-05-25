import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop_size = min(5 * dim, max(10, budget // 4))
        pop_size = max(pop_size, 5)  # ensure at least 5 for DE
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0
        # initial evaluations
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
        # DE loop
        F = 0.8
        CR = 0.9
        while fcalls < budget:
            for i in range(pop_size):
                if fcalls >= budget:
                    break
                # select three distinct indices not equal to i
                candidates = list(range(pop_size))
                candidates.remove(i)
                r0, r1, r2 = np.random.choice(candidates, 3, replace=False)
                mutant = pop[r0] + F * (best_x - pop[r0]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                trial = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
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