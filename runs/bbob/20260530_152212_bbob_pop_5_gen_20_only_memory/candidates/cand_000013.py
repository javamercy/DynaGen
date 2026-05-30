import numpy as np
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 2, 4 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        dim = self.dim
        pop_size = self.pop_size
        F = 0.5
        CR = 0.5

        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                idxs = [j for j in range(pop_size) if j != i]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(0, dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()

        # local search via coordinate descent
        if evals < budget:
            step = (ub - lb).max() * 0.1
            x = best_x.copy()
            f = best_val
            for _ in range(100):
                if evals >= budget:
                    break
                improved = False
                for d in np.random.permutation(dim):
                    if evals >= budget:
                        break
                    for delta in [step, -step]:
                        trial = x.copy()
                        trial[d] += delta
                        trial = np.clip(trial, lb, ub)
                        val = func(trial)
                        evals += 1
                        if val < f:
                            f = val
                            x = trial.copy()
                            improved = True
                            if val < best_val:
                                best_val = val
                                best_x = x.copy()
                            break
                step *= 0.5
                if step < 1e-15 or not improved:
                    break
        return best_val, best_x