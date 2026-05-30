import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # smaller population for exploitation
        self.pop_size = max(3, min(budget // 2, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        F = 0.7
        CR = 0.9

        # initial population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # initial evaluations
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
                report_best(best_val, best_x)

        # main DE loop
        while evals < budget:
            # find best index
            best_idx = np.argmin(fitness)
            best = pop[best_idx].copy()
            # generate offspring
            for i in range(pop_size):
                if evals >= budget:
                    break
                # select two distinct random indices different from i and best_idx
                idxs = [j for j in range(pop_size) if j != i and j != best_idx]
                a, b = np.random.choice(idxs, 2, replace=False)
                mutant = best + F * (pop[a] - pop[b])
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
                        report_best(best_val, best_x)
            # local search around best
            if evals < budget:
                # determine step size scale
                sigma = 0.1 * (ub - lb) * (1 - evals / budget)
                for _ in range(min(5, budget - evals)):
                    candidate = best_x + sigma * np.random.randn(dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        # reduce step size on improvement
                        sigma *= 0.9
                    else:
                        sigma *= 0.95
                    if evals >= budget:
                        break
        return best_val, best_x