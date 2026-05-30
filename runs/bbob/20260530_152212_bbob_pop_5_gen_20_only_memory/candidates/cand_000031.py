import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(3, min(budget // 4, 5 * dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        F = 0.8
        CR = 0.9

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        while evals < budget:
            # DE generation (best/1/bin)
            best_idx = np.argmin(fitness)
            best = pop[best_idx]
            for i in range(pop_size):
                if evals >= budget:
                    break
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

            # Intensive local search from best point
            step = (ub - lb).max() * 0.1
            local_evals = 0
            ls_budget = min(budget - evals, 20 * dim)
            while evals < budget and local_evals < ls_budget:
                pert = best_x + np.random.normal(0, step, size=dim)
                pert = np.clip(pert, lb, ub)
                val = func(pert)
                evals += 1
                local_evals += 1
                if val < best_val:
                    best_val = val
                    best_x = pert.copy()
                    report_best(best_val, best_x)
                step *= 0.95
                if step < 1e-10:
                    break

        return best_val, best_x