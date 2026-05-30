import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # Small population for exploitation
        self.pop_size = max(3, min(10, dim))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        F = 0.5
        CR = 0.9
        step_size = 0.01 * (ub - lb)  # small step for local search

        # Initialize population uniformly
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_val = np.inf

        # Initial evaluations
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

        # Main DE loop with local search
        while evals < budget:
            best_idx = np.argmin(fitness)
            best = pop[best_idx].copy()
            # DE generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices different from i and best_idx
                idxs = [j for j in range(pop_size) if j != i and j != best_idx]
                if len(idxs) < 2:
                    continue
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
                        best = best_x  # update best for subsequent mutations

            # Local search around the current best
            if evals < budget:
                n_local = min(5, budget - evals)
                for _ in range(n_local):
                    if evals >= budget:
                        break
                    perturbation = np.random.normal(0, step_size, dim)
                    x_new = best + perturbation
                    x_new = np.clip(x_new, lb, ub)
                    val = func(x_new)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_new.copy()
                        best = best_x
                        report_best(best_val, best_x)

        return best_val, best_x