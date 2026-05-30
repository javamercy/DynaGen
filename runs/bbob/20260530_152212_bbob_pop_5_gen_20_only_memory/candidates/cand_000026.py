import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        # small population for exploitation
        self.pop_size = max(3, min(dim, budget // 10))

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        pop_size = self.pop_size
        evals = 0
        best_val = np.inf
        best_x = None

        # initial population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
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

        # one DE generation with best/1 and dithering
        if evals < budget:
            best_idx = np.argmin(fitness)
            best = pop[best_idx].copy()
            F = 0.5 + 0.5 * np.random.rand()
            CR = 0.9
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if best_idx in candidates:
                    candidates.remove(best_idx)
                if len(candidates) < 2:
                    continue
                a, b = np.random.choice(candidates, 2, replace=False)
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

        # intensive coordinate-wise pattern search
        step = 0.1 * (ub - lb)
        shrink = 0.5
        while evals < budget:
            improved = False
            for d in range(dim):
                if evals >= budget:
                    break
                # positive step
                candidate = best_x.copy()
                candidate[d] = min(ub[d], candidate[d] + step[d])
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                    improved = True
                # negative step
                if evals >= budget:
                    break
                candidate = best_x.copy()
                candidate[d] = max(lb[d], candidate[d] - step[d])
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                    improved = True
            if not improved:
                step *= shrink
            if np.max(step) < 1e-15:
                break

        return best_val, best_x