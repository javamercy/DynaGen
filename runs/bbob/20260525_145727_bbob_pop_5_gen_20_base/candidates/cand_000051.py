import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget

        pop_size = max(4, min(10*dim, budget // 4))
        if pop_size > budget:
            pop_size = budget

        budget_de = int(0.8 * budget)
        if budget_de < pop_size:
            budget_de = budget

        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

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

        initial_evals = evals
        de_ops = max(0, budget_de - initial_evals)
        de_evals_done = 0

        while evals < budget_de and evals < budget and de_evals_done < de_ops:
            fraction = de_evals_done / max(1, de_ops)
            F = 0.9 + (0.2 - 0.9) * fraction
            CR = 0.2 + (0.9 - 0.2) * fraction
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]
                mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                    else:
                        trial[j] = pop[i][j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                de_evals_done += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if de_evals_done >= de_ops:
                    break

        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            span = ub - lb
            for k in range(remaining):
                fraction = k / max(1, remaining)
                sigma = (0.2 + (0.01 - 0.2) * fraction) * span
                perturb = rng.normal(0, sigma, dim)
                candidate = best_x + perturb
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x