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

        pop_size = max(4, min(15*dim, budget // 3))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 2:
            pop_size = budget

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

        if evals == 0:
            return best_val, best_x

        budget_de = int(0.6 * budget)
        no_improve_evals = 0
        restart_threshold = max(pop_size, budget // 4)

        while evals < budget_de and evals < budget:
            if no_improve_evals >= restart_threshold and evals < budget_de:
                restart_size = max(1, pop_size // 2)
                indices = rng.choice(pop_size, restart_size, replace=False)
                sigma = 0.3 * (ub - lb)
                for idx in indices:
                    candidate = best_x + rng.normal(0, sigma, dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    fitness[idx] = val
                    pop[idx] = candidate
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                no_improve_evals = 0

            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]
                if rng.rand() < 0.5:
                    F = rng.uniform(0.3, 0.9)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    F = rng.uniform(0.3, 0.9)
                    mutant = best_x + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                CR = rng.uniform(0.5, 1.0)
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
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        no_improve_evals = 0
                else:
                    no_improve_evals += 1

        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.05 * (ub - lb)
            for _ in range(remaining):
                perturb = rng.standard_cauchy(dim) * sigma
                candidate = best_x + perturb
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x