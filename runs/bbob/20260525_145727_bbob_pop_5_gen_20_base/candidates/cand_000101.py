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

        if budget == 0:
            return np.inf, np.zeros(dim)

        pop_size = max(8, min(20 * dim, budget // 3))
        pop_size = min(pop_size, budget)
        if pop_size < 4:
            pop_size = max(1, budget)

        budget_de = int(0.7 * budget)
        if budget_de < pop_size:
            budget_de = pop_size
        budget_de = min(budget_de, budget)
        budget_local = budget - budget_de

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

        CR = 0.9
        F_low = 0.2
        F_high = 0.9

        while evals < budget_de:
            for i in range(pop_size):
                if evals >= budget_de:
                    break
                fraction = evals / max(1, budget_de)
                F = F_low + (F_high - F_low) * (0.5 + 0.5 * np.sin(2 * np.pi * fraction + rng.rand() * 2 * np.pi))

                strategy = rng.randint(2)
                if strategy == 0:
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b, c = candidates[:3]
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b = candidates[:2]
                    mutant = best_x + F * (pop[a] - pop[b])

                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
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

        if budget_local > 0 and best_x is not None:
            sigma_init = 0.2 * (ub - lb)
            for k in range(budget_local):
                sigma = sigma_init * (1 - k / budget_local)
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