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

        pop_size = max(4, min(10 * dim, budget // 4))
        pop_size = min(pop_size, budget)
        if pop_size < 4:
            pop_size = max(1, budget)

        budget_de = int(0.8 * budget)
        if budget_de < pop_size:
            budget_de = pop_size
        budget_de = min(budget_de, budget)

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

        while evals < budget_de:
            for i in range(pop_size):
                if evals >= budget_de:
                    break
                fraction = evals / max(1, budget_de)
                F = 0.5 + 0.3 * np.sin(np.pi * fraction)

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
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            total_remaining = remaining
            for step_count in range(remaining):
                frac = step_count / max(1, total_remaining)
                step_size = 0.1 * (ub - lb) * (1 - 0.9 * frac)  # decays from 0.1 to 0.01
                pert = rng.uniform(-step_size, step_size, dim)
                candidate = best_x + pert
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x