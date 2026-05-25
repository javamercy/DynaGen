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
            return np.inf, None

        pop_size = max(4, min(10 * dim, budget // 4))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 2:
            pop_size = budget

        budget_de = int(0.8 * budget)
        if budget_de < pop_size:
            budget_de = max(pop_size, 1)
        if budget_de > budget:
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

        if evals == 0:
            return np.inf, None

        CR = 0.9
        stagnation_counter = 0
        stagnation_limit = 5 * pop_size
        last_improvement_evals = evals

        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                fraction = evals / budget_de
                F = 0.5 + 0.5 * np.sin(np.pi * fraction)

                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b, c = candidates[:3]

                mutant = pop[a] + F * (pop[b] - pop[c])
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
                        last_improvement_evals = evals

            # Check stagnation
            if evals - last_improvement_evals >= stagnation_limit:
                # Reinitialize worst half of population (excluding best)
                idx = np.argsort(fitness)
                worst_idx = idx[-pop_size//2:]
                for j in worst_idx:
                    if fitness[j] == best_val and pop[j] is best_x:
                        continue
                    pop[j] = lb + rng.rand(dim) * (ub - lb)
                    fitness[j] = np.inf
                last_improvement_evals = evals

        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.2 * (ub - lb)
            for _ in range(remaining):
                if rng.rand() < 0.5:
                    candidate = best_x + rng.normal(0, sigma, dim)
                else:
                    candidate = lb + rng.rand(dim) * (ub - lb)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x