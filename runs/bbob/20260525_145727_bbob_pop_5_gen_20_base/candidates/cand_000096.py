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

        # Adaptive population size
        pop_size = max(4, min(10 * dim, budget // 4))
        if pop_size > budget:
            pop_size = budget
        if pop_size < 4:
            pop_size = max(1, budget)

        # If population too small for DE, fall back to random + local search
        if pop_size < 3:
            best_val = np.inf
            best_x = None
            evals = 0
            # random sampling
            n_random = min(budget, pop_size)
            for _ in range(n_random):
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            # local refinement
            remaining = budget - evals
            if remaining > 0 and best_x is not None:
                sigma = 0.1 * (ub - lb)
                for _ in range(remaining):
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

        # Normal case: DE with current-to-best/1
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Initial evaluation
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

        CR = 0.85

        while evals < budget:
            # One generation
            for i in range(pop_size):
                if evals >= budget:
                    break
                fraction = evals / max(1, budget)
                F = 0.5 + 0.3 * np.sin(np.pi * fraction)

                # Choose three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]

                # DE/current-to-best/1
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
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

            # After each generation, local search around best
            local_steps = max(1, min(5, budget // (pop_size * 10)))
            for _ in range(local_steps):
                if evals >= budget:
                    break
                sigma = 0.05 * (ub - lb)
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