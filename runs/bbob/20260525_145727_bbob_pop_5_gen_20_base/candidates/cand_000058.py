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

        # Population size
        pop_size = max(4, min(10*dim, budget // 4))
        if pop_size > budget:
            pop_size = budget

        # Latin hypercube
        def lhs(d, n, l, u, rng):
            samples = np.empty((n, d))
            for i in range(d):
                edges = np.linspace(0, 1, n+1)
                points = edges[:-1] + rng.rand(n) * (edges[1] - edges[:-1])
                rng.shuffle(points)
                samples[:, i] = points
            return l + samples * (u - l)

        pop = lhs(dim, pop_size, lb, ub, rng)
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

        F, CR = 0.7, 0.5
        stagnation_counter = 0
        stagnation_limit = max(10, int(0.2 * budget / pop_size))
        de_budget = int(0.7 * budget)
        restart_budget = int(0.2 * budget)
        local_budget = budget - evals

        # Main DE loop
        while evals < de_budget and evals < budget:
            improved = False
            for i in range(pop_size):
                if evals >= de_budget or evals >= budget:
                    break
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
                        improved = True
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit:
                    # Restart: keep best, regenerate rest via LHS
                    new_pop = lhs(dim, pop_size-1, lb, ub, rng)
                    pop = np.vstack((best_x, new_pop))
                    # Reevaluate all except best
                    for i in range(1, pop_size):
                        if evals >= budget:
                            break
                        val = func(pop[i])
                        evals += 1
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = pop[i].copy()
                            report_best(best_val, best_x)
                    stagnation_counter = 0

        # Local search on best
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