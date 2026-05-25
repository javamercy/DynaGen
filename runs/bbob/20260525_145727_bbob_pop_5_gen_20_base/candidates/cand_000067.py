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

        # population size
        pop_size = max(4, min(10 * dim, budget // 3))
        if pop_size > budget:
            pop_size = budget

        # DE budget fraction
        de_budget = int(0.8 * budget)
        if de_budget < pop_size:
            de_budget = budget

        # initial population
        pop = lb + rng.rand(pop_size, dim) * (ub - lb)
        fitness = np.full(pop_size, np.inf)
        best_x = None
        best_val = np.inf
        evals = 0

        # initial evaluation
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

        # DE main loop
        while evals < de_budget and evals < budget:
            for i in range(pop_size):
                if evals >= de_budget or evals >= budget:
                    break
                frac = evals / de_budget  # progress in DE phase

                # F exponentially decreasing: start 0.9, end 0.2
                F = 0.2 + 0.7 * np.exp(-2 * frac)
                # CR linearly increasing from 0.5 to 0.9
                CR = 0.5 + 0.4 * frac

                # mutation strategy: switch at halfway
                if frac < 0.5:
                    # DE/rand/1
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b, c = candidates[:3]
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    # DE/best/1
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    rng.shuffle(candidates)
                    a, b = candidates[:2]
                    mutant = best_x + F * (pop[a] - pop[b])

                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
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

        # local refinement
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.05 * (ub - lb)  # smaller step
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