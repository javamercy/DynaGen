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

        # population size: smaller for exploitation
        pop_size = max(4, min(5*dim, budget // 6))
        pop_size = min(pop_size, budget)
        if pop_size < 4:
            pop_size = max(1, budget)

        # allocate 60% of budget for DE, rest for local search
        budget_de = int(0.6 * budget)
        if budget_de < pop_size:
            budget_de = budget

        # initialize population
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

        # DE parameters: exploitative
        F = 0.3
        CR = 0.9

        # main DE loop
        while evals < budget_de and evals < budget and best_x is not None:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]
                # DE/best/1
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

        # local refinement: (1+1)-ES with step-size adaptation
        remaining = budget - evals
        if remaining > 0 and best_x is not None:
            sigma = 0.2 * (ub - lb)  # initial step size
            success_ratio = 0.2  # target 1/5 success rate
            # adaptation parameters
            c_success = 0.2  # success count window? we use simple exponential
            # Actually, we'll use a simpler adaptation: if improvement, increase sigma; else decrease
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
                    sigma *= 1.2  # increase step size after success
                else:
                    sigma *= 0.8  # decrease step size after failure
                # ensure sigma not too small
                sigma = np.maximum(sigma, 1e-10 * (ub - lb))

        return best_val, best_x