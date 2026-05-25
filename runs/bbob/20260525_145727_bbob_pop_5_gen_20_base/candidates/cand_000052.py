import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(4, min(10*dim, budget // 4))
        if self.pop_size > budget:
            self.pop_size = budget

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng
        budget = self.budget
        pop_size = self.pop_size

        # Latin hypercube initialization
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

        budget_de = int(0.8 * budget)
        if budget_de < pop_size:
            budget_de = budget

        CR = 0.9
        F_max = 0.9
        F_min = 0.4

        while evals < budget_de and evals < budget:
            for i in range(pop_size):
                if evals >= budget_de or evals >= budget:
                    break
                frac = evals / budget_de
                # Sigmoid schedule: transitioning from high to low F
                sig = 1.0 / (1.0 + np.exp(-10.0 * (frac - 0.5)))
                F = F_min + (F_max - F_min) * (1.0 - sig)

                candidates = list(range(pop_size))
                candidates.remove(i)
                rng.shuffle(candidates)
                a, b = candidates[:2]

                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
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