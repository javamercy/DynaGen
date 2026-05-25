import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = min(max(4 * dim, 20), budget // 2)
        pop_size = max(pop_size, 4)

        def lhs(n, d, lb, ub):
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    points[i, j] = lb[j] + (perm[i] + rng.uniform()) * (ub[j] - lb[j]) / n
            return points

        pop = lhs(pop_size, dim, lb, ub)
        fitness = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_f = np.inf

        for i in range(pop_size):
            if evals >= budget:
                break
            f = func(pop[i])
            evals += 1
            fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        stagnation_counter = 0
        max_stagnation_gen = pop_size * 5
        C = np.eye(dim)
        use_cov_prob = 0.5

        while evals < budget:
            if stagnation_counter >= max_stagnation_gen and pop_size > 1:
                new_pop = lhs(pop_size - 1, dim, lb, ub)
                pop = np.vstack([best_x, new_pop])
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    f = func(pop[i])
                    evals += 1
                    fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = pop[i].copy()
                        report_best(best_f, best_x)
                stagnation_counter = 0
                continue

            if pop_size >= 4:
                sorted_idx = np.argsort(fitness)
                half = pop_size // 2
                best_half = pop[sorted_idx[:half]]
                cov = np.cov(best_half, rowvar=False)
                cov += 1e-10 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(cov)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
            else:
                L = np.eye(dim)

            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)

                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.3, 0.9)

                if rng.uniform() < use_cov_prob and pop_size >= 4:
                    mutant = pop[r1] + F * (L @ (pop[r2] - pop[r3]))
                else:
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)

                # exponential crossover
                trial = pop[i].copy()
                j_start = rng.randint(dim)
                j = j_start
                while True:
                    trial[j] = mutant[j]
                    j = (j + 1) % dim
                    if rng.uniform() >= CR or j == j_start:
                        break

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        stagnation_counter = 0
            stagnation_counter += 1

        return best_f, best_x