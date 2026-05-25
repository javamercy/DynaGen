import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        rng = self.rng
        budget = self.budget

        pop_size = min(budget, max(4, min(5 * dim, budget // 3)))
        if pop_size < 4:
            pop_size = min(budget, 4)

        # Latin Hypercube Sampling
        points = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = rng.permutation(pop_size)
            u = rng.rand(pop_size)
            points[:, i] = lb[i] + (perm + u) / pop_size * (ub[i] - lb[i])

        pop_fitness = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            pop_fitness[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        F = 0.5
        CR = 0.9

        stagnation_limit = max(1, 5 * dim)
        stagnation_counter = 0

        sigma = 0.1 * np.mean(ub - lb)
        sigma_min = 1e-5 * np.mean(ub - lb)
        sigma_max = 0.2 * np.mean(ub - lb)

        while evals < budget:
            improved_in_generation = False
            indices = rng.permutation(pop_size)
            for target_idx in indices:
                if evals >= budget:
                    break
                possible = list(range(pop_size))
                possible.remove(target_idx)
                if len(possible) < 3:
                    continue
                selected = rng.choice(possible, 3, replace=False)
                a, b, c = selected
                mutant = points[a] + F * (points[b] - points[c])
                trial = points[target_idx].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_fitness[target_idx]:
                    points[target_idx] = trial
                    pop_fitness[target_idx] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        improved_in_generation = True

            if improved_in_generation:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit and evals + pop_size <= budget:
                old_best_f = best_f
                if pop_size > 1:
                    cov = np.cov(points.T)
                    cov += 1e-8 * np.eye(dim)
                else:
                    cov = np.eye(dim)
                new_points = rng.multivariate_normal(best_x, sigma**2 * cov, size=pop_size)
                new_points = np.clip(new_points, lb, ub)
                new_fitness = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    x = new_points[i]
                    f = func(x)
                    evals += 1
                    new_fitness[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                if evals >= budget:
                    break
                points = new_points
                pop_fitness = new_fitness
                if best_f < old_best_f:
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)
                stagnation_counter = 0

        return best_f, best_x