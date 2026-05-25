import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        # Population size: slightly larger for exploration
        pop_size = min(30, max(5, budget // 3))
        if budget < pop_size:
            pop_size = max(1, budget)

        if budget <= 0:
            best_x = np.zeros(dim)
            best_f = float('inf')
            report_best(best_f, best_x)
            return best_f, best_x

        # Initialize uniform population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1
        if budget <= 0:
            best_idx = np.argmin(pop_f)
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]
            report_best(best_f, best_x)
            return best_f, best_x

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        CR = 0.9
        stagnation_limit = max(1, budget // (4 * pop_size))
        stagnation_counter = 0
        diversity_threshold = 0.02 * np.mean(ub - lb)

        while budget > 0:
            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Select two distinct random indices different from i
                indices = [j for j in range(pop_size) if j != i]
                r1, r2 = rng.choice(indices, 2, replace=False)
                # Sample F uniformly per individual
                F = rng.uniform(0.5, 1.0)
                # Mutation: rand-to-best
                mut = pop[i] + F * (pop[r1] - pop[r2]) + F * (best_x - pop[i])
                # Exponential crossover
                trial = pop[i].copy()
                j0 = rng.randint(dim)
                j = j0
                L = 0
                while True:
                    trial[j] = mut[j]
                    j = (j + 1) % dim
                    L += 1
                    if L == dim or rng.rand() > CR:
                        break
                trial = np.clip(trial, lb, ub)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        improved = True
            # Compute population diversity
            pop_std = np.mean(np.std(pop, axis=0))
            if not improved:
                stagnation_counter += 1
                if (stagnation_counter >= stagnation_limit) or (stagnation_counter >= stagnation_limit//2 and pop_std < diversity_threshold):
                    if budget >= pop_size:
                        # Restart: keep best, then 50% Gaussian perturbed, 50% uniform
                        half = (pop_size - 1) // 2
                        # Gaussian perturbed points (including perhaps the best itself is kept)
                        pert_std = 0.03 * (ub - lb)
                        gauss_points = np.clip(best_x + rng.randn(half, dim) * pert_std, lb, ub)
                        # Evaluate Gaussian points
                        gauss_f = np.full(half, np.inf)
                        for k in range(half):
                            if budget <= 0:
                                break
                            gauss_f[k] = func(gauss_points[k])
                            budget -= 1
                        # Uniform points for the rest
                        uniform_count = pop_size - 1 - half
                        uniform_points = rng.uniform(lb, ub, size=(uniform_count, dim))
                        uniform_f = np.full(uniform_count, np.inf)
                        for k in range(uniform_count):
                            if budget <= 0:
                                break
                            uniform_f[k] = func(uniform_points[k])
                            budget -= 1
                        # Combine: best, gauss, uniform
                        new_pop = np.vstack((best_x.reshape(1, -1), gauss_points, uniform_points))
                        new_f = np.concatenate(([best_f], gauss_f, uniform_f))
                        pop = new_pop
                        pop_f = new_f
                        stagnation_counter = 0
            else:
                stagnation_counter = 0

        return best_f, best_x