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

        def clip(x):
            return np.clip(x, lb, ub)

        # Larger population for exploration
        pop_size = max(10, min(50, int(4 * dim)))
        pop_size = min(pop_size, budget - 1)

        # Initial population
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F = 0.8
        CR = 0.9
        stagnation_counter = 0
        max_stagnation = 5

        # Differential Evolution generations
        while budget > 0 and pop_size > 1:
            # Check for stagnation and restart
            if stagnation_counter >= max_stagnation:
                # Replace worst half of population with random points
                worst_indices = np.argsort(pop_f)[-pop_size//2:]
                new_points = rng.uniform(low=lb, high=ub, size=(len(worst_indices), dim))
                for j, idx in enumerate(worst_indices):
                    if budget <= 0:
                        break
                    pop[idx] = new_points[j]
                    pop_f[idx] = func(pop[idx])
                    budget -= 1
                    if pop_f[idx] < best_f:
                        best_x = pop[idx].copy()
                        best_f = pop_f[idx]
                        report_best(best_f, best_x)
                stagnation_counter = 0

            if budget <= 0:
                break

            # One generation of DE
            new_pop = pop.copy()
            new_pop_f = pop_f.copy()
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    new_pop[i] = trial
                    new_pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
                        stagnation_counter = 0
            pop = new_pop
            pop_f = new_pop_f
            stagnation_counter += 1

        # Local refinement via Gaussian perturbations (explorative start)
        sigma = 0.2 * (ub - lb)
        while budget > 0:
            pert = rng.normal(0, sigma, size=dim)
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)
                sigma *= 0.9  # shrink on improvement
            # else keep sigma same for more exploration

        return best_f, best_x