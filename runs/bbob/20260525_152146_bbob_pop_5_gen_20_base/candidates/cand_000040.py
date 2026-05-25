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

        # Population size: adaptive, capped at 10
        pop_size = max(4, min(10, dim))
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

        # DE/best/1/bin
        while budget > 0:
            # Generate one trial per individual per iteration
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Choose two distinct random indices different from i
                candidates = [j for j in range(pop_size) if j != i]
                a, b = rng.choice(candidates, 2, replace=False)
                # Mutation towards best
                mut = best_x + F * (pop[a] - pop[b])
                # Crossover
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)

        # Local refinement via Cauchy perturbations
        scale = 0.1 * (ub - lb)  # same order as initial sigma
        while budget > 0:
            pert = rng.standard_cauchy(size=dim) * scale
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)
                scale *= 0.9

        return best_f, best_x