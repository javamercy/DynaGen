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

        # Small population for exploitation
        pop_size = max(2, min(4, budget // 10))
        pop_size = min(pop_size, budget - 1)

        # Initial random population
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # DE phase with exploitation parameters
        F = 0.5
        CR = 0.2
        max_gen = min(20, budget // pop_size)
        for gen in range(max_gen):
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
                mut = clip(mut)
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
            if budget <= 0:
                break

        # Local refinement with adaptive Gaussian step
        sigma = 0.1 * (ub - lb)
        success_streak = 0
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
                success_streak += 1
                if success_streak >= 3:
                    sigma *= 1.2  # increase step on repeated success
                    success_streak = 0
            else:
                sigma *= 0.85  # decrease step on failure
                success_streak = 0
            # Ensure sigma doesn't become too small
            sigma = np.maximum(sigma, 1e-6 * (ub - lb))

        return best_f, best_x