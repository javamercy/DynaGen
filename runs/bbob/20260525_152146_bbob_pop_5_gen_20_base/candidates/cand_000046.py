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

        pop_size = min(30, max(5, budget // 3))
        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f[:pop_size])
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        CR = 0.9
        no_improve = 0

        while budget > 0:
            # Dither F per generation
            F = rng.uniform(0.5, 1.0)
            # Restart if stuck
            if no_improve >= 2:
                # Keep best, reinit rest
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    pop[i] = rng.uniform(low=lb, high=ub, size=dim)
                    # Re-evaluations happen in mutation loops below; no extra budget here
                no_improve = 0

            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b, c, d, e = rng.choice(indices, 5, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
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

            if improved:
                no_improve = 0
            else:
                no_improve += 1

        return best_f, best_x