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

        pop_size = min(20, budget // 2)
        if pop_size < 4:
            pop_size = 4

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

        no_improve = 0
        restart_threshold = max(5, int(0.1 * self.budget))

        while budget > 0:
            frac = 1.0 - (budget / self.budget)
            F = 0.5 + 0.5 * np.sin(2 * np.pi * frac)
            CR = 0.9 - 0.8 * frac

            improved = False
            for i in range(pop_size):
                if budget <= 0:
                    break
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mut = pop[a] + F * (pop[b] - pop[c])
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

            if not improved:
                no_improve += 1
            else:
                no_improve = 0

            if no_improve >= restart_threshold and budget > pop_size:
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]
                n_replace = pop_size // 2
                for i in range(n_replace, pop_size):
                    if budget <= 0:
                        break
                    pop[i] = rng.uniform(low=lb, high=ub)
                    pop_f[i] = func(pop[i])
                    budget -= 1
                    if pop_f[i] < best_f:
                        best_x = pop[i].copy()
                        best_f = pop_f[i]
                        report_best(best_f, best_x)
                scale = 0.1 * (ub - lb)
                for i in range(1, n_replace):
                    if budget <= 0:
                        break
                    noise = rng.standard_cauchy(size=dim) * scale
                    candidate = best_x + noise
                    candidate = np.clip(candidate, lb, ub)
                    candidate_f = func(candidate)
                    budget -= 1
                    pop[i] = candidate
                    pop_f[i] = candidate_f
                    if candidate_f < best_f:
                        best_x = candidate.copy()
                        best_f = candidate_f
                        report_best(best_f, best_x)
                no_improve = 0

        return best_f, best_x