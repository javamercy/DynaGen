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

        # population size
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget - 1)

        # initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if budget <= 0:
                break
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # DE parameters: sample F and CR adaptively
        F_mean = 0.8
        F_std = 0.1
        CR_mean = 0.9
        CR_std = 0.1

        # main DE loop
        max_gen = budget // pop_size if pop_size > 0 else 0
        for gen in range(max_gen):
            if budget <= 0:
                break
            F = np.clip(rng.normal(F_mean, F_std), 0, 2)
            CR = np.clip(rng.normal(CR_mean, CR_std), 0, 1)
            for i in range(pop_size):
                if budget <= 0:
                    break
                a, b = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mut = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
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

        # local refinement with exponentially decaying step size
        sigma0 = 0.1 * (ub - lb)
        sigma = sigma0.copy()
        decay = 0.95
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
                sigma = sigma0.copy()
            else:
                sigma *= decay

        return best_f, best_x