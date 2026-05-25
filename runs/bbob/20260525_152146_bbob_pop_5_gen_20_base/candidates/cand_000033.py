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

        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget - 1)

        pop = rng.uniform(low=lb, high=ub, size=(pop_size, dim))
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            pop_f[i] = func(pop[i])
            budget -= 1

        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        F_init = 0.8
        CR_init = 0.9
        F_end = 0.2
        CR_end = 0.2

        max_gen = budget // pop_size
        for gen in range(max_gen):
            if budget <= 0:
                break
            progress = gen / max_gen if max_gen > 0 else 1.0
            F = F_init + (F_end - F_init) * progress
            CR = CR_init + (CR_end - CR_init) * progress
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
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)

        sigma_init = 0.2 * (ub - lb)
        sigma_end = 0.01 * (ub - lb)
        total_evals = self.budget
        used_evals = self.budget - budget
        while budget > 0:
            frac = used_evals / total_evals
            sigma = sigma_init + (sigma_end - sigma_init) * frac
            pert = rng.normal(0, sigma)
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            used_evals += 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)

        return best_f, best_x