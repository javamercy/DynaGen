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

        # Population size adaptive to dimension
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
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

        # Linear mean schedules for F and CR
        F_start, F_end = 0.9, 0.3
        CR_start, CR_end = 0.9, 0.1
        F_std, CR_std = 0.1, 0.1

        max_gen = budget // pop_size if budget > 0 else 0
        for gen in range(max_gen):
            if budget <= 0:
                break
            frac = gen / max_gen if max_gen > 0 else 0
            F_mean = F_start + (F_end - F_start) * frac
            CR_mean = CR_start + (CR_end - CR_start) * frac
            # Sample F and CR from truncated normal
            F = np.clip(rng.normal(F_mean, F_std), 0, 2)
            CR = np.clip(rng.normal(CR_mean, CR_std), 0, 1)
            for i in range(pop_size):
                if budget <= 0:
                    break
                # DE/rand/1/bin
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

        # Local refinement via Gaussian perturbations
        sigma0 = 0.1 * (ub - lb)
        sigma = sigma0.copy()
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
                sigma = sigma0.copy()  # reset on success
            else:
                sigma *= 0.95  # decay on failure

        return best_f, best_x