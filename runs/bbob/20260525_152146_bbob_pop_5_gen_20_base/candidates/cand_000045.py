import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        rng = np.random.RandomState(self.seed)

        # initial population size
        NP = max(4, min(20, int(2*dim), budget//3))
        NP = max(1, min(NP, budget-1))  # ensure at least 1 and not exceed budget-1
        pop = rng.uniform(low=lb, high=ub, size=(NP, dim))
        pop_f = np.full(NP, np.inf)
        for i in range(NP):
            pop_f[i] = func(pop[i])
            budget -= 1
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]
        report_best(best_f, best_x)

        # DE/best/1/bin if enough individuals and budget left
        if NP >= 4 and budget >= NP:
            F = 0.7
            CR = 0.8
            max_gen = budget // NP
            for gen in range(max_gen):
                for i in range(NP):
                    if budget <= 0:
                        break
                    indices = [j for j in range(NP) if j != i]
                    a, b = rng.choice(indices, 2, replace=False)
                    mut = best_x + F * (pop[a] - pop[b])
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
                if budget <= 0:
                    break

        # (1+1)-ES adaptive local search
        if budget > 0:
            sigma = 0.2 * (ub - lb)
            success_counter = 0
            while budget > 0:
                pert = rng.normal(0, sigma)
                candidate = best_x + pert
                candidate = np.clip(candidate, lb, ub)
                cand_f = func(candidate)
                budget -= 1
                if cand_f < best_f:
                    best_x = candidate.copy()
                    best_f = cand_f
                    report_best(best_f, best_x)
                    success_counter += 1
                    if success_counter % 5 == 0:
                        sigma *= 1.2
                else:
                    success_counter = 0
                    sigma *= 0.85
                sigma = np.clip(sigma, 1e-12*(ub-lb), 0.5*(ub-lb))

        return best_f, best_x