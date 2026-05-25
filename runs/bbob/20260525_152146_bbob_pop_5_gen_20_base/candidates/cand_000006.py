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

        F = 0.8
        CR = 0.9

        # Reserve budget for local search
        local_reserve = min(100, max(10, self.budget // 5))
        de_budget = budget - local_reserve
        if de_budget > 0:
            max_gen = de_budget // pop_size
            for gen in range(max_gen):
                for i in range(pop_size):
                    if budget <= local_reserve:
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
                if budget <= local_reserve:
                    break

        # Local refinement with restarts
        sigma = 0.1 * (ub - lb)
        stagnation_limit = max(5, int(self.budget * 0.01))
        stagnation_counter = 0
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
                sigma = sigma * 0.9
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                if stagnation_counter >= stagnation_limit:
                    # Restart with new random point
                    candidate = rng.uniform(lb, ub)
                    cand_f = func(candidate)
                    budget -= 1
                    if cand_f < best_f:
                        best_x = candidate.copy()
                        best_f = cand_f
                        report_best(best_f, best_x)
                    sigma = 0.1 * (ub - lb)
                    stagnation_counter = 0
        return best_f, best_x