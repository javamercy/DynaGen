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

        # Adaptive population size, cap to leave budget for local search
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget // 2)
        if pop_size < 1:
            pop_size = 1

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

        # DE parameters with dither
        CR = 0.9

        # Main DE loop
        max_gen = budget // pop_size if pop_size > 0 else 0
        for gen in range(max_gen):
            # Dynamic F
            F = 0.9 + 0.2 * rng.rand()
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
            if budget <= 0:
                break

        # Exploration phase: random sampling and large perturbations
        while budget > 0:
            # Random point
            rand_point = rng.uniform(lb, ub, size=dim)
            rand_f = func(rand_point)
            budget -= 1
            if rand_f < best_f:
                best_x = rand_point.copy()
                best_f = rand_f
                report_best(best_f, best_x)

            if budget > 0:
                # Large Gaussian perturbation around best
                sigma = 0.3 * (ub - lb)
                pert = rng.normal(0, sigma)
                candidate = best_x + pert
                candidate = clip(candidate)
                candidate_f = func(candidate)
                budget -= 1
                if candidate_f < best_f:
                    best_x = candidate.copy()
                    best_f = candidate_f
                    report_best(best_f, best_x)

        return best_f, best_x