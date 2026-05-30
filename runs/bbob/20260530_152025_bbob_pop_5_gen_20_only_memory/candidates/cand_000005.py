import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        def clip(x):
            return np.clip(x, lb, ub)

        # initial population size, safe for small budgets
        pop_size = min(budget // 3, max(5, dim * 2))
        pop_size = max(pop_size, 2)  # at least 2
        pop = [rng.uniform(lb, ub, dim) for _ in range(pop_size)]
        f_pop = np.array([func(x) for x in pop])
        evals = pop_size
        best_idx = np.argmin(f_pop)
        best_x = pop[best_idx].copy()
        best_f = f_pop[best_idx]
        report_best(best_f, best_x)

        # DE parameters
        F = 0.8
        CR = 0.9

        # number of full DE generations possible
        remaining = budget - evals
        max_gens = remaining // pop_size

        for gen in range(max_gens):
            for i in range(pop_size):
                # select three distinct random indices not equal to i
                candidates = [j for j in range(pop_size) if j != i]
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = clip(pop[r1] + F * (pop[r2] - pop[r3]))
                # crossover
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = clip(trial)
                f_trial = func(trial)
                evals += 1
                if f_trial < f_pop[i]:
                    pop[i] = trial
                    f_pop[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)

        # remaining budget for local random perturbations
        while evals < budget:
            step = 0.1 * (ub - lb) * (1 - (evals / budget))  # decreasing step
            x_new = best_x + rng.normal(0, step)
            x_new = clip(x_new)
            f_new = func(x_new)
            evals += 1
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                report_best(best_f, best_x)

        return best_f, best_x