import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        budget = self.budget
        rng = self.rng

        # Handle extremely small budgets
        if budget < 3:
            while evals < budget:
                x = lb + (ub - lb) * rng.rand(self.dim)
                val = func(x)
                evals += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = x.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x

        # Population size: larger for exploration, capped
        pop_size = max(5, min(30, budget // 10))
        stagnation_limit = max(1, budget // 20)
        stagnation_counter = 0

        # Initialize population with Latin hypercube sampling for better coverage
        pop = np.zeros((pop_size, self.dim))
        for i in range(pop_size):
            for d in range(self.dim):
                u = (i + rng.uniform()) / pop_size
                pop[i, d] = lb[d] + (ub[d] - lb[d]) * u
        rng.shuffle(pop)  # shuffle to avoid ordering bias
        pop_fit = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)

        # Main loop
        while evals < budget:
            # Restart condition: full restart (except best) to increase diversity
            if stagnation_counter >= stagnation_limit:
                # Keep best solution, reinitialize others
                best_idx = np.argmin(pop_fit)
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    if evals >= budget:
                        break
                    pop[i] = lb + (ub - lb) * rng.rand(self.dim)
                    pop_fit[i] = func(pop[i])
                    evals += 1
                    if pop_fit[i] < self.best_val:
                        self.best_val = pop_fit[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                stagnation_counter = 0

            # Update each individual
            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Generate mutant using rand/1 with Cauchy step for larger jumps
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                # Cauchy-distributed scale factor (exploration)
                F = 0.5 + 0.5 * rng.standard_cauchy()
                F = np.clip(F, 0.1, 2.0)  # bound to avoid extreme steps
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover with lower CR for more diversity
                CR = 0.5
                cross_points = rng.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Evaluate trial
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                # Greedy selection
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
            pop = new_pop

            # Additional diversity: sometimes inject random points
            if rng.rand() < 0.1 and evals < budget:
                idx = rng.randint(pop_size)
                pop[idx] = lb + (ub - lb) * rng.rand(self.dim)
                pop_fit[idx] = func(pop[idx])
                evals += 1
                if pop_fit[idx] < self.best_val:
                    self.best_val = pop_fit[idx]
                    self.best_x = pop[idx].copy()
                    report_best(self.best_val, self.best_x)

        return self.best_val, self.best_x