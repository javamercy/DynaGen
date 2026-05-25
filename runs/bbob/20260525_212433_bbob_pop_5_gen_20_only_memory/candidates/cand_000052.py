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

        pop_size = min(budget, max(3, min(20, budget // 5)))
        stagnation_limit = max(1, budget // 10)
        local_budget = max(1, budget // 10)
        self.pop_size = pop_size
        self.stagnation_limit = stagnation_limit
        self.local_budget = local_budget

        pop = lb + (ub - lb) * rng.rand(pop_size, self.dim)
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

        stagnation_counter = 0
        main_budget = budget - local_budget
        while evals < main_budget:
            if stagnation_counter >= stagnation_limit:
                num_restart = pop_size // 2
                restart_idx = rng.choice(pop_size, num_restart, replace=False)
                for idx in restart_idx:
                    if evals >= main_budget:
                        break
                    pop[idx] = lb + (ub - lb) * rng.rand(self.dim)
                    pop_fit[idx] = func(pop[idx])
                    evals += 1
                    if pop_fit[idx] < self.best_val:
                        self.best_val = pop_fit[idx]
                        self.best_x = pop[idx].copy()
                        report_best(self.best_val, self.best_x)
                stagnation_counter = 0

            new_pop = pop.copy()
            for i in range(pop_size):
                if evals >= main_budget:
                    break
                exploration_rate = max(0.1, 1.0 - 0.9 * evals / budget)
                if rng.rand() < exploration_rate:
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b, c = rng.choice(idxs, 3, replace=False)
                    F = 0.5 + 0.5 * rng.rand()
                    mutant = pop[a] + F * (pop[b] - pop[c])
                else:
                    idxs = [j for j in range(pop_size) if j != i]
                    a, b = rng.choice(idxs, 2, replace=False)
                    F = 0.8
                    mutant = pop[i] + F * (self.best_x - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                CR = 0.9
                cross_points = rng.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
            pop = new_pop

        sigma = 0.1 * (ub - lb).mean()
        while evals < budget:
            candidate = self.best_x + sigma * rng.randn(self.dim)
            candidate = np.clip(candidate, lb, ub)
            fit = func(candidate)
            evals += 1
            if fit < self.best_val:
                self.best_val = fit
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
            sigma *= 0.95

        return self.best_val, self.best_x