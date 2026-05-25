import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(5, min(20, budget // 20))
        if self.pop_size > budget:
            self.pop_size = budget
        if self.pop_size < 3:
            self.pop_size = 3
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # Initial population
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            val = func(pop[i])
            evals += 1
            pop_fit[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # DE phase: 75% budget
        de_budget = int(0.75 * self.budget)
        # Additional budget for restarts (up to 10% of total)
        restart_budget = int(0.1 * self.budget)
        de_evals_limit = de_budget - restart_budget  # reserve some evaluations for restarts
        if de_evals_limit < pop_size:
            de_evals_limit = pop_size
        generation = 0
        stall_count = 0
        best_prev_gen = self.best_val
        while evals < min(self.budget, de_evals_limit):
            generation += 1
            fraction = evals / max(1, de_evals_limit)
            F = 0.9 - fraction * 0.5  # F ends at 0.4
            CR = 0.9 - fraction * 0.3  # CR ends at 0.6
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            for i in range(self.pop_size):
                if evals >= min(self.budget, de_evals_limit):
                    break
                # Decide: with 10% chance, generate random point for exploration
                if self.rng.rand() < 0.1:
                    trial = lb + (ub - lb) * self.rng.rand(self.dim)
                else:
                    idxs = [j for j in range(self.pop_size) if j != i]
                    a, b, c = self.rng.choice(idxs, 3, replace=False)
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, lb, ub)
                    cross_points = self.rng.rand(self.dim) < CR
                    if not np.any(cross_points):
                        cross_points[self.rng.randint(self.dim)] = True
                    trial = np.where(cross_points, mutant, pop[i])
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            pop = new_pop
            pop_fit = new_fit
            # Check for stagnation
            if self.best_val < best_prev_gen:
                stall_count = 0
                best_prev_gen = self.best_val
            else:
                stall_count += 1
            if stall_count >= 3 and evals + pop_size // 3 <= min(self.budget, de_evals_limit):
                # Replace worst 30% with random points
                worst_indices = np.argsort(pop_fit)[-max(1, int(0.3 * self.pop_size)):]
                for idx in worst_indices:
                    if evals >= min(self.budget, de_evals_limit):
                        break
                    new_point = lb + (ub - lb) * self.rng.rand(self.dim)
                    val = func(new_point)
                    evals += 1
                    pop[idx] = new_point
                    pop_fit[idx] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = new_point.copy()
                        report_best(self.best_val, self.best_x)
                stall_count = 0
        # Local search phase: remaining budget
        while evals < self.budget:
            if self.rng.rand() < 0.2:
                candidate = lb + (ub - lb) * self.rng.rand(self.dim)
            else:
                fraction = evals / self.budget
                sigma = 0.1 * (1 - fraction) * (ub - lb).mean()
                candidate = self.best_x + sigma * self.rng.randn(self.dim)
                candidate = np.clip(candidate, lb, ub)
            val = func(candidate)
            evals += 1
            if val < self.best_val:
                self.best_val = val
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x