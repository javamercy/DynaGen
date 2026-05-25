import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.de_budget = int(0.6 * budget)
        if self.de_budget < 4:
            self.de_budget = budget // 2
        self.ls_budget = budget - self.de_budget
        self.pop_size = max(4, min(10, self.de_budget // 30))
        if self.pop_size > self.de_budget:
            self.pop_size = self.de_budget
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        # Phase 1: DE with exponential crossover
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.de_budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        while evals < self.de_budget:
            F = 0.5 + 0.5 * self.rng.rand()
            new_pop = pop.copy()
            best_idx = np.argmin(pop_fit)
            for i in range(self.pop_size):
                if evals >= self.de_budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                trial = pop[i].copy()
                j0 = self.rng.randint(self.dim)
                L = 1
                while self.rng.rand() < 0.9 and L < self.dim:
                    L += 1
                for k in range(L):
                    trial[(j0 + k) % self.dim] = mutant[(j0 + k) % self.dim]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    pop_fit[i] = trial_fit
            pop = new_pop
        # Phase 2: Local refinement
        sigma_max = 0.1 * (ub - lb)
        sigma_min = 1e-3 * (ub - lb)
        ls_evals = 0
        while evals < self.budget:
            if self.ls_budget > 0:
                t = ls_evals / self.ls_budget
            else:
                t = 1.0
            sigma = sigma_max * (1 - t) + sigma_min * t
            candidate = self.best_x + self.rng.randn(self.dim) * sigma
            candidate = np.clip(candidate, lb, ub)
            candidate_fit = func(candidate)
            evals += 1
            ls_evals += 1
            if candidate_fit < self.best_val:
                self.best_val = candidate_fit
                self.best_x = candidate.copy()
                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x