import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        if budget >= 50:
            self.pop_size = max(4, min(20, int(budget / (dim + 1))))
        else:
            self.pop_size = max(4, int(budget / 5))
        self.F = 0.8
        self.CR = 0.9
        self.stagnation_limit = max(5, dim)
        self.best_val = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        pop_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            pop_fit[i] = func(pop[i])
            evals += 1
            if pop_fit[i] < self.best_val:
                self.best_val = pop_fit[i]
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        no_improve = 0
        while evals < self.budget:
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = self.rng.rand(self.dim) < self.CR
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
            if np.min(new_fit) < self.best_val:
                no_improve = 0
            else:
                no_improve += 1
            pop = new_pop
            pop_fit = new_fit
            if no_improve >= self.stagnation_limit and evals < self.budget:
                best_idx = np.argmin(pop_fit)
                pop[0] = pop[best_idx].copy()
                pop_fit[0] = pop_fit[best_idx]
                for i in range(1, self.pop_size):
                    if evals >= self.budget:
                        break
                    pop[i] = lb + (ub - lb) * self.rng.rand(self.dim)
                    pop_fit[i] = func(pop[i])
                    evals += 1
                    if pop_fit[i] < self.best_val:
                        self.best_val = pop_fit[i]
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                no_improve = 0
        return self.best_val, self.best_x