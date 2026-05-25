import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(12, budget // 25))
        if self.pop_size < 1:
            self.pop_size = 1
        self.F = 0.5
        self.CR = 0.7
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
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
        if self.best_x is None:
            # fallback: return first point
            self.best_x = pop[0].copy()
        stags = 0
        while evals < self.budget:
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            improve = False
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                r1, r2 = self.rng.choice(idxs, 2, replace=False)
                mutant = self.best_x + self.F * (pop[r1] - pop[r2])
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
                    improve = True
                if trial_fit < new_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
            pop = new_pop
            pop_fit = new_fit
            if improve:
                stags = 0
            else:
                stags += 1
            if stags >= 3 and evals < self.budget:
                radius = 0.2 * (ub - lb)
                new_pop_list = [self.best_x]
                new_fit_list = [self.best_val]
                for _ in range(self.pop_size - 1):
                    if evals >= self.budget:
                        break
                    point = self.best_x + self.rng.uniform(-1, 1, self.dim) * radius
                    point = np.clip(point, lb, ub)
                    fit = func(point)
                    evals += 1
                    if fit < self.best_val:
                        self.best_val = fit
                        self.best_x = point.copy()
                        report_best(self.best_val, self.best_x)
                    new_pop_list.append(point)
                    new_fit_list.append(fit)
                pop = np.array(new_pop_list)
                pop_fit = np.array(new_fit_list)
                stags = 0
        return self.best_val, self.best_x