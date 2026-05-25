import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(6, min(30, budget // 15))
        self.F = 0.9
        self.CR = 0.5
        self.stall_limit = 10
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None

    def __call__(self, func):
        if self.budget == 0:
            return float('inf'), None
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
        no_improve_gen = 0
        while evals < self.budget:
            # Restart if stalled
            if no_improve_gen >= self.stall_limit:
                # Reinitialize worst half of population (excluding best)
                order = np.argsort(pop_fit)
                best_idx = order[0]
                worst_indices = order[self.pop_size//2:]
                for idx in worst_indices:
                    if idx == best_idx:
                        continue
                    pop[idx] = lb + (ub - lb) * self.rng.rand(self.dim)
                    pop_fit[idx] = func(pop[idx])
                    evals += 1
                    if pop_fit[idx] < self.best_val:
                        self.best_val = pop_fit[idx]
                        self.best_x = pop[idx].copy()
                        report_best(self.best_val, self.best_x)
                    if evals >= self.budget:
                        break
                no_improve_gen = 0
                if evals >= self.budget:
                    break
            # Normal DE generation
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            improved = False
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # rand/2 mutation: select 5 distinct indices different from i
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c, d, e = self.rng.choice(idxs, 5, replace=False)
                mutant = pop[a] + self.F * (pop[b] - pop[c]) + self.F * (pop[d] - pop[e])
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
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
                if trial_fit < new_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    improved = True
            pop = new_pop
            pop_fit = new_fit
            if improved:
                no_improve_gen = 0
            else:
                no_improve_gen += 1
            # Ensure best survives (elitism)
            best_idx_gen = np.argmin(pop_fit)
            if pop_fit[best_idx_gen] < self.best_val:
                self.best_val = pop_fit[best_idx_gen]
                self.best_x = pop[best_idx_gen].copy()
                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x