import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(10, min(30, budget // 10))
        if self.pop_size > budget:
            self.pop_size = budget
        self.CR = 0.95
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None
        self.patience = 5

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
        gen_no_improve = 0
        while evals < self.budget:
            # generate random F for this generation
            F = 0.5 + 0.5 * self.rng.rand()  # U[0.5,1]
            new_pop = pop.copy()
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b, c = self.rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
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
                    pop_fit[i] = trial_fit
                    gen_no_improve = 0
                else:
                    gen_no_improve += 1  # approximate: per individual
            pop = new_pop
            # restart if stagnant
            if gen_no_improve >= self.patience * self.pop_size:
                # reinitialize worst half
                idx_order = np.argsort(pop_fit)
                worst = idx_order[self.pop_size // 2:]
                for i in worst:
                    if evals >= self.budget:
                        break
                    new_x = lb + (ub - lb) * self.rng.rand(self.dim)
                    new_fit = func(new_x)
                    evals += 1
                    pop[i] = new_x
                    pop_fit[i] = new_fit
                    if new_fit < self.best_val:
                        self.best_val = new_fit
                        self.best_x = new_x.copy()
                        report_best(self.best_val, self.best_x)
                gen_no_improve = 0
        return self.best_val, self.best_x