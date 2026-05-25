import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(20, budget // 20))
        if self.pop_size > budget:
            self.pop_size = budget
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
        while evals < self.budget:
            F = 0.5 + 0.5 * self.rng.rand()
            new_pop = pop.copy()
            best_idx = np.argmin(pop_fit)
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                a, b = self.rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[a] - pop[b])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
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
        return self.best_val, self.best_x