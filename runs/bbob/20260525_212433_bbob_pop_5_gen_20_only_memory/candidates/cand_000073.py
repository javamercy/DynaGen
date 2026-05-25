import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(40, budget // 10))
        if self.pop_size > budget:
            self.pop_size = budget
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None
        self.w_start = 0.9
        self.w_end = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        evals = 0
        pop = lb + (ub - lb) * self.rng.rand(self.pop_size, self.dim)
        vel = (self.rng.rand(self.pop_size, self.dim) - 0.5) * (ub - lb) * 0.1
        pbest = pop.copy()
        pbest_fit = np.full(self.pop_size, np.inf)
        # initial evaluation
        for i in range(self.pop_size):
            if evals >= self.budget:
                break
            fit = func(pop[i])
            evals += 1
            pbest_fit[i] = fit
            if fit < self.best_val:
                self.best_val = fit
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # neighborhood best (ring topology)
        nbest = np.zeros_like(pop)
        nbest_fit = np.full(self.pop_size, np.inf)
        for i in range(self.pop_size):
            left = (i - 1) % self.pop_size
            right = (i + 1) % self.pop_size
            neighbors = [i, left, right]
            best_idx = neighbors[np.argmin(pbest_fit[neighbors])]
            nbest[i] = pbest[best_idx]
            nbest_fit[i] = pbest_fit[best_idx]
        gen = 0
        while evals < self.budget:
            w = self.w_start - (self.w_start - self.w_end) * gen / (self.budget // self.pop_size)
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                r1 = self.rng.rand(self.dim)
                r2 = self.rng.rand(self.dim)
                vel[i] = w * vel[i] + self.c1 * r1 * (pbest[i] - pop[i]) + self.c2 * r2 * (nbest[i] - pop[i])
                pop[i] = pop[i] + vel[i]
                pop[i] = np.clip(pop[i], lb, ub)
                fit = func(pop[i])
                evals += 1
                if fit < pbest_fit[i]:
                    pbest[i] = pop[i].copy()
                    pbest_fit[i] = fit
                    if fit < self.best_val:
                        self.best_val = fit
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
            # update neighborhood bests
            for i in range(self.pop_size):
                left = (i - 1) % self.pop_size
                right = (i + 1) % self.pop_size
                neighbors = [i, left, right]
                best_idx = neighbors[np.argmin(pbest_fit[neighbors])]
                nbest[i] = pbest[best_idx]
                nbest_fit[i] = pbest_fit[best_idx]
            gen += 1
        return self.best_val, self.best_x