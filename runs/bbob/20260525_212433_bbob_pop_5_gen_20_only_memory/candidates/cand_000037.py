import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.pop_size = max(4, min(20, budget // 20))
        if self.pop_size > budget:
            self.pop_size = budget
        self.CR = 0.9
        self.rng = np.random.RandomState(seed)
        self.best_val = float('inf')
        self.best_x = None
        # stagnation parameters
        self.restart_threshold = max(5, 10 + dim // 5)
        self.no_improve_gens = 0

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
        gen = 0
        while evals < self.budget:
            F = 0.5 + 0.5 * self.rng.rand()
            # check for stagnation every generation after at least one restart is possible
            if gen > 0 and self.no_improve_gens >= self.restart_threshold:
                # restart: replace all but the best with random points
                for i in range(self.pop_size):
                    if i == 0:  # keep best at index 0
                        pop[i] = self.best_x.copy()
                        pop_fit[i] = self.best_val
                    else:
                        pop[i] = lb + (ub - lb) * self.rng.rand(self.dim)
                        if evals < self.budget:
                            pop_fit[i] = func(pop[i])
                            evals += 1
                            if pop_fit[i] < self.best_val:
                                self.best_val = pop_fit[i]
                                self.best_x = pop[i].copy()
                                report_best(self.best_val, self.best_x)
                self.no_improve_gens = 0
            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            improved = False
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                idxs = [j for j in range(self.pop_size) if j != i]
                if len(idxs) < 5:
                    continue
                # select 5 distinct indices for rand/2
                selected = self.rng.choice(idxs, 5, replace=False)
                a, b, c, d, e = selected
                mutant = pop[a] + F * (pop[b] - pop[c]) + F * (pop[d] - pop[e])
                mutant = np.clip(mutant, lb, ub)
                # exponential crossover
                trial = pop[i].copy()
                start = self.rng.randint(self.dim)
                L = 0
                while self.rng.rand() < self.CR and L < self.dim:
                    idx = (start + L) % self.dim
                    trial[idx] = mutant[idx]
                    L += 1
                trial_fit = func(trial)
                evals += 1
                if trial_fit < self.best_val:
                    self.best_val = trial_fit
                    self.best_x = trial.copy()
                    report_best(self.best_val, self.best_x)
                if trial_fit < pop_fit[i]:
                    new_pop[i] = trial
                    new_fit[i] = trial_fit
                    improved = True
            if improved:
                self.no_improve_gens = 0
            else:
                self.no_improve_gens += 1
            pop = new_pop
            pop_fit = new_fit
            gen += 1
        return self.best_val, self.best_x