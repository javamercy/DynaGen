import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0
        self.stag_gen = 0
        self.tol_stag = max(10, dim)

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        self._init_population(func)
        while self.calls < self.budget:
            F = np.random.uniform(self.F_l, self.F_u, self.NP)
            CR = np.random.uniform(self.CR_l, self.CR_u, self.NP)
            improved = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # jDE adaptation
                if np.random.rand() < self.tau_F:
                    F[i] = np.random.uniform(self.F_l, self.F_u)
                if np.random.rand() < self.tau_CR:
                    CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                # mutation
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F[i] * (self.pop[r1] - self.pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, self.pop[i])
                trial[j_rand] = mutant[j_rand]
                # evaluate
                val = func(trial)
                self.calls += 1
                if val < self.fitness[i]:
                    self.pop[i] = trial
                    self.fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        improved = True
                        report_best(self.best_val, self.best_x)
            if improved:
                self.stag_gen = 0
            else:
                self.stag_gen += 1
                if self.stag_gen >= self.tol_stag:
                    # restart
                    self._init_population(func)
                    self.stag_gen = 0
        return self.best_val, self.best_x

    def _init_population(self, func):
        self.pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        self.fitness = np.full(self.NP, float('inf'))
        for i in range(self.NP):
            if self.calls >= self.budget:
                break
            val = func(self.pop[i])
            self.calls += 1
            self.fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = self.pop[i].copy()
                report_best(self.best_val, self.best_x)
        self.stag_gen = 0