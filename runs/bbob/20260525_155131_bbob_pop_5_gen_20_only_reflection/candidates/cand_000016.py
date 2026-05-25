import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = min(100, max(4, int(budget / (2 * dim))))
        self.tau_F_init = 0.2
        self.tau_CR_init = 0.2
        self.F_l_init = 0.1
        self.F_u_init = 0.9
        self.F_l_final = 0.2
        self.F_u_final = 0.6
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.NP, self.dim))
        fitness = np.full(self.NP, float('inf'))
        for i in range(self.NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        F = np.random.uniform(self.F_l_init, self.F_u_init, self.NP)
        CR = np.random.uniform(0, 1, self.NP)
        generation = 0
        while self.calls < self.budget:
            tau_F = self.tau_F_init - (self.tau_F_init - 0.01) * (self.calls / self.budget)
            tau_CR = self.tau_CR_init - (self.tau_CR_init - 0.01) * (self.calls / self.budget)
            F_l = self.F_l_init + (self.F_l_final - self.F_l_init) * (self.calls / self.budget)
            F_u = self.F_u_init + (self.F_u_final - self.F_u_init) * (self.calls / self.budget)
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                if np.random.rand() < tau_F:
                    F[i] = np.random.uniform(F_l, F_u)
                if np.random.rand() < tau_CR:
                    CR[i] = np.random.uniform(0, 1)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
            generation += 1
        return self.best_val, self.best_x