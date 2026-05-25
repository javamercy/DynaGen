import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = min(100, max(4, int(budget / (2 * dim))))
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0
        self.stagnation_counter = 0
        self.stagnation_limit = 50

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

        # F and CR arrays
        F = np.random.uniform(self.F_l, self.F_u, self.NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, self.NP)

        best_val_prev = self.best_val
        while self.calls < self.budget:
            # check stagnation
            if self.stagnation_counter >= self.stagnation_limit:
                # restart: reinitialize population except best
                pop = np.random.uniform(lb, ub, (self.NP - 1, self.dim))
                # keep best solution in population
                pop = np.vstack((self.best_x, pop))
                # reset fitness for new points (fitness for best preserved)
                for i in range(1, self.NP):
                    if self.calls >= self.budget:
                        break
                    val = func(pop[i])
                    self.calls += 1
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                # reset stagnation counter
                self.stagnation_counter = 0
                # reset F and CR
                F = np.random.uniform(self.F_l, self.F_u, self.NP)
                CR = np.random.uniform(self.CR_l, self.CR_u, self.NP)
                best_val_prev = self.best_val

            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # adapt F and CR
                if np.random.rand() < self.tau_F:
                    F[i] = np.random.uniform(self.F_l, self.F_u)
                if np.random.rand() < self.tau_CR:
                    CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                # select distinct indices for mutation
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

            # update stagnation counter
            if self.best_val < best_val_prev:
                self.stagnation_counter = 0
                best_val_prev = self.best_val
            else:
                self.stagnation_counter += 1

        return self.best_val, self.best_x