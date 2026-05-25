import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0
        self.last_improvement_calls = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        # initialize population
        pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        fitness = np.full(self.NP, float('inf'))
        for i in range(self.NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        self.last_improvement_calls = self.calls
        # initialize F and CR arrays
        F = np.full(self.NP, 0.5)
        CR = np.full(self.NP, 0.9)
        while self.calls < self.budget:
            # stagnation detection
            if self.calls - self.last_improvement_calls > 0.1 * self.budget:
                if self.calls + 1 <= self.budget:
                    # diversity-guided restart: reinitialize 20% of population (excluding best)
                    num_restart = max(1, int(0.2 * self.NP))
                    # local search on best
                    for _ in range(5):
                        if self.calls >= self.budget:
                            break
                        step_size = 0.01 * (self.ub - self.lb)
                        candidate = self.best_x + np.random.normal(0, step_size, self.dim)
                        candidate = np.clip(candidate, self.lb, self.ub)
                        val = func(candidate)
                        self.calls += 1
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = candidate.copy()
                            report_best(self.best_val, self.best_x)
                            self.last_improvement_calls = self.calls
                    # reinitialize a subset of population with Cauchy around best
                    indices = np.random.choice(range(1, self.NP), num_restart, replace=False)
                    for idx in indices:
                        if self.calls >= self.budget:
                            break
                        # Cauchy distribution
                        scale = 0.1 * (self.ub - self.lb)
                        perturbation = np.random.standard_cauchy(self.dim) * scale
                        new_point = self.best_x + perturbation
                        new_point = np.clip(new_point, self.lb, self.ub)
                        val = func(new_point)
                        self.calls += 1
                        pop[idx] = new_point
                        fitness[idx] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = new_point.copy()
                            report_best(self.best_val, self.best_x)
                            self.last_improvement_calls = self.calls
                # reset parameters
                F = np.full(self.NP, 0.5)
                CR = np.full(self.NP, 0.9)
                self.last_improvement_calls = self.calls
            # evolve each target
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # adapt F and CR
                r = np.random.rand()
                if r < self.tau1:
                    F[i] = self.F_l + np.random.rand() * (self.F_u - self.F_l)
                r = np.random.rand()
                if r < self.tau2:
                    CR[i] = self.CR_l + np.random.rand() * (self.CR_u - self.CR_l)
                # mutation (best/1)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # binomial crossover
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
                        self.last_improvement_calls = self.calls
        return self.best_val, self.best_x