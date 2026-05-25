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
        F = np.full(self.NP, 0.5)
        CR = np.full(self.NP, 0.9)
        while self.calls < self.budget:
            if self.calls - self.last_improvement_calls > 0.1 * self.budget:
                # restart if enough budget left for re-evaluation
                if self.calls + self.NP - 1 <= self.budget:
                    # diversity-guided reinitialization
                    new_pop = np.empty_like(pop)
                    new_pop[0] = self.best_x.copy()
                    for j in range(1, self.NP):
                        r = np.random.uniform(self.lb, self.ub)
                        new_pop[j] = self.best_x + 0.5 * (r - self.best_x)  # bias towards best
                        new_pop[j] = np.clip(new_pop[j], self.lb, self.ub)
                        val = func(new_pop[j])
                        self.calls += 1
                        fitness[j] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = new_pop[j].copy()
                            report_best(self.best_val, self.best_x)
                            self.last_improvement_calls = self.calls
                    pop = new_pop
                    fitness[0] = self.best_val
                # reset parameters
                F = np.full(self.NP, 0.5)
                CR = np.full(self.NP, 0.9)
                # local search after restart
                local_search_budget = max(1, int(0.05 * self.budget))
                for _ in range(local_search_budget):
                    if self.calls >= self.budget:
                        break
                    step = np.random.randn(self.dim) * 0.01 * (self.ub - self.lb)
                    trial = np.clip(self.best_x + step, self.lb, self.ub)
                    val = func(trial)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                        self.last_improvement_calls = self.calls
                self.last_improvement_calls = self.calls
            # evolve each target
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                r = np.random.rand()
                if r < self.tau1:
                    F[i] = self.F_l + np.random.rand() * (self.F_u - self.F_l)
                r = np.random.rand()
                if r < self.tau2:
                    CR[i] = self.CR_l + np.random.rand() * (self.CR_u - self.CR_l)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
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