import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.CR = 0.9
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        # initial population
        pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        fitness = np.full(self.NP, float('inf'))
        for i in range(self.NP):
            if self.calls >= self.budget:
                break
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # if budget exhausted after initial evaluations
        if self.calls >= self.budget:
            return self.best_val, self.best_x
        stagnation = 0
        stagnation_limit = max(5, self.NP // 2)
        restarts = 0
        max_restarts = 3
        lambda_param = 0.5
        while self.calls < self.budget:
            improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # mutation: current-to-best/1
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                mutant = pop[i] + lambda_param * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
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
                        improved_this_gen = True
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= stagnation_limit and restarts < max_restarts and self.calls < self.budget:
                restarts += 1
                stagnation = 0
                # reinitialize population except best: use Cauchy distribution
                new_pop = np.empty((self.NP - 1, self.dim))
                # center at best with scale 0.1*(ub-lb)
                scale = 0.1 * (self.ub - self.lb)
                for j in range(self.NP - 1):
                    new_pop[j] = self.best_x + scale * np.random.standard_cauchy(self.dim)
                    new_pop[j] = np.clip(new_pop[j], self.lb, self.ub)
                new_fitness = np.full(self.NP - 1, float('inf'))
                for j, x in enumerate(new_pop):
                    if self.calls >= self.budget:
                        break
                    val = func(x)
                    self.calls += 1
                    new_fitness[j] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = x.copy()
                        report_best(self.best_val, self.best_x)
                pop = np.vstack((self.best_x.reshape(1, -1), new_pop))
                fitness = np.concatenate(([self.best_val], new_fitness))
                # local refinement after restart
                local_budget = max(1, int(0.025 * (self.budget - self.calls)))
                sigma = 0.2 * (self.ub - self.lb)
                for _ in range(local_budget):
                    if self.calls >= self.budget:
                        break
                    candidate = self.best_x + sigma * np.random.randn(self.dim)
                    candidate = np.clip(candidate, self.lb, self.ub)
                    val = func(candidate)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = candidate.copy()
                        report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x