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
        generation = 0
        stagnation = 0
        stagnation_limit = max(5, self.NP // 2)
        restarts = 0
        max_restarts = 3
        while self.calls < self.budget:
            improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # Mutation: DE/current-to-best/1 with lambda=0.5
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                lam = 0.5
                mutant = pop[i] + lam * (self.best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # Binomial crossover
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
            generation += 1
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= stagnation_limit and restarts < max_restarts and self.calls < self.budget:
                restarts += 1
                stagnation = 0
                # Restart: keep best, reinitialize others with diversity injection
                # 50% Cauchy around best, 50% uniform
                new_pop = []
                new_fitness = []
                half = (self.NP - 1) // 2
                # Cauchy perturbations
                cauchy_scale = 0.5 * (self.ub - self.lb)
                for _ in range(half):
                    if len(new_pop) >= self.NP - 1:
                        break
                    x = self.best_x + cauchy_scale * np.random.standard_cauchy(self.dim)
                    x = np.clip(x, self.lb, self.ub)
                    new_pop.append(x)
                # fill rest with uniform
                while len(new_pop) < self.NP - 1:
                    x = np.random.uniform(self.lb, self.ub, self.dim)
                    new_pop.append(x)
                new_pop = np.array(new_pop)
                for x in new_pop:
                    if self.calls >= self.budget:
                        break
                    val = func(x)
                    self.calls += 1
                    new_fitness.append(val)
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = x.copy()
                        report_best(self.best_val, self.best_x)
                # Pad fitness if budget exceeded
                while len(new_fitness) < self.NP - 1:
                    new_fitness.append(float('inf'))
                pop = np.vstack((self.best_x.reshape(1, -1), new_pop[:self.NP-1]))
                fitness = np.concatenate(([self.best_val], np.array(new_fitness[:self.NP-1])))
                # Local refinement after restart (2.5% of remaining budget)
                local_budget = max(1, int(0.025 * (self.budget - self.calls)))
                sigma = 0.1 * (self.ub - self.lb)
                for _ in range(local_budget):
                    if self.calls >= self.budget:
                        break
                    # Random direction line search
                    direction = np.random.randn(self.dim)
                    direction = direction / np.linalg.norm(direction)
                    step_size = np.random.uniform(0, sigma[0])  # sigma is array, use first dim? use scalar
                    # Actually sigma is array, adapt
                    step_size = np.random.uniform(0, 0.1 * (self.ub[0]-self.lb[0]))
                    candidate = self.best_x + step_size * direction
                    candidate = np.clip(candidate, self.lb, self.ub)
                    val = func(candidate)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = candidate.copy()
                        report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x