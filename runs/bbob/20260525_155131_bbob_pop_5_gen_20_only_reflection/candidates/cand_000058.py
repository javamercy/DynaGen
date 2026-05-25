import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # Increase population size for diversity
        self.NP = max(10, min(int(budget/2), 20*dim))
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
        stagnation_limit = max(5, self.NP // 2, self.dim)
        restarts = 0
        max_restarts = 3
        partial_restart_threshold = stagnation_limit // 2
        while self.calls < self.budget:
            improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                # Mix of best/1 and rand/1 strategies for diversity
                if np.random.rand() < 0.5:
                    mutant = self.best_x + F * (pop[r1] - pop[r2])
                else:
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, self.lb, self.ub)
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
            # Partial restart to boost diversity before full restart
            if stagnation >= partial_restart_threshold and stagnation < stagnation_limit:
                # Replace worst 30% with random points
                num_replace = int(0.3 * self.NP)
                indices = np.argsort(fitness)[-num_replace:]
                for idx in indices:
                    if self.calls >= self.budget:
                        break
                    new_x = np.random.uniform(self.lb, self.ub)
                    val = func(new_x)
                    self.calls += 1
                    pop[idx] = new_x
                    fitness[idx] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = new_x.copy()
                        report_best(self.best_val, self.best_x)
            # Full restart with Cauchy emphasis
            if stagnation >= stagnation_limit and restarts < max_restarts and self.calls < self.budget:
                restarts += 1
                stagnation = 0
                new_pop = np.empty((self.NP, self.dim))
                for j in range(self.NP):
                    if np.random.rand() < 0.3:  # 30% uniform, 70% Cauchy
                        new_pop[j] = np.random.uniform(self.lb, self.ub)
                    else:
                        step = np.random.standard_cauchy(self.dim) * 0.3 * (self.ub - self.lb)
                        new_pop[j] = np.clip(self.best_x + step, self.lb, self.ub)
                new_fitness = np.full(self.NP, float('inf'))
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
                pop = new_pop
                fitness = new_fitness
                # Local refinement with varied step sizes
                local_steps = max(1, min(20, int(self.budget / 50)))
                for _ in range(local_steps):
                    if self.calls >= self.budget:
                        break
                    if np.random.rand() < 0.8:
                        step = np.random.randn(self.dim) * 0.1 * (self.ub - self.lb)
                    else:
                        step = np.random.standard_cauchy(self.dim) * 0.3 * (self.ub - self.lb)
                    candidate = np.clip(self.best_x + step, self.lb, self.ub)
                    val = func(candidate)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = candidate.copy()
                        report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x