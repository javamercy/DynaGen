import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.NP = min(self.NP, budget)
        self.CR = 0.9
        self.CR_memory = []
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        # Fallback for very small budget
        if self.budget < 3:
            self.best_x = np.random.uniform(self.lb, self.ub)
            self.best_val = func(self.best_x)
            self.calls = 1
            report_best(self.best_val, self.best_x)
            for _ in range(self.budget - 1):
                if self.calls >= self.budget:
                    break
                candidate = np.random.uniform(self.lb, self.ub)
                val = func(candidate)
                self.calls += 1
                if val < self.best_val:
                    self.best_val = val
                    self.best_x = candidate.copy()
                    report_best(self.best_val, self.best_x)
            return self.best_val, self.best_x
        # Initialize population
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
        if self.calls >= self.budget:
            return self.best_val, self.best_x
        generation = 0
        stagnation = 0
        stagnation_limit = max(5, self.NP // 2)
        restarts = 0
        max_restarts = 3
        while self.calls < self.budget:
            improved_this_gen = False
            # Update CR adaptively from memory
            if len(self.CR_memory) >= 10:
                self.CR = np.median(self.CR_memory[-20:])
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # Choose mutation scheme
                use_rand = np.random.rand() < 0.3
                candidate_indices = list(range(self.NP))
                candidate_indices.remove(i)
                if len(candidate_indices) < 2:
                    # Fallback: random search
                    candidate = np.random.uniform(self.lb, self.ub)
                    val = func(candidate)
                    self.calls += 1
                    if val < fitness[i]:
                        pop[i] = candidate
                        fitness[i] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = candidate.copy()
                            report_best(self.best_val, self.best_x)
                            improved_this_gen = True
                    continue
                r1, r2 = np.random.choice(candidate_indices, 2, replace=False)
                if use_rand or self.best_val == float('inf'):
                    # DE/rand/1
                    r3 = np.random.choice([j for j in candidate_indices if j not in (r1, r2)], 1)[0]
                    F = 0.5 + 0.5 * np.random.rand()
                    mutant = pop[r3] + F * (pop[r1] - pop[r2])
                else:
                    # DE/best/1 with dither
                    F = 0.5 + 0.5 * np.random.rand()
                    mutant = self.best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    self.CR_memory.append(self.CR)
                    if len(self.CR_memory) > 20:
                        self.CR_memory.pop(0)
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
                # Keep best, reinitialize rest
                new_size = self.NP - 1
                new_pop = np.random.uniform(self.lb, self.ub, (new_size, self.dim))
                new_fitness = np.full(new_size, float('inf'))
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
                if self.calls >= self.budget:
                    break
                pop = np.vstack((self.best_x.reshape(1, -1), new_pop))
                fitness = np.concatenate(([self.best_val], new_fitness))
                # Local refinement after restart with larger step
                remaining = self.budget - self.calls
                local_budget = max(1, int(0.05 * remaining))
                local_budget = min(local_budget, remaining)
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