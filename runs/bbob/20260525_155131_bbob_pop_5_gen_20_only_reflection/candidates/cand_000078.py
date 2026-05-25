import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.mu_CR = 0.9
        self.mu_F = 0.5
        self.M = 10
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
        local_scale = 0.1
        success_CR = []
        success_F = []
        while self.calls < self.budget:
            improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # Sample CR and F from memory
                if np.random.rand() < 0.1 or len(success_CR) == 0:
                    CR_i = self.mu_CR
                else:
                    CR_i = np.random.choice(success_CR)
                if np.random.rand() < 0.1 or len(success_F) == 0:
                    F_i = self.mu_F
                else:
                    F_i = np.random.choice(success_F)
                # Dither F around sampled value
                F_i = F_i + 0.1 * np.random.randn()
                F_i = np.clip(F_i, 0.1, 1.0)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F_i * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR_i, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    success_CR.append(CR_i)
                    success_F.append(F_i)
                    if len(success_CR) > self.M:
                        success_CR.pop(0)
                        success_F.pop(0)
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                        improved_this_gen = True
            generation += 1
            # Update memory with mean of successes
            if len(success_CR) > 0:
                self.mu_CR = np.mean(success_CR)
                self.mu_F = np.mean(success_F)
            # Stagnation and diversity checks
            pop_var = np.var(pop, axis=0)
            max_var = np.max(pop_var)
            diversity_low = max_var < 1e-6 * np.prod(self.ub - self.lb)
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1
            if (stagnation >= stagnation_limit or diversity_low) and restarts < max_restarts and self.calls < self.budget:
                restarts += 1
                stagnation = 0
                # Reset memory
                self.mu_CR = 0.9
                self.mu_F = 0.5
                success_CR = []
                success_F = []
                # Reinitialize population: mix uniform and Cauchy around best
                new_pop = np.empty((self.NP - 1, self.dim))
                for j in range(self.NP - 1):
                    if np.random.rand() < 0.5:
                        new_pop[j] = np.random.uniform(self.lb, self.ub)
                    else:
                        step = np.random.standard_cauchy(self.dim) * 0.2 * (self.ub - self.lb)
                        new_pop[j] = np.clip(self.best_x + step, self.lb, self.ub)
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
                # Local refinement with adaptive step
                local_steps = max(1, min(10, int(self.budget / 100)))
                for _ in range(local_steps):
                    if self.calls >= self.budget:
                        break
                    if np.random.rand() < 0.9:
                        step = np.random.randn(self.dim) * 0.01 * (self.ub - self.lb)
                    else:
                        step = np.random.standard_cauchy(self.dim) * local_scale * (self.ub - self.lb)
                    candidate = np.clip(self.best_x + step, self.lb, self.ub)
                    val = func(candidate)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = candidate.copy()
                        report_best(self.best_val, self.best_x)
                        pop[0] = self.best_x
                        fitness[0] = self.best_val
                        local_scale = min(1.0, local_scale * 1.2)
                    else:
                        local_scale = max(0.01, local_scale * 0.9)
        return self.best_val, self.best_x