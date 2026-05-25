import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget / 2), 10 * dim))
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0
        self.H = 5
        self.M_F = np.ones(self.H) * 0.5
        self.M_CR = np.ones(self.H) * 0.9
        self.mem_idx = 0
        self.success_F = []
        self.success_CR = []
        self.success_fitness_diff = []

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        bounds_range = self.ub - self.lb
        bounds_range[bounds_range == 0] = 1.0
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
        stagnation = 0
        restarts = 0
        max_restarts = 3
        first_gen = True
        while self.calls < self.budget:
            # Diversity trigger (skip first generation)
            if not first_gen:
                div = self._diversity(pop, bounds_range)
                if div < 0.05 and stagnation > 2 and restarts < max_restarts:
                    pop, fitness = self._restart(func)
                    restarts += 1
                    stagnation = 0
                    first_gen = True
                    continue
            # Generate offspring
            improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                idx = np.random.randint(self.H)
                F = self._sample_cauchy(self.M_F[idx], 0.1)
                CR = self._sample_cauchy(self.M_CR[idx], 0.1)
                F = np.clip(F, 0.1, 0.9)
                CR = np.clip(CR, 0.0, 1.0)
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                mutant = self.best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])
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
                    self.success_F.append(F)
                    self.success_CR.append(CR)
                    diff = max(fitness[i] - val, 0)
                    self.success_fitness_diff.append(diff)
            # Update memory
            if len(self.success_F) > 0:
                weights = np.array(self.success_fitness_diff)
                if np.sum(weights) > 0:
                    w = weights / np.sum(weights)
                    mean_F = np.sum(w * np.array(self.success_F)**2) / np.sum(w * np.array(self.success_F))
                    mean_CR = np.sum(w * np.array(self.success_CR))
                else:
                    mean_F = np.mean(self.success_F)
                    mean_CR = np.mean(self.success_CR)
                self.M_F[self.mem_idx] = mean_F
                self.M_CR[self.mem_idx] = mean_CR
                self.mem_idx = (self.mem_idx + 1) % self.H
                self.success_F.clear()
                self.success_CR.clear()
                self.success_fitness_diff.clear()
            # Stagnation counter
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1
            # Stagnation restart
            if stagnation >= 10 and restarts < max_restarts and self.calls < self.budget:
                pop, fitness = self._restart(func)
                restarts += 1
                stagnation = 0
                first_gen = True
                continue
            first_gen = False
        return self.best_val, self.best_x

    def _restart(self, func):
        NP_new = self.NP - 1
        new_pop = np.random.uniform(self.lb, self.ub, (NP_new, self.dim))
        cauchy_scale = 0.1 * (self.ub - self.lb)
        cauchy_samples = cauchy_scale * np.random.standard_cauchy((NP_new, self.dim))
        cauchy_samples = np.clip(self.best_x + cauchy_samples, self.lb, self.ub)
        mask = np.random.rand(NP_new) < 0.5
        new_pop[mask] = cauchy_samples[mask]
        new_fitness = np.full(NP_new, float('inf'))
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
        # Local refinement
        local_budget = max(1, int(0.05 * (self.budget - self.calls)))
        sigma = 0.1 * (self.ub - self.lb)
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
                sigma *= 1.1
            else:
                sigma *= 0.9
        return pop, fitness

    def _diversity(self, pop, bounds_range):
        n = len(pop)
        if n <= 1:
            return 1.0
        sum_dist = 0.0
        for i in range(n):
            for j in range(i+1, n):
                diff = pop[i] - pop[j]
                dist = np.sqrt(np.sum((diff / bounds_range)**2))
                sum_dist += dist
        num_pairs = n * (n - 1) / 2
        return sum_dist / num_pairs

    def _sample_cauchy(self, location, scale):
        return location + scale * np.random.standard_cauchy()