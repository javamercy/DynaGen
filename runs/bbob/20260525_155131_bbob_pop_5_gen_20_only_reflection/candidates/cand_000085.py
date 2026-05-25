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
        self.success_window = np.array([])
        self.window_size = int(self.NP * 0.5)

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
            successes = []
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                mutant = self.best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                j_rand = np.random.randint(self.dim)
                cr = self.CR
                trial = np.where(np.random.rand(self.dim) < cr, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    successes.append(True)
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                        improved_this_gen = True
                else:
                    successes.append(False)
            if len(successes) > 0:
                success_rate = np.mean(successes)
                self.success_window = np.append(self.success_window, success_rate)
                if len(self.success_window) > self.window_size:
                    self.success_window = self.success_window[-self.window_size:]
                if len(self.success_window) >= 10:
                    overall_success = np.mean(self.success_window)
                    if overall_success > 0.4:
                        self.CR = min(0.95, self.CR + 0.05)
                    elif overall_success < 0.2:
                        self.CR = max(0.2, self.CR - 0.05)
            generation += 1
            if self.calls < self.budget and restarts < max_restarts:
                diversity = np.mean(np.linalg.norm(pop - self.best_x, axis=1))
                diversity_threshold = 0.05 * np.linalg.norm(self.ub - self.lb)
                if not improved_this_gen:
                    stagnation += 1
                else:
                    stagnation = 0
                if (stagnation >= stagnation_limit or diversity < diversity_threshold) and self.calls < self.budget:
                    restarts += 1
                    stagnation = 0
                    new_pop = np.random.uniform(self.lb, self.ub, (self.NP - 1, self.dim))
                    cauchy_scale = 0.1 * (self.ub - self.lb)
                    cauchy = np.random.standard_cauchy((self.NP - 1, self.dim))
                    cauchy_samples = cauchy_scale * cauchy
                    cauchy_samples = np.clip(self.best_x + cauchy_samples, self.lb, self.ub)
                    mask = np.random.rand(self.NP - 1) < 0.5
                    new_pop[mask] = cauchy_samples[mask]
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
        return self.best_val, self.best_x