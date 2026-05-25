import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(budget // 2, 10 * dim))
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        # Initialize population
        pop = np.random.uniform(self.lb, self.ub, (self.NP, self.dim))
        fitness = np.full(self.NP, float('inf'))
        F = np.full(self.NP, 0.5)
        CR = np.full(self.NP, 0.9)
        # Evaluate initial population
        for i in range(self.NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # Main loop
        gen = 0
        last_improvement_gen = 0
        while self.calls < self.budget:
            old_best_val = self.best_val
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # Select two distinct indices different from i
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                # jDE: update F and CR with probability 0.1
                if np.random.rand() < 0.1:
                    F[i] = 0.1 + 0.9 * np.random.rand()
                if np.random.rand() < 0.1:
                    CR[i] = np.random.rand()
                # Mutation: best/1
                mutant = self.best_x + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # Binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluate
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
            # Check stagnation
            gen += 1
            if self.best_val < old_best_val:
                last_improvement_gen = gen
            elif gen - last_improvement_gen >= 5:
                # Restart: keep best, reinitialize others
                for i in range(1, self.NP):
                    if self.calls >= self.budget:
                        break
                    pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                    val = func(pop[i])
                    self.calls += 1
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = pop[i].copy()
                        report_best(self.best_val, self.best_x)
                    F[i] = 0.5
                    CR[i] = 0.9
                last_improvement_gen = gen
        return self.best_val, self.best_x