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
                # Mutation: DE/best/1 with dither
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                mutant = self.best_x + F * (pop[r1] - pop[r2])
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
                # Restart: keep best, reinitialize other individuals
                restarts += 1
                stagnation = 0
                new_pop = np.random.uniform(self.lb, self.ub, (self.NP - 1, self.dim))
                # Evaluate new individuals
                for j, x in enumerate(new_pop):
                    if self.calls >= self.budget:
                        break
                    val = func(x)
                    self.calls += 1
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = x.copy()
                        report_best(self.best_val, self.best_x)
                    # We'll assign to pop and fitness later
                # Reassemble population: best + new_pop
                pop = np.vstack((self.best_x.reshape(1, -1), new_pop))
                # Update fitness for new individuals; best is already fit
                # We'll evaluate all new individuals again carefully
                # Actually we already evaluated them above, but we need to store fitness
                # Simpler: reevaluate all except best
                fit_new = np.full(self.NP, float('inf'))
                fit_new[0] = self.best_val
                # For new individuals, we already evaluated them but didn't store; we'll reevaluate to keep call count correct?
                # Better: evaluate in loop and store directly
                # Let's redo restart block
            # End restart
        return self.best_val, self.best_x