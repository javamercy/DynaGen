import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.F = np.full(self.NP, 0.5)
        self.CR = np.full(self.NP, 0.9)
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        dim = self.dim
        NP = self.NP
        # Initialize population
        pop = np.random.uniform(self.lb, self.ub, (NP, dim))
        fitness = np.full(NP, float('inf'))
        for i in range(NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # Evolution loop
        stagnation_limit = max(10, int(0.05 * self.budget))
        no_improve_eval = 0
        while self.calls < self.budget:
            for i in range(NP):
                if self.calls >= self.budget:
                    break
                # Self-adaptive F and CR
                new_F = np.random.uniform(0.1, 1.0) if np.random.rand() < 0.1 else self.F[i]
                new_CR = np.random.uniform(0, 1) if np.random.rand() < 0.1 else self.CR[i]
                # Mutation
                candidates = [j for j in range(NP) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = pop[r1] + new_F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, self.lb, self.ub)
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < new_CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluation and selection
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    self.F[i] = new_F
                    self.CR[i] = new_CR
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        report_best(self.best_val, self.best_x)
                        no_improve_eval = 0
                    else:
                        no_improve_eval += 1
                else:
                    no_improve_eval += 1
                # Check stagnation
                if no_improve_eval >= stagnation_limit and self.calls < self.budget:
                    n_restart = max(1, int(0.2 * NP))
                    # Find best individual index to protect
                    best_idx = None
                    for idx, f in enumerate(fitness):
                        if f == self.best_val and np.allclose(pop[idx], self.best_x):
                            best_idx = idx
                            break
                    # Choose restart indices excluding best
                    candidates_restart = [idx for idx in range(NP) if idx != best_idx]
                    if len(candidates_restart) > n_restart:
                        restart_idx = np.random.choice(candidates_restart, n_restart, replace=False)
                    else:
                        restart_idx = np.array(candidates_restart)
                    for idx in restart_idx:
                        if self.calls >= self.budget:
                            break
                        pop[idx] = np.random.uniform(self.lb, self.ub, dim)
                        self.F[idx] = 0.5
                        self.CR[idx] = 0.9
                        val = func(pop[idx])
                        self.calls += 1
                        fitness[idx] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = pop[idx].copy()
                            report_best(self.best_val, self.best_x)
                    no_improve_eval = 0
        return self.best_val, self.best_x