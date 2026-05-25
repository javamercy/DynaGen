import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.F_l = 0.1
        self.F_u = 0.9
        self.tau_F = 0.1
        self.CR_l = 0.0
        self.CR_u = 1.0
        self.tau_CR = 0.1
        self.stagnation_limit = 40
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        NP = self.NP
        # Initial population
        pop = np.random.uniform(lb, ub, (NP, self.dim))
        fitness = np.full(NP, float('inf'))
        for i in range(NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # Initialize F and CR arrays
        F = np.random.uniform(self.F_l, self.F_u, NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, NP)
        best_val_history = self.best_val
        stagnation_counter = 0
        # Main loop
        while self.calls < self.budget:
            for i in range(NP):
                if self.calls >= self.budget:
                    break
                # Self-adapt F and CR
                if np.random.rand() < self.tau_F:
                    F[i] = self.F_l + np.random.rand() * (self.F_u - self.F_l)
                if np.random.rand() < self.tau_CR:
                    CR[i] = np.random.rand()
                # Mutation
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                mutant = pop[r1] + F[i] * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluation and selection
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
            if self.best_val < best_val_history:
                best_val_history = self.best_val
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            # Restart if stagnation
            if stagnation_counter >= self.stagnation_limit:
                # Replace worst half (except best) with random points
                sorted_idx = np.argsort(fitness)
                best_idx = sorted_idx[0]
                worst_idx = sorted_idx[-(NP//2):]
                for idx in worst_idx:
                    if idx == best_idx:
                        continue
                    if self.calls >= self.budget:
                        break
                    pop[idx] = np.random.uniform(lb, ub, self.dim)
                    val = func(pop[idx])
                    self.calls += 1
                    fitness[idx] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = pop[idx].copy()
                        report_best(self.best_val, self.best_x)
                stagnation_counter = 0
        return self.best_val, self.best_x