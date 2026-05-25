import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.F_l = 0.1
        self.F_u = 0.9
        self.CR_l = 0.0
        self.CR_u = 1.0
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
        F = np.random.uniform(self.F_l, self.F_u, self.NP)
        CR = np.random.uniform(self.CR_l, self.CR_u, self.NP)
        generation = 0
        gen_no_improve = 0
        max_gen_stag = 10
        while self.calls < self.budget:
            best_improved_this_gen = False
            for i in range(self.NP):
                if self.calls >= self.budget:
                    break
                # Adaptive parameters
                if np.random.rand() < self.tau_F:
                    F[i] = np.random.uniform(self.F_l, self.F_u)
                if np.random.rand() < self.tau_CR:
                    CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                # Mutation: best or random base
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                if np.random.rand() < 0.1:
                    base = pop[r1]
                else:
                    base = self.best_x
                mutant = base + F[i] * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # Binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.where(np.random.rand(self.dim) < CR[i], mutant, pop[i])
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
                        best_improved_this_gen = True
            if not best_improved_this_gen:
                gen_no_improve += 1
            else:
                gen_no_improve = 0
            # Stagnation trigger
            if gen_no_improve >= max_gen_stag:
                # Keep best, reinitialize others if enough budget
                if self.calls + self.NP - 1 <= self.budget:
                    for i in range(self.NP):
                        if i == np.argmin(fitness):
                            continue
                        pop[i] = np.random.uniform(self.lb, self.ub, self.dim)
                        val = func(pop[i])
                        self.calls += 1
                        fitness[i] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = pop[i].copy()
                            report_best(self.best_val, self.best_x)
                        F[i] = np.random.uniform(self.F_l, self.F_u)
                        CR[i] = np.random.uniform(self.CR_l, self.CR_u)
                    gen_no_improve = 0
            generation += 1
        return self.best_val, self.best_x