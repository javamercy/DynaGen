import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget / 2), 10 * dim))
        self.F = None
        self.CR = None
        self.tau_F = 0.1
        self.tau_CR = 0.1
        self.lb = None
        self.ub = None
        self.best_x = None
        self.best_val = float('inf')
        self.calls = 0
        self.stall_generations = 0
        self.gen = 0

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        NP = self.NP
        dim = self.dim
        # Initialize population
        pop = np.random.uniform(self.lb, self.ub, (NP, dim))
        fitness = np.full(NP, np.inf)
        for i in range(NP):
            val = func(pop[i])
            self.calls += 1
            fitness[i] = val
            if val < self.best_val:
                self.best_val = val
                self.best_x = pop[i].copy()
                report_best(self.best_val, self.best_x)
        # Initialize F and CR per individual
        self.F = np.random.uniform(0.1, 0.9, NP)
        self.CR = np.random.uniform(0.0, 1.0, NP)
        # Main loop
        while self.calls < self.budget:
            # Check for restart
            if self.gen > 0 and self.stall_generations >= 20:
                # Restart: increase population size, reinitialize part of population
                new_NP = min(int(NP * 1.5), self.budget // 2)  # ensure enough budget for at least one more generation
                if new_NP > NP:
                    # Add new individuals
                    n_new = new_NP - NP
                    new_pop = np.random.uniform(self.lb, self.ub, (n_new, dim))
                    pop = np.vstack((pop, new_pop))
                    fitness = np.append(fitness, np.full(n_new, np.inf))
                    # Reinitialize 30% of old individuals (excluding best)
                    n_reinit = max(1, int(0.3 * NP))
                    reinit_idx = np.random.choice(np.arange(NP), n_reinit, replace=False)
                    for idx in reinit_idx:
                        if idx != np.argmin(fitness):
                            pop[idx] = np.random.uniform(self.lb, self.ub, dim)
                            fitness[idx] = np.inf
                    NP = new_NP
                    # Reset F and CR for new individuals
                    self.F = np.append(self.F, np.random.uniform(0.1, 0.9, n_new))
                    self.CR = np.append(self.CR, np.random.uniform(0.0, 1.0, n_new))
                    # Reinitialize F and CR for reinitialized individuals
                    for idx in reinit_idx:
                        self.F[idx] = np.random.uniform(0.1, 0.9)
                        self.CR[idx] = np.random.uniform(0.0, 1.0)
                self.stall_generations = 0
            # Check diversity: if mean distance to best is below threshold, restart similarly
            if NP > 1:
                distances = np.mean(np.abs(pop - self.best_x), axis=1)
                diversity = np.mean(distances)
                threshold = 0.01 * (self.ub - self.lb).mean()
                if diversity < threshold:
                    # Same restart as above
                    new_NP = min(int(NP * 1.5), self.budget // 2)
                    if new_NP > NP:
                        n_new = new_NP - NP
                        new_pop = np.random.uniform(self.lb, self.ub, (n_new, dim))
                        pop = np.vstack((pop, new_pop))
                        fitness = np.append(fitness, np.full(n_new, np.inf))
                        n_reinit = max(1, int(0.3 * NP))
                        reinit_idx = np.random.choice(np.arange(NP), n_reinit, replace=False)
                        for idx in reinit_idx:
                            if idx != np.argmin(fitness):
                                pop[idx] = np.random.uniform(self.lb, self.ub, dim)
                                fitness[idx] = np.inf
                        NP = new_NP
                        self.F = np.append(self.F, np.random.uniform(0.1, 0.9, n_new))
                        self.CR = np.append(self.CR, np.random.uniform(0.0, 1.0, n_new))
                        for idx in reinit_idx:
                            self.F[idx] = np.random.uniform(0.1, 0.9)
                            self.CR[idx] = np.random.uniform(0.0, 1.0)
            # Evolution for one generation
            for i in range(NP):
                if self.calls >= self.budget:
                    break
                # Generate new F, CR via jDE
                r = np.random.random()
                if r < self.tau_F:
                    new_F = np.random.uniform(0.1, 0.9)
                else:
                    new_F = self.F[i]
                r = np.random.random()
                if r < self.tau_CR:
                    new_CR = np.random.random()
                else:
                    new_CR = self.CR[i]
                # Mutation: best/1 with occasional random base (10% probability)
                if np.random.random() < 0.1:
                    # random base
                    base_idx = np.random.randint(NP)
                    base = pop[base_idx]
                else:
                    base = self.best_x
                # Select two distinct random indices not equal to i and not base_idx
                idxs = [j for j in range(NP) if j != i]
                np.random.shuffle(idxs)
                r1, r2 = idxs[:2]
                mutant = base + new_F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, self.lb, self.ub)
                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.random(dim) < new_CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Selection
                val = func(trial)
                self.calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < self.best_val:
                        self.best_val = val
                        self.best_x = trial.copy()
                        self.stall_generations = 0
                        report_best(self.best_val, self.best_x)
                    # Update F, CR if improved
                    self.F[i] = new_F
                    self.CR[i] = new_CR
                else:
                    self.stall_generations += 1
            self.gen += 1
        return self.best_val, self.best_x