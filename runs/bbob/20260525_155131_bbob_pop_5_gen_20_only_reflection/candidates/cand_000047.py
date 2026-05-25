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
                # Mutation: mix best/1 and rand/1
                candidates = list(range(self.NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                if np.random.rand() < 0.5:
                    # best/1
                    mutant = self.best_x + F * (pop[r1] - pop[r2])
                else:
                    # rand/1
                    r0 = np.random.randint(self.NP)
                    mutant = pop[r0] + F * (pop[r1] - pop[r2])
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
                restarts += 1
                stagnation = 0
                # Restart: keep best, reinitialize others
                new_pop = np.random.uniform(self.lb, self.ub, (self.NP - 1, self.dim))
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
                # Local refinement after restart: Nelder-Mead simplex around best
                local_budget = max(self.dim, int(0.05 * (self.budget - self.calls)))
                if local_budget > 0:
                    # Build initial simplex
                    delta = 0.05 * (self.ub - self.lb)
                    simplex = np.zeros((self.dim + 1, self.dim))
                    simplex[0] = self.best_x.copy()
                    for d in range(self.dim):
                        x = self.best_x.copy()
                        x[d] = np.clip(x[d] + delta[d], self.lb[d], self.ub[d])
                        simplex[d+1] = x
                    simplex_vals = np.full(self.dim + 1, float('inf'))
                    for s in range(self.dim + 1):
                        if self.calls >= self.budget:
                            break
                        val = func(simplex[s])
                        self.calls += 1
                        simplex_vals[s] = val
                        if val < self.best_val:
                            self.best_val = val
                            self.best_x = simplex[s].copy()
                            report_best(self.best_val, self.best_x)
                    # Nelder-Mead iterations
                    for _ in range(local_budget):
                        if self.calls >= self.budget:
                            break
                        # Sort
                        idx = np.argsort(simplex_vals)
                        simplex = simplex[idx]
                        simplex_vals = simplex_vals[idx]
                        centroid = np.mean(simplex[:-1], axis=0)
                        # Reflection
                        reflect = centroid + (centroid - simplex[-1])
                        reflect = np.clip(reflect, self.lb, self.ub)
                        val_ref = func(reflect)
                        self.calls += 1
                        if val_ref < simplex_vals[0]:
                            # Expansion
                            expand = centroid + 2 * (reflect - centroid)
                            expand = np.clip(expand, self.lb, self.ub)
                            val_exp = func(expand)
                            self.calls += 1
                            if val_exp < val_ref:
                                simplex[-1] = expand
                                simplex_vals[-1] = val_exp
                                if val_exp < self.best_val:
                                    self.best_val = val_exp
                                    self.best_x = expand.copy()
                                    report_best(self.best_val, self.best_x)
                            else:
                                simplex[-1] = reflect
                                simplex_vals[-1] = val_ref
                                if val_ref < self.best_val:
                                    self.best_val = val_ref
                                    self.best_x = reflect.copy()
                                    report_best(self.best_val, self.best_x)
                        elif val_ref < simplex_vals[-2]:
                            simplex[-1] = reflect
                            simplex_vals[-1] = val_ref
                            if val_ref < self.best_val:
                                self.best_val = val_ref
                                self.best_x = reflect.copy()
                                report_best(self.best_val, self.best_x)
                        else:
                            # Contraction
                            contract = centroid + 0.5 * (simplex[-1] - centroid)
                            contract = np.clip(contract, self.lb, self.ub)
                            val_cont = func(contract)
                            self.calls += 1
                            if val_cont < simplex_vals[-1]:
                                simplex[-1] = contract
                                simplex_vals[-1] = val_cont
                                if val_cont < self.best_val:
                                    self.best_val = val_cont
                                    self.best_x = contract.copy()
                                    report_best(self.best_val, self.best_x)
                            else:
                                # Shrink
                                for s in range(1, self.dim + 1):
                                    if self.calls >= self.budget:
                                        break
                                    simplex[s] = simplex[0] + 0.5 * (simplex[s] - simplex[0])
                                    simplex[s] = np.clip(simplex[s], self.lb, self.ub)
                                    val_shrink = func(simplex[s])
                                    self.calls += 1
                                    simplex_vals[s] = val_shrink
                                    if val_shrink < self.best_val:
                                        self.best_val = val_shrink
                                        self.best_x = simplex[s].copy()
                                        report_best(self.best_val, self.best_x)
                                # Update best in simplex
                                best_idx = np.argmin(simplex_vals)
                                self.best_x = simplex[best_idx].copy()
                                self.best_val = simplex_vals[best_idx]
                                report_best(self.best_val, self.best_x)
        return self.best_val, self.best_x