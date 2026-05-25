import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        NP = self.NP
        dim = self.dim
        calls = 0
        best_val = float('inf')
        best_x = None

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.full(NP, float('inf'))
        for i in range(NP):
            val = func(pop[i])
            calls += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        stagnation = 0
        stagnation_limit = max(5, NP // 2)
        restarts = 0
        max_restarts = 3

        while calls < self.budget:
            improved_gen = False
            # Compute probability of using rand/1 based on remaining budget
            frac = calls / self.budget
            p_rand = max(0.2, 0.5 - 0.3 * frac)
            for i in range(NP):
                if calls >= self.budget:
                    break
                # Mutation strategy selection
                if np.random.rand() < p_rand:
                    # DE/rand/1/bin
                    idxs = [j for j in range(NP) if j != i]
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    F = 0.5 + 0.5 * np.random.rand()
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                else:
                    # DE/best/1/bin
                    idxs = [j for j in range(NP) if j != i]
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    F = 0.5 + 0.5 * np.random.rand()
                    mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                calls += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        improved_gen = True

            if improved_gen:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= stagnation_limit and restarts < max_restarts and calls < self.budget:
                restarts += 1
                stagnation = 0
                # Reinitialize population except best
                new_pop = np.random.uniform(lb, ub, (NP - 1, dim))
                new_fitness = np.full(NP - 1, float('inf'))
                for j, x in enumerate(new_pop):
                    if calls >= self.budget:
                        break
                    val = func(x)
                    calls += 1
                    new_fitness[j] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Assemble population
                pop = np.vstack((best_x.reshape(1, -1), new_pop))
                fitness = np.concatenate(([best_val], new_fitness))

                # Local refinement on best
                step_size = 0.01 * (ub - lb)
                for _ in range(min(5, (self.budget - calls) // dim + 1)):
                    if calls >= self.budget:
                        break
                    perturb = step_size * np.random.randn(dim)
                    candidate = best_x + perturb
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)

        return best_val, best_x