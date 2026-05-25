import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.NP = max(4, min(int(budget/2), 10*dim))
        self.CR = 0.9

    def __call__(self, func):
        np.random.seed(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        NP = self.NP
        CR = self.CR
        best_x = np.random.uniform(lb, ub, dim)
        best_val = func(best_x)
        calls = 1
        report_best(best_val, best_x)
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.full(NP, float('inf'))
        for i in range(NP):
            if calls >= self.budget:
                break
            x = pop[i]
            val = func(x)
            calls += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        generation = 0
        stagnation = 0
        stagnation_limit = max(5, NP // 2, dim)
        restarts = 0
        max_restarts = 3
        while calls < self.budget:
            improved_this_gen = False
            for i in range(NP):
                if calls >= self.budget:
                    break
                candidates = list(range(NP))
                candidates.remove(i)
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                F = 0.5 + 0.5 * np.random.rand()
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
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
                        improved_this_gen = True
            generation += 1
            if improved_this_gen:
                stagnation = 0
            else:
                stagnation += 1
            if stagnation >= stagnation_limit and restarts < max_restarts and calls < self.budget:
                restarts += 1
                stagnation = 0
                new_pop = np.empty((NP - 1, dim))
                for j in range(NP - 1):
                    if np.random.rand() < 0.5:
                        new_pop[j] = np.random.uniform(lb, ub)
                    else:
                        step = np.random.standard_cauchy(dim) * 0.2 * (ub - lb)
                        new_pop[j] = np.clip(best_x + step, lb, ub)
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
                pop = np.vstack((best_x.reshape(1, -1), new_pop))
                fitness = np.concatenate(([best_val], new_fitness))
                # local refinement with adaptive step sizes
                gauss_scale = 0.01 * (ub - lb)
                cauchy_scale = 0.2 * (ub - lb)
                local_steps = max(1, min(10, int(self.budget / 100)))
                for _ in range(local_steps):
                    if calls >= self.budget:
                        break
                    if np.random.rand() < 0.9:
                        step = np.random.randn(dim) * gauss_scale
                    else:
                        step = np.random.standard_cauchy(dim) * cauchy_scale
                    candidate = np.clip(best_x + step, lb, ub)
                    val = func(candidate)
                    calls += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        pop[0] = best_x
                        fitness[0] = best_val
                        if np.random.rand() < 0.9:
                            gauss_scale = np.clip(gauss_scale * 1.1, 1e-6, 1.0 * (ub - lb))
                        else:
                            cauchy_scale = np.clip(cauchy_scale * 1.1, 1e-6, 1.0 * (ub - lb))
                    else:
                        if np.random.rand() < 0.9:
                            gauss_scale = np.clip(gauss_scale * 0.9, 1e-6, 1.0 * (ub - lb))
                        else:
                            cauchy_scale = np.clip(cauchy_scale * 0.9, 1e-6, 1.0 * (ub - lb))
        return best_val, best_x