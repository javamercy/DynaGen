import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.popsize = max(4, min(4 * dim, budget // 2))
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        # initialize population
        pop = self.rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0
        # evaluate initial population
        for i in range(popsize):
            if evaluations >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        # main loop
        while evaluations < self.budget:
            best_before_gen = best_val
            # one generation
            for i in range(popsize):
                if evaluations >= self.budget:
                    break
                # mutation with dithering F
                F = self.rng.uniform(0.5, 1.0)
                indices = list(range(popsize))
                indices.remove(i)
                a, b, c = self.rng.choice(indices, 3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # crossover
                cross_points = self.rng.random(dim) < self.CR
                if not np.any(cross_points):
                    cross_points[self.rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            # check stagnation
            if best_val == best_before_gen and evaluations < self.budget and popsize > 1:
                # restart: keep best, replace others with random points
                new_pop = [best_x.copy()]
                new_fitness = [best_val]
                for _ in range(popsize - 1):
                    if evaluations >= self.budget:
                        break
                    x = self.rng.uniform(lb, ub, size=dim)
                    val = func(x)
                    evaluations += 1
                    new_pop.append(x)
                    new_fitness.append(val)
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = np.array(new_pop)
                fitness = np.array(new_fitness)
        return best_val, best_x