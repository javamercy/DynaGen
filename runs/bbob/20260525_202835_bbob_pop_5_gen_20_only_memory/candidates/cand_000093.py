import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        # Increased population size for exploration
        self.popsize = max(8, min(8 * dim, budget // 2))
        # Stagnation threshold
        self.stagnation_limit = 5

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        popsize = self.popsize
        budget = self.budget
        rng = self.rng

        # Initialize population uniformly
        pop = rng.uniform(lb, ub, size=(popsize, dim))
        fitness = np.full(popsize, np.inf)
        best_x = None
        best_val = np.inf
        evaluations = 0

        # Evaluate initial population
        for i in range(popsize):
            if evaluations >= budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Main evolution loop
        stagnation_counter = 0
        while evaluations < budget:
            # Check for restart condition
            if stagnation_counter >= self.stagnation_limit:
                # Keep best individual, reinitialize others
                pop[0] = best_x.copy()
                for i in range(1, popsize):
                    pop[i] = rng.uniform(lb, ub, size=dim)
                stagnation_counter = 0
                # Evaluate reinitialized individuals (except the best)
                for i in range(1, popsize):
                    if evaluations >= budget:
                        break
                    x = pop[i]
                    val = func(x)
                    evaluations += 1
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                # Recompute fitness for best (unchanged)
                fitness[0] = best_val
                continue  # start next generation

            # Determine best index for current population
            best_idx = np.argmin(fitness)
            generation_improved = False

            for i in range(popsize):
                if evaluations >= budget:
                    break
                # Dithering: F and CR drawn uniformly per generation (can be per individual)
                F = rng.uniform(0.5, 1.0)
                CR = rng.uniform(0.5, 1.0)

                # Current-to-pbest mutation: pbest = top 20% individuals
                sorted_idx = np.argsort(fitness)
                pbest_max = max(1, int(0.2 * popsize))
                pbest_idx = rng.integers(0, pbest_max)
                pbest = sorted_idx[pbest_idx]

                # Select two distinct random indices different from i and pbest
                indices = [j for j in range(popsize) if j != i and j != pbest]
                if len(indices) < 2:
                    r1, r2 = rng.integers(0, popsize, size=2)
                else:
                    r1, r2 = rng.choice(indices, 2, replace=False)

                # Mutation
                mutant = pop[i] + F * (pop[pbest] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)

                # Binomial crossover
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
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
                        generation_improved = True

            if generation_improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1

        return best_val, best_x