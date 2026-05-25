import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        self.pop_size = max(3, min(2*dim, budget//10))
        self.restart_threshold = max(5, dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget
        rng = self.rng

        # Evaluate at least one point
        best_x = rng.uniform(lb, ub, dim)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        if budget == 1:
            return best_val, best_x

        # Initialize population
        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            # DE/best/1 mutation
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    continue
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                F = rng.uniform(0.3, 0.6)
                mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                CR = 0.95
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved_this_gen = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # Coordinate-wise local refinement around best
            if evals < budget:
                step = 0.02 * (ub - lb)
                local_evals = min(2*dim, budget - evals)
                for d in range(dim):
                    if evals >= budget:
                        break
                    # Try positive step
                    x_plus = best_x.copy()
                    x_plus[d] += step[d]
                    x_plus[d] = np.clip(x_plus[d], lb[d], ub[d])
                    val = func(x_plus)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_plus.copy()
                        report_best(best_val, best_x)
                    # Try negative step
                    x_minus = best_x.copy()
                    x_minus[d] -= step[d]
                    x_minus[d] = np.clip(x_minus[d], lb[d], ub[d])
                    val = func(x_minus)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x_minus.copy()
                        report_best(best_val, best_x)

            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.restart_threshold:
                # Reinitialize around best with Gaussian
                sigma = 0.1 * (ub - lb)
                new_pop = rng.normal(loc=best_x, scale=sigma, size=(pop_size, dim))
                new_pop = np.clip(new_pop, lb, ub)
                new_pop[0] = best_x.copy()
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i]
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                no_improve = 0
            generation += 1

        return best_val, best_x