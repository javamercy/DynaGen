import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size, at least 4, scaled by dim but limited by budget
        self.pop_size = max(4, min(4 * dim, budget // 2))
        # restart threshold: number of generations without improvement
        self.restart_threshold = max(5, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        # Handle degenerate case where pop_size <= 0 (fallback to random search)
        if pop_size <= 0:
            best_x = np.random.uniform(lb, ub, dim)
            best_val = func(best_x)
            report_best(best_val, best_x)
            evals = 1
            while evals < budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialization
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            x = pop[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Parameters
        F = 0.5          # initial mutation scale
        CR = 0.9         # crossover rate
        local_sigma_factor = 0.01  # relative perturbation for local search
        no_improve = 0   # generations without improvement
        generation = 0

        while evals < budget:
            # Check if we can do a full generation (pop_size evaluations)
            if evals + pop_size > budget:
                # Not enough budget for full generation; do a few more random points
                remaining = budget - evals
                for _ in range(remaining):
                    x = np.random.uniform(lb, ub, dim)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                break

            improved_this_gen = False
            # DE/best/1 mutation with adaptive F
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select two distinct random indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 2:
                    # Not enough individuals for mutation; skip or use random
                    # Fallback: random mutation
                    mutant = best_x + F * np.random.randn(dim)
                else:
                    r1, r2 = np.random.choice(candidates, size=2, replace=False)
                    # Mutation: best + F * (pop[r1] - pop[r2])
                    mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                # Evaluate trial
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

            # Local search around best (intensification)
            if evals < budget:
                # Use up to 2 evaluations for local search per generation
                local_evals = min(2, budget - evals)
                for _ in range(local_evals):
                    sigma = local_sigma_factor * (ub - lb)
                    x = best_x + sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # Adapt F based on whether any trial improved
            if improved_this_gen:
                F *= 1.1
                F = min(F, 0.9)
                no_improve = 0
            else:
                F *= 0.9
                F = max(F, 0.1)
                no_improve += 1

            # Restart if stagnation
            if no_improve >= self.restart_threshold:
                # Focused restart: 30% around best, 70% random
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = max(1, int(0.3 * pop_size))
                    for j in range(num_focused):
                        new_pop[j] = best_x + 0.1 * np.random.randn(dim) * (ub - lb)
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                # Evaluate new population (except best if already known)
                new_fitness = np.full(pop_size, np.inf)
                new_fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    x = new_pop[i].copy()
                    val = func(x)
                    evals += 1
                    new_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fitness = new_fitness
                F = 0.5
                no_improve = 0

            generation += 1

        return best_val, best_x