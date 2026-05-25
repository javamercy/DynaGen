import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # Dynamic population size: at least 10, up to 5*dim or budget/4
        self.pop_size = max(10, min(5 * dim, budget // 4))
        # Restart threshold based on dim
        self.restart_threshold = max(10, 3 * dim)
        # Probability of using DE/rand/1; decays when no improvement
        self.p_rand = 0.7
        # Mutation scale range
        self.F_min = 0.2
        self.F_max = 1.2
        # Crossover rate
        self.CR = 0.9

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.pop_size
        budget = self.budget

        # Fallback for tiny populations (shouldn't happen but safe)
        if pop_size < 4:
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

        # Initialize population uniformly
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

        # Tracking variables
        no_improve = 0
        generation = 0
        max_gen = (budget - evals) // pop_size if pop_size > 0 else 0

        while evals < budget and generation < max_gen:
            improved_this_gen = False
            # Adaptation of p_rand based on improvement
            if no_improve > 0:
                self.p_rand = max(0.1, self.p_rand * 0.95)
            else:
                self.p_rand = min(0.9, self.p_rand * 1.05)

            for i in range(pop_size):
                if evals >= budget:
                    break
                # Select three distinct random indices (for rand/1) or two for best/1
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                # Mutation scale with dither
                F = np.random.uniform(self.F_min, self.F_max)
                # Choose mutation strategy based on p_rand
                if np.random.rand() < self.p_rand:
                    # DE/rand/1
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                else:
                    # DE/best/1
                    mutant = best_x + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                # Binomial crossover
                cross_points = np.random.rand(dim) < self.CR
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
            # Local perturbation of best (mild exploration)
            if evals < budget and best_x is not None:
                for _ in range(min(2, budget - evals)):
                    sigma = 0.05 * (ub - lb)
                    x = best_x + sigma * np.random.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
            # Track improvement
            if improved_this_gen:
                no_improve = 0
            else:
                no_improve += 1
            # Restart if stagnation
            if no_improve >= self.restart_threshold:
                # Mix: 30% around best, 70% uniform random
                new_pop = np.random.uniform(lb, ub, (pop_size, dim))
                if best_x is not None:
                    num_focused = max(1, int(0.3 * pop_size))
                    for j in range(num_focused):
                        perturbation = 0.2 * np.random.randn(dim) * (ub - lb)
                        new_pop[j] = best_x + perturbation
                        new_pop[j] = np.clip(new_pop[j], lb, ub)
                    new_pop[0] = best_x.copy()
                # Evaluate new population (skip first if best already known)
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
                no_improve = 0
                self.p_rand = 0.7  # reset probability
            generation += 1

        return best_val, best_x