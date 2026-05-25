import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # population size: at least 4, at most budget/2, scales with dim
        self.initial_pop_size = max(4, min(4 * dim, budget // 2))
        self.min_pop_size = max(4, dim // 2)
        self.restart_threshold = max(5, int(budget / (4 * self.initial_pop_size)) if self.initial_pop_size > 0 else 5)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        pop_size = self.initial_pop_size
        evals = 0
        best_val = np.inf
        best_x = None

        # Handle degenerate pop_size
        if pop_size <= 0:
            x = np.random.uniform(lb, ub, dim)
            best_val = func(x)
            best_x = x.copy()
            report_best(best_val, best_x)
            evals = 1
            while evals < self.budget:
                x = np.random.uniform(lb, ub, dim)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= self.budget:
                break
            x = pop[i]
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Constants
        F = 0.5
        max_gen = (self.budget - evals) // pop_size if pop_size > 0 else 0
        no_improve = 0
        gen = 0
        # For local search radius
        initial_radius = 0.1 * (ub - lb).mean()
        while evals < self.budget and gen < max_gen:
            # Shrink population size
            target_pop = max(self.min_pop_size, self.initial_pop_size - gen * (self.initial_pop_size - self.min_pop_size) // (max_gen if max_gen > 0 else 1))
            if pop_size > target_pop:
                # Remove worst individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:target_pop]].copy()
                fitness = fitness[sorted_idx[:target_pop]].copy()
                pop_size = target_pop

            # Adaptive CR: decreases rapidly
            CR = 0.5 - 0.4 * (gen / max_gen) if max_gen > 0 else 0.5
            CR = np.clip(CR, 0.1, 0.5)
            improved = False
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # Select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                if len(candidates) < 3:
                    continue
                a, b, c = np.random.choice(candidates, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Exponential crossover
                cross_points = np.random.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
            if improved:
                no_improve = 0
            else:
                no_improve += 1
                # Local search around best
                if no_improve >= 2 and evals < self.budget:
                    num_local = min(5, self.budget - evals)
                    radius = initial_radius * (1 - gen / (max_gen if max_gen > 0 else 1))
                    for _ in range(num_local):
                        if evals >= self.budget:
                            break
                        x = best_x + np.random.randn(dim) * radius
                        x = np.clip(x, lb, ub)
                        val = func(x)
                        evals += 1
                        if val < best_val:
                            best_val = val
                            best_x = x.copy()
                            report_best(best_val, best_x)
                            break  # restart local search from new best
            # Restart if stagnation detected
            if no_improve >= self.restart_threshold:
                # Retain best individual only
                sorted_indices = np.argsort(fitness)
                n_keep = 1
                keep_indices = sorted_indices[:n_keep]
                new_pop = np.empty_like(pop)
                new_fitness = np.full(pop_size, np.inf)
                for idx, keep_idx in enumerate(keep_indices):
                    new_pop[idx] = pop[keep_idx].copy()
                    new_fitness[idx] = fitness[keep_idx]
                for i in range(n_keep, pop_size):
                    if evals >= self.budget:
                        break
                    x = np.random.uniform(lb, ub, dim)
                    new_pop[i] = x
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
            gen += 1
        return best_val, best_x