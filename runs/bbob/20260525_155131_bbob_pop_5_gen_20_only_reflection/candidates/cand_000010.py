import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        budget = self.budget
        rng = self.rng

        # Evaluate initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        if budget <= 1:
            return best_val, best_x

        # Allocate budget: population for DE, then local refinement
        pop_size = min(20, max(3, budget // 10))
        local_budget = max(1, budget // 4)
        global_budget = budget - local_budget

        # Build initial population with random points
        population = [best_x.copy()]
        while len(population) < pop_size and evals < global_budget:
            x = rng.uniform(lb, ub)
            val = func(x)
            evals += 1
            population.append(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # If population too small for DE, do local search with remaining budget
        if len(population) < 3:
            remaining = budget - evals
            if remaining > 0:
                step_size = np.linalg.norm(ub - lb) * 0.1
                for _ in range(remaining):
                    candidate = best_x + rng.normal(0, step_size, size=dim)
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
            return best_val, best_x

        pop = np.array(population)
        F = 0.8
        CR = 0.9
        remaining_global = global_budget - evals
        max_gens = remaining_global // pop_size

        for gen in range(max_gens):
            if evals >= global_budget:
                break
            new_pop = np.empty_like(pop)
            for i in range(pop_size):
                if evals >= global_budget:
                    break
                indices = list(range(pop_size))
                indices.remove(i)
                a, b, c = rng.choice(indices, size=3, replace=False)
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.rand(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.randint(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = trial.copy()
                    report_best(best_val, best_x)
                new_pop[i] = trial
            pop = new_pop

        # Local refinement
        remaining = budget - evals
        if remaining > 0:
            step_size = np.linalg.norm(ub - lb) * 0.05
            for _ in range(remaining):
                candidate = best_x + rng.normal(0, step_size, size=dim)
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)

        return best_val, best_x