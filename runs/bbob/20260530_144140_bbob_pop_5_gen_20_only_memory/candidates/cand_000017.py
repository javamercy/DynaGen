import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Smaller population to focus exploitation
        self.NP = max(4, min(5*dim, budget // 5))
        self.CR = 0.9
        self.F_start = 0.8
        self.F_end = 0.2

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub

        # Initialization
        pop = rng.uniform(bounds_lb, bounds_ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], bounds_lb, bounds_ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # DE loop with decreasing F
        gen = 0
        max_gen = self.budget // NP  # conservative estimate
        while budget > 0 and NP > 1:
            F = self.F_start + (self.F_end - self.F_start) * gen / max(1, max_gen)
            gen += 1
            for i in range(NP):
                if budget <= 0:
                    break
                # Choose two distinct indices different from i
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 2:
                    break
                r1, r2 = rng.choice(indices, size=2, replace=False)
                # Mutation: DE/current-to-best/1
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, bounds_lb, bounds_ub)
                # Crossover binomial
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Evaluate
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Local search: pattern search along each dimension
        if budget > 0 and best_x is not None:
            step = 0.2 * (bounds_ub - bounds_lb)  # initial step size
            while budget > 0:
                improved = False
                for d in range(dim):
                    if budget <= 0:
                        break
                    # Try positive step
                    candidate = best_x.copy()
                    candidate[d] = np.clip(candidate[d] + step[d], bounds_lb[d], bounds_ub[d])
                    val = func(candidate)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        improved = True
                        continue
                    # Try negative step
                    candidate = best_x.copy()
                    candidate[d] = np.clip(candidate[d] - step[d], bounds_lb[d], bounds_ub[d])
                    val = func(candidate)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        improved = True
                if not improved:
                    # Reduce step size
                    step *= 0.9
                    if np.max(step) < 1e-12:
                        break
                else:
                    # Expand step slightly if improvement found
                    step *= 1.1
        
        # Additional random perturbations if budget remains
        if budget > 0 and best_x is not None:
            step = 0.1 * (bounds_ub - bounds_lb)
            while budget > 0:
                perturbation = rng.uniform(-step, step, size=dim)
                candidate = best_x + perturbation
                candidate = np.clip(candidate, bounds_lb, bounds_ub)
                val = func(candidate)
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                step *= 0.95
                if np.max(step) < 1e-12:
                    break

        if best_x is None:
            x = rng.uniform(bounds_lb, bounds_ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x