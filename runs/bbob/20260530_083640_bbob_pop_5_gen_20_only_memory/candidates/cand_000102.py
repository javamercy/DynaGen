import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        # Initial random point
        best_x = rng.uniform(lb, ub)
        best_val = func(best_x)
        evals = 1
        report_best(best_val, best_x)

        # Phase 1: Differential Evolution
        de_budget = int(0.8 * budget)
        if de_budget > 1 and evals < budget:
            # Determine population size based on dimension and budget
            NP = max(4, min(20, dim * 3))
            # Ensure NP is at least 2 and not too large given de_budget
            NP = min(NP, max(2, de_budget // 2))
            if NP < 4:
                NP = 4
            # Initialize population
            pop = rng.uniform(lb, ub, size=(NP, dim))
            fitness = np.full(NP, np.inf)
            for i in range(NP):
                if evals >= de_budget:
                    break
                val = func(pop[i])
                evals += 1
                fitness[i] = val
                if val < best_val:
                    best_val = val
                    best_x = pop[i].copy()
                    report_best(best_val, best_x)
            # DE generations
            F = 0.8
            CR = 0.9
            max_generations = (de_budget - evals) // NP
            for gen in range(max_generations):
                if evals >= de_budget:
                    break
                for i in range(NP):
                    if evals >= de_budget:
                        break
                    indices = list(range(NP))
                    indices.remove(i)
                    rng.shuffle(indices)
                    a, b, c = indices[0], indices[1], indices[2]
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                    j_rand = rng.randint(dim)
                    trial[j_rand] = mutant[j_rand]
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < fitness[i]:
                        pop[i] = trial
                        fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = trial.copy()
                            report_best(best_val, best_x)

        # Phase 2: Local Random Search
        remaining = budget - evals
        if remaining > 0:
            step_size = 0.1 * (ub - lb)
            for _ in range(remaining):
                if evals >= budget:
                    break
                perturbation = rng.normal(0, step_size, dim)
                candidate = best_x + perturbation
                candidate = np.clip(candidate, lb, ub)
                val = func(candidate)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = candidate.copy()
                    report_best(best_val, best_x)
                else:
                    step_size *= 0.9
                    # Reset step size if too small
                    if np.any(step_size < 1e-15 * (ub - lb)):
                        step_size = 0.1 * (ub - lb)

        return best_val, best_x