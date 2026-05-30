import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        # Small population for exploitation
        self.NP = max(4, min(5 * dim, budget // 4))
        self.CR = 0.9
        self.F = 0.5

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initial population
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

        # Evaluate initial points
        for i in range(NP):
            if budget <= 0:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            budget -= 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            pop[i] = x

        # DE/best/1/bin loop
        while budget > 0 and NP > 1:
            for i in range(NP):
                if budget <= 0:
                    break
                candidates = [j for j in range(NP) if j != i]
                if len(candidates) < 2:
                    break
                r1, r2 = rng.choice(candidates, size=2, replace=False)
                mutant = best_x + self.F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < self.CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                val = func(trial)
                budget -= 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Advanced local search: coordinate-wise line search with decaying step
        if budget > 0 and best_x is not None:
            step = (ub - lb) * 0.01  # initial step
            while budget > 0:
                # Randomize order of dimensions
                dims = rng.permutation(dim)
                for d in dims:
                    if budget <= 0:
                        break
                    # Generate candidate by perturbing only dimension d
                    direction = 1 if rng.rand() < 0.5 else -1
                    candidate = best_x.copy()
                    candidate[d] += direction * step[d]
                    candidate = np.clip(candidate, lb, ub)
                    val = func(candidate)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = candidate.copy()
                        report_best(best_val, best_x)
                        # keep step, maybe increase? but we keep
                    else:
                        # shrink step on failure for this dimension
                        step[d] *= 0.9
                # After a full cycle, possibly restart with larger step if no improvement?
                # But we keep simple
                # If step too small, break?
                if np.max(step) < 1e-12:
                    break

        # Fallback if best_x is None
        if best_x is None:
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x

        return best_val, best_x