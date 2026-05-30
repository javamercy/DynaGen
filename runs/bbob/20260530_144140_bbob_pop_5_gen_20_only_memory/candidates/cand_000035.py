import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.NP = max(4, min(5*dim, budget // 5))
        self.CR = 0.9
        self.F_start = 0.8
        self.F_end = 0.2

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        NP = self.NP
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialization
        pop = rng.uniform(lb, ub, size=(NP, dim))
        fitness = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None

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

        # DE loop
        gen = 0
        max_gen = self.budget // NP if NP > 0 else 1
        while budget > 0 and NP > 1:
            F = self.F_start + (self.F_end - self.F_start) * gen / max(1, max_gen)
            gen += 1
            for i in range(NP):
                if budget <= 0:
                    break
                indices = [j for j in range(NP) if j != i]
                if len(indices) < 2:
                    break
                r1, r2 = rng.choice(indices, size=2, replace=False)
                mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
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

        # Local refinement
        if best_x is not None:
            # Coordinate-wise pattern search with multiple step sizes
            step_scales = [0.2, 0.1, 0.05, 0.025]
            for scale in step_scales:
                if budget <= 0:
                    break
                step = scale * (ub - lb)
                for d in range(dim):
                    if budget <= 0:
                        break
                    # Positive step
                    cand = best_x.copy()
                    cand[d] = np.clip(cand[d] + step[d], lb[d], ub[d])
                    val = func(cand)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = cand.copy()
                        report_best(best_val, best_x)
                        continue
                    # Negative step
                    cand = best_x.copy()
                    cand[d] = np.clip(cand[d] - step[d], lb[d], ub[d])
                    val = func(cand)
                    budget -= 1
                    if val < best_val:
                        best_val = val
                        best_x = cand.copy()
                        report_best(best_val, best_x)

            # Cauchy random perturbations
            step = 0.1 * (ub - lb)
            while budget > 0:
                if np.max(step) < 1e-12:
                    break
                perturbation = rng.standard_cauchy(size=dim)
                # Scale perturbation to avoid too large jumps
                perturbation = np.clip(perturbation, -10, 10) * step
                cand = best_x + perturbation
                cand = np.clip(cand, lb, ub)
                val = func(cand)
                budget -= 1
                if val < best_val:
                    best_val = val
                    best_x = cand.copy()
                    report_best(best_val, best_x)
                step *= 0.95

        if best_x is None:
            x = rng.uniform(lb, ub)
            best_val = func(x)
            best_x = x
        return best_val, best_x