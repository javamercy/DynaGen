import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        rng = self.rng

        def clip(x):
            return np.clip(x, lb, ub)

        # Smart initialization: evaluate up to 5 random points, pick best
        n_init = min(5, budget // 2)
        candidates = rng.uniform(lb, ub, size=(n_init, dim))
        vals = np.full(n_init, np.inf)
        best_x = None
        best_f = np.inf
        for i in range(n_init):
            vals[i] = func(candidates[i])
            budget -= 1
            if vals[i] < best_f:
                best_f = vals[i]
                best_x = candidates[i].copy()
                report_best(best_f, best_x)
        # Initial population: include the best point and additional random points
        pop_size = max(4, min(20, int(np.sqrt(dim) * 5)))
        pop_size = min(pop_size, budget + n_init)  # ensure we have budget for remaining
        if pop_size < n_init:
            pop_size = n_init
        # Reuse evaluated points: keep all evaluated points if needed, else keep best and randomly select others
        # For simplicity, we create a list of evaluated points and their values, then generate extra if needed
        evals = list(zip(candidates, vals))
        evals.sort(key=lambda x: x[1])
        # Ensure we have pop_size individuals
        if len(evals) < pop_size:
            needed = pop_size - len(evals)
            extra = rng.uniform(lb, ub, size=(needed, dim))
            for i in range(needed):
                f = func(extra[i])
                budget -= 1
                evals.append((extra[i], f))
                if f < best_f:
                    best_f = f
                    best_x = extra[i].copy()
                    report_best(best_f, best_x)
                if budget <= 0:
                    break
        # Build population arrays from evals (keep sorted by fitness)
        evals.sort(key=lambda x: x[1])
        pop = np.array([e[0] for e in evals[:pop_size]])
        pop_f = np.array([e[1] for e in evals[:pop_size]])
        # Update best from population
        best_idx = 0
        best_f = pop_f[0]
        best_x = pop[0].copy()
        report_best(best_f, best_x)

        # DE parameters
        F_mean = 0.8
        F_std = 0.1
        CR_mean = 0.9
        CR_std = 0.1

        # Main DE loop
        while budget > 0:
            # Sample parameters for this generation
            F = np.clip(rng.normal(F_mean, F_std), 0, 2)
            CR = np.clip(rng.normal(CR_mean, CR_std), 0, 1)
            for i in range(pop_size):
                if budget <= 0:
                    break
                # Current-to-best/1 mutation
                a, b = rng.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                mut = pop[i] + F * (best_x - pop[i]) + F * (pop[a] - pop[b])
                # Binary crossover
                cross = rng.rand(dim) < CR
                if not cross.any():
                    cross[rng.randint(dim)] = True
                trial = np.where(cross, mut, pop[i])
                trial = clip(trial)
                trial_f = func(trial)
                budget -= 1
                if trial_f < pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f
                    if trial_f < best_f:
                        best_x = trial.copy()
                        best_f = trial_f
                        report_best(best_f, best_x)
            if budget <= 0:
                break
            # Update best index after generation
            best_idx = np.argmin(pop_f)
            if pop_f[best_idx] < best_f:
                best_f = pop_f[best_idx]
                best_x = pop[best_idx].copy()
                report_best(best_f, best_x)

        # Local refinement with exponentially decaying step size
        sigma0 = 0.1 * (ub - lb)
        sigma = sigma0.copy()
        decay = 0.95
        while budget > 0:
            pert = rng.normal(0, sigma, size=dim)
            candidate = best_x + pert
            candidate = clip(candidate)
            cand_f = func(candidate)
            budget -= 1
            if cand_f < best_f:
                best_x = candidate.copy()
                best_f = cand_f
                report_best(best_f, best_x)
                sigma = sigma0.copy()
            else:
                sigma *= decay

        return best_f, best_x