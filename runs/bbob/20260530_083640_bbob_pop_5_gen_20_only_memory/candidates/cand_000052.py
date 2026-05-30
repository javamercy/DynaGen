import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)

        best_val = np.inf
        best_x = None
        evals = 0

        # Population size for DE
        pop_size = max(4, min(10 * dim, budget // 3))
        if pop_size < 5:
            pop_size = max(4, pop_size)

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)
        # Initialize F and CR for each individual
        F = rng.uniform(0.1, 0.9, pop_size)
        CR = rng.uniform(0, 1, pop_size)
        # Store successful parameters
        sf = []
        scr = []

        def clip_point(x):
            return np.clip(x, lb, ub)

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            val = func(x)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            return val

        # Evaluate initial pop
        for i in range(pop_size):
            v = evaluate(pop[i])
            if v is None:
                break
            pop_fit[i] = v
        if evals >= budget:
            return best_val, best_x

        # jDE parameters
        tau1 = 0.1
        tau2 = 0.1
        Fl = 0.1
        Fu = 0.9

        max_gen = max(1, int(0.6 * budget / pop_size))
        for gen in range(max_gen):
            if evals >= budget:
                break
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR[i], mutant, pop[i])
                trial = clip_point(trial)
                # Selection
                v = evaluate(trial)
                if v is None:
                    break
                if v <= pop_fit[i]:
                    pop[i] = trial
                    pop_fit[i] = v
                    sf.append(F[i])
                    scr.append(CR[i])
                else:
                    # Update F, CR with probability
                    if rng.rand() < tau1:
                        F[i] = Fl + rng.rand() * (Fu - Fl)
                    if rng.rand() < tau2:
                        CR[i] = rng.rand()
            if evals >= budget:
                break

        # Local search: Nelder-Mead if enough budget
        remaining = budget - evals
        if remaining > dim + 1 and best_x is not None:
            # Build simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.05 * (ub[i] - lb[i])
                if step == 0:
                    step = 0.01
                x = best_x.copy()
                x[i] = min(ub[i], max(lb[i], x[i] + step))
                simplex[i+1] = x
            simplex_vals = np.full(dim + 1, np.inf)
            simplex_vals[0] = best_val
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                val = evaluate(simplex[i])
                if val is None:
                    break
                simplex_vals[i] = val
            # Nelder-Mead iterations
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100
            for _ in range(max_iter):
                if evals >= budget:
                    break
                # Sort
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = clip_point(xr)
                yr = evaluate(xr)
                if yr is None:
                    break
                if yr < simplex_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = clip_point(xe)
                    ye = evaluate(xe)
                    if ye is None:
                        break
                    if ye < yr:
                        simplex[-1] = xe
                        simplex_vals[-1] = ye
                    else:
                        simplex[-1] = xr
                        simplex_vals[-1] = yr
                elif yr < simplex_vals[-2]:
                    simplex[-1] = xr
                    simplex_vals[-1] = yr
                else:
                    # Contraction
                    if yr < simplex_vals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = clip_point(xc)
                    yc = evaluate(xc)
                    if yc is None:
                        break
                    if yc < simplex_vals[-1]:
                        simplex[-1] = xc
                        simplex_vals[-1] = yc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if evals >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = clip_point(simplex[i])
                            y = evaluate(simplex[i])
                            if y is None:
                                break
                            simplex_vals[i] = y
                # Update best from simplex
                idx_min = np.argmin(simplex_vals)
                if simplex_vals[idx_min] < best_val:
                    best_val = simplex_vals[idx_min]
                    best_x = simplex[idx_min].copy()
                    report_best(best_val, best_x)
        # Final random perturbations
        while evals < budget:
            if best_x is None:
                x = lb + (ub - lb) * rng.rand(dim)
            else:
                step = 0.01 * (ub - lb) * (1 - (evals / budget))
                x = best_x + rng.randn(dim) * step
                x = clip_point(x)
            val = evaluate(x)
            if val is None:
                break
        return best_val, best_x