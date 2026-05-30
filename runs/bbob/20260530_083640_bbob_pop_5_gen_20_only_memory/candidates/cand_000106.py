import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def _sobol(self, n, d):
        # Simple linear sequence as placeholder for Sobol
        # Since Sobol may not be available, use Latin Hypercube
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = self.rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + self.rng.uniform(0, 1 / n)
        return lhs

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        best_val = np.inf
        best_x = None
        evals = 0

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            x_clipped = np.clip(x, lb, ub)
            val = func(x_clipped)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_clipped.copy()
                report_best(best_val, best_x)
            return val

        # Initial point
        x0 = lb + (ub - lb) * rng.rand(dim)
        evaluate(x0)
        if evals >= budget:
            return best_val, best_x

        # Population size
        pop_size = max(4, min(20, (budget * 7) // 10 // 2))
        pop_size = min(pop_size, budget)

        # Sobol-like initial population
        lhs = self._sobol(pop_size, dim)
        pop = lb + (ub - lb) * lhs
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            val = evaluate(pop[i])
            if val is None:
                break
            pop_fitness[i] = val

        # Self-adaptive DE (jDE)
        # Initialize F and CR for each individual
        F = 0.5 + 0.3 * rng.rand(pop_size)
        CR = 0.5 + 0.4 * rng.rand(pop_size)
        tau1 = 0.1
        tau2 = 0.1
        Fl = 0.1
        Fu = 0.9

        de_budget = (budget * 7) // 10
        while evals < de_budget:
            for i in range(pop_size):
                if evals >= de_budget:
                    break
                # Mutate F and CR
                if rng.rand() < tau1:
                    F[i] = Fl + rng.rand() * (Fu - Fl)
                if rng.rand() < tau2:
                    CR[i] = rng.rand()
                # Mutation: DE/rand/1
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F[i] * (pop[b] - pop[c])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR[i], mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = evaluate(trial)
                if val is None:
                    break
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                # else keep, but F and CR already possibly updated
        
        # Nelder-Mead local search
        nm_budget = budget - evals
        if nm_budget > 0 and best_x is not None:
            # Initialize simplex with best point and perturbations
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.05 * (ub[i] - lb[i])
                if step == 0:
                    step = 0.01
                simplex[i+1] = best_x.copy()
                simplex[i+1, i] = np.clip(best_x[i] + step, lb[i], ub[i])
            simplex_vals = np.full(dim + 1, np.inf)
            simplex_vals[0] = best_val
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                val = evaluate(simplex[i])
                if val is None:
                    break
                simplex_vals[i] = val
            
            # Adaptive Nelder-Mead coefficients
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            # Track success to adapt
            success_count = 0
            max_iter = nm_budget
            for _ in range(max_iter):
                if evals >= budget:
                    break
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget: break
                yr = evaluate(xr)
                if yr is None: break
                if yr < simplex_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget: break
                    ye = evaluate(xe)
                    if ye is None: break
                    if ye < yr:
                        simplex[-1] = xe
                        simplex_vals[-1] = ye
                        success_count += 1
                    else:
                        simplex[-1] = xr
                        simplex_vals[-1] = yr
                        success_count += 1
                elif yr < simplex_vals[-2]:
                    simplex[-1] = xr
                    simplex_vals[-1] = yr
                    success_count += 1
                else:
                    # Contraction
                    if yr < simplex_vals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    if evals >= budget: break
                    yc = evaluate(xc)
                    if yc is None: break
                    if yc < simplex_vals[-1]:
                        simplex[-1] = xc
                        simplex_vals[-1] = yc
                        success_count += 1
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if evals >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            val = evaluate(simplex[i])
                            if val is None:
                                break
                            simplex_vals[i] = val
                # Update best
                idx = np.argmin(simplex_vals)
                if simplex_vals[idx] < best_val:
                    best_val = simplex_vals[idx]
                    best_x = simplex[idx].copy()
                    report_best(best_val, best_x)
        return best_val, best_x