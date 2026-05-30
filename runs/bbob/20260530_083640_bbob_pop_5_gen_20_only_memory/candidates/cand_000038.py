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

        # Reserve evaluations for Nelder-Mead (need at least dim+1 points)
        reserve_nm = max(10, dim + 1)
        de_budget = budget - reserve_nm
        if de_budget < 4:
            # If budget too small, just random search
            best_val = np.inf
            best_x = None
            for _ in range(budget):
                x = lb + (ub - lb) * rng.rand(dim)
                val = func(x)
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # DE parameters
        F = 0.8
        CR = 0.9
        pop_size = max(4, min(20, de_budget // 2))
        max_gens = de_budget // pop_size
        if max_gens < 1:
            max_gens = 1
            pop_size = de_budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        # Evaluate initial population
        for i in range(pop_size):
            val = func(pop[i])
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE generations
        for gen in range(max_gens):
            if evals >= de_budget:
                break
            for i in range(pop_size):
                if evals >= de_budget:
                    break
                # Mutation
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Nelder-Mead local search
        remaining = budget - evals
        if remaining > 0:
            # Build simplex around best
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
                val = func(simplex[i])
                evals += 1
                simplex_vals[i] = val
                if val < best_val:
                    best_val = val
                    best_x = simplex[i].copy()
                    report_best(best_val, best_x)

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100
            for _ in range(max_iter):
                if evals >= budget:
                    break
                # Sort simplex
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                yr = func(xr)
                evals += 1
                if yr < simplex_vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    ye = func(xe)
                    evals += 1
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
                    xc = np.clip(xc, lb, ub)
                    yc = func(xc)
                    evals += 1
                    if yc < simplex_vals[-1]:
                        simplex[-1] = xc
                        simplex_vals[-1] = yc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if evals >= budget:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            y = func(simplex[i])
                            evals += 1
                            simplex_vals[i] = y
                            if y < best_val:
                                best_val = y
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)

                # Update best after iteration
                current_best_idx = np.argmin(simplex_vals)
                if simplex_vals[current_best_idx] < best_val:
                    best_val = simplex_vals[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
                    report_best(best_val, best_x)

                if evals >= budget:
                    break

        return best_val, best_x