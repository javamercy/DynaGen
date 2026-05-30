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

        # Reserve evaluations for Nelder-Mead
        reserve_nm = min(dim + 5, budget // 3)
        de_budget = budget - reserve_nm
        if de_budget < 2:
            de_budget = max(1, budget - 1)
            reserve_nm = budget - de_budget

        if budget <= dim + 1 or de_budget < 2:
            for _ in range(budget):
                if evals >= budget:
                    break
                x = lb + (ub - lb) * rng.rand(dim)
                evaluate(x)
            return best_val, best_x

        # DE parameters
        F = 0.5
        CR = 0.5
        pop_size = max(5, min(20, de_budget // 5))
        if pop_size * 3 > de_budget:
            pop_size = max(2, de_budget // 3)
        max_gens = de_budget // pop_size
        if max_gens == 0:
            max_gens = 1
            pop_size = de_budget

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if evals >= de_budget:
                break
            val = evaluate(pop[i])
            if val is None:
                break
            pop_fitness[i] = val

        # Track stagnation
        no_improve_gen = 0
        prev_best = best_val

        # DE generations
        for gen in range(max_gens):
            if evals >= de_budget:
                break
            # Check stagnation and restart worst half
            if best_val < prev_best:
                no_improve_gen = 0
                prev_best = best_val
            else:
                no_improve_gen += 1
                if no_improve_gen >= 3:
                    # Restart worst half
                    order = np.argsort(pop_fitness)
                    half = pop_size // 2
                    worst_indices = order[half:]
                    for idx in worst_indices:
                        if evals >= de_budget:
                            break
                        # Replace with random point
                        pop[idx] = lb + (ub - lb) * rng.rand(dim)
                        val = evaluate(pop[idx])
                        if val is None:
                            break
                        pop_fitness[idx] = val
                    no_improve_gen = 0

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
                val = evaluate(trial)
                if val is None:
                    break
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val

        # Nelder-Mead local search
        if evals < budget and best_x is not None:
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
                val = evaluate(simplex[i])
                if val is None:
                    break
                simplex_vals[i] = val

            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_iter = 100
            for _ in range(max_iter):
                if evals >= budget:
                    break
                order = np.argsort(simplex_vals)
                simplex = simplex[order]
                simplex_vals = simplex_vals[order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                if evals >= budget:
                    break
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                yr = evaluate(xr)
                if yr is None:
                    break
                if yr < simplex_vals[0]:
                    # Expansion
                    if evals >= budget:
                        break
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
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
                    if evals >= budget:
                        break
                    if yr < simplex_vals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
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
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            val = evaluate(simplex[i])
                            if val is None:
                                break
                            simplex_vals[i] = val

                current_best_idx = np.argmin(simplex_vals)
                if simplex_vals[current_best_idx] < best_val:
                    best_val = simplex_vals[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
                    report_best(best_val, best_x)

        return best_val, best_x