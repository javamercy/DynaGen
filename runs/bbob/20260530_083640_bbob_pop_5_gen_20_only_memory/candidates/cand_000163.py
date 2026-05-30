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

        # Initial feasible point
        x0 = lb + (ub - lb) * rng.rand(dim)
        evaluate(x0)

        if budget <= dim + 2:
            for _ in range(budget - evals):
                if evals >= budget:
                    break
                x = lb + (ub - lb) * rng.rand(dim)
                evaluate(x)
            return best_val, best_x

        # Adaptive parameters
        pop_start = min(20, (budget * 2) // 3 // 2)
        pop_end = 4
        pop_size = max(pop_start, pop_end)
        F_start = 0.9
        F_end = 0.2
        CR_start = 0.9
        CR_end = 0.3

        # Budget split: 2/3 DE, 1/3 NM
        de_max_evals = (budget * 2) // 3
        if de_max_evals < pop_size:
            de_max_evals = pop_size

        # Initialize population
        pop = lb + (ub - lb) * rng.rand(pop_size, dim)
        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= de_max_evals:
                break
            val = evaluate(pop[i])
            if val is None:
                break
            pop_fitness[i] = val

        gen = 0
        while evals < de_max_evals:
            frac = evals / de_max_evals if de_max_evals > 0 else 0
            # Exponential decay
            F = F_end + (F_start - F_end) * np.exp(-2.0 * frac)
            CR = CR_end + (CR_start - CR_end) * np.exp(-2.0 * frac)
            # Population size deterministic exponential decay
            pop_target = int(round(pop_end + (pop_start - pop_end) * np.exp(-2.0 * frac)))
            pop_target = max(pop_end, min(pop_target, pop_start))
            # Adjust population if needed
            if pop_target < pop_size:
                order = np.argsort(pop_fitness)
                pop = pop[order[:pop_target]]
                pop_fitness = pop_fitness[order[:pop_target]]
                pop_size = pop_target
            elif pop_target > pop_size:
                n_new = pop_target - pop_size
                new_pop = lb + (ub - lb) * rng.rand(n_new, dim)
                new_fit = np.full(n_new, np.inf)
                for i in range(n_new):
                    if evals >= de_max_evals:
                        break
                    val = evaluate(new_pop[i])
                    if val is None:
                        break
                    new_fit[i] = val
                pop = np.vstack([pop, new_pop])
                pop_fitness = np.concatenate([pop_fitness, new_fit])
                pop_size = len(pop)

            # DE iteration
            for i in range(pop_size):
                if evals >= de_max_evals:
                    break
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = evaluate(trial)
                if val is None:
                    break
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
            gen += 1

        # Nelder-Mead local search (more aggressive)
        if evals < budget and best_x is not None:
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.05 * (ub[i] - lb[i])
                if step == 0:
                    step = 0.01
                x = best_x.copy()
                x[i] = np.clip(x[i] + step, lb[i], ub[i])
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

            alpha = 1.5
            gamma = 2.5
            rho = 0.75
            sigma = 0.25
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