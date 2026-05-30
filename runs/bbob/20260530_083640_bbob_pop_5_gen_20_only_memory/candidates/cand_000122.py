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

        # Phase 1: Differential Evolution with restart
        de_budget = int(0.7 * budget)
        if de_budget > 1:
            NP = max(5, min(30, dim * 5))
            if NP < 5:
                NP = 5
            if de_budget > NP:
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
                F = 0.9
                CR = 0.95
                stagnation_counter = 0
                best_fitness_ever = best_val
                max_generations = (de_budget - evals) // NP
                for gen in range(max_generations):
                    if evals >= de_budget:
                        break
                    # Check stagnation and restart
                    if best_val < best_fitness_ever:
                        best_fitness_ever = best_val
                        stagnation_counter = 0
                    else:
                        stagnation_counter += 1
                    if stagnation_counter >= 5:
                        # Replace worst half of population with random points
                        order = np.argsort(fitness)[::-1]
                        half = NP // 2
                        for idx in order[:half]:
                            if evals >= de_budget:
                                break
                            pop[idx] = rng.uniform(lb, ub)
                            val = func(pop[idx])
                            evals += 1
                            fitness[idx] = val
                            if val < best_val:
                                best_val = val
                                best_x = pop[idx].copy()
                                report_best(best_val, best_x)
                        stagnation_counter = 0
                        best_fitness_ever = best_val
                    # DE generation
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

        # Phase 2: Nelder-Mead local search
        remaining = budget - evals
        if remaining > 1:
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = best_x.copy()
            for i in range(dim):
                step = 0.1 * (ub[i] - lb[i])
                x = best_x.copy()
                x[i] = np.clip(x[i] + step, lb[i], ub[i])
                simplex[i + 1] = x

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

            alpha = 1.0
            gamma = 2.5
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

                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                yr = func(xr)
                evals += 1
                if yr < simplex_vals[0]:
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

                current_best_idx = np.argmin(simplex_vals)
                if simplex_vals[current_best_idx] < best_val:
                    best_val = simplex_vals[current_best_idx]
                    best_x = simplex[current_best_idx].copy()
                    report_best(best_val, best_x)

                if evals >= budget:
                    break

        # Phase 3: budget leftover perturbations
        if evals < budget:
            for _ in range(min(10, budget - evals)):
                if evals >= budget:
                    break
                perturb = best_x + 0.2 * rng.uniform(lb - best_x, ub - best_x)
                perturb = np.clip(perturb, lb, ub)
                val = func(perturb)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = perturb.copy()
                    report_best(best_val, best_x)

        return best_val, best_x