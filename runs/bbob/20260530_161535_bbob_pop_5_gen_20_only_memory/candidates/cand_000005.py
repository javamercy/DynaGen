import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
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

        # Latin Hypercube Sampling
        n_init = max(2, budget // 4)
        points = np.empty((n_init, dim))
        for i in range(dim):
            points[:, i] = rng.uniform(lb[i], ub[i], size=n_init)
        # Shuffle each column to create LHS
        for i in range(dim):
            rng.shuffle(points[:, i])
        # Evaluate
        best_x = points[0].copy()
        best_f = func(best_x)
        evals = 1
        for i in range(1, n_init):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Differential Evolution style global search
        pop_size = max(5, dim * 2)
        pop = np.empty((pop_size, dim))
        for i in range(pop_size):
            pop[i] = rng.uniform(lb, ub, size=dim)
        # Evaluate pop (if not already evaluated)
        pop_f = np.full(pop_size, np.inf)
        for i in range(pop_size):
            if evals >= budget:
                break
            f = func(pop[i])
            evals += 1
            pop_f[i] = f
            if f < best_f:
                best_f = f
                best_x = pop[i].copy()
                report_best(best_f, best_x)

        # Main DE loop
        max_gen = 10
        for gen in range(max_gen):
            if evals >= budget:
                break
            new_pop = np.empty_like(pop)
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: choose three distinct indices
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mutant = pop[a] + 0.8 * (pop[b] - pop[c])
                # Clip
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                cross_prob = 0.9
                trial = np.where(rng.rand(dim) < cross_prob, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_f[i]:
                    new_pop[i] = trial
                    pop_f[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                else:
                    new_pop[i] = pop[i]
            pop = new_pop

        # Local refinement with Nelder-Mead from best
        n_local = min(budget - evals, budget // 4)
        if n_local > 0:
            # Build simplex around best_x
            simplex = np.zeros((dim+1, dim))
            simplex[0] = best_x
            for i in range(dim):
                delta = 0.05 * (ub[i] - lb[i])
                if delta == 0:
                    delta = 0.001
                x = best_x.copy()
                x[i] += delta
                x[i] = np.clip(x[i], lb[i], ub[i])
                simplex[i+1] = x
            # Evaluate simplex
            simplex_f = np.full(dim+1, np.inf)
            for i in range(dim+1):
                if evals >= budget:
                    break
                f = func(simplex[i])
                evals += 1
                simplex_f[i] = f
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
                    report_best(best_f, best_x)
            # Nelder-Mead iterations
            for _ in range(n_local):
                if evals >= budget:
                    break
                # Order simplex
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + 2.0 * (centroid - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evals += 1
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            f = func(simplex[i])
                            evals += 1
                            simplex_f[i] = f
                # Update best
                idx_min = np.argmin(simplex_f)
                if simplex_f[idx_min] < best_f:
                    best_f = simplex_f[idx_min]
                    best_x = simplex[idx_min].copy()
                    report_best(best_f, best_x)

        return best_f, best_x