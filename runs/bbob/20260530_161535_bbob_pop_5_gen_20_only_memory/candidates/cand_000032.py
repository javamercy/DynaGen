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

        best_x = None
        best_f = np.inf
        evals = 0

        # Latin Hypercube Sampling (initialization)
        n_init = max(2, budget // 6)
        lhs = np.empty((n_init, dim))
        for i in range(dim):
            lhs[:, i] = np.linspace(lb[i], ub[i], n_init + 1)[:-1] + (ub[i]-lb[i])/(2*n_init)
        for i in range(dim):
            rng.shuffle(lhs[:, i])
        for i in range(n_init):
            if evals >= budget:
                break
            x = lhs[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Differential Evolution with restarts
        pop_size = max(10, dim * 3)
        pop = np.empty((pop_size, dim))
        for i in range(pop_size):
            pop[i] = rng.uniform(lb, ub, size=dim)
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

        max_gen = 100
        no_improve_count = 0
        for gen in range(max_gen):
            if evals >= budget:
                break
            # Check for stagnation and restart if needed
            if gen > 0 and no_improve_count >= 2:
                # Restart worst 50% of population
                sorted_idx = np.argsort(pop_f)
                n_restart = pop_size // 2
                for j in range(pop_size - n_restart, pop_size):
                    pop[sorted_idx[j]] = rng.uniform(lb, ub, size=dim)
                    if evals >= budget:
                        break
                    f = func(pop[sorted_idx[j]])
                    evals += 1
                    pop_f[sorted_idx[j]] = f
                    if f < best_f:
                        best_f = f
                        best_x = pop[sorted_idx[j]].copy()
                        report_best(best_f, best_x)
                no_improve_count = 0

            new_pop = np.empty_like(pop)
            new_pop_f = np.empty(pop_size)
            improved = False
            for i in range(pop_size):
                if evals >= budget:
                    break
                # Mutation: DE/rand/1
                indices = [j for j in range(pop_size) if j != i]
                a, b, c = rng.choice(indices, 3, replace=False)
                mutant = pop[a] + 0.9 * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Crossover: binomial
                cross_prob = 0.9
                trial = np.where(rng.rand(dim) < cross_prob, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1
                if f_trial < pop_f[i]:
                    new_pop[i] = trial
                    new_pop_f[i] = f_trial
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        improved = True
                else:
                    new_pop[i] = pop[i]
                    new_pop_f[i] = pop_f[i]
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            pop = new_pop
            pop_f = new_pop_f

        # Local refinement with Nelder-Mead from best (if budget remains)
        remaining = budget - evals
        if remaining > 0 and remaining >= dim:
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
            for _ in range(min(remaining, 20)):
                if evals >= budget:
                    break
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
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evals += 1
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        for i in range(1, dim+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            f = func(simplex[i])
                            evals += 1
                            simplex_f[i] = f
                idx_min = np.argmin(simplex_f)
                if simplex_f[idx_min] < best_f:
                    best_f = simplex_f[idx_min]
                    best_x = simplex[idx_min].copy()
                    report_best(best_f, best_x)

        return best_f, best_x