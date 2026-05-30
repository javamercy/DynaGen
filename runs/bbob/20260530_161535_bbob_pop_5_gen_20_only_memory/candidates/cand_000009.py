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

        # Helper functions
        def lhs_sample(n, dim, lb, ub):
            samples = np.empty((n, dim))
            for i in range(dim):
                samples[:, i] = rng.uniform(lb[i], ub[i], size=n)
                rng.shuffle(samples[:, i])
            return samples

        # Initial LHS population
        pop_size = max(5, min(2*dim, budget // 4))
        pop = lhs_sample(pop_size, dim, lb, ub)
        pop_f = np.full(pop_size, np.inf)
        evals = 0
        best_x = None
        best_f = np.inf

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

        # Differential Evolution with restart
        local_budget = budget // 4
        final_budget = budget // 8
        de_budget = budget - evals - local_budget - final_budget
        if de_budget < 0:
            de_budget = 0
        stagnation = 0
        max_stagnation = max(10, dim * 5)
        gen_evals = 0
        while evals < budget - local_budget - final_budget:
            if evals >= budget:
                break
            # One DE generation
            new_pop = np.empty_like(pop)
            new_pop_f = np.empty(pop_size)
            for i in range(pop_size):
                if evals >= budget - local_budget - final_budget:
                    break
                # Mutation: select three distinct indices
                idxs = list(range(pop_size))
                idxs.remove(i)
                a, b, c = rng.choice(idxs, 3, replace=False)
                mutant = pop[a] + 0.8 * (pop[b] - pop[c])
                mutant = np.clip(mutant, lb, ub)
                # Crossover
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
                else:
                    new_pop[i] = pop[i]
                    new_pop_f[i] = pop_f[i]
            pop = new_pop
            pop_f = new_pop_f
            # Check improvement
            if pop_f.min() < best_f:
                stagnation = 0
            else:
                stagnation += 1
            # Restart if stagnation
            if stagnation >= max_stagnation:
                # Replace population except best
                best_idx = np.argmin(pop_f)
                for i in range(pop_size):
                    if i == best_idx:
                        continue
                    pop[i] = rng.uniform(lb, ub, size=dim)
                    f_i = func(pop[i])
                    evals += 1
                    pop_f[i] = f_i
                    if f_i < best_f:
                        best_f = f_i
                        best_x = pop[i].copy()
                        report_best(best_f, best_x)
                stagnation = 0

        # Nelder-Mead local refinement
        nm_evals = 0
        if evals < budget - final_budget:
            # Build simplex around best_x
            simplex = np.zeros((dim+1, dim))
            simplex[0] = best_x.copy()
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
                if evals >= budget - final_budget:
                    break
                f = func(simplex[i])
                evals += 1
                nm_evals += 1
                simplex_f[i] = f
                if f < best_f:
                    best_f = f
                    best_x = simplex[i].copy()
                    report_best(best_f, best_x)
            # NM iterations
            while evals < budget - final_budget:
                # Order
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
                            fs = func(simplex[i])
                            evals += 1
                            simplex_f[i] = fs
                # Update best
                idx_min = np.argmin(simplex_f)
                if simplex_f[idx_min] < best_f:
                    best_f = simplex_f[idx_min]
                    best_x = simplex[idx_min].copy()
                    report_best(best_f, best_x)

        # Final random refinement
        ref_radius = 0.01 * (ub - lb)
        while evals < budget:
            candidate = best_x + rng.normal(0, ref_radius, size=dim)
            candidate = np.clip(candidate, lb, ub)
            f = func(candidate)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = candidate.copy()
                report_best(best_f, best_x)
                ref_radius *= 0.9
                ref_radius = np.maximum(ref_radius, 1e-10 * (ub - lb))

        return best_f, best_x