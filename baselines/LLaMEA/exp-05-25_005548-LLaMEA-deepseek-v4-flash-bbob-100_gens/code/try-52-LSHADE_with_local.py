import numpy as np

class LSHADE_with_local:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Budget for local search
        local_budget = max(8 * dim, int(0.15 * budget))
        main_budget = budget - local_budget
        if main_budget < 20:  # tiny budget: random search
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ----- Latin hypercube initialisation -----
        NP0 = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP0

        def lhs(n, d, low, high):
            x = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                x[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return x

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        # Best so far
        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive (for current/old population, size = NP)
        archive = np.empty((0, dim))
        archive_size = NP

        # Memory for CR and F (jSO style: H=5)
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ----- Main DE loop (jSO variant) -----
        while fevals < main_budget:
            remaining = main_budget - fevals
            # Linear population reduction (LSHADE style)
            NP_new = max(4, int(4 + (NP0 - 4) * (remaining / main_budget)))
            if NP_new < NP:
                # Keep best individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                archive_size = NP

            pbest_ratio = 0.1  # fixed as in jSO
            pbest_num = max(1, int(pbest_ratio * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # For memory update
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # Archive needs to be at least 1 for r2 selection
            if len(archive) == 0:
                archive = pop.copy()

            for i in range(NP):
                # Select random memory entry
                r = np.random.randint(H)

                # Generate CR from normal distribution around M_CR[r], clamp to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # Generate F from Cauchy distribution around M_F[r], scale 0.1
                F = M_F[r] + 0.1 * np.random.standard_cauchy()
                while F <= 0.0:
                    F = M_F[r] + 0.1 * np.random.standard_cauchy()
                F = min(F, 1.0)

                # pbest selection
                pbest = pop[np.random.choice(pbest_pool)]

                # Choose r1 (different from i)
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Choose r2 from union of pop and archive (different from i and r1)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx != i and idx != r1:
                        break
                r2_vec = combined[idx]

                # Mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflective boundary handling
                low = lb
                high = ub
                out_low = u < low
                out_high = u > high
                u[out_low] = 2 * low[out_low] - u[out_low]
                u[out_high] = 2 * high[out_high] - u[out_high]
                # Second chance if still outside
                still_low = u < low
                still_high = u > high
                u[still_low] = np.random.uniform(low[still_low], high[still_low])
                u[still_high] = np.random.uniform(low[still_high], high[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > archive_size:
                        # Remove random element
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update memory if there were successful mutations
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Weighted Lehmer mean for F (jSO style)
                sum_F_sq = np.sum(w * np.array(S_F) ** 2)
                sum_F = np.sum(w * np.array(S_F))
                mean_F = sum_F_sq / sum_F if sum_F > 1e-30 else 0.5
                # Weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))

                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ----- Local search: bounded Nelder-Mead simplex -----
        def nelder_mead(x0, step, max_evals):
            # Build initial simplex (reflection and expansion factors standard)
            x0 = np.clip(x0, lb, ub)
            n = dim
            simplex = np.zeros((n+1, n))
            simplex[0] = x0.copy()
            for i in range(n):
                x = x0.copy()
                x[i] = min(ub[i], x[i] + step[i])
                simplex[i+1] = x
            fvals = np.array([func(p) for p in simplex])
            evals = n+1

            # Sort by fitness
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while evals < max_evals:
                # Centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                x_worst = simplex[-1]

                # Reflection
                xr = centroid + alpha * (centroid - x_worst)
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1

                if fvals[0] <= fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                elif fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals += 1
                        if fc < fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            # re-sort
                            idx = np.argsort(fvals)
                            simplex = simplex[idx]
                            fvals = fvals[idx]
                            continue
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - x_worst)
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i])
                                evals += 1
                                if evals >= max_evals:
                                    break
                            # Re-sort
                            idx = np.argsort(fvals)
                            simplex = simplex[idx]
                            fvals = fvals[idx]
                            continue
                # Re-sort after reflection/expansion/contraction
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        local_evals_used = 0
        x = self.best_x.copy()
        f = self.best_f

        # Initial step: 0.1 of the range per dimension
        step = 0.1 * (ub - lb)
        max_nm_evals = local_budget // 2  # use half of local budget for Nelder-Mead
        if max_nm_evals > dim*3:  # only if enough evaluations
            x_new, f_new = nelder_mead(x, step, max_nm_evals)
            local_evals_used = max_nm_evals  # actual evaluations are counted inside, approximate for stop
            if f_new < f:
                f = f_new
                x = x_new

        # Remaining local budget: coordinate descent
        remaining_local = local_budget - local_evals_used
        if remaining_local > 0:
            evals = 0
            step = 0.1 * (ub - lb)
            while evals < remaining_local and np.any(step > 1e-12):
                improved = False
                for d in np.random.permutation(dim):
                    # positive
                    x_new = x.copy()
                    x_new[d] = min(ub[d], x_new[d] + step[d])
                    f_new = func(x_new)
                    evals += 1
                    if f_new < f:
                        f = f_new
                        x = x_new
                        improved = True
                        if evals >= remaining_local:
                            break
                        continue
                    # negative
                    x_new = x.copy()
                    x_new[d] = max(lb[d], x_new[d] - step[d])
                    f_new = func(x_new)
                    evals += 1
                    if f_new < f:
                        f = f_new
                        x = x_new
                        improved = True
                    if evals >= remaining_local:
                        break
                if not improved:
                    step *= 0.5

        if f < self.best_f:
            self.best_f = f
            self.best_x = x

        return self.best_f, self.best_x