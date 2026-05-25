import numpy as np

class LSHADE_restart_local:
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

        # Reserve a small portion for final local search
        final_local_budget = max(10 * dim, int(0.05 * budget))
        main_budget = budget - final_local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init

        def lhs(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive & memory
        archive = np.empty((0, dim))
        max_archive = NP
        H = 30  # increased memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart and local search parameters
        no_improvement = 0
        max_no_improvement = max(5, int(0.1 * NP))
        diversity_threshold = 0.02 * (ub - lb).mean()
        restart_min_evals = int(0.2 * main_budget)  # don't restart too early

        # Main DE loop with linear population reduction and restarts
        while fevals < main_budget:
            remaining = main_budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio
            ratio = 0.2 - 0.15 * (1 - remaining / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_f = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0, 1)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling with random fallback
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_f.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        no_improvement = 0
                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update memory
            if S_CR:
                w = np.array(delta_f) / np.sum(delta_f)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Check for stagnation and trigger local search
            if np.min(fitness) == self.best_f:
                no_improvement += 1
            else:
                no_improvement = 0

            if no_improvement >= max_no_improvement and fevals > restart_min_evals:
                # Adaptive local search from best (small budget)
                local_budget = min(5 * dim, int(0.02 * remaining))
                if local_budget > 0:
                    # Simple Nelder-Mead from best
                    x0 = self.best_x.copy()
                    f0 = self.best_f
                    n = dim
                    step = 0.05 * (ub - lb)
                    simplex = np.zeros((n + 1, n))
                    simplex[0] = x0
                    for i in range(n):
                        simplex[i + 1] = x0.copy()
                        simplex[i + 1][i] += step[i]
                    simplex = np.clip(simplex, lb, ub)
                    fvals = np.array([func(x) for x in simplex])
                    evals_ls = n + 1
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    best_ls = fvals[0]
                    best_x_ls = simplex[0].copy()

                    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                    while evals_ls < local_budget:
                        centroid = np.mean(simplex[:-1], axis=0)
                        xr = centroid + alpha * (centroid - simplex[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        evals_ls += 1
                        if fr < fvals[0]:
                            xe = centroid + gamma * (centroid - simplex[-1])
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            evals_ls += 1
                            if fe < fr:
                                simplex[-1] = xe; fvals[-1] = fe
                            else:
                                simplex[-1] = xr; fvals[-1] = fr
                        elif fr < fvals[-2]:
                            simplex[-1] = xr; fvals[-1] = fr
                        else:
                            if fr < fvals[-1]:
                                xc = centroid + rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                evals_ls += 1
                                if fc <= fr:
                                    simplex[-1] = xc; fvals[-1] = fc
                                else:
                                    for i in range(1, n+1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i])
                                        evals_ls += 1
                            else:
                                xc = centroid - rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                evals_ls += 1
                                if fc < fvals[-1]:
                                    simplex[-1] = xc; fvals[-1] = fc
                                else:
                                    for i in range(1, n+1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i])
                                        evals_ls += 1
                        order = np.argsort(fvals)
                        simplex = simplex[order]
                        fvals = fvals[order]
                        if fvals[0] < best_ls:
                            best_ls = fvals[0]
                            best_x_ls = simplex[0].copy()
                    if best_ls < self.best_f:
                        self.best_f = best_ls
                        self.best_x = best_x_ls.copy()
                no_improvement = 0

            # Diversity-based restart
            if fevals > restart_min_evals and fevals < 0.9 * main_budget:
                pop_std = np.std(pop, axis=0).mean()
                if pop_std < diversity_threshold:
                    # Keep best, reinitialize rest
                    NP = NP_init
                    new_pop = np.zeros((NP, dim))
                    new_pop[0] = self.best_x.copy()
                    new_pop[1:] = lhs(NP - 1, dim, lb, ub)
                    fitness = np.array([func(x) for x in new_pop])
                    fevals += NP
                    pop = new_pop
                    archive = np.empty((0, dim))
                    max_archive = NP
                    M_CR[:] = 0.5
                    M_F[:] = 0.5
                    mem_idx = 0
                    # Re-evaluate best
                    best_idx = np.argmin(fitness)
                    f_best = fitness[best_idx]
                    if f_best < self.best_f:
                        self.best_f = f_best
                        self.best_x = pop[best_idx].copy()
                    no_improvement = 0
                    # Continue loop without reducing budget prematurely
                    continue

        # Final local search with remaining budget
        if final_local_budget > 0:
            x0 = self.best_x.copy()
            f0 = self.best_f
            n = dim
            step = 0.05 * (ub - lb)
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step[i]
            simplex = np.clip(simplex, lb, ub)
            fvals = np.array([func(x) for x in simplex])
            evals_ls = n + 1
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            best_ls = fvals[0]
            best_x_ls = simplex[0].copy()

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            while evals_ls < final_local_budget:
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals_ls += 1
                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals_ls += 1
                    if fe < fr:
                        simplex[-1] = xe; fvals[-1] = fe
                    else:
                        simplex[-1] = xr; fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr; fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        xc = centroid + rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals_ls += 1
                        if fc <= fr:
                            simplex[-1] = xc; fvals[-1] = fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i])
                                evals_ls += 1
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals_ls += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc; fvals[-1] = fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i])
                                evals_ls += 1
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                if fvals[0] < best_ls:
                    best_ls = fvals[0]
                    best_x_ls = simplex[0].copy()
            if best_ls < self.best_f:
                self.best_f = best_ls
                self.best_x = best_x_ls.copy()

        return self.best_f, self.best_x