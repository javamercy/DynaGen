import numpy as np

class jSO_restart_local:
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

        # reserve budget for local search
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            # pure random search
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling ----
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init

        def lhs(n, d, low, high):
            res = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                res[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return res

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # jSO parameters
        H = 5
        M_CR = 0.8 * np.ones(H)
        M_F = 0.3 * np.ones(H)
        mem_idx = 0

        archive = np.empty((0, dim))
        max_archive = int(2.6 * NP)
        stagnation = 0
        restart_threshold = max(100, int(0.05 * main_budget))

        # ---- Main jSO loop with linear population reduction and restart ----
        while fevals < main_budget:
            remaining = main_budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                max_archive = int(2.6 * NP)
                if len(archive) > max_archive:
                    np.random.shuffle(archive)
                    archive = archive[:max_archive]

            # jSO adaptive pbest ratio
            p = 0.2 - 0.1 * (1.0 - remaining / main_budget)
            p = max(0.1, min(0.2, p))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)

            S_CR = []
            S_F = []
            delta_f = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # generate CR from normal with mean M_CR[r], std 0.1
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)
                # generate F from Cauchy with location M_F[r], scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # mutation: current-to-pbest/1 with archive
                pbest = pop[np.random.choice(sorted_idx[:pbest_num])]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or (idx < NP and idx == r1):
                        continue
                    break
                r2 = combined[idx]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # boundary handling: reflect and then random if still outside
                o_low = u < lb
                o_high = u > ub
                u[o_low] = 2 * lb[o_low] - u[o_low]
                u[o_high] = 2 * ub[o_high] - u[o_high]
                o_low = u < lb
                o_high = u > ub
                u[o_low] = np.random.uniform(lb[o_low], ub[o_low])
                u[o_high] = np.random.uniform(lb[o_high], ub[o_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_f.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # add to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        del_idx = np.random.randint(len(archive))
                        archive = np.delete(archive, del_idx, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stagnation = 0
                # else: no improvement

                if fevals >= main_budget:
                    break

            # update population
            pop = new_pop
            fitness = new_fitness

            # update memory if any successful updates
            if S_CR:
                w = np.array(delta_f) / np.sum(delta_f)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_F2 = np.sum(w * np.array(S_F) ** 2)
                sum_F = np.sum(w * np.array(S_F))
                mean_F = sum_F2 / sum_F if sum_F > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # stagnation detection and restart
            if np.min(fitness) < self.best_f:
                stagnation = 0
            else:
                stagnation += (main_budget - remaining) - (main_budget - remaining - NP)
            if stagnation >= restart_threshold:
                # soft restart: reinitialize worst half of population (except the best)
                restarts = NP // 2
                best_point = self.best_x.copy()
                # keep best in population
                pop[0] = best_point
                fitness[0] = self.best_f
                # random points for restarts
                random_pts = lhs(restarts, dim, lb, ub)
                for i in range(restarts):
                    idx = i + 1  # start from index 1
                    if idx < NP:
                        pop[idx] = random_pts[i]
                        fitness[idx] = func(random_pts[i])
                        fevals += 1
                # reset memory
                M_CR[:] = 0.8
                M_F[:] = 0.3
                mem_idx = 0
                archive = np.empty((0, dim))
                stagnation = 0
                if fevals >= main_budget:
                    break

        # ---- Nelder-Mead local search using remaining budget ----
        if local_budget > 0:
            x0 = self.best_x.copy()
            f0 = self.best_f
            step = 0.05 * (ub - lb)
            n = dim
            # initial simplex
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step[i]
                simplex[i + 1] = np.clip(simplex[i + 1], lb, ub)
            fvals = np.array([func(x) for x in simplex])
            evals = n + 1

            # sort
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]
            best_local_f = fvals[0]
            best_local_x = simplex[0].copy()
            if best_local_f < self.best_f:
                self.best_f = best_local_f
                self.best_x = best_local_x.copy()

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            while evals < local_budget:
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr); evals += 1

                if fr < fvals[0]:
                    # expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe); evals += 1
                    if fe < fr:
                        simplex[-1] = xe; fvals[-1] = fe
                    else:
                        simplex[-1] = xr; fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr; fvals[-1] = fr
                else:
                    if fr < fvals[-1]:
                        # outside contraction
                        xc = centroid + rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc); evals += 1
                        if fc <= fr:
                            simplex[-1] = xc; fvals[-1] = fc
                        else:
                            # shrink
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i]); evals += 1
                    else:
                        # inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc); evals += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc; fvals[-1] = fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i]); evals += 1
                # re-sort
                order = np.argsort(fvals)
                simplex = simplex[order]; fvals = fvals[order]
                if fvals[0] < best_local_f:
                    best_local_f = fvals[0]
                    best_local_x = simplex[0].copy()
                if fvals[0] < best_local_f:
                    best_local_f = fvals[0]
                    best_local_x = simplex[0].copy()
                if evals >= local_budget:
                    break

            if best_local_f < self.best_f:
                self.best_f = best_local_f
                self.best_x = best_local_x

        return self.best_f, self.best_x