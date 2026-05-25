import numpy as np

class LSHADE_enhanced_local:
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

        # allocate budget: main DE and local search (20% to CMA-ES)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization ----
        NP_init = max(10, min(200, 20 * int(np.log(dim)) if dim > 1 else 20))
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

        archive = np.empty((0, dim))
        max_archive = NP * 2  # larger archive for diversity
        H = 30
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main jSO-inspired DE loop ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP * 2

            # adaptive pbest ratio (jSO style)
            ratio = 0.25 - 0.20 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
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
                    if idx < NP:
                        if idx != i and idx != r1:
                            break
                    else:
                        break
                r2_vec = combined[idx]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # reflected boundary handling
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
                    S_df.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # update memory with Lehmer mean for F, arithmetic for CR
            if S_CR:
                w = np.array(S_df) / np.sum(S_df)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- CMA-ES Local Search on best solution ----
        if local_budget > 0 and dim > 1:
            self._cma_es_local(func, lb, ub, local_budget)
        elif local_budget > 0 and dim == 1:
            # simple 1D local search
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = 0.1
            while evals < local_budget:
                for sign in [1, -1]:
                    cand = x_best + sign * step
                    cand = np.clip(cand, lb, ub)
                    f_c = func(cand)
                    evals += 1
                    if f_c < f_best:
                        x_best, f_best = cand, f_c
                        step *= 1.2
                        break
                else:
                    step *= 0.5
                if step < 1e-10:
                    break
            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best

        return self.best_f, self.best_x

    def _cma_es_local(self, func, lb, ub, budget):
        """Simple CMA-ES local optimizer with restart."""
        dim = self.dim
        x_mean = self.best_x.copy()
        f_best = self.best_f
        x_best = x_mean.copy()
        evals = 0

        # CMA-ES parameters
        lam = max(4, 4 + int(3 * np.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)

        # Strategy constants
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance related
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Restart loop
        while evals < budget:
            # Initialize for this restart
            mean = x_best.copy()
            sigma = 0.5 * (ub - lb)  # range is 10, so 5 maybe too large
            sigma = 2.0  # fixed step
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            evals_restart = 0
            best_in_restart = f_best
            best_x_in_restart = x_best.copy()

            while evals < budget:
                # Eigen decomposition
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                # Ensure positive definite
                eigenvalues = np.maximum(eigenvalues, 1e-30)
                D = np.sqrt(eigenvalues)
                B = eigenvectors

                # Sample offspring
                z = np.random.randn(lam, dim)
                y = z @ (B * D).T  # each row: B * D * z
                x = mean + sigma * y   # shape (lam, dim)
                # Clip to bounds
                x = np.clip(x, lb, ub)

                # Evaluate
                f_vals = np.array([func(x[i]) for i in range(lam)])
                evals += lam
                evals_restart += lam

                # Sort
                order = np.argsort(f_vals)
                f_vals = f_vals[order]
                x = x[order]

                # Update best
                if f_vals[0] < best_in_restart:
                    best_in_restart = f_vals[0]
                    best_x_in_restart = x[0].copy()
                    if f_vals[0] < f_best:
                        f_best = f_vals[0]
                        x_best = x[0].copy()
                        self.best_f = f_best
                        self.best_x = x_best

                # Update mean
                y_best = y[order[:mu]]  # mu best y
                mean_new = mean + sigma * np.dot(weights, y_best)

                # Update evolution paths
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (mean_new - mean) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals_restart // lam))) /
                        chiN < 1.4 + 2.0 / (dim + 1))
                hsig = float(hsig)
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (mean_new - mean) / sigma

                # Update covariance matrix
                y_mu = (x[order[:mu]] - mean) / sigma  # using x directly to have y vectors
                # weighted outer sum
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * np.dot(weights * y_mu.T, y_mu)

                # Update sigma
                sigma = sigma * np.exp((np.linalg.norm(ps) / chiN - 1) * cs / damps)

                mean = mean_new

                # Check break conditions for restart
                if sigma < 1e-10 * (ub - lb).mean() or evals_restart > budget / 2:
                    break

            # After restart: if best improved enough, maybe continue; else restart with new mean
            if best_in_restart < f_best - 1e-8:
                # continue from best of restart
                x_best = best_x_in_restart.copy()
                f_best = best_in_restart
            else:
                # restart with random perturbation
                x_best = x_best + 0.1 * np.random.randn(dim) * (ub - lb)
                x_best = np.clip(x_best, lb, ub)
            if evals >= budget:
                break