import numpy as np

class CMA_ES_Restart_LS:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        dim = self.dim
        lb = -5.0
        ub = 5.0

        # Initial population size and CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)

        # Strategy parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        # Initialization
        sigma = 2.0
        mean = np.random.uniform(lb, ub, dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)

        evals = 0
        gen = 0
        best_f = np.inf
        best_x = mean.copy()

        # Restart parameters
        max_restarts = 5
        restart_count = 0
        stagnation_count = 0
        prev_best_f = np.inf
        no_improve_gens = 0
        max_no_improve = 10 + int(30 * dim / lambda_)

        # Local search parameters
        ls_freq = max(5, int(0.1 * (self.budget / lambda_)))
        ls_counter = 0

        while evals < self.budget:
            gen += 1
            # Sample new population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.randn(dim)
                pop[i] = mean + sigma * (B @ (D * z))
                pop[i] = np.clip(pop[i], lb, ub)

            # Evaluate
            fitness = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if evals >= self.budget:
                    break
                fitness[i] = func(pop[i])
                evals += 1
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = pop[i].copy()
                    if best_f < self.f_opt:
                        self.f_opt = best_f
                        self.x_opt = best_x.copy()

            # Sort
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]

            # Update mean
            old_mean = mean.copy()
            mean = np.dot(weights, pop[:mu])

            # Cumulation: update evolution paths
            z_mean = (mean - old_mean) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ z_mean)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*gen))) / (1.4 + 2.0/(dim+1))
            hsig = 1 if hsig < 1 else 0
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * z_mean

            # Update covariance matrix
            artmp = (pop[:mu] - old_mean) / sigma
            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) \
                + cmu * (artmp.T @ np.diag(weights) @ artmp)

            # Update sigma
            sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(dim) - 1 + 1.0/(5*dim)))
            sigma = np.clip(sigma, 0.01, 10.0)

            # SVD: compute B, D, invsqrtC
            try:
                D, B = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                D = np.ones(dim)
                B = np.eye(dim)
            D = np.sqrt(np.maximum(D, 1e-20))
            invsqrtC = B @ np.diag(1.0 / D) @ B.T

            # Check stagnation and restart
            if best_f < prev_best_f - 1e-10:
                prev_best_f = best_f
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            if no_improve_gens >= max_no_improve or sigma < 0.01:
                # Restart with bigger population
                restart_count += 1
                if restart_count >= max_restarts:
                    break
                lambda_ = int(lambda_ * 2)  # IPOP-style
                mu = lambda_ // 2
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()
                mueff = 1.0 / np.sum(weights**2)
                cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3)**2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
                damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
                sigma = 2.0
                mean = best_x + np.random.randn(dim) * 0.5
                mean = np.clip(mean, lb, ub)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                invsqrtC = np.eye(dim)
                no_improve_gens = 0
                prev_best_f = np.inf
                gen = 0

            # Local search every ls_freq generations on best point
            ls_counter += 1
            if ls_counter >= ls_freq and evals < self.budget - 2*dim:
                ls_counter = 0
                # Nelder-Mead simplex on best solution
                x0 = best_x.copy()
                f0 = best_f
                # Build initial simplex
                simplex = np.zeros((dim+1, dim))
                simplex[0] = x0
                delta = 0.05 * (ub - lb)
                for i in range(dim):
                    x = x0.copy()
                    x[i] += delta
                    x = np.clip(x, lb, ub)
                    simplex[i+1] = x
                f_vals = np.full(dim+1, np.inf)
                f_vals[0] = f0
                for i in range(1, dim+1):
                    if evals >= self.budget:
                        break
                    f_vals[i] = func(simplex[i])
                    evals += 1
                    if f_vals[i] < best_f:
                        best_f = f_vals[i]
                        best_x = simplex[i].copy()
                        if best_f < self.f_opt:
                            self.f_opt = best_f
                            self.x_opt = best_x.copy()

                # Nelder-Mead iterations (limited)
                max_iter_nm = min(2*dim, (self.budget - evals) // 2)
                for _ in range(max_iter_nm):
                    if evals + 2 >= self.budget:
                        break
                    # Order
                    order = np.argsort(f_vals)
                    simplex = simplex[order]
                    f_vals = f_vals[order]
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr)
                    evals += 1
                    if fr < best_f:
                        best_f = fr
                        best_x = xr.copy()
                        if best_f < self.f_opt:
                            self.f_opt = best_f
                            self.x_opt = best_x.copy()
                    if fr < f_vals[0]:
                        # Expansion
                        xe = centroid + 2*(xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        evals += 1
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                            if best_f < self.f_opt:
                                self.f_opt = best_f
                                self.x_opt = best_x.copy()
                        if fe < fr:
                            simplex[-1] = xe
                            f_vals[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_vals[-1] = fr
                    elif fr < f_vals[-2]:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                    else:
                        # Contraction
                        if fr < f_vals[-1]:
                            xc = centroid + 0.5*(xr - centroid)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc < best_f:
                                best_f = fc
                                best_x = xc.copy()
                                if best_f < self.f_opt:
                                    self.f_opt = best_f
                                    self.x_opt = best_x.copy()
                            if fc < fr:
                                simplex[-1] = xc
                                f_vals[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    if evals >= self.budget:
                                        break
                                    f_vals[i] = func(simplex[i])
                                    evals += 1
                                    if f_vals[i] < best_f:
                                        best_f = f_vals[i]
                                        best_x = simplex[i].copy()
                                        if best_f < self.f_opt:
                                            self.f_opt = best_f
                                            self.x_opt = best_x.copy()
                        else:
                            xc = centroid - 0.5*(centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc < best_f:
                                best_f = fc
                                best_x = xc.copy()
                                if best_f < self.f_opt:
                                    self.f_opt = best_f
                                    self.x_opt = best_x.copy()
                            if fc < f_vals[-1]:
                                simplex[-1] = xc
                                f_vals[-1] = fc
                            else:
                                # Shrink
                                for i in range(1, dim+1):
                                    simplex[i] = simplex[0] + 0.5*(simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    if evals >= self.budget:
                                        break
                                    f_vals[i] = func(simplex[i])
                                    evals += 1
                                    if f_vals[i] < best_f:
                                        best_f = f_vals[i]
                                        best_x = simplex[i].copy()
                                        if best_f < self.f_opt:
                                            self.f_opt = best_f
                                            self.x_opt = best_x.copy()

                # Inject best found into CMA-ES mean
                if best_f < self.f_opt:
                    self.f_opt = best_f
                    self.x_opt = best_x.copy()
                mean = best_x.copy()

        return self.f_opt, self.x_opt