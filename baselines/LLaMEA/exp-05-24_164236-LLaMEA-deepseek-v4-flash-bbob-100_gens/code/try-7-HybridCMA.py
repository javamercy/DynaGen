import numpy as np

class HybridCMA:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # --- CMA-ES parameters ---
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        mu = lambda_ // 2
        # recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)                     # effective selection mass

        # strategy parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)  # cumulation for covariance
        cs = (mu_eff + 2) / (dim + mu_eff + 5)                  # cumulation for step size
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)                     # learning rate for rank-1
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))  # learning rate for rank-mu
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs  # damping for step size

        # initial distribution
        x_mean = np.random.uniform(lb, ub, dim)
        sigma = (ub - lb).max() / 4.0              # initial step size
        pc = np.zeros(dim)                         # evolution path for C
        ps = np.zeros(dim)                         # evolution path for sigma
        B = np.eye(dim)                            # eigenvectors of C
        D = np.ones(dim)                           # eigenvalues of C
        C = np.eye(dim)                            # covariance matrix

        evals = 0
        best_f = np.inf
        best_x = None

        # helpers for restarts
        max_f_eval = budget
        restarts = 0
        eval_before_restart = 0

        while evals < budget:
            # --- generate offspring ---
            arz = np.random.randn(lambda_, dim)
            arx = x_mean.reshape(1, -1) + sigma * (arz @ (B * D).T)  # apply C matrix
            # repair bounds: clip and reflect
            arx = np.clip(arx, lb, ub)

            # evaluate
            arf = np.zeros(lambda_)
            for i in range(lambda_):
                if evals >= budget:
                    break
                arf[i] = func(arx[i])
                evals += 1
                if arf[i] < best_f:
                    best_f = arf[i]
                    best_x = arx[i].copy()

            # --- selection and recombination ---
            idx = np.argsort(arf)[:mu]           # indices of best mu individuals
            x_old = x_mean.copy()
            x_mean = np.dot(w, arx[idx])         # weighted mean

            # --- step size control ---
            z_mean = np.dot(w, arz[idx])         # weighted mean in z-space
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (B @ (z_mean * D))
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # --- covariance matrix adaptation ---
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * evals / lambda_)) /
                    np.sqrt(dim) < 1.4 + 2 / (dim + 1))
            hsig = 1.0  # simplified (always update rank-1)

            # update evolution path for covariance
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (x_mean - x_old) / sigma

            # update covariance matrix
            artmp = (arx[idx] - x_old) / sigma
            delta_hsig = (1 - hsig) * cc * (2 - cc)   # correction term
            # rank-1 update
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + delta_hsig * C)
            # rank-mu update
            for k in range(mu):
                C += cmu * w[k] * np.outer(artmp[k], artmp[k])

            # enforce symmetry and numerical stability
            C = (C + C.T) / 2
            # eigen decomposition (for next generation)
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.abs(D))          # eigenvalues > 0
            B = B[:, np.argsort(D)[::-1]]   # sort descending? Actually order for multiplication later
            # Reorder: B and D must correspond such that C = B * diag(D^2) * B^T
            # We keep D sorted ascending for stable multiplication with arz (default eigh returns ascending)
            # But we need to use them consistently. We'll compute later properly.
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.abs(D))
            # Ensure ordering: typically ascending, we can keep that

            # --- stagnation check and restart ---
            if evals > eval_before_restart + 10 * lambda_ and best_f == self.f_opt:
                # no improvement -> restart
                # Increase population size
                lambda_ = min(4 * lambda_, int(budget / 20))
                mu = lambda_ // 2
                w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                w /= w.sum()
                mu_eff = 1.0 / np.sum(w**2)
                # reset parameters
                cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
                cs = (mu_eff + 2) / (dim + mu_eff + 5)
                c1 = 2 / ((dim + 1.3)**2 + mu_eff)
                cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
                damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
                # reinitialize mean and covariance
                x_mean = np.random.uniform(lb, ub, dim)
                sigma = (ub - lb).max() / 4.0
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                restarts += 1
                eval_before_restart = evals

            # --- periodic Nelder-Mead local search on best ---
            if evals < budget and (evals % (budget // 20)) == 0:
                nm_evals = 0
                max_nm_evals = min(10 * dim, budget - evals)
                if max_nm_evals > 0:
                    x_best = best_x.copy()
                    f_best = best_f
                    step = (ub - lb) * 0.05
                    simplex = np.zeros((dim + 1, dim))
                    simplex[0] = x_best
                    for k in range(dim):
                        x = x_best.copy()
                        x[k] = np.clip(x[k] + step[k], lb[k], ub[k])
                        simplex[k + 1] = x
                    f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim + 1)])
                    evals += dim
                    nm_evals += dim
                    while nm_evals < max_nm_evals:
                        order = np.argsort(f_simplex)
                        simplex = simplex[order]
                        f_simplex = f_simplex[order]
                        centroid = np.mean(simplex[:-1], axis=0)
                        # reflection
                        xr = centroid + (centroid - simplex[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        evals += 1
                        nm_evals += 1
                        if fr < f_simplex[0]:
                            # expansion
                            xe = centroid + 2 * (centroid - simplex[-1])
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            evals += 1
                            nm_evals += 1
                            if fe < fr:
                                simplex[-1] = xe
                                f_simplex[-1] = fe
                            else:
                                simplex[-1] = xr
                                f_simplex[-1] = fr
                        elif fr < f_simplex[-2]:
                            simplex[-1] = xr
                            f_simplex[-1] = fr
                        else:
                            # contraction
                            if fr < f_simplex[-1]:
                                xc = centroid + 0.5 * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                evals += 1
                                nm_evals += 1
                                if fc < fr:
                                    simplex[-1] = xc
                                    f_simplex[-1] = fc
                                else:
                                    # shrink
                                    for i in range(1, dim + 1):
                                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        f_simplex[i] = func(simplex[i])
                                        evals += 1
                                        nm_evals += 1
                            else:
                                xc = centroid - 0.5 * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                evals += 1
                                nm_evals += 1
                                if fc < f_simplex[-1]:
                                    simplex[-1] = xc
                                    f_simplex[-1] = fc
                                else:
                                    for i in range(1, dim + 1):
                                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        f_simplex[i] = func(simplex[i])
                                        evals += 1
                                        nm_evals += 1
                        min_idx = np.argmin(f_simplex)
                        if f_simplex[min_idx] < best_f:
                            best_f = f_simplex[min_idx]
                            best_x = simplex[min_idx].copy()
                    # update global best
                    if best_f < self.f_opt:
                        self.f_opt = best_f
                        self.x_opt = best_x.copy()

            # update global best
            if best_f < self.f_opt:
                self.f_opt = best_f
                self.x_opt = best_x.copy()

        return self.f_opt, self.x_opt