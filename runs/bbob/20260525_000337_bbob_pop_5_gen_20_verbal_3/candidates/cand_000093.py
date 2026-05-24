import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = self.rng
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)

        # Initial point
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)

        # CMA-ES parameters
        lam = 4 + int(2 * np.log(dim))
        lam = max(2, min(lam, budget - evals))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**1.5)

        # State variables
        m = best_x.copy()
        sigma = (ub - lb).mean() / 4.0  # initial sigma
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)

        # Stagnation tracking
        no_improve_evals = 0
        restart_threshold = int(0.15 * budget)

        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # Restart: reinitialize mean uniformly, larger sigma, reset covariances
                m = rng.uniform(lb, ub, size=dim)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                # Temporarily double population for one generation
                lam_gen = min(2 * lam, budget - evals)
            else:
                lam_gen = min(lam, budget - evals)

            if lam_gen < 2:
                break

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            # Sample population
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)

            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen

            # Update best and stagnation
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1

            # Sort by fitness
            idx = np.argsort(vals)

            # Update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma

            # Update rank-one component
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta

            # Update rank-mu component (active)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)

            # Step-size adaptation
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)

            # Update covariance matrix
            ccov_mu = ccov * (mueff - 1 + 1/mueff) / (dim + 2)**1.5
            ccov_mu = min(1.0, ccov_mu)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)

        return best_value, best_x