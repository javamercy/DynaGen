import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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
        x0 = rng.uniform(lb, ub, size=dim)
        best_x = x0.copy()
        best_value = func(best_x)
        evals = 1
        report_best(best_value, best_x)
        
        # CMA-ES parameters
        lam = 6 + int(3 * np.log(dim))
        lam = max(2, min(lam, budget - evals))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3)**1.5)
        ccov_mu = min(1.0, ccov * (mueff - 1 + 1/mueff) / (dim + 2)**1.5)
        m = best_x.copy()
        sigma = (ub - lb).mean() / 5.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        
        # Restart threshold in number of CMA generations without improvement
        restart_gen_threshold = max(1, int(0.2 * budget / lam))
        cma_no_improve = 0
        
        # Local search parameters
        local_search_frac = 0.03
        local_step = 0.05  # fraction of sigma
        
        while evals < budget:
            # Check restart condition based on CMA generations only
            if cma_no_improve >= restart_gen_threshold and evals < budget - 2:
                # Restart near best with random perturbation
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                cma_no_improve = 0
                # Local search after restart
                local_evals = min(int(local_search_frac * budget), budget - evals)
                for _ in range(local_evals):
                    point = best_x + local_step * sigma * rng.randn(dim)
                    point = np.clip(point, lb, ub)
                    val = func(point)
                    evals += 1
                    if val < best_value:
                        best_value = val
                        best_x = point.copy()
                        report_best(best_value, best_x)
                if evals >= budget:
                    break
                # Reinitialize mean to best after local search
                m = best_x.copy()
                # Reset sigma based on distance from bounds? Not needed.
                continue
            
            # Determine generation size
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            # Sample
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # Update best
            improved = False
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    report_best(best_value, best_x)
                    improved = True
            # Update no_improve counter only based on CMA generation
            if not improved:
                cma_no_improve += 1
            else:
                cma_no_improve = 0
            # Sort
            idx = np.argsort(vals)
            # Update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # Update step-size and covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # Active rank-mu
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            # Cumulative step-size
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            sigma *= np.exp((cs / damps) * (np.linalg.norm(p_s) / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # Covariance update
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)
        return best_value, best_x