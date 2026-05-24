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
        no_improve_evals = 0
        restart_threshold = int(0.2 * budget)
        # local refinement budget
        local_budget_min = 10
        local_budget_frac = 0.1
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # local CMA refinement before restart
                local_budget = max(local_budget_min, int(local_budget_frac * (budget - evals)))
                if local_budget > 0:
                    lam_local = max(2, lam // 2)
                    mu_local = lam_local // 2
                    w_local = np.log(mu_local + 0.5) - np.log(np.arange(1, mu_local+1))
                    w_local = w_local / w_local.sum()
                    mueff_local = 1.0 / np.sum(w_local**2)
                    cc_local = 4.0 / (dim + 4.0)
                    cs_local = (mueff_local + 2.0) / (dim + mueff_local + 5.0)
                    damps_local = 1.0 + 2.0 * max(0.0, np.sqrt((mueff_local - 1.0) / (dim + 1.0)) - 1.0) + cs_local
                    ccov_local = 2.0 / ((dim + 1.3)**1.5)
                    ccov_mu_local = min(1.0, ccov_local * (mueff_local - 1 + 1/mueff_local) / (dim + 2)**1.5)
                    m_local = best_x.copy()
                    sigma_local = 0.1 * (ub - lb).mean()
                    C_local = np.eye(dim)
                    p_c_local = np.zeros(dim)
                    p_s_local = np.zeros(dim)
                    local_evals = 0
                    while local_evals < local_budget and evals + local_evals < budget:
                        lam_gen = min(lam_local, local_budget - local_evals)
                        if lam_gen < 2:
                            break
                        try:
                            L_local = np.linalg.cholesky(C_local)
                        except np.linalg.LinAlgError:
                            C_local = np.eye(dim)
                            L_local = np.eye(dim)
                        pop = np.zeros((lam_gen, dim))
                        for i in range(lam_gen):
                            z = rng.randn(dim)
                            pop[i] = m_local + sigma_local * L_local @ z
                        np.clip(pop, lb, ub, out=pop)
                        vals = np.array([func(p) for p in pop])
                        local_evals += lam_gen
                        for i in range(lam_gen):
                            if vals[i] < best_value:
                                best_value = vals[i]
                                best_x = pop[i].copy()
                                no_improve_evals = 0
                                report_best(best_value, best_x)
                        idx = np.argsort(vals)
                        x_old = m_local.copy()
                        m_local = w_local @ pop[idx[:mu_local]]
                        delta = (m_local - x_old) / sigma_local
                        p_c_local = (1 - cc_local) * p_c_local + np.sqrt(cc_local * (2 - cc_local) * mueff_local) * delta
                        C_mu_local = np.zeros((dim, dim))
                        for i in range(mu_local):
                            z = (pop[idx[i]] - x_old) / sigma_local
                            C_mu_local += w_local[i] * np.outer(z, z)
                        C_mu_local = C_mu_local - np.outer(delta, delta)
                        invL_local = np.linalg.solve(L_local, np.eye(dim))
                        delta_Cinv_local = invL_local @ delta
                        p_s_local = (1 - cs_local) * p_s_local + np.sqrt(cs_local * (2 - cs_local) * mueff_local) * delta_Cinv_local
                        norm_p_s = np.linalg.norm(p_s_local)
                        sigma_local *= np.exp((cs_local / damps_local) * (norm_p_s / np.sqrt(dim) - 1.0))
                        sigma_local = max(sigma_local, 1e-10)
                        C_local = (1 - ccov_local) * C_local + ccov_local * np.outer(p_c_local, p_c_local) + ccov_mu_local * C_mu_local
                        C_local += 1e-10 * np.eye(dim)
                    evals += local_evals
                # reset global CMA state with refined best
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
            else:
                lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1
            idx = np.argsort(vals)
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)
        return best_value, best_x