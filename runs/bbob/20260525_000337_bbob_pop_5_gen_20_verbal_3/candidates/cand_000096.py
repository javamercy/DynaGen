import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        rng = np.random.RandomState(self.seed)
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        # initial point
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
        no_improve_evals = 0
        restart_threshold = int(0.2 * budget)
        # main loop
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # restart
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                np.clip(m, lb, ub, out=m)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                # local refinement using a few CMA-ES generations with small population
                local_budget = min(2*dim, budget - evals)
                if local_budget >= dim+1:
                    lam_loc = max(dim+1, 4)
                    lam_loc = min(lam_loc, local_budget)
                    mu_loc = lam_loc // 2
                    w_loc = np.log(mu_loc + 0.5) - np.log(np.arange(1, mu_loc+1))
                    w_loc = w_loc / w_loc.sum()
                    mueff_loc = 1.0 / np.sum(w_loc**2)
                    cc_loc = 4.0 / (dim + 4.0)
                    cs_loc = (mueff_loc + 2.0) / (dim + mueff_loc + 5.0)
                    damps_loc = 1.0 + 2.0 * max(0.0, np.sqrt((mueff_loc - 1.0) / (dim + 1.0)) - 1.0) + cs_loc
                    ccov_loc = 2.0 / ((dim + 1.3)**1.5)
                    ccov_mu_loc = min(1.0, ccov_loc * (mueff_loc - 1 + 1/mueff_loc) / (dim + 2)**1.5)
                    m_loc = best_x.copy()
                    sigma_loc = 0.3 * sigma
                    C_loc = np.eye(dim)
                    p_c_loc = np.zeros(dim)
                    p_s_loc = np.zeros(dim)
                    local_evals = 0
                    while evals < budget and local_evals < local_budget:
                        # sample
                        try:
                            L_loc = np.linalg.cholesky(C_loc)
                        except np.linalg.LinAlgError:
                            C_loc = np.eye(dim)
                            L_loc = np.eye(dim)
                        pop = np.zeros((lam_loc, dim))
                        for i in range(lam_loc):
                            z = rng.randn(dim)
                            pop[i] = m_loc + sigma_loc * L_loc @ z
                        np.clip(pop, lb, ub, out=pop)
                        vals = np.array([func(p) for p in pop])
                        evals += lam_loc
                        local_evals += lam_loc
                        for i in range(lam_loc):
                            if vals[i] < best_value:
                                best_value = vals[i]
                                best_x = pop[i].copy()
                                report_best(best_value, best_x)
                                no_improve_evals = 0
                        # update
                        idx = np.argsort(vals)
                        x_old = m_loc.copy()
                        m_loc = w_loc @ pop[idx[:mu_loc]]
                        delta = (m_loc - x_old) / sigma_loc
                        p_c_loc = (1 - cc_loc) * p_c_loc + np.sqrt(cc_loc * (2 - cc_loc) * mueff_loc) * delta
                        C_mu = np.zeros((dim, dim))
                        for i in range(mu_loc):
                            z = (pop[idx[i]] - x_old) / sigma_loc
                            C_mu += w_loc[i] * np.outer(z, z)
                        C_mu = C_mu - np.outer(delta, delta)
                        invL_loc = np.linalg.solve(L_loc, np.eye(dim))
                        delta_Cinv = invL_loc @ delta
                        p_s_loc = (1 - cs_loc) * p_s_loc + np.sqrt(cs_loc * (2 - cs_loc) * mueff_loc) * delta_Cinv
                        norm_p_s_loc = np.linalg.norm(p_s_loc)
                        sigma_loc *= np.exp((cs_loc / damps_loc) * (norm_p_s_loc / np.sqrt(dim) - 1.0))
                        sigma_loc = max(sigma_loc, 1e-10)
                        C_loc = (1 - ccov_loc) * C_loc + ccov_loc * np.outer(p_c_loc, p_c_loc) + ccov_mu_loc * C_mu
                        C_loc += 1e-10 * np.eye(dim)
                        if evals >= budget:
                            break
                lam_gen = min(lam, budget - evals)
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
            # sample population
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            # evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # update best
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1
            # sort
            idx = np.argsort(vals)
            # update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # update rank-one covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # update rank-mu covariance (active)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            # cumulative step-size adaptation
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # update covariance
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)
        return best_value, best_x