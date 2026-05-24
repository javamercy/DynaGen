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
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # restart near best with random perturbation and larger sigma
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                # local covariance-adaptive refinement
                local_sigma = sigma * 0.5
                local_lam = 2
                local_max_gen = min(5, (budget - evals) // local_lam)
                if local_max_gen > 0:
                    local_m = m.copy()
                    local_C = C.copy()
                    local_p_c = p_c.copy()
                    local_p_s = p_s.copy()
                    local_mueff = 1.0  # mu=1
                    local_cc = cc
                    local_cs = (local_mueff + 2.0) / (dim + local_mueff + 5.0)
                    local_damps = 1.0 + 2.0 * max(0.0, np.sqrt((local_mueff - 1.0) / (dim + 1.0)) - 1.0) + local_cs
                    local_ccov = 2.0 / ((dim + 1.3)**1.5)
                    local_ccov_mu = min(1.0, local_ccov * (local_mueff - 1 + 1/local_mueff) / (dim + 2)**1.5)
                    for _ in range(local_max_gen):
                        try:
                            L = np.linalg.cholesky(local_C)
                        except np.linalg.LinAlgError:
                            local_C = np.eye(dim)
                            L = np.eye(dim)
                        z = rng.randn(dim)
                        trial1 = local_m + local_sigma * L @ z
                        trial1 = np.clip(trial1, lb, ub)
                        val1 = func(trial1)
                        evals += 1
                        z = rng.randn(dim)
                        trial2 = local_m + local_sigma * L @ z
                        trial2 = np.clip(trial2, lb, ub)
                        val2 = func(trial2)
                        evals += 1
                        if val1 < val2:
                            best_idx = 0
                        else:
                            best_idx = 1
                        pop = [trial1, trial2]
                        vals = [val1, val2]
                        # update best
                        for i in range(2):
                            if vals[i] < best_value:
                                best_value = vals[i]
                                best_x = pop[i].copy()
                                no_improve_evals = 0
                                report_best(best_value, best_x)
                        # update mean (mu=1, so best point)
                        x_old = local_m.copy()
                        local_m = pop[best_idx].copy()
                        delta = (local_m - x_old) / local_sigma
                        # update p_c
                        local_p_c = (1 - local_cc) * local_p_c + np.sqrt(local_cc * (2 - local_cc) * local_mueff) * delta
                        # update p_s, sigma
                        try:
                            invL = np.linalg.solve(L, np.eye(dim))
                        except np.linalg.LinAlgError:
                            invL = np.eye(dim)
                        delta_Cinv = invL @ delta
                        local_p_s = (1 - local_cs) * local_p_s + np.sqrt(local_cs * (2 - local_cs) * local_mueff) * delta_Cinv
                        norm_p_s = np.linalg.norm(local_p_s)
                        local_sigma *= np.exp((local_cs / local_damps) * (norm_p_s / np.sqrt(dim) - 1.0))
                        local_sigma = max(local_sigma, 1e-10)
                        # update C (rank-one only, mu=1)
                        local_C = (1 - local_ccov) * local_C + local_ccov * np.outer(local_p_c, local_p_c) + 1e-10 * np.eye(dim)
                    # after local refinement, transfer state back to main
                    m = local_m.copy()
                    C = local_C.copy()
                    p_c = local_p_c.copy()
                    p_s = local_p_s.copy()
                    sigma = local_sigma
                # continue with main loop
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