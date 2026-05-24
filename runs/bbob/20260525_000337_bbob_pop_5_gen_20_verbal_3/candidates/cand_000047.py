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
        lam = 4 + 2 * int(np.log(dim))
        lam = max(2, min(lam, budget - evals))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = 4.0 / (dim + 4.0)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        ccov = 2.0 / ((dim + 1.3) ** 1.5)
        ccov_mu = min(1.0, ccov * (mueff - 1 + 1 / mueff) / (dim + 2) ** 1.5)
        m = best_x.copy()
        sigma = (ub - lb).mean() / 5.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        no_improve_evals = 0
        restart_threshold = int(0.2 * budget)
        local_budget = min(50, max(10, int(0.05 * budget)))
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                lam_local = max(2, int(2 + np.log(dim)))
                mu_local = lam_local // 2
                if mu_local < 1:
                    mu_local = 1
                w_local = np.log(mu_local + 0.5) - np.log(np.arange(1, mu_local + 1))
                w_local = w_local / w_local.sum()
                mueff_local = 1.0 / np.sum(w_local ** 2)
                m_local = best_x.copy()
                sigma_local = (ub - lb).mean() / 20.0
                C_local = np.eye(dim)
                p_c_local = np.zeros(dim)
                p_s_local = np.zeros(dim)
                local_evals = 0
                while local_evals < local_budget and evals + local_evals < budget:
                    lam_curr = min(lam_local, local_budget - local_evals)
                    if lam_curr < 2:
                        break
                    try:
                        L = np.linalg.cholesky(C_local)
                    except np.linalg.LinAlgError:
                        C_local = np.eye(dim)
                        L = np.eye(dim)
                    pop = np.zeros((lam_curr, dim))
                    for i in range(lam_curr):
                        z = rng.randn(dim)
                        pop[i] = m_local + sigma_local * L @ z
                    np.clip(pop, lb, ub, out=pop)
                    vals = np.array([func(p) for p in pop])
                    local_evals += lam_curr
                    for i in range(lam_curr):
                        if vals[i] < best_value:
                            best_value = vals[i]
                            best_x = pop[i].copy()
                            no_improve_evals = 0
                            report_best(best_value, best_x)
                    idx = np.argsort(vals)
                    x_old = m_local.copy()
                    m_local = w_local @ pop[idx[:mu_local]]
                    delta = (m_local - x_old) / sigma_local
                    p_c_local = (1 - cc) * p_c_local + np.sqrt(cc * (2 - cc) * mueff_local) * delta
                    C_mu = np.zeros((dim, dim))
                    for i in range(mu_local):
                        z = (pop[idx[i]] - x_old) / sigma_local
                        C_mu += w_local[i] * np.outer(z, z)
                    C_mu = C_mu - np.outer(delta, delta)
                    invL = np.linalg.solve(L, np.eye(dim))
                    delta_Cinv = invL @ delta
                    p_s_local = (1 - cs) * p_s_local + np.sqrt(cs * (2 - cs) * mueff_local) * delta_Cinv
                    norm_ps = np.linalg.norm(p_s_local)
                    sigma_local *= np.exp((cs / damps) * (norm_ps / np.sqrt(dim) - 1.0))
                    sigma_local = max(sigma_local, 1e-10)
                    C_local = (1 - ccov) * C_local + ccov * np.outer(p_c_local, p_c_local) + ccov_mu * C_mu
                    C_local += 1e-10 * np.eye(dim)
                evals += local_evals
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                np.clip(m, lb, ub, out=m)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
            else:
                lam_curr = min(lam, budget - evals)
                if lam_curr < 2:
                    break
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)
                pop = np.zeros((lam_curr, dim))
                for i in range(lam_curr):
                    z = rng.randn(dim)
                    pop[i] = m + sigma * L @ z
                np.clip(pop, lb, ub, out=pop)
                vals = np.array([func(p) for p in pop])
                evals += lam_curr
                improved = False
                for i in range(lam_curr):
                    if vals[i] < best_value:
                        best_value = vals[i]
                        best_x = pop[i].copy()
                        no_improve_evals = 0
                        report_best(best_value, best_x)
                        improved = True
                if not improved:
                    no_improve_evals += lam_curr
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
                norm_ps = np.linalg.norm(p_s)
                sigma *= np.exp((cs / damps) * (norm_ps / np.sqrt(dim) - 1.0))
                sigma = max(sigma, 1e-10)
                C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
                C += 1e-10 * np.eye(dim)
        return best_value, best_x