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
        # sub-CMA budget for refinement after restart
        sub_budget = min(50, max(10, int(0.1 * budget)))
        while evals < budget:
            if no_improve_evals >= restart_threshold and evals < budget:
                # short CMA sub-run around best_x for refinement
                sub_lam = max(2, lam // 2)
                sub_mu = sub_lam // 2
                sub_w = np.log(sub_mu + 0.5) - np.log(np.arange(1, sub_mu+1))
                sub_w = sub_w / sub_w.sum()
                sub_m = best_x.copy()
                sub_sigma = (ub - lb).mean() / 20.0
                sub_C = np.eye(dim)
                sub_pc = np.zeros(dim)
                sub_ps = np.zeros(dim)
                sub_evals = 0
                while sub_evals < sub_budget and evals + sub_evals < budget:
                    sub_lam_now = min(sub_lam, budget - evals - sub_evals)
                    if sub_lam_now < 2:
                        break
                    try:
                        L = np.linalg.cholesky(sub_C)
                    except np.linalg.LinAlgError:
                        sub_C = np.eye(dim)
                        L = np.eye(dim)
                    pop = np.zeros((sub_lam_now, dim))
                    for i in range(sub_lam_now):
                        z = rng.randn(dim)
                        pop[i] = sub_m + sub_sigma * L @ z
                    np.clip(pop, lb, ub, out=pop)
                    vals = np.array([func(p) for p in pop])
                    sub_evals += sub_lam_now
                    for i in range(sub_lam_now):
                        if vals[i] < best_value:
                            best_value = vals[i]
                            best_x = pop[i].copy()
                            report_best(best_value, best_x)
                    idx = np.argsort(vals)
                    x_old = sub_m.copy()
                    sub_m = sub_w @ pop[idx[:sub_mu]]
                    delta = (sub_m - x_old) / sub_sigma
                    sub_pc = (1 - cc) * sub_pc + np.sqrt(cc * (2 - cc) * sub_mu) * delta
                    C_mu = np.zeros((dim, dim))
                    for i in range(sub_mu):
                        z = (pop[idx[i]] - x_old) / sub_sigma
                        C_mu += sub_w[i] * np.outer(z, z)
                    C_mu = C_mu - np.outer(delta, delta)
                    invL = np.linalg.solve(L, np.eye(dim))
                    delta_Cinv = invL @ delta
                    sub_ps = (1 - cs) * sub_ps + np.sqrt(cs * (2 - cs) * sub_mu) * delta_Cinv
                    norm_ps = np.linalg.norm(sub_ps)
                    sub_sigma *= np.exp((cs / damps) * (norm_ps / np.sqrt(dim) - 1.0))
                    sub_sigma = max(sub_sigma, 1e-10)
                    sub_C = (1 - ccov) * sub_C + ccov * np.outer(sub_pc, sub_pc) + ccov_mu * C_mu
                    sub_C += 1e-10 * np.eye(dim)
                evals += sub_evals
                # reset main CMA state near best_x
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