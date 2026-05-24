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
        restart_threshold = int(0.15 * budget)  # earlier restart
        local_search_budget = min(50, max(20, int(0.1 * budget)))
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # pattern search local refinement
                pattern_step = (ub - lb).mean() / 20.0
                for _ in range(local_search_budget):
                    if evals >= budget:
                        break
                    improved = False
                    for d in range(dim):
                        for sign in [1, -1]:
                            candidate = best_x.copy()
                            candidate[d] += sign * pattern_step
                            candidate = np.clip(candidate, lb, ub)
                            val = func(candidate)
                            evals += 1
                            if val < best_value:
                                best_value = val
                                best_x = candidate.copy()
                                report_best(best_value, best_x)
                                improved = True
                                break
                        if improved:
                            break
                    if not improved:
                        pattern_step *= 0.5
                        if pattern_step < 1e-12 * (ub - lb).mean():
                            break
                # restart CMA-ES near best with small sigma
                m = best_x.copy() + sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 10.0  # smaller initial sigma
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                lam_gen = min(lam, budget - evals)
                if lam_gen < 2:
                    break
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