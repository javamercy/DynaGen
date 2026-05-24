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
        if lam < 2:
            # no further evaluations possible
            return best_value, best_x
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
                # restart
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0

                # local refinement: mini CMA-ES with population 2
                local_steps = min(5, (budget - evals) // 2)
                for _ in range(local_steps):
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        C = np.eye(dim)
                        L = np.eye(dim)
                    z1 = rng.randn(dim)
                    z2 = rng.randn(dim)
                    x1 = m + sigma * 0.5 * L @ z1
                    x2 = m + sigma * 0.5 * L @ z2
                    x1 = np.clip(x1, lb, ub)
                    x2 = np.clip(x2, lb, ub)
                    v1 = func(x1)
                    evals += 1
                    if v1 < best_value:
                        best_value = v1
                        best_x = x1.copy()
                        report_best(best_value, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                    if evals >= budget:
                        break
                    v2 = func(x2)
                    evals += 1
                    if v2 < best_value:
                        best_value = v2
                        best_x = x2.copy()
                        report_best(best_value, best_x)
                        no_improve_evals = 0
                    else:
                        no_improve_evals += 1
                    if evals >= budget:
                        break
                    x_old = m.copy()
                    if v1 <= v2:
                        m = x1.copy()
                        y = x1 - x_old
                    else:
                        m = x2.copy()
                        y = x2 - x_old
                    delta = y / (sigma * 0.5)
                    p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc)) * delta
                    C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
                    C += 1e-10 * np.eye(dim)
                    invL = np.linalg.solve(L, np.eye(dim))
                    delta_Cinv = invL @ delta
                    p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs)) * delta_Cinv
                    norm_p_s = np.linalg.norm(p_s)
                    sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
                    sigma = max(sigma, 1e-10)
                # end local refinement
                # recompute lam_gen for main loop
                lam_gen = min(lam, budget - evals)
            else:
                lam_gen = min(lam, budget - evals)

            if lam_gen < 2:
                # not enough evaluations for CMA-ES update, try a single point
                if lam_gen == 1:
                    x_try = m + sigma * rng.randn(dim)
                    x_try = np.clip(x_try, lb, ub)
                    v_try = func(x_try)
                    evals += 1
                    if v_try < best_value:
                        best_value = v_try
                        best_x = x_try.copy()
                        report_best(best_value, best_x)
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

            # update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta

            # update p_s
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)

            # update C (rank-one + rank-mu)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)

        return best_value, best_x