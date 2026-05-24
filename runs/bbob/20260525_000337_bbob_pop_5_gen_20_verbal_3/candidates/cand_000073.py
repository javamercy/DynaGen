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
        # CMA-ES parameters (exploration-oriented)
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
        # refinement phase control
        refinement_generations = 0
        max_refinement_gens = 5
        
        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # restart near best with random perturbation and moderate sigma
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                refinement_generations = 0  # start refinement phase
                lam_gen = min(lam, budget - evals)
            else:
                lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # refinement phase: reduce sigma and lambda for first few generations after restart
            if refinement_generations < max_refinement_gens and sigma > 1e-8:
                local_sigma = sigma * 0.5
                local_lam = max(2, lam // 2)
                local_lam = min(local_lam, budget - evals)
            else:
                local_sigma = sigma
                local_lam = lam_gen
            lam_eff = min(local_lam, budget - evals)
            if lam_eff < 2:
                break
            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            # sample population
            pop = np.zeros((lam_eff, dim))
            for i in range(lam_eff):
                z = rng.randn(dim)
                pop[i] = m + local_sigma * L @ z
            np.clip(pop, lb, ub, out=pop)
            # evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_eff
            # update best
            for i in range(lam_eff):
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
            delta = (m - x_old) / local_sigma
            # update rank-one covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # update rank-mu covariance (active)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / local_sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            # cumulative step-size adaptation (use actual sigma for step size update? use local_sigma? use sigma for consistency)
            # Actually step-size adaptation should use the current sigma (local_sigma) but we should update global sigma consistently
            # We'll compute using local_sigma for the adaptation but update sigma accordingly
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # update covariance
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)
            refinement_generations += 1
        return best_value, best_x