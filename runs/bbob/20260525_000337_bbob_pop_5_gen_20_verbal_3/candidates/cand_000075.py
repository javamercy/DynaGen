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
        m = best_x.copy()
        sigma = (ub - lb).mean() / 6.0
        C = np.eye(dim)
        p_c = np.zeros(dim)
        p_s = np.zeros(dim)
        # Restart parameters
        no_improve_evals = 0
        restart_threshold = int(0.3 * budget)  # initial, will be overridden adaptively
        restart_count = 0
        while evals < budget:
            # Adaptive restart threshold: 15% of remaining budget, clamped
            remaining = budget - evals
            adaptive_threshold = int(0.15 * remaining) + 10
            adaptive_threshold = max(50, min(adaptive_threshold, 300))
            if no_improve_evals >= adaptive_threshold:
                # Diverse restart
                if restart_count % 2 == 0:
                    m = best_x.copy() + sigma * rng.randn(dim)
                    m = np.clip(m, lb, ub)
                    sigma = (ub - lb).mean() / 12.0
                else:
                    m = rng.uniform(lb, ub, size=dim)
                    sigma = (ub - lb).mean() / 3.0
                # Local refinement after restart: sample 5 small perturbations
                for _ in range(min(5, budget - evals)):
                    point = m + 0.01 * (ub - lb) * rng.randn(dim)
                    point = np.clip(point, lb, ub)
                    val = func(point)
                    evals += 1
                    if val < best_value:
                        best_value = val
                        best_x = point.copy()
                        report_best(best_value, best_x)
                # Reset CMA-ES state
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                restart_count += 1
                continue
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break
            # Sample
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
            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen
            # Update best
            improved = False
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    improved = True
                    report_best(best_value, best_x)
            if improved:
                no_improve_evals = 0
            else:
                no_improve_evals += lam_gen
            # Sort
            idx = np.argsort(vals)
            # Update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / sigma
            # Update p_c
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta
            # Update p_s
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            # Update sigma with aggressive adaptation factor 1.2
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp(1.2 * (cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)
            # Update C
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c)
            C += 1e-10 * np.eye(dim)
        return best_value, best_x