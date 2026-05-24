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
        refinement_phase = False
        refine_evals_limit = max(2 * dim, 20)
        refine_evals_spent = 0
        lam_refine = max(2, lam // 2)

        while evals < budget:
            if no_improve_evals >= restart_threshold:
                # Restart near best with random perturbation and larger sigma
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0
                refinement_phase = True
                refine_evals_spent = 0

            # Adjust parameters during refinement phase
            if refinement_phase:
                if refine_evals_spent >= refine_evals_limit:
                    refinement_phase = False
                else:
                    # Use smaller sigma and smaller population
                    current_sigma = sigma * 0.5
                    current_lam = min(lam_refine, budget - evals)
                else:
                    current_sigma = sigma
                    current_lam = min(lam, budget - evals)
            else:
                current_sigma = sigma
                current_lam = min(lam, budget - evals)

            if current_lam < 2:
                break

            # Cholesky decomposition
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            # Sample population
            pop = np.zeros((current_lam, dim))
            for i in range(current_lam):
                z = rng.randn(dim)
                pop[i] = m + current_sigma * L @ z
            np.clip(pop, lb, ub, out=pop)

            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += current_lam

            # Update best
            for i in range(current_lam):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1

            # Sort
            idx = np.argsort(vals)

            # Update mean
            x_old = m.copy()
            m = w @ pop[idx[:mu]]
            delta = (m - x_old) / current_sigma

            # Update rank-one covariance
            p_c = (1 - cc) * p_c + np.sqrt(cc * (2 - cc) * mueff) * delta

            # Update rank-mu covariance (active)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / current_sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)

            # Cumulative step-size adaptation
            invL = np.linalg.solve(L, np.eye(dim))
            delta_Cinv = invL @ delta
            p_s = (1 - cs) * p_s + np.sqrt(cs * (2 - cs) * mueff) * delta_Cinv
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)

            # Update covariance
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)

            # Track refinement phase evals
            if refinement_phase:
                refine_evals_spent += current_lam

        return best_value, best_x