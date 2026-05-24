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

        # CMA-ES parameters
        lam = max(2, 4 + int(3 * np.log(dim)))
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
            # Diversity check: mean distance of population to mean
            if evals > lam:  # need at least one full population
                # compute mean distance of last population (if available)
                pass  # we'll compute after sampling

            # Decision to restart based on no improvement or low diversity
            restart = False
            if no_improve_evals >= restart_threshold:
                restart = True
            # compute diversity if we have at least one full population
            if evals > lam and not restart:
                # population diversity measured as mean distance to mean
                # We'll compute after sampling in next iteration? Better to check after a generation.
                # We store last generation points and compute diversity
                pass  # will be done after sampling

            if restart:
                # Restart: reinitialize mean, sigma, C, p_c, p_s
                m = best_x.copy() + 1.5 * sigma * rng.randn(dim)
                m = np.clip(m, lb, ub)
                sigma = (ub - lb).mean() / 4.0
                C = np.eye(dim)
                p_c = np.zeros(dim)
                p_s = np.zeros(dim)
                no_improve_evals = 0

                # Local refinement: 2 random perturbations
                for _ in range(2):
                    if evals >= budget:
                        break
                    z = rng.randn(dim)
                    x_try = best_x + 0.5 * sigma * z
                    x_try = np.clip(x_try, lb, ub)
                    val = func(x_try)
                    evals += 1
                    if val < best_value:
                        best_value = val
                        best_x = x_try.copy()
                        no_improve_evals = 0
                        report_best(best_value, best_x)
                continue  # restart loop, then continue main loop

            # Main CMA-ES generation
            lam_gen = min(lam, budget - evals)
            if lam_gen < 2:
                break

            # Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)

            # Sample population
            pop = np.zeros((lam_gen, dim))
            for i in range(lam_gen):
                z = rng.randn(dim)
                pop[i] = m + sigma * L @ z
            np.clip(pop, lb, ub, out=pop)

            # Evaluate
            vals = np.array([func(p) for p in pop])
            evals += lam_gen

            # Update best
            for i in range(lam_gen):
                if vals[i] < best_value:
                    best_value = vals[i]
                    best_x = pop[i].copy()
                    no_improve_evals = 0
                    report_best(best_value, best_x)
                else:
                    no_improve_evals += 1

            # Compute diversity (mean distance to m) for next generation
            if lam_gen >= 2:
                mean_dist = np.mean(np.linalg.norm(pop - m, axis=1))
                if mean_dist < 0.5 * sigma:
                    # diversity too low, trigger restart after this generation
                    # We'll set a flag to restart in next iteration
                    # But we need to break current generation update? We'll set restart flag for next loop
                    # Actually we can immediately restart after this generation
                    # To avoid modifying update, we set a flag and handle after this iteration?
                    # Better: compute diversity before update and if low, skip update and restart
                    # But we already evaluated pop. So we can just continue update, but set no_improve_evals high to trigger restart next iteration.
                    no_improve_evals = restart_threshold  # force restart next loop

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
            norm_p_s = np.linalg.norm(p_s)
            sigma *= np.exp((cs / damps) * (norm_p_s / np.sqrt(dim) - 1.0))
            sigma = max(sigma, 1e-10)

            # Update C (rank-one + rank-mu)
            C_mu = np.zeros((dim, dim))
            for i in range(mu):
                z = (pop[idx[i]] - x_old) / sigma
                C_mu += w[i] * np.outer(z, z)
            C_mu = C_mu - np.outer(delta, delta)
            C = (1 - ccov) * C + ccov * np.outer(p_c, p_c) + ccov_mu * C_mu
            C += 1e-10 * np.eye(dim)

        return best_value, best_x