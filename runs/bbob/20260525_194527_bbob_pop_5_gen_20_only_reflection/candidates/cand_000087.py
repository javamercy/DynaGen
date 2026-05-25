import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    def __call__(self, func):
        d = self.dim
        lb, ub = func.bounds.lb, func.bounds.ub
        x0 = self.rs.uniform(lb, ub, d)
        fbest = func(x0)
        xbest = x0.copy()
        report_best(fbest, xbest)
        evals = 1

        pop_mult = 2.0  # start larger to increase exploration
        while evals < self.budget:
            # Diversification: sample extra random points to possibly update xbest
            n_diversify = min(20, self.budget - evals)
            if n_diversify > 0:
                for _ in range(n_diversify):
                    if evals >= self.budget:
                        break
                    xi = self.rs.uniform(lb, ub, d)
                    fi = func(xi)
                    evals += 1
                    if fi < fbest:
                        fbest = fi
                        xbest = xi.copy()
                        report_best(fbest, xbest)

            if evals >= self.budget:
                break

            lam = int((4 + int(3 * np.log(d))) * pop_mult)
            lam = min(lam, self.budget - evals)
            if lam <= 0:
                break

            # Initialize CMA-ES state from perturbed best
            xmean = xbest + self.rs.uniform(-0.2, 0.2, d) * (ub - lb)
            xmean = np.clip(xmean, lb, ub)
            sigma = 0.4 * np.mean(ub - lb)  # larger initial step
            C = np.eye(d)
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3)**2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0/mueff) / ((d + 2.0)**2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0)/(d + 1.0)) - 1.0) + cs
            pc = np.zeros(d)
            ps = np.zeros(d)
            norm_expected = np.sqrt(d) * (1.0 - 1.0/(4.0*d) + 1.0/(21.0*d*d))
            best_in_run = fbest
            no_improve = 0
            max_no_improve = 50 + int(0.2 * d)
            cond_threshold = 1e14
            gen_count = 0  # count generations for diversification injection

            while evals + lam <= self.budget:
                # Diversification injection every 5 generations if stagnating
                if gen_count > 0 and gen_count % 5 == 0 and no_improve > 10 and self.budget - evals >= 5:
                    for _ in range(5):
                        if evals >= self.budget:
                            break
                        xi = xbest + 0.5 * self.rs.randn(d) * (ub - lb)
                        xi = np.clip(xi, lb, ub)
                        fi = func(xi)
                        evals += 1
                        if fi < fbest:
                            fbest = fi
                            xbest = xi.copy()
                            report_best(fbest, xbest)
                            best_in_run = fbest
                            no_improve = 0

                # Condition number check
                try:
                    eigvals = np.linalg.eigh(C, eigvals_only=True)
                    cond = eigvals[-1] / max(eigvals[0], 1e-20)
                except:
                    cond = 1.0
                if cond > cond_threshold:
                    break
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                z = self.rs.randn(lam, d)
                x_raw = xmean + sigma * np.dot(z, B.T)
                x = np.empty_like(x_raw)
                for i in range(lam):
                    xi = x_raw[i]
                    attempts = 0
                    while np.any(xi < lb) or np.any(xi > ub):
                        attempts += 1
                        if attempts > 10:
                            xi = np.clip(xi, lb, ub)
                            break
                        zi = self.rs.randn(d)
                        xi = xmean + sigma * np.dot(zi, B.T)
                    x[i] = xi
                fvals = np.empty(lam)
                for i in range(lam):
                    fvals[i] = func(x[i])
                    evals += 1
                    if fvals[i] < fbest:
                        fbest = fvals[i]
                        xbest = x[i].copy()
                        report_best(fbest, xbest)
                min_f = np.min(fvals)
                if min_f < best_in_run:
                    best_in_run = min_f
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= max_no_improve:
                    break
                idx = np.argsort(fvals)
                x_sorted = x[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                invB = np.linalg.solve(B, np.eye(d))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * np.dot(invB, x_diff)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp((cs/ds) * (ps_norm / norm_expected - 1.0))
                gen_count += 1

            # Local refinement with adaptive step
            if evals < self.budget:
                local_budget = min(int(0.2 * (self.budget - evals)), 20)
                sigma_local = 0.01 * np.mean(ub - lb)
                for _ in range(local_budget):
                    if evals >= self.budget:
                        break
                    xi = xbest + sigma_local * self.rs.randn(d)
                    xi = np.clip(xi, lb, ub)
                    fi = func(xi)
                    evals += 1
                    if fi < fbest:
                        fbest = fi
                        xbest = xi.copy()
                        report_best(fbest, xbest)
                        sigma_local *= 1.5
                    else:
                        sigma_local *= 0.85

            # Update population multiplier
            if pop_mult < 2.0:
                pop_mult *= 2.0
            else:
                pop_mult *= 1.5

        return fbest, xbest