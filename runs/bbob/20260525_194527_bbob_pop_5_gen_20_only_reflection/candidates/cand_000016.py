import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rs = np.random.RandomState(seed)

    def __call__(self, func):
        d = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        best_f = np.inf
        best_x = None
        total_evals = 0
        
        # initial point
        x0 = self.rs.uniform(lb, ub, d)
        f0 = func(x0)
        total_evals += 1
        best_f = f0
        best_x = x0.copy()
        report_best(best_f, best_x)
        
        # CMA-ES parameters
        lam0 = 4 + int(3 * np.log(d))
        lam = lam0
        max_restarts = 5
        restart_count = 0
        
        # restart loop
        while total_evals < self.budget and restart_count <= max_restarts:
            # initialize for current restart
            if restart_count == 0:
                xmean = x0.copy()
            else:
                xmean = self.rs.uniform(lb, ub, d)
            sigma = 0.3 * np.mean(ub - lb)
            C = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            
            # population size for this restart
            lam = lam0 * (2 ** restart_count)
            lam = min(lam, max(1, self.budget // 5))
            mu = max(lam // 2, 1)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = 4.0 / (d + 4.0)
            cs = (mueff + 2.0) / (d + mueff + 5.0)
            c1 = 2.0 / ((d + 1.3) ** 2 + mueff)
            cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((d + 2.0) ** 2 + mueff))
            ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (d + 1.0)) - 1.0) + cs
            norm_expected = np.sqrt(d) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d * d))
            
            restart_best = best_f
            gen_no_improve = 0
            
            while total_evals + lam <= self.budget:
                # compute Cholesky decomposition
                try:
                    B = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-9 * np.eye(d)
                    B = np.linalg.cholesky(C)
                invB = np.linalg.solve(B, np.eye(d))
                
                x_off = np.empty((lam, d))
                f_off = np.empty(lam)
                for i in range(lam):
                    # resample until inside bounds
                    for _ in range(100):
                        z = self.rs.randn(d)
                        x = xmean + sigma * B.dot(z)
                        if np.all(x >= lb) and np.all(x <= ub):
                            break
                    else:
                        x = np.clip(x, lb, ub)
                    x_off[i] = x
                    f_off[i] = func(x)
                    total_evals += 1
                    if f_off[i] < best_f:
                        best_f = f_off[i]
                        best_x = x_off[i].copy()
                        report_best(best_f, best_x)
                
                idx = np.argsort(f_off)
                x_sorted = x_off[idx]
                old_xmean = xmean.copy()
                xmean = np.dot(weights, x_sorted[:mu])
                x_diff = (xmean - old_xmean) / sigma
                
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * x_diff
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invB.dot(x_diff)
                
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                diff = x_sorted[:mu] - old_xmean
                diff_norm = diff / sigma
                C += cmu * np.dot(diff_norm.T, np.dot(np.diag(weights), diff_norm))
                C = (C + C.T) / 2.0
                
                ps_norm = np.linalg.norm(ps)
                sigma *= np.exp((cs / ds) * (ps_norm / norm_expected - 1.0))
                
                # detect stagnation
                gen_best = f_off.min()
                if gen_best < restart_best - 1e-12:
                    restart_best = gen_best
                    gen_no_improve = 0
                else:
                    gen_no_improve += 1
                
                if gen_no_improve > 10 + d // 2:
                    break
            
            restart_count += 1
        
        return best_f, best_x