import numpy as np
import math

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        np.random.seed(self.seed)
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub
        range_ = ub - lb
        best_val = None
        best_x = None
        evals = 0
        # initial point
        x0 = np.random.uniform(lb, ub)
        best_val = func(x0)
        best_x = x0.copy()
        evals = 1
        report_best(best_val, best_x)
        if self.budget <= 1:
            return best_val, best_x
        # parameters
        lam = max(2, 10 + int(5 * math.log(dim)))
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        # exploration-friendly: lower learning rates
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff) * 0.5
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff)) * 0.5
        damps = 1 + 2 * max(0, math.sqrt((mueff-1)/(dim+1)) - 1) + cs
        # restart loop
        while evals < self.budget:
            # initialize for this restart
            mean = best_x.copy()
            sigma = 0.5 * np.mean(range_)
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            local_evals = 0
            last_improvement_evals = evals
            # adjust lambda for remaining budget
            lam_cur = min(lam, self.budget - evals)
            if lam_cur < 2:
                break
            while evals < self.budget:
                # sample
                try:
                    A = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    A = np.eye(dim)
                candidates = []
                for _ in range(lam_cur):
                    z = np.random.randn(dim)
                    y = mean + sigma * A @ z
                    y = np.clip(y, lb, ub)
                    candidates.append(y)
                # evaluate
                vals = []
                for cand in candidates:
                    if evals >= self.budget:
                        break
                    val = func(cand)
                    vals.append(val)
                    evals += 1
                    local_evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = cand.copy()
                        last_improvement_evals = evals
                        report_best(best_val, best_x)
                if len(vals) == 0:
                    break
                # sort
                idx = np.argsort(vals)
                candidates = [candidates[i] for i in idx]
                # update mean
                old_mean = mean.copy()
                mean = np.zeros(dim)
                for i in range(mu):
                    mean += weights[i] * candidates[i]
                mean = np.clip(mean, lb, ub)
                # update paths
                z_mean = (mean - old_mean) / sigma
                try:
                    invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
                except:
                    invsqrtC = np.eye(dim)
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ z_mean
                hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2*local_evals/lam_cur)) < (1.4 + 2/(dim+1))
                pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * z_mean
                # update covariance
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                for i in range(mu):
                    z = (candidates[i] - old_mean) / sigma
                    C += cmu * weights[i] * np.outer(z, z)
                C = (C + C.T) / 2
                # update sigma
                sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / math.sqrt(dim) - 1))
                # clamp sigma
                if sigma < 1e-10 * np.mean(range_):
                    sigma = 0.1 * np.mean(range_)
                    C = np.eye(dim)
                    pc = np.zeros(dim)
                    ps = np.zeros(dim)
                # adjust lambda
                lam_cur = min(lam, self.budget - evals)
                if lam_cur < 2:
                    break
                # restart condition: no improvement for a while
                if evals - last_improvement_evals > 0.2 * self.budget:
                    break
            # restart by reinitializing mean to a new random point?
            # But we don't want to lose the best, so just continue from new random mean
            # However, current best is already captured; we can start fresh from random
            mean = np.random.uniform(lb, ub)
            # but we shouldn't evaluate it again if we already have best? We can just continue loop
            # Actually, we will break out of inner while and start new restart
        return best_val, best_x