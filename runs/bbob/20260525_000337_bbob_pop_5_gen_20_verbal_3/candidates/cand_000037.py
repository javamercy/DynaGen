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
        # Parameters
        lam = 4 + int(2 * np.log(dim)) if dim > 1 else 4
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        ccov = 2.0 / ((dim + 1.3) ** 1.5)
        ccov_neg = 0.5 * ccov * (1 - (mu - 1) / (dim + 2))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        # State
        m = rng.uniform(lb, ub)
        sigma = 0.3 * (ub - lb).mean()
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        invsqrtC = np.eye(dim)
        evals = 0
        best_x = None
        best_val = np.inf
        evals_since_improve = 0
        restart_threshold = int(0.15 * budget)
        while evals < budget:
            # Generate offspring
            arz = rng.randn(lam, dim)
            arx = np.empty((lam, dim))
            arf = np.empty(lam)
            for i in range(lam):
                if evals >= budget:
                    break
                x = m + sigma * (B @ (D * arz[i]))
                x = np.clip(x, lb, ub)
                arx[i] = x
                val = func(x)
                arf[i] = val
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    evals_since_improve = 0
                    report_best(best_val, best_x)
            if evals >= budget:
                break
            # Sort by fitness
            idx = np.argsort(arf)
            arx = arx[idx]
            arf = arf[idx]
            # Update mean
            m_old = m.copy()
            m = np.dot(weights, arx[:mu])
            # Update evolution paths
            diff = (m - m_old) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ diff
            ps_norm = np.linalg.norm(ps)
            hsig = (ps_norm / chiN) < (1.4 + 2.0 / (dim + 1))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * diff
            # Update covariance matrix
            artmp = (arx[:mu] - m_old) / sigma
            C = (1 - ccov) * C + ccov * np.outer(pc, pc)
            for i in range(mu):
                C += ccov * weights[i] * np.outer(artmp[i], artmp[i])
            # Active correction
            if ccov_neg > 0 and lam > mu:
                neg_weights = np.zeros(lam)
                for i in range(mu, lam):
                    neg_weights[i] = -(mu - i - 0.5) / (mu * (mu + 0.5))
                neg_weights /= -np.sum(neg_weights[mu:])
                for i in range(mu, lam):
                    artmp_neg = (arx[i] - m_old) / sigma
                    C += ccov_neg * neg_weights[i] * np.outer(artmp_neg, artmp_neg)
            # Enforce symmetry
            C = np.triu(C) + np.triu(C, 1).T
            # Update B and D
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-10))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T
            except np.linalg.LinAlgError:
                pass
            # Update step size
            sigma *= np.exp((cs / damps) * (ps_norm / chiN - 1))
            # Stagnation
            evals_since_improve += lam
            # Restart if necessary
            if evals_since_improve >= restart_threshold and evals < budget:
                m = rng.uniform(lb, ub)
                sigma = 0.3 * (ub - lb).mean()
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                invsqrtC = np.eye(dim)
                evals_since_improve = 0
        return best_val, best_x