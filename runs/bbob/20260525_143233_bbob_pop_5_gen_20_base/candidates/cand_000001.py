import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, func):
        dim = self.dim
        bounds_lb = func.bounds.lb
        bounds_ub = func.bounds.ub
        
        # initial point
        x0 = np.random.uniform(bounds_lb, bounds_ub)
        best_x = x0.copy()
        best_val = func(x0)
        self.calls = 1
        report_best(best_val, best_x)
        
        # CMA-ES parameters
        n = dim
        lambda_ = max(4, int(4 + 3 * np.log(n)))
        # adjust to budget: ensure at least 2 generations
        lambda_ = min(lambda_, self.budget // 2)
        if lambda_ < 4:
            lambda_ = 4
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # strategy parameters
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        damps = 1 + 2*max(0, np.sqrt((mueff-1)/(n+1)) - 1) + cs
        
        # initialize dynamic state
        xmean = x0.copy()
        sigma = 0.5 * (bounds_ub - bounds_lb).mean()
        pc = np.zeros(n)
        ps = np.zeros(n)
        B = np.eye(n)
        D = np.ones(n)
        C = B @ np.diag(D**2) @ B.T
        invsqrtC = B @ np.diag(1/D) @ B.T
        eigeneval = 0
        
        while self.calls < self.budget:
            # generate offspring
            arz = np.random.randn(lambda_, n)
            arx = np.zeros((lambda_, n))
            for i in range(lambda_):
                arx[i] = xmean + sigma * (B @ (D * arz[i]))
            # clip to bounds
            arx = np.clip(arx, bounds_lb, bounds_ub)
            # evaluate
            arf = np.full(lambda_, np.inf)
            for i in range(lambda_):
                if self.calls >= self.budget:
                    break
                val = func(arx[i])
                self.calls += 1
                arf[i] = val
                if val < best_val:
                    best_val = val
                    best_x = arx[i].copy()
                    report_best(best_val, best_x)
            if self.calls >= self.budget:
                break
            # sort by fitness
            idx = np.argsort(arf)
            arf = arf[idx]
            arx = arx[idx]
            # update xmean
            xold = xmean.copy()
            xmean = weights @ arx[:mu]
            # update evolution paths
            zmean = weights @ arz[idx[:mu]]
            ps = (1 - cs) * ps + np.sqrt(cs*(2-cs)*mueff) * (invsqrtC @ zmean)
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1-cs)**(2*self.calls/lambda_)) < 1.4 + 2/(n+1)
            pc = (1 - cc) * pc + hs * np.sqrt(cc*(2-cc)*mueff) * (zmean @ invsqrtC.T)  # actually: pc = (1-cc)*pc + hs*sqrt(cc*(2-cc)*mueff) * (B @ (D * zmean))  ?
            # Standard CMA update: pc = (1-cc)*pc + hs*sqrt(cc*(2-cc)*mueff) * (xmean-xold)/sigma
            # Simplify: use arithmetics
            pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
            # update covariance matrix
            artmp = (arx[:mu] - xold) / sigma
            C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc) + cmu * (weights @ np.array([np.outer(artmp[i], artmp[i]) for i in range(mu)]))
            # update step-size
            sigma *= np.exp((cs/damps) * (np.linalg.norm(ps)/np.sqrt(n) - 1))
            # enforce positive definiteness and symmetrize
            C = (C + C.T) / 2
            # update B and D
            if self.calls - eigeneval > lambda_ / (c1+cmu) / n / 10:
                eigeneval = self.calls
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                invsqrtC = B @ np.diag(1/D) @ B.T
        return best_val, best_x