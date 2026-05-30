import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.lam = 4 + int(3 * np.log(dim))
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        domain_range = ub - lb
        best_f = np.inf
        best_x = None
        count = 0
        # initial evaluation
        x0 = self.rng.uniform(lb, ub)
        f0 = func(x0)
        count += 1
        if f0 < best_f:
            best_f = f0
            best_x = x0.copy()
            report_best(f0, best_x)
        # number of restarts, ensure at least 1
        n_restarts = max(1, int(np.sqrt(self.budget / (self.dim * 10))))
        # allocate budget per restart, leaving some for pattern search
        for restart in range(n_restarts):
            if count >= self.budget:
                break
            remaining = self.budget - count
            budget_per_restart = max(self.lam + 1, remaining // (n_restarts - restart + 1))
            # initialize CMA-ES parameters
            sigma = 0.3 * np.mean(domain_range)
            xmean = self.rng.uniform(lb, ub)
            pc = np.zeros(self.dim)
            ps = np.zeros(self.dim)
            C = np.eye(self.dim)
            B = np.eye(self.dim)
            D = np.ones(self.dim)
            invsqrtC = np.eye(self.dim)
            eigen_eval = 0
            # local best tracking
            local_best_f = np.inf
            local_best_x = None
            # first evaluate xmean
            f = func(xmean)
            count += 1
            if f < best_f:
                best_f = f
                best_x = xmean.copy()
                report_best(f, best_x)
            if f < local_best_f:
                local_best_f = f
                local_best_x = xmean.copy()
            # main CMA-ES loop for this restart
            while count + self.lam <= self.budget and count < self.budget:
                arx = []
                arf = []
                for _ in range(self.lam):
                    z = self.rng.normal(0, 1, self.dim)
                    y = B @ (D * z)
                    x = xmean + sigma * y
                    x = np.clip(x, lb, ub)
                    arx.append(x)
                    f = func(x)
                    count += 1
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(f, best_x)
                    if f < local_best_f:
                        local_best_f = f
                        local_best_x = x.copy()
                    if count >= self.budget:
                        break
                if count >= self.budget:
                    break
                arf = np.array(arf)
                idx = np.argsort(arf)
                xold = xmean.copy()
                xmean = np.sum(self.weights[:, None] * np.array(arx)[idx[:self.mu]], axis=0)
                dmean = xmean - xold
                ps = (1 - self.cs) * ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (invsqrtC @ dmean / sigma)
                pc = (1 - self.cc) * pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (dmean / sigma)
                C = (1 - self.c1 - self.cmu) * C
                C += self.c1 * np.outer(pc, pc)
                for i in range(self.mu):
                    diff = (np.array(arx)[idx[i]] - xold) / sigma
                    C += self.cmu * self.weights[i] * np.outer(diff, diff)
                sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(ps) / (np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))) - 1))
                if count - eigen_eval > self.dim:
                    eigen_eval = count
                    C = np.triu(C) + np.triu(C, 1).T
                    D2, B = np.linalg.eigh(C)
                    D2 = np.abs(D2)
                    D2 = np.maximum(D2, 1e-30)
                    D = np.sqrt(D2)
                    invsqrtC = B @ np.diag(1/D) @ B.T
                # early local convergence check: if sigma small, break
                if sigma < 1e-10 * np.mean(domain_range):
                    break
            # After CMA-ES, apply pattern search on local best
            if local_best_x is not None:
                x_current = local_best_x.copy()
                f_current = local_best_f
                # ensure eigenvectors are up-to-date
                C = np.triu(C) + np.triu(C, 1).T
                D2, B = np.linalg.eigh(C)
                D2 = np.abs(D2)
                D2 = np.maximum(D2, 1e-30)
                D = np.sqrt(D2)
                step_init = 0.1 * np.mean(domain_range) * D
                factor = 1.0
                while count < self.budget:
                    improved = False
                    for i in range(self.dim):
                        direction = B[:, i]
                        step = factor * step_init[i]
                        # positive step
                        x_new = x_current + step * direction
                        x_new = np.clip(x_new, lb, ub)
                        f_new = func(x_new)
                        count += 1
                        if f_new < f_current:
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                                report_best(f_new, best_x)
                            f_current = f_new
                            x_current = x_new
                            improved = True
                        if count >= self.budget:
                            break
                        # negative step
                        x_new = x_current - step * direction
                        x_new = np.clip(x_new, lb, ub)
                        f_new = func(x_new)
                        count += 1
                        if f_new < f_current:
                            if f_new < best_f:
                                best_f = f_new
                                best_x = x_new.copy()
                                report_best(f_new, best_x)
                            f_current = f_new
                            x_current = x_new
                            improved = True
                        if count >= self.budget:
                            break
                    if not improved or count >= self.budget:
                        factor *= 0.5
                    if factor < 1e-10 or count >= self.budget:
                        break
        # final random fill if budget remains
        while count < self.budget:
            x = self.rng.uniform(lb, ub)
            f = func(x)
            count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(f, best_x)
        return best_f, best_x