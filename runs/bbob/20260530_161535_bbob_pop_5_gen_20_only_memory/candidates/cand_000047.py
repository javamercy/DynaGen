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
        self.sigma = None
        self.xmean = None
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)
        self.eigen_eval = 0
        self.count = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        domain_range = self.ub - self.lb
        self.sigma = 0.3 * np.mean(domain_range)
        self.xmean = self.rng.uniform(self.lb, self.ub, size=self.dim)
        if self.count < self.budget:
            f = func(self.xmean)
            self.count += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = self.xmean.copy()
                report_best(f, self.best_x)
        while self.count + self.lam <= self.budget:
            arx = []
            arf = []
            for k in range(self.lam):
                z = self.rng.normal(0, 1, self.dim)
                y = self.B @ (self.D * z)
                x = self.xmean + self.sigma * y
                x = np.clip(x, self.lb, self.ub)
                arx.append(x)
                f = func(x)
                self.count += 1
                arf.append(f)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
                    report_best(f, self.best_x)
            if self.count >= self.budget:
                break
            arf = np.array(arf)
            arx = np.array(arx)
            idx = np.argsort(arf)
            xold = self.xmean.copy()
            self.xmean = np.sum(self.weights[:, None] * arx[idx[:self.mu]], axis=0)
            dmean = self.xmean - xold
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * (self.invsqrtC @ dmean / self.sigma)
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (dmean / self.sigma)
            self.C *= (1 - self.c1 - self.cmu)
            self.C += self.c1 * np.outer(self.pc, self.pc)
            for i in range(self.mu):
                diff = (arx[idx[i]] - xold) / self.sigma
                self.C += self.cmu * self.weights[i] * np.outer(diff, diff)
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / (np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))) - 1))
            if self.count - self.eigen_eval > self.dim:
                self.eigen_eval = self.count
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.abs(self.D)
                self.D = np.maximum(self.D, 1e-30)
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(1/self.D) @ self.B.T
        # Adaptive coordinate descent along eigenvectors
        if self.best_x is not None:
            x_current = self.best_x.copy()
            f_current = self.best_f
            # Ensure eigenvectors are up-to-date
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            D, B = np.linalg.eigh(self.C)
            D = np.abs(D)
            D = np.maximum(D, 1e-30)
            D = np.sqrt(D)
            step_sizes = 0.1 * np.mean(domain_range) * D
            inc = 1.2
            dec = 0.5
            max_cycles = 2 * self.dim
            cycle = 0
            while self.count < self.budget and cycle < max_cycles:
                improved_cycle = False
                permutation = self.rng.permutation(self.dim)
                for i in permutation:
                    direction = B[:, i]
                    step = step_sizes[i]
                    # positive step
                    x_new = x_current + step * direction
                    x_new = np.clip(x_new, self.lb, self.ub)
                    f_new = func(x_new)
                    self.count += 1
                    if f_new < f_current:
                        if f_new < self.best_f:
                            self.best_f = f_new
                            self.best_x = x_new.copy()
                            report_best(f_new, self.best_x)
                        f_current = f_new
                        x_current = x_new
                        step_sizes[i] *= inc
                        improved_cycle = True
                        if self.count >= self.budget:
                            break
                        continue
                    # negative step
                    x_new = x_current - step * direction
                    x_new = np.clip(x_new, self.lb, self.ub)
                    f_new = func(x_new)
                    self.count += 1
                    if f_new < f_current:
                        if f_new < self.best_f:
                            self.best_f = f_new
                            self.best_x = x_new.copy()
                            report_best(f_new, self.best_x)
                        f_current = f_new
                        x_current = x_new
                        step_sizes[i] *= inc
                        improved_cycle = True
                        if self.count >= self.budget:
                            break
                    else:
                        step_sizes[i] *= dec
                if not improved_cycle or self.count >= self.budget:
                    if np.max(step_sizes) < 1e-15:
                        break
                cycle += 1
        if self.best_x is None:
            while self.count < self.budget:
                x = self.rng.uniform(self.lb, self.ub)
                f = func(x)
                self.count += 1
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
                    report_best(f, self.best_x)
        return self.best_f, self.best_x