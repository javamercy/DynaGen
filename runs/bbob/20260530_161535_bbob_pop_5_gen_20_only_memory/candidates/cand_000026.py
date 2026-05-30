import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
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
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff))
        self.pc = np.zeros(dim)
        self.B = np.eye(dim)
        self.D = np.ones(dim)
        self.C = np.eye(dim)
        self.invsqrtC = np.eye(dim)
        self.eigen_eval = 0
        self.count = 0
        self.best_x = None
        self.best_f = np.inf
        self.xmean = None
        self.sigma = None
        self.f_mean = None

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        domain_range = self.ub - self.lb
        self.sigma = 0.3 * np.mean(domain_range)
        self.xmean = self.rng.uniform(self.lb, self.ub, size=self.dim)
        # evaluate initial mean
        if self.count < self.budget:
            f = func(self.xmean)
            self.count += 1
            self.f_mean = f
            if f < self.best_f:
                self.best_f = f
                self.best_x = self.xmean.copy()
                report_best(f, self.best_x)
        generation = 0
        while True:
            if generation == 0:
                f_parent = self.f_mean
            else:
                # evaluate current mean at start of generation (except first)
                if self.count >= self.budget:
                    break
                f = func(self.xmean)
                self.count += 1
                self.f_mean = f
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = self.xmean.copy()
                    report_best(f, self.best_x)
                f_parent = f
            # check if enough budget for lam offspring
            if self.count + self.lam > self.budget:
                break
            # sample lambda offspring
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
            # selection and recombination
            arf = np.array(arf)
            arx = np.array(arx)
            idx = np.argsort(arf)
            xold = self.xmean.copy()
            self.xmean = np.sum(self.weights[:, None] * arx[idx[:self.mu]], axis=0)
            # update evolution path for covariance
            dmean = self.xmean - xold
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (dmean / self.sigma)
            # update covariance
            self.C *= (1 - self.c1 - self.cmu)
            self.C += self.c1 * np.outer(self.pc, self.pc)
            for i in range(self.mu):
                diff = (arx[idx[i]] - xold) / self.sigma
                self.C += self.cmu * self.weights[i] * np.outer(diff, diff)
            # step-size adaptation using 1/5th rule
            success_count = np.sum(arf < f_parent)
            success_rate = success_count / self.lam
            if success_rate > 0.2:
                self.sigma *= np.exp(1 / self.dim)
            else:
                self.sigma *= np.exp(-1 / self.dim)
            self.sigma = max(self.sigma, 1e-10)
            # eigen decomposition
            if self.count - self.eigen_eval > self.dim:
                self.eigen_eval = self.count
                self.C = np.triu(self.C) + np.triu(self.C, 1).T
                self.D, self.B = np.linalg.eigh(self.C)
                self.D = np.abs(self.D)
                self.D = np.maximum(self.D, 1e-30)
                self.D = np.sqrt(self.D)
                self.invsqrtC = self.B @ np.diag(1/self.D) @ self.B.T
            generation += 1
        # local refinement via pattern search on best point
        step = 0.1 * np.mean(domain_range) * (1 - self.count / self.budget) ** 2
        while self.count < self.budget:
            improved = False
            for d in range(self.dim):
                if self.count >= self.budget:
                    break
                # try positive direction
                trial = self.best_x.copy()
                trial[d] = np.clip(trial[d] + step, self.lb[d], self.ub[d])
                f_trial = func(trial)
                self.count += 1
                if f_trial < self.best_f:
                    self.best_f = f_trial
                    self.best_x = trial
                    report_best(f_trial, trial)
                    step *= 1.1
                    improved = True
                    break
                # try negative direction
                trial = self.best_x.copy()
                trial[d] = np.clip(trial[d] - step, self.lb[d], self.ub[d])
                f_trial = func(trial)
                self.count += 1
                if f_trial < self.best_f:
                    self.best_f = f_trial
                    self.best_x = trial
                    report_best(f_trial, trial)
                    step *= 1.1
                    improved = True
                    break
            if not improved:
                step *= 0.9
            # ensure step does not get too small
            step = max(step, 1e-10)
        return self.best_f, self.best_x