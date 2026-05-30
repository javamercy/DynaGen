import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        # CMA-ES parameters
        self.lam = int(4 + 3 * np.log(dim))
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights**2)
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2 / ((dim + 1.3)**2 + self.mu_eff)
        self.cmu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((dim + 2)**2 + self.mu_eff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1)/(dim + 1)) - 1) + self.cs
        # Initialize mean in bounds
        self.mean = np.random.uniform(0, 1, size=dim)  # will be scaled later
        self.sigma = 0.5 * (np.random.uniform(0,1) + 0.5)  # step size
        self.C = np.eye(dim)
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.best_x = None
        self.best_val = np.inf
        # For initial scaling, we need bounds, but they are not available here.
        # We'll store them when func is called in __call__
        self.lb = None
        self.ub = None

    def __call__(self, func):
        self.lb = func.bounds.lb
        self.ub = func.bounds.ub
        # Scale mean to bounds
        self.mean = self.lb + (self.ub - self.lb) * np.random.uniform(0, 1, size=self.dim)
        # Initial evaluations: evaluate mean
        val = func(self.mean)
        self.best_val = val
        self.best_x = self.mean.copy()
        report_best(self.best_val, self.best_x)
        evals = 1
        generation = 0
        while evals + self.lam <= self.budget:
            generation += 1
            # Sample lambda points
            eigvals, eigvecs = np.linalg.eigh(self.C)
            # Ensure positive definite
            eigvals = np.maximum(eigvals, 1e-20)
            D = np.diag(np.sqrt(eigvals))
            B = eigvecs
            # Generate z vectors
            Z = np.random.randn(self.dim, self.lam)
            X = self.mean[:, np.newaxis] + self.sigma * (B @ D @ Z)
            # Clip to bounds
            X = np.clip(X, self.lb[:, np.newaxis], self.ub[:, np.newaxis])
            # Evaluate each point
            F = np.zeros(self.lam)
            for i in range(self.lam):
                F[i] = func(X[:, i])
                evals += 1
            # Sort by fitness
            idx = np.argsort(F)
            F = F[idx]
            X = X[:, idx]
            # Check for improvement
            if F[0] < self.best_val:
                self.best_val = F[0]
                self.best_x = X[:, 0].copy()
                report_best(self.best_val, self.best_x)
            # Update mean
            old_mean = self.mean.copy()
            self.mean = X[:, :self.mu] @ self.weights
            # Update evolution paths
            zmean = np.linalg.solve(B @ D, self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * zmean
            hsig = (np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs)**(2*(evals/self.lam))) < (1.4 + 2/(self.dim+1))) * 1.0
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * (self.mean - old_mean) / self.sigma
            # Update covariance matrix
            artmp = (X[:, :self.mu] - old_mean[:, np.newaxis]) / self.sigma
            self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C) + self.cmu * (artmp @ np.diag(self.weights) @ artmp.T)
            # Enforce symmetry (numerical safety)
            self.C = (self.C + self.C.T) / 2
            # Update step size
            self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        return self.best_val, self.best_x