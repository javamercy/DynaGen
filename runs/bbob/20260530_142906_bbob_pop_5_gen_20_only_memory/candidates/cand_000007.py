import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.rng = np.random.RandomState(seed)
        # CMA-ES parameters with larger population for exploration
        self.lam = max(2, 2 * (4 + int(3 * np.log(dim))))  # double default
        self.mu = self.lam // 2
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= self.weights.sum()
        self.mueff = 1.0 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.dim) / (self.dim + 4 + 2 * self.mueff / self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        self.ccov = (2 / (self.dim + 1.4) ** 2 + (1 - 1 / self.mueff) * min(1, (2 * self.mueff - 1) / ((self.dim + 2) ** 2 + self.mueff)))
        self.sigma = 0.6  # larger initial step size
        self.mean = None
        self.pc = np.zeros(dim)
        self.ps = np.zeros(dim)
        self.C = np.eye(dim)
        self.best_x = None
        self.best_f = np.inf
        self.num_evals = 0
        self.initial_mean = None
        self.no_improve_count = 0
        self.restart_threshold = max(0.15 * budget, 10)  # evaluations without improvement before restart

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        bounds_span = ub - lb
        # Initialize mean
        self.mean = lb + self.rng.rand(self.dim) * bounds_span
        self.initial_mean = self.mean.copy()
        # Evaluate initial point
        f = func(self.mean)
        self.num_evals += 1
        self.best_x = self.mean.copy()
        self.best_f = f
        report_best(self.best_f, self.best_x)
        # Precompute eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(self.C)
        B = eigvecs
        D = np.sqrt(np.maximum(eigvals, 1e-12))
        # Main loop
        while self.num_evals + self.lam <= self.budget:
            # Check for restart condition
            if self.sigma < 1e-4 or self.no_improve_count >= self.restart_threshold:
                # Restart: reinitialize mean, covariance, step-size
                self.mean = lb + self.rng.rand(self.dim) * bounds_span
                self.pc = np.zeros(self.dim)
                self.ps = np.zeros(self.dim)
                self.C = np.eye(self.dim)
                self.sigma = 0.6
                self.no_improve_count = 0
                # Recompute eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(self.C)
                B = eigvecs
                D = np.sqrt(np.maximum(eigvals, 1e-12))
            # Sample lambda candidates
            z = self.rng.randn(self.lam, self.dim)
            y = z @ (B * D).T
            x = self.mean + self.sigma * y
            x = np.clip(x, lb, ub)
            # Evaluate all
            f_vals = np.array([func(x[i]) for i in range(self.lam)])
            self.num_evals += self.lam
            # Sort
            idx = np.argsort(f_vals)
            f_vals_sorted = f_vals[idx]
            x_sorted = x[idx]
            y_sorted = y[idx]
            # Update mean
            old_mean = self.mean.copy()
            self.mean = old_mean + self.sigma * (self.weights @ y_sorted[:self.mu])
            # Update evolution paths
            invsqrtC = (B * (1.0 / D).T) @ B.T
            y_mean = (self.mean - old_mean) / self.sigma
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC @ y_mean
            self.pc = (1 - self.cc) * self.pc + np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean
            # Update covariance matrix
            self.C = (1 - self.ccov) * self.C + self.ccov * np.outer(self.pc, self.pc)
            self.C = (self.C + self.C.T) / 2
            # Update step size
            ps_norm = np.linalg.norm(self.ps)
            chi_n = np.sqrt(self.dim) * (1 - 1.0 / (4 * self.dim) + 1.0 / (21 * self.dim ** 2))
            self.sigma = self.sigma * np.exp((self.cs / self.damps) * (ps_norm / chi_n - 1))
            self.sigma = max(1e-8, min(self.sigma, 10))
            # Recompute eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                B = np.eye(self.dim)
                D = np.ones(self.dim)
            # Update best and no_improve_count
            improved = False
            for fval, xval in zip(f_vals_sorted, x_sorted):
                if fval < self.best_f:
                    self.best_f = fval
                    self.best_x = xval.copy()
                    report_best(self.best_f, self.best_x)
                    improved = True
            if improved:
                self.no_improve_count = 0
            else:
                self.no_improve_count += self.lam
        # Use remaining budget for random evaluations
        while self.num_evals < self.budget:
            x = lb + self.rng.rand(self.dim) * bounds_span
            f = func(x)
            self.num_evals += 1
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()
                report_best(self.best_f, self.best_x)
        return self.best_f, self.best_x