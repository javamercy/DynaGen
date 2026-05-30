import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.pop_size = max(4, min(5 * dim, budget // 10))
        self.mean_F = 0.5
        self.mean_CR = 0.9
        self.F_history = []
        self.CR_history = []
        self.C = np.eye(dim)
        self.c_learn = 1.0 / (dim + 5)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        pop = self.rng.uniform(lb, ub, size=(self.pop_size, self.dim))
        fit = np.full(self.pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(self.pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(val, x)
            if evals >= self.budget:
                return best_val, best_x

        while evals < self.budget:
            # Determine mean F and CR from history
            if len(self.F_history) > 0:
                meanF = np.mean(self.F_history)
                meanCR = np.mean(self.CR_history)
            else:
                meanF = self.mean_F
                meanCR = self.mean_CR

            # Compute square root of C (numerical stability)
            try:
                eigvals, eigvecs = np.linalg.eigh(self.C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(self.dim)

            # List to store successful steps
            successful_steps = []
            new_F = []
            new_CR = []

            for i in range(self.pop_size):
                if evals >= self.budget:
                    break
                # Sample F and CR
                F = np.clip(self.rng.normal(meanF, 0.1), 0.1, 0.9)
                CR = np.clip(self.rng.normal(meanCR, 0.1), 0.1, 1.0)

                # Choose indices for mutation (DE/rand/1)
                indices = [j for j in range(self.pop_size) if j != i]
                if len(indices) < 3:
                    break
                r1, r2, r3 = self.rng.choice(indices, size=3, replace=False)
                diff = pop[r2] - pop[r3]
                # Transform diff using covariance
                diff_adapt = C_sqrt @ diff
                mutant = pop[r1] + F * diff_adapt
                mutant = np.clip(mutant, lb, ub)

                # Crossover
                cross_points = self.rng.random(self.dim) < CR
                cross_points[self.rng.integers(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                val = func(trial)
                evals += 1
                if val < fit[i]:
                    pop[i] = trial
                    fit[i] = val
                    # Record successful step for covariance update
                    successful_steps.append(F * diff_adapt)
                    new_F.append(F)
                    new_CR.append(CR)
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(val, trial)

            # Update histories
            self.F_history.extend(new_F)
            self.CR_history.extend(new_CR)
            if len(self.F_history) > 50:
                self.F_history = self.F_history[-50:]
                self.CR_history = self.CR_history[-50:]

            # Update covariance matrix
            if len(successful_steps) > 0:
                steps = np.array(successful_steps)
                # Weight each step equally (can be improved)
                weights = np.ones(len(successful_steps)) / len(successful_steps)
                # Rank-mu update
                self.C = (1 - self.c_learn) * self.C + self.c_learn * (steps.T @ np.diag(weights) @ steps)
                # Symmetrize and enforce positive definiteness
                self.C = (self.C + self.C.T) / 2
                try:
                    eigvals, _ = np.linalg.eigh(self.C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                except np.linalg.LinAlgError:
                    pass

        return best_val, best_x