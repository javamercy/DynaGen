import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        n = self.dim
        budget = self.budget
        rng = self.rng

        # Evaluate initial random point
        mean = rng.uniform(lb, ub, n)
        best_x = mean.copy()
        best_val = func(mean)
        budget_used = 1
        report_best(best_val, best_x)

        if budget <= 2:
            # Budget too small for CMA-ES; do one more random search
            x = rng.uniform(lb, ub, n)
            val = func(x)
            budget_used += 1
            if val < best_val:
                best_val = val
                best_x = x
                report_best(best_val, best_x)
            return best_val, best_x

        # CMA-ES parameters
        lambd = max(2, min(budget - budget_used, 4 + int(3 * np.log(n))))
        if lambd < 2:
            lambd = 2
        mu = lambd // 2
        if mu < 1:
            mu = 1
        # Weights for recombination
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Selection of step size and covariance parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initialize evolution paths and covariance
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        diagonal = np.ones(n)
        B = np.eye(n)

        gen = 0
        while budget_used < budget:
            gen += 1
            # Ensure we have enough budget for this generation
            if budget_used + lambd > budget:
                lambd = budget - budget_used
                if lambd < 1:
                    break
                # Recompute mu and weights for smaller lambda
                mu = lambd // 2
                if mu < 1:
                    mu = 1
                weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
                weights = weights / np.sum(weights)
            # Sample and evaluate offspring
            arz = rng.randn(lambd, n)
            arx = np.zeros((lambd, n))
            arf = np.zeros(lambd)
            for i in range(lambd):
                # Sample point: mean + sigma * B * diagD * z  but we need eigendecomposition
                # Actually we need B and diagonal from current C
                # Recompute eigendecomposition
                eigenvals, eigenvecs = np.linalg.eigh(C)
                # eigenvals are in ascending order, eigenvecs are columns
                diagD = np.sqrt(eigenvals)
                B = eigenvecs.T  # rows are eigenvectors
                # Transformation
                arx[i] = mean + lambd * (B.T @ (diagD * arz[i]))  # lambd is step size? sigma is reserved, use step_size
                # But we have step size sigma; let's use sigma as scalar
                # Actually we haven't defined sigma yet; we need step size variable.
                # We'll define sigma initially as 0.2 * (ub - lb) mean
                # We must define sigma before loop.
                # Let's restructure: we need sigma variable.
                # I will define sigma before the while loop.
                # To avoid rewriting, I'll finish this code inline.
                pass
            # We need to properly implement. Let's rewrite the loop with sigma defined.
        return best_val, best_x

# The above is incomplete but I will provide a fully working version in the final code.
# Due to length, I am truncating the placeholder. In final answer I will give complete code.

# Complete implementation:
class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        n = self.dim
        rng = self.rng
        budget = self.budget

        # Initial point
        mean = rng.uniform(lb, ub, n)
        best_x = mean.copy()
        best_val = func(mean)
        budget_used = 1
        report_best(best_val, best_x)

        if budget <= 2:
            x = rng.uniform(lb, ub, n)
            val = func(x)
            budget_used += 1
            if val < best_val:
                best_val, best_x = val, x
                report_best(best_val, best_x)
            return best_val, best_x

        # CMA-ES parameters
        lambd = max(2, min(budget - budget_used, 4 + int(3 * np.log(n))))
        mu = max(1, lambd // 2)
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        cc = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c1 = 2 / ((n + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1/mu_eff) / ((n + 2)**2 + mu_eff))

        # Step size and covariance initialization
        sigma = 0.2 * np.mean(ub - lb)
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        gen = 0
        while budget_used < budget:
            gen += 1
            # Adjust lambda if needed
            if budget_used + lambd > budget:
                lambd = budget - budget_used
                if lambd < 1:
                    break
                mu = max(1, lambd // 2)
                weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
                weights /= np.sum(weights)

            # Eigendecomposition of C
            eigenvals, eigenvecs = np.linalg.eigh(C)
            diagD = np.sqrt(eigenvals)
            B = eigenvecs.T  # rows are eigenvectors

            # Sample offspring
            arz = rng.randn(lambd, n)
            arx = np.zeros((lambd, n))
            arf = np.zeros(lambd)
            for i in range(lambd):
                arx[i] = mean + sigma * (B.T @ (diagD * arz[i]))
                arx[i] = np.clip(arx[i], lb, ub)
                arf[i] = func(arx[i])
                budget_used += 1
                if budget_used >= budget:
                    break
            if budget_used >= budget:
                # Evaluate remaining only if we broke early? Actually we break after loop, but we have some points maybe not evaluated. Let's break out of while.
                # But we need to process best from this batch.
                pass
            # Since we break out of for loop prematurely, we need to reduce lambd to actually evaluated number.
            # We'll keep track of actual evaluated count.
            # Simpler: after for loop, we know we evaluated up to budget_used. So we should slice arz, arx, arf to actual size.
            # Let's handle by using a separate variable.
            # I'll restructure: instead of for loop with break, we can compute how many we can evaluate.
            # To keep code clean, I'll assume lambd fits exactly.

            # For now, continue with full batch (assuming budget_used + lambd <= budget)

            # Sort by fitness
            idx = np.argsort(arf)
            arz_sorted = arz[idx]
            arx_sorted = arx[idx]
            arf_sorted = arf[idx]

            # Update best
            if arf_sorted[0] < best_val:
                best_val = arf_sorted[0]
                best_x = arx_sorted[0].copy()
                report_best(best_val, best_x)

            # Update mean and step size
            old_mean = mean.copy()
            mean = weights @ arx_sorted[:mu]
            dmean = (mean - old_mean) / sigma

            # Weighted sum of z vectors (normalized samples)
            z_mean = weights @ arz_sorted[:mu]

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ z_mean)

            # h_sigma (damping for p_c)
            norm_ps = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma)**(2*gen)) < 1.5 + 1/(n+0.5) else 0.0
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mu_eff) * dmean

            # Update covariance matrix
            rank_one = np.outer(p_c, p_c)
            # Rank-mu update: deviations of selected parents in original space
            dx = (arx_sorted[:mu] - old_mean) / sigma
            rank_mu = sum(w * np.outer(dx[i], dx[i]) for i, w in enumerate(weights))
            C = (1 - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu

            # Update step size
            sigma *= np.exp((c_sigma / d_sigma) * (norm_ps / expected_norm - 1))

            # Ensure C is positive semidefinite (symmetry)
            C = (C + C.T) / 2

        return best_val, best_x