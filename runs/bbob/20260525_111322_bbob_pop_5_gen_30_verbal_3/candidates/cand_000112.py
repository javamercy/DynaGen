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
        dim = self.dim
        n = dim + 1
        
        # Latin Hypercube Sampling for initial simplex
        points = np.zeros((n, dim))
        for i in range(dim):
            perm = self.rng.permutation(n)
            u = self.rng.rand(n)
            points[:, i] = lb[i] + (perm + u) / n * (ub[i] - lb[i])
        
        fitness = np.full(n, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0
        
        def evaluate(x):
            nonlocal evals, best_f, best_x
            if evals >= self.budget:
                return None
            x = np.clip(x, lb, ub)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)
            return f
        
        for i in range(n):
            f = evaluate(points[i])
            if f is None:
                return best_f, best_x
            fitness[i] = f
        
        # Sort
        order = np.argsort(fitness)
        points = points[order]
        fitness = fitness[order]
        
        # Nelder-Mead parameters
        rho = 1.0
        chi = 2.0
        gamma = 0.5
        sigma_shrink = 0.5
        
        stagnation = 0
        patience = max(5, dim * 3)
        restart_scale = 0.5  # initial scaling for restart noise
        
        while evals < self.budget:
            # Check for stagnation
            if stagnation >= patience:
                # Restart: anisotropic sampling
                # Compute covariance of current simplex (n points)
                pts = np.array(points)
                cov = np.cov(pts.T) + 1e-8 * np.eye(dim)
                # Generate new points around best
                new_pts = []
                for _ in range(n - 1):
                    sample = self.rng.multivariate_normal(np.zeros(dim), restart_scale * cov)
                    new_x = points[0] + sample
                    new_pts.append(new_x)
                # Evaluate and insert
                for new_x in new_pts:
                    if evals >= self.budget:
                        break
                    f = evaluate(new_x)
                    if f is None:
                        break
                    # Replace worst if better
                    if f < fitness[-1]:
                        points[-1] = new_x
                        fitness[-1] = f
                        # Re-sort
                        order = np.argsort(fitness)
                        points = points[order]
                        fitness = fitness[order]
                stagnation = 0
                # Adapt scale: if restart improved best, keep scale; else reduce
                # We can track best before and after, but for simplicity just reduce a bit
                restart_scale *= 0.9
                # Continue to next iteration
                continue
            
            # One Nelder-Mead iteration
            # Reflect worst
            worst_idx = n - 1
            centroid = np.mean(points[:worst_idx], axis=0)
            xr = centroid + rho * (centroid - points[worst_idx])
            fr = evaluate(xr)
            if fr is None:
                break
            if fr < fitness[0]:
                # Expand
                xe = centroid + chi * (xr - centroid)
                fe = evaluate(xe)
                if fe is None:
                    break
                if fe < fr:
                    points[worst_idx] = xe
                    fitness[worst_idx] = fe
                else:
                    points[worst_idx] = xr
                    fitness[worst_idx] = fr
                stagnation = 0
            elif fr < fitness[-2]:
                # Accept reflection
                points[worst_idx] = xr
                fitness[worst_idx] = fr
                stagnation = 0
            else:
                # Contract or shrink
                if fr < fitness[-1]:
                    # Outside contraction
                    xc = centroid + gamma * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - gamma * (centroid - points[worst_idx])
                fc = evaluate(xc)
                if fc is None:
                    break
                if fc < min(fitness[-1], fr):
                    points[worst_idx] = xc
                    fitness[worst_idx] = fc
                    stagnation = 0
                else:
                    # Shrink
                    best_point = points[0].copy()
                    for i in range(1, n):
                        points[i] = best_point + sigma_shrink * (points[i] - best_point)
                        f = evaluate(points[i])
                        if f is None:
                            break
                        fitness[i] = f
                    stagnation = 0
            # Re-sort
            order = np.argsort(fitness)
            points = points[order]
            fitness = fitness[order]
            # Increment stagnation if best not improved (we can check if fitness[0] unchanged)
            # Simplified: increment each iteration, but reset when improved. Actually we reset on improvement above.
            stagnation += 1
        
        return best_f, best_x