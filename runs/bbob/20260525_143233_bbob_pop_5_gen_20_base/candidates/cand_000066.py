import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.restart_threshold = max(10, 2 * dim)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        n_simplex = dim + 1

        # Initialize simplex
        simplex = rng.uniform(lb, ub, (n_simplex, dim))
        fitness = np.full(n_simplex, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(n_simplex):
            if evals >= budget:
                break
            x = simplex[i].copy()
            val = func(x)
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # Ensure best is saved
        if best_x is None:
            return best_val, best_x

        no_improve = 0
        while evals < budget:
            # Sort simplex by fitness
            order = np.argsort(fitness)
            simplex = simplex[order]
            fitness = fitness[order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            alpha = 1.0
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, lb, ub)
            if evals < budget:
                fr = func(xr)
                evals += 1
                if fr < fitness[0]:  # better than best
                    # Expansion
                    gamma = 2.0
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals < budget:
                        fe = func(xe)
                        evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fitness[-1] = fe
                            if fe < best_val:
                                best_val = fe
                                best_x = xe.copy()
                                report_best(best_val, best_x)
                        else:
                            simplex[-1] = xr
                            fitness[-1] = fr
                            if fr < best_val:
                                best_val = fr
                                best_x = xr.copy()
                                report_best(best_val, best_x)
                elif fr < fitness[-2]:  # better than second worst
                    simplex[-1] = xr
                    fitness[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                else:
                    # Contraction (outside)
                    rho = 0.5
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, lb, ub)
                    if evals < budget:
                        fc = func(xc)
                        evals += 1
                        if fc < fitness[-1]:
                            simplex[-1] = xc
                            fitness[-1] = fc
                            if fc < best_val:
                                best_val = fc
                                best_x = xc.copy()
                                report_best(best_val, best_x)
                        else:
                            # Restart
                            evals, simplex, fitness, no_improve, best_val, best_x = self._restart(simplex, fitness, best_val, best_x, lb, ub, func, evals, budget)
                            continue  # continue outer loop after restart

            # Local perturbation around best
            if evals < budget:
                for _ in range(min(2, budget - evals)):
                    sigma = 0.05 * (ub - lb)
                    x = best_x + sigma * rng.randn(dim)
                    x = np.clip(x, lb, ub)
                    val = func(x)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)
                        # Insert into simplex replacing worst point
                        simplex[-1] = x
                        fitness[-1] = val

            no_improve += 1
            if no_improve >= self.restart_threshold:
                evals, simplex, fitness, no_improve, best_val, best_x = self._restart(simplex, fitness, best_val, best_x, lb, ub, func, evals, budget)

        return best_val, best_x

    def _restart(self, simplex, fitness, best_val, best_x, lb, ub, func, evals, budget):
        dim = self.dim
        rng = self.rng
        n_simplex = dim + 1
        new_simplex = np.zeros((n_simplex, dim))
        new_simplex[0] = best_x.copy()
        for i in range(1, n_simplex):
            sigma = 0.2 * (ub - lb)
            x = best_x + sigma * rng.randn(dim)
            new_simplex[i] = np.clip(x, lb, ub)
        new_fitness = np.full(n_simplex, np.inf)
        new_fitness[0] = best_val
        for i in range(1, n_simplex):
            if evals >= budget:
                break
            x = new_simplex[i].copy()
            val = func(x)
            evals += 1
            new_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        return evals, new_simplex, new_fitness, 0, best_val, best_x