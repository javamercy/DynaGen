import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.stall_limit = max(5 * dim, budget // 20)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        n = dim + 1
        if self.budget < n:
            x0 = self.rng.uniform(lb, ub)
            val0 = func(x0)
            report_best(val0, x0)
            return val0, x0
        simplex = self.rng.uniform(lb, ub, size=(n, dim))
        values = np.full(n, np.inf)
        evaluations = 0
        best_x = None
        best_val = np.inf
        for i in range(n):
            if evaluations >= self.budget:
                break
            x = simplex[i]
            val = func(x)
            evaluations += 1
            values[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
        last_improvement_eval = evaluations
        while evaluations < self.budget:
            idx_sorted = np.argsort(values)
            simplex = simplex[idx_sorted]
            values = values[idx_sorted]
            best_idx = 0
            worst_idx = n - 1
            centroid = np.mean(simplex[:-1], axis=0)
            # Reflection
            xr = centroid + (centroid - simplex[worst_idx])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evaluations += 1
            if evaluations >= self.budget:
                break
            if fr < values[best_idx]:
                xe = centroid + 2.0 * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evaluations += 1
                if evaluations >= self.budget:
                    break
                if fe < fr:
                    simplex[worst_idx] = xe
                    values[worst_idx] = fe
                    if fe < best_val:
                        best_val = fe
                        best_x = xe.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
                else:
                    simplex[worst_idx] = xr
                    values[worst_idx] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
            elif fr < values[worst_idx - 1]:
                simplex[worst_idx] = xr
                values[worst_idx] = fr
                if fr < best_val:
                    best_val = fr
                    best_x = xr.copy()
                    report_best(best_val, best_x)
                    last_improvement_eval = evaluations
            else:
                if fr < values[worst_idx]:
                    xc = centroid + 0.5 * (xr - centroid)
                else:
                    xc = centroid - 0.5 * (centroid - simplex[worst_idx])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evaluations += 1
                if evaluations >= self.budget:
                    break
                if fc < values[worst_idx]:
                    simplex[worst_idx] = xc
                    values[worst_idx] = fc
                    if fc < best_val:
                        best_val = fc
                        best_x = xc.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
                else:
                    for i in range(1, n):
                        simplex[i] = simplex[best_idx] + 0.5 * (simplex[i] - simplex[best_idx])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        val_i = func(simplex[i])
                        evaluations += 1
                        if evaluations >= self.budget:
                            break
                        values[i] = val_i
                        if val_i < best_val:
                            best_val = val_i
                            best_x = simplex[i].copy()
                            report_best(best_val, best_x)
                            last_improvement_eval = evaluations
                    if evaluations >= self.budget:
                        break
            if evaluations - last_improvement_eval > self.stall_limit:
                new_simplex = self.rng.uniform(lb, ub, size=(n, dim))
                new_simplex[0] = best_x
                for i in range(1, n):
                    if evaluations >= self.budget:
                        break
                    x_new = new_simplex[i]
                    val_new = func(x_new)
                    evaluations += 1
                    new_simplex[i] = x_new.copy()
                    values[i] = val_new
                    if val_new < best_val:
                        best_val = val_new
                        best_x = x_new.copy()
                        report_best(best_val, best_x)
                        last_improvement_eval = evaluations
                simplex = new_simplex
                values[0] = best_val
                last_improvement_eval = evaluations
        return best_val, best_x