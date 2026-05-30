import numpy as np

class Optimizer:
    def __init__(self, budget, dim, seed):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        n_init = max(2, budget // 5)
        points = np.empty((n_init, dim))
        for i in range(dim):
            points[:, i] = rng.uniform(lb[i], ub[i], size=n_init)
        for i in range(dim):
            rng.shuffle(points[:, i])
        best_x = points[0].copy()
        best_f = func(best_x)
        evals = 1
        report_best(best_f, best_x)
        for i in range(1, n_init):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        simplex = np.zeros((dim+1, dim))
        simplex[0] = best_x.copy()
        for i in range(dim):
            delta = 0.05 * (ub[i] - lb[i])
            if delta == 0:
                delta = 0.001
            x = best_x.copy()
            x[i] += delta
            x[i] = np.clip(x[i], lb[i], ub[i])
            simplex[i+1] = x
        simplex_f = np.full(dim+1, np.inf)
        for i in range(dim+1):
            if evals >= budget:
                break
            f = func(simplex[i])
            evals += 1
            simplex_f[i] = f
            if f < best_f:
                best_f = f
                best_x = simplex[i].copy()
                report_best(best_f, best_x)

        while evals < budget:
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            centroid = np.mean(simplex[:-1], axis=0)

            if evals < budget:
                xr = centroid + 1.0 * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1
                if fr < simplex_f[0]:
                    if evals < budget:
                        xe = centroid + 2.0 * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            simplex_f[-1] = fe
                        else:
                            simplex[-1] = xr
                            simplex_f[-1] = fr
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    if evals < budget:
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        evals += 1
                        if fc < simplex_f[-1]:
                            simplex[-1] = xc
                            simplex_f[-1] = fc
                        else:
                            for i in range(1, dim+1):
                                if evals >= budget:
                                    break
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                f = func(simplex[i])
                                evals += 1
                                simplex_f[i] = f
                    else:
                        # no budget, just continue
                        pass

            idx_min = np.argmin(simplex_f)
            if simplex_f[idx_min] < best_f:
                best_f = simplex_f[idx_min]
                best_x = simplex[idx_min].copy()
                report_best(best_f, best_x)

        return best_f, best_x