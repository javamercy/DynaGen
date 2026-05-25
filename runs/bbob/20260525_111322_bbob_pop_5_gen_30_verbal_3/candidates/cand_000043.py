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
        bud = self.budget
        rng = self.rng

        # Latin hypercube sampling
        n_init = min(bud, max(2 * dim, 5))
        best_x = None
        best_f = np.inf
        evals = 0

        # Generate LHS points
        lhs_points = np.empty((n_init, dim))
        for i in range(dim):
            seq = np.arange(n_init) / n_init + rng.rand(n_init) / n_init
            rng.shuffle(seq)
            lhs_points[:, i] = seq
        lhs_points = lb + lhs_points * (ub - lb)

        for i in range(n_init):
            if evals >= bud:
                break
            x = lhs_points[i]
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if best_x is None:
            x = lb + rng.rand(dim) * (ub - lb)
            f = func(x)
            evals += 1
            best_f = f
            best_x = x.copy()
            report_best(best_f, best_x)

        # Pattern search
        step = 0.1 * np.mean(ub - lb)
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        restart_done = False
        while evals < bud:
            improved = False
            for d in directions:
                if evals >= bud:
                    break
                candidate = best_x + step * d
                candidate = np.clip(candidate, lb, ub)
                f = func(candidate)
                evals += 1
                if f < best_f:
                    best_f = f
                    best_x = candidate.copy()
                    improved = True
                    report_best(best_f, best_x)
                    break
            if improved:
                step *= 1.2
            else:
                step *= 0.5
                if step < 1e-12 and not restart_done and evals < bud - 2*dim:
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_f = func(new_x)
                    evals += 1
                    if new_f < best_f:
                        best_f = new_f
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                    step = 0.1 * np.mean(ub - lb)
                    restart_done = True
                if step < 1e-15:
                    break

        return best_f, best_x