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
        budget = self.budget
        rng = self.rng

        # Determine number of independent runs
        if budget >= 30 * dim:
            k = 3
        elif budget >= 10 * dim:
            k = 2
        else:
            k = 1
        # Ensure each run has at least 5*dim evaluations
        run_budget_min = max(5 * dim, 20)
        while k > 1 and budget // k < run_budget_min:
            k -= 1
        run_budget_list = [budget // k] * k
        for i in range(budget % k):
            run_budget_list[i] += 1

        # Generate k LHS points (one per run)
        lhs_points = np.zeros((k, dim))
        for d in range(dim):
            perm = rng.permutation(k)
            u = rng.rand(k)
            lhs_points[:, d] = (perm + u) / k
        lhs_points = lb + lhs_points * (ub - lb)

        best_f = np.inf
        best_x = None
        evals_used = 0

        for run_idx in range(k):
            run_budget_this = run_budget_list[run_idx]
            if run_budget_this <= 0:
                continue
            run_evals = 0

            # Initial evaluation at LHS point
            x_curr = lhs_points[run_idx].copy()
            f_curr = func(x_curr)
            evals_used += 1
            run_evals += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
                report_best(best_f, best_x)

            best_f_run = f_curr
            best_x_run = x_curr.copy()
            history = [best_x_run.copy()]
            step = 0.1 * np.mean(ub - lb)

            # Coordinate search directions
            coord_dirs = []
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                coord_dirs.append(e)
                coord_dirs.append(-e)
            eig_dirs = []

            # Pattern search loop
            while run_evals < run_budget_this and evals_used < budget:
                # Update eigen directions periodically
                if len(history) >= dim and (run_evals % (10 * dim) == 0):
                    arr = np.array(history)
                    cov = np.cov(arr, rowvar=False)
                    if cov.shape[0] == dim:
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        eig_dirs = []
                        for v in eigvecs.T:
                            eig_dirs.append(v)
                            eig_dirs.append(-v)

                dirs = coord_dirs + eig_dirs
                improved = False
                for d in dirs:
                    if run_evals >= run_budget_this or evals_used >= budget:
                        break
                    candidate = best_x_run + step * d
                    candidate = np.clip(candidate, lb, ub)
                    f_val = func(candidate)
                    evals_used += 1
                    run_evals += 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                    if f_val < best_f_run:
                        best_f_run = f_val
                        best_x_run = candidate.copy()
                        improved = True
                        history.append(best_x_run.copy())
                        if len(history) > 3 * dim:
                            history.pop(0)
                        break
                if improved:
                    step *= 1.2
                else:
                    step *= 0.5

        return best_f, best_x