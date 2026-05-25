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

        # Number of independent runs
        if budget >= 30 * dim:
            k = 3
        elif budget >= 10 * dim:
            k = 2
        else:
            k = 1
        run_budget_min = max(5 * dim, 20)
        while k > 1 and budget // k < run_budget_min:
            k -= 1
        run_budget_list = [budget // k] * k
        for i in range(budget % k):
            run_budget_list[i] += 1

        # LHS points for each run
        lhs_points = np.zeros((k, dim))
        for d in range(dim):
            perm = rng.permutation(k)
            u = rng.rand(k)
            lhs_points[:, d] = (perm + u) / k
        lhs_points = lb + lhs_points * (ub - lb)

        best_f = np.inf
        best_x = None
        total_evals = 0

        for run_idx in range(k):
            run_budget = run_budget_list[run_idx]
            if run_budget <= 0:
                continue
            run_evals = 0

            # Initial point
            x_curr = lhs_points[run_idx].copy()
            f_curr = func(x_curr)
            total_evals += 1
            run_evals += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
                report_best(best_f, best_x)

            best_f_run = f_curr
            best_x_run = x_curr.copy()
            history = [best_x_run.copy()]
            initial_step = 0.1 * np.mean(ub - lb)
            step = initial_step

            # Coordinate directions
            coord_dirs = []
            for i in range(dim):
                e = np.zeros(dim)
                e[i] = 1.0
                coord_dirs.append(e)
                coord_dirs.append(-e)
            eig_dirs = []

            # Stagnation tracking
            no_improve_count = 0
            max_no_improve = 10 * dim

            while run_evals < run_budget and total_evals < budget:
                # Update eigen directions every 5*dim evals
                if len(history) >= dim and (run_evals % (5 * dim) == 0):
                    arr = np.array(history)
                    if arr.shape[0] >= dim:
                        # Weighted covariance: more weight to recent points
                        n = arr.shape[0]
                        weights = np.exp(np.linspace(-1, 0, n))
                        weights /= weights.sum()
                        mean = np.average(arr, axis=0, weights=weights)
                        diff = arr - mean
                        cov = np.dot((weights.reshape(-1,1) * diff).T, diff)
                        try:
                            eigvals, eigvecs = np.linalg.eigh(cov)
                            eig_dirs = []
                            for v in eigvecs.T:
                                eig_dirs.append(v)
                                eig_dirs.append(-v)
                        except np.linalg.LinAlgError:
                            pass

                dirs = coord_dirs + eig_dirs
                improved = False
                for d in dirs:
                    if run_evals >= run_budget or total_evals >= budget:
                        break
                    candidate = best_x_run + step * d
                    candidate = np.clip(candidate, lb, ub)
                    f_val = func(candidate)
                    total_evals += 1
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
                        no_improve_count = 0
                        break
                if improved:
                    step *= 1.2
                else:
                    step *= 0.5
                    no_improve_count += len(dirs)

                # Intra-run restart if step too small or stagnation
                if step < initial_step * 1e-8 or no_improve_count > max_no_improve:
                    step = initial_step
                    # Perturb best point within bounds
                    pert = rng.uniform(low=-0.2, high=0.2, size=dim) * (ub - lb)
                    new_x = np.clip(best_x_run + pert, lb, ub)
                    f_new = func(new_x)
                    total_evals += 1
                    run_evals += 1
                    if f_new < best_f:
                        best_f = f_new
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                    if f_new < best_f_run:
                        best_f_run = f_new
                        best_x_run = new_x.copy()
                        history.append(best_x_run.copy())
                        if len(history) > 3 * dim:
                            history.pop(0)
                    no_improve_count = 0

        return best_f, best_x