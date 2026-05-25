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
        range_len = ub - lb

        # Determine number of restarts
        if budget >= 40 * dim:
            n_restarts = 3
        elif budget >= 15 * dim:
            n_restarts = 2
        else:
            n_restarts = 1
        run_budget = budget // n_restarts
        # Adjust to ensure each run has at least 3*dim evaluations
        while n_restarts > 1 and run_budget < 3 * dim:
            n_restarts -= 1
            run_budget = budget // n_restarts
        # Distribute remaining evaluations
        remainder = budget - run_budget * n_restarts

        best_x = None
        best_f = np.inf
        total_evals = 0

        for run in range(n_restarts):
            run_budget_this = run_budget + (1 if run < remainder else 0)
            if run_budget_this <= 0:
                continue

            # LHS start point
            lhs_point = np.zeros(dim)
            for d in range(dim):
                u = rng.uniform(0, 1)
                lhs_point[d] = lb[d] + (rng.permutation(1)[0] + u) / 1 * (ub[d] - lb[d])
            # Actually we want a single point, so just uniform
            # Simpler: use random point in [lb, ub] but with LHS coverage across runs? Not needed.
            # We'll just sample uniformly per run
            x_curr = lb + rng.rand(dim) * (ub - lb)
            f_curr = func(x_curr)
            total_evals += 1
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
                report_best(best_f, best_x)

            best_f_run = f_curr
            best_x_run = x_curr.copy()
            step = 0.1 * np.mean(range_len)
            C = np.eye(dim)
            lr = 0.1
            success_history = [best_x_run.copy()]

            # Pre-compute coordinate directions
            coord_dirs = []
            for i in range(dim):
                d = np.zeros(dim)
                d[i] = 1.0
                coord_dirs.append(d)
                coord_dirs.append(-d)

            while total_evals < budget and run_budget_this > 0:
                # Generate random directions from covariance
                num_random = min(2 * dim, min(budget - total_evals, run_budget_this))
                if num_random > 0:
                    try:
                        C_reg = C + 1e-15 * np.eye(dim)
                        samples = rng.multivariate_normal(np.zeros(dim), C_reg, size=num_random)
                        norms = np.linalg.norm(samples, axis=1, keepdims=True) + 1e-15
                        random_dirs = samples / norms
                    except:
                        random_dirs = np.zeros((0, dim))
                else:
                    random_dirs = np.zeros((0, dim))

                # Combine directions
                directions = coord_dirs + [random_dirs[i] for i in range(len(random_dirs))]
                # Shuffle directions? Not strictly necessary, but helps diversity
                rng.shuffle(directions)

                improved = False
                for d in directions:
                    if total_evals >= budget or run_budget_this <= 0:
                        break
                    candidate = best_x_run + step * d
                    candidate = np.clip(candidate, lb, ub)
                    f_val = func(candidate)
                    total_evals += 1
                    run_budget_this -= 1
                    if f_val < best_f:
                        best_f = f_val
                        best_x = candidate.copy()
                        report_best(best_f, best_x)
                    if f_val < best_f_run:
                        best_f_run = f_val
                        best_x_run = candidate.copy()
                        improved = True
                        # Update covariance with direction
                        d_norm = np.linalg.norm(d) + 1e-15
                        d_unit = d / d_norm
                        C = (1 - lr) * C + lr * np.outer(d_unit, d_unit)
                        C += 1e-15 * np.eye(dim)
                        success_history.append(best_x_run.copy())
                        if len(success_history) > 2 * dim:
                            success_history.pop(0)
                        break

                if improved:
                    step *= 1.2
                else:
                    step *= 0.5
                    if step < 1e-12 * np.mean(range_len) and total_evals < budget:
                        # Restart within this run: LHS sample point
                        x_new = lb + rng.rand(dim) * (ub - lb)
                        f_new = func(x_new)
                        total_evals += 1
                        run_budget_this -= 1
                        if f_new < best_f:
                            best_f = f_new
                            best_x = x_new.copy()
                            report_best(best_f, best_x)
                        if f_new < best_f_run:
                            best_f_run = f_new
                            best_x_run = x_new.copy()
                        step = 0.1 * np.mean(range_len)
                        C = np.eye(dim)
                        success_history = [best_x_run.copy()]

        return best_f, best_x