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
        mean_range = np.mean(range_len)

        # Restart schedule
        if budget >= 40 * dim:
            n_restarts = 3
        elif budget >= 15 * dim:
            n_restarts = 2
        else:
            n_restarts = 1
        run_budget = budget // n_restarts
        while n_restarts > 1 and run_budget < 3 * dim:
            n_restarts -= 1
            run_budget = budget // n_restarts
        remainder = budget - run_budget * n_restarts

        best_x = None
        best_f = np.inf
        total_evals = 0

        for run in range(n_restarts):
            run_budget_this = run_budget + (1 if run < remainder else 0)
            if run_budget_this <= 0:
                continue

            # Initialize simplex with random points
            n_vertices = dim + 1
            simplex = np.zeros((n_vertices, dim))
            for i in range(n_vertices):
                simplex[i] = rng.uniform(lb, ub, size=dim)
            f_simplex = np.full(n_vertices, np.inf)

            # Evaluate initial simplex
            for i in range(n_vertices):
                if total_evals >= budget:
                    break
                x = simplex[i]
                f = func(x)
                total_evals += 1
                run_budget_this -= 1
                f_simplex[i] = f
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                    report_best(best_f, best_x)

            if total_evals >= budget:
                continue

            # Sort initial simplex
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            max_no_improve = max(5, dim)
            no_improve_count = 0

            # Covariance and step for pattern search
            C = np.eye(dim)
            lr = 0.1
            step = 0.1 * mean_range

            # Precompute coordinate directions
            coord_dirs = []
            for i in range(dim):
                d = np.zeros(dim)
                d[i] = 1.0
                coord_dirs.append(d)
                coord_dirs.append(-d)

            while run_budget_this > 0 and total_evals < budget:
                old_best_f = best_f

                # Nelder-Mead step
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                total_evals += 1
                run_budget_this -= 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                    report_best(best_f, best_x)
                if fr < f_simplex[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    total_evals += 1
                    run_budget_this -= 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                        report_best(best_f, best_x)
                    if fe < fr:
                        simplex[-1] = xe
                        f_simplex[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_simplex[-1] = fr
                    no_improve_count = 0
                elif fr < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                    no_improve_count = 0
                else:
                    # Contraction
                    if fr < f_simplex[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    total_evals += 1
                    run_budget_this -= 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                        report_best(best_f, best_x)
                    if fc < min(fr, f_simplex[-1]):
                        simplex[-1] = xc
                        f_simplex[-1] = fc
                        no_improve_count = 0
                    else:
                        # Shrink
                        for i in range(1, n_vertices):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                        for i in range(1, n_vertices):
                            fi = func(simplex[i])
                            total_evals += 1
                            run_budget_this -= 1
                            f_simplex[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                                report_best(best_f, best_x)
                        no_improve_count += 1

                # Update covariance if improved
                if best_f < old_best_f:
                    direction = best_x - (simplex[0] if old_best_f == f_simplex[0] else best_x.copy())
                    direction = simplex[0] if old_best_f == f_simplex[0] else best_x - (old_best_x if 'old_best_x' in locals() else centroid)
                    # Compute direction from previous best to new best
                    # Store previous best_x before update

                # If no improvement for too long, do pattern search
                if no_improve_count >= max_no_improve and run_budget_this > 0:
                    # Pattern search from best_x
                    improved = False
                    for attempt in range(min(2 * dim, run_budget_this)):
                        if total_evals >= budget:
                            break
                        # Generate random directions from C
                        num_random = min(2 * dim, run_budget_this)
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
                        directions = coord_dirs + [random_dirs[i] for i in range(len(random_dirs))]
                        rng.shuffle(directions)
                        found = False
                        for d in directions[:max(1, len(directions)//2)]:
                            if total_evals >= budget or run_budget_this <= 0:
                                break
                            candidate = best_x + step * d
                            candidate = np.clip(candidate, lb, ub)
                            f_val = func(candidate)
                            total_evals += 1
                            run_budget_this -= 1
                            if f_val < best_f:
                                best_f = f_val
                                best_x = candidate.copy()
                                report_best(best_f, best_x)
                                # Update covariance
                                d_norm = np.linalg.norm(d) + 1e-15
                                d_unit = d / d_norm
                                C = (1 - lr) * C + lr * np.outer(d_unit, d_unit)
                                C += 1e-15 * np.eye(dim)
                                step = min(step * 1.2, 0.5 * mean_range)
                                improved = True
                                found = True
                                break
                        if found:
                            break
                        else:
                            step *= 0.5
                    if improved:
                        no_improve_count = 0
                        # Replace worst vertex with new best
                        f_simplex[-1] = best_f
                        simplex[-1] = best_x.copy()
                        # Re-sort
                        order = np.argsort(f_simplex)
                        simplex = simplex[order]
                        f_simplex = f_simplex[order]
                    else:
                        # Random restart
                        # Keep best vertex
                        simplex[0] = best_x
                        f_simplex[0] = best_f
                        for i in range(1, n_vertices):
                            simplex[i] = rng.uniform(lb, ub, size=dim)
                            fi = func(simplex[i])
                            total_evals += 1
                            run_budget_this -= 1
                            f_simplex[i] = fi
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                                report_best(best_f, best_x)
                        # Reset covariance and step
                        C = np.eye(dim)
                        step = 0.1 * mean_range
                        no_improve_count = 0

                # Re-sort simplex after each iteration
                order = np.argsort(f_simplex)
                simplex = simplex[order]
                f_simplex = f_simplex[order]

        return best_f, best_x