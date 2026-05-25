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

        # LHS initialization for simplex
        n_vertices = dim + 1
        points = np.zeros((n_vertices, dim))
        for j in range(dim):
            intervals = np.linspace(lb[j], ub[j], n_vertices + 1)
            samples = rng.uniform(intervals[:-1], intervals[1:], size=n_vertices)
            rng.shuffle(samples)
            points[:, j] = samples

        # Evaluate initial simplex
        f_vals = np.full(n_vertices, np.inf)
        best_x = points[0].copy()
        best_f = np.inf
        evals = 0
        for i in range(n_vertices):
            if evals >= budget:
                break
            x = points[i]
            f = func(x)
            evals += 1
            f_vals[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # If initial evaluation exhausted budget, return
        if evals >= budget:
            return best_f, best_x

        # Parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0

        # DE mutation parameters
        F = 0.5  # initial mutation factor
        F_low = 0.1
        F_high = 0.9
        success_count = 0
        fail_count = 0
        success_rate_window = 10
        mutation_interval = max(5, dim)  # apply mutation every few iterations
        iteration = 0

        while evals < budget:
            iteration += 1
            # Order vertices by fitness
            order = np.argsort(f_vals)
            points = points[order]
            f_vals = f_vals[order]

            # Compute centroid of all but worst
            centroid = np.mean(points[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - points[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if evals >= budget:
                break
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)

            if f_vals[0] <= fr < f_vals[-2]:
                # Accept reflection
                points[-1] = xr
                f_vals[-1] = fr
                no_improve_count = 0
            elif fr < f_vals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, lb, ub)
                fe = func(xe)
                evals += 1
                if evals >= budget:
                    break
                if fe < best_f:
                    best_f = fe
                    best_x = xe.copy()
                    report_best(best_f, best_x)
                if fe < fr:
                    points[-1] = xe
                    f_vals[-1] = fe
                else:
                    points[-1] = xr
                    f_vals[-1] = fr
                no_improve_count = 0
            else:
                # Contraction
                if fr < f_vals[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - points[-1])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if evals >= budget:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    report_best(best_f, best_x)
                if fc < min(fr, f_vals[-1]):
                    points[-1] = xc
                    f_vals[-1] = fc
                    no_improve_count = 0
                else:
                    # Shrink
                    for i in range(1, n_vertices):
                        points[i] = points[0] + sigma * (points[i] - points[0])
                        points[i] = np.clip(points[i], lb, ub)
                        fi = func(points[i])
                        evals += 1
                        if evals >= budget:
                            break
                        f_vals[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = points[i].copy()
                            report_best(best_f, best_x)
                    no_improve_count += 1

            # Stagnation check and restart
            if no_improve_count >= max_no_improve and evals < budget:
                # Keep best, reinitialize rest randomly with DE mutation
                points[0] = best_x
                f_vals[0] = best_f
                for i in range(1, n_vertices):
                    # DE/rand/1 mutation
                    idxs = [j for j in range(n_vertices) if j != i]
                    a, b, c = rng.choice(idxs, 3, replace=False)
                    trial = points[a] + F * (points[b] - points[c])
                    trial = np.clip(trial, lb, ub)
                    ft = func(trial)
                    evals += 1
                    if evals >= budget:
                        break
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                    if ft < f_vals[i]:
                        points[i] = trial
                        f_vals[i] = ft
                        success_count += 1
                    else:
                        points[i] = rng.uniform(lb, ub, size=dim)
                        fi = func(points[i])
                        evals += 1
                        if evals >= budget:
                            break
                        f_vals[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = points[i].copy()
                            report_best(best_f, best_x)
                        fail_count += 1
                # Adapt F based on success rate
                total = success_count + fail_count
                if total > success_rate_window:
                    success_rate = success_count / total
                    if success_rate > 0.4:
                        F = min(F_high, F * 1.1)
                    else:
                        F = max(F_low, F * 0.9)
                    success_count = 0
                    fail_count = 0
                no_improve_count = 0

            # Also occasionally apply DE mutation to best point for local refinement
            if iteration % mutation_interval == 0 and evals < budget:
                # Generate one trial around best
                idxs = [j for j in range(n_vertices)]
                a, b, c = rng.choice(idxs, 3, replace=False)
                trial = points[a] + F * (points[b] - points[c])
                trial = np.clip(trial, lb, ub)
                ft = func(trial)
                evals += 1
                if evals >= budget:
                    break
                if ft < best_f:
                    best_f = ft
                    best_x = trial.copy()
                    report_best(best_f, best_x)
                    # Replace worst if better
                    if ft < f_vals[-1]:
                        points[-1] = trial
                        f_vals[-1] = ft

        return best_f, best_x