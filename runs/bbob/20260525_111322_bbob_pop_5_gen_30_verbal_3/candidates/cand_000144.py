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

        # LHS for initial simplex of size dim+1
        def lhs_sample(n, dim, lb, ub):
            points = np.zeros((n, dim))
            for i in range(dim):
                perm = rng.permutation(n)
                u = rng.rand(n)
                points[:, i] = lb[i] + (perm + u) / n * (ub[i] - lb[i])
            return points

        n = dim + 1
        simplex = lhs_sample(n, dim, lb, ub)
        fits = np.full(n, np.inf)
        best_x = None
        best_f = np.inf
        evals = 0

        # Evaluate initial simplex
        for i in range(n):
            if evals >= budget:
                break
            x = simplex[i]
            f = func(x)
            evals += 1
            fits[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        # Stagnation limit
        stagnation_limit = max(1, budget // (10 * n))
        no_improve_evals = 0

        # DE mutation parameters
        F = 0.5
        CR = 0.9

        while evals < budget:
            # Sort simplex by fitness
            order = np.argsort(fits)
            simplex = simplex[order]
            fits = fits[order]
            best = simplex[0]
            best_f_current = fits[0]
            worst = simplex[-1]
            second_worst = simplex[-2]

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, lb, ub)
            if evals >= budget:
                break
            fr = func(xr)
            evals += 1
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)
                no_improve_evals = 0
            else:
                no_improve_evals += 1

            if fr < fits[-2]:  # better than second worst
                if fr < fits[0]:  # better than best -> expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget:
                        break
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fits[-1] = fe
                        if fe < best_f:
                            best_f = fe
                            best_x = xe.copy()
                            report_best(best_f, best_x)
                            no_improve_evals = 0
                    else:
                        simplex[-1] = xr
                        fits[-1] = fr
                else:
                    simplex[-1] = xr
                    fits[-1] = fr
            else:  # fr >= second worst
                if fr < fits[-1]:  # better than worst but not second worst
                    simplex[-1] = xr
                    fits[-1] = fr
                else:
                    # Contraction
                    if fr < fits[-1]:  # outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:  # inside contraction
                        xc = centroid - rho * (centroid - worst)
                    xc = np.clip(xc, lb, ub)
                    if evals >= budget:
                        break
                    fc = func(xc)
                    evals += 1
                    if fc < fits[-1]:
                        simplex[-1] = xc
                        fits[-1] = fc
                        if fc < best_f:
                            best_f = fc
                            best_x = xc.copy()
                            report_best(best_f, best_x)
                            no_improve_evals = 0
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if evals >= budget:
                                break
                            f = func(simplex[i])
                            evals += 1
                            fits[i] = f
                            if f < best_f:
                                best_f = f
                                best_x = simplex[i].copy()
                                report_best(best_f, best_x)
                                no_improve_evals = 0

            # Check stagnation and restart
            if evals >= budget:
                break
            if no_improve_evals >= stagnation_limit:
                # Anisotropic restart: generate new simplex via multivariate normal
                # Estimate mean and covariance from current simplex (excluding worst? include all)
                mean = np.mean(simplex, axis=0)
                cov = np.cov(simplex, rowvar=False)
                # Add small diagonal to ensure positive definiteness
                cov += 1e-10 * np.eye(dim)
                # Use best point as center for sampling (more aggressive)
                center = best_x if best_x is not None else simplex[0]
                new_simplex = np.zeros_like(simplex)
                for i in range(n):
                    sample = rng.multivariate_normal(center, cov)
                    sample = np.clip(sample, lb, ub)
                    new_simplex[i] = sample
                # Evaluate new points (except if best retained? We'll evaluate all)
                new_fits = np.full(n, np.inf)
                for i in range(n):
                    if evals >= budget:
                        break
                    x = new_simplex[i]
                    f = func(x)
                    evals += 1
                    new_fits[i] = f
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                        report_best(best_f, best_x)
                # Apply DE/rand/1 mutation to each point (except best? do all)
                for i in range(n):
                    if evals >= budget:
                        break
                    target = new_simplex[i]
                    # choose three distinct random indices
                    candidates = list(range(n))
                    candidates.remove(i)
                    if len(candidates) < 3:
                        continue
                    a, b, c = rng.choice(candidates, 3, replace=False)
                    mutant = new_simplex[a] + F * (new_simplex[b] - new_simplex[c])
                    trial = target.copy()
                    j_rand = rng.randint(dim)
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                    trial = np.clip(trial, lb, ub)
                    ft = func(trial)
                    evals += 1
                    if ft < new_fits[i]:
                        new_simplex[i] = trial
                        new_fits[i] = ft
                        if ft < best_f:
                            best_f = ft
                            best_x = trial.copy()
                            report_best(best_f, best_x)
                # Replace simplex
                simplex = new_simplex
                fits = new_fits
                # Reset stagnation
                no_improve_evals = 0
                # Resample F and CR randomly
                F = rng.uniform(0.2, 0.9)
                CR = rng.uniform(0.5, 1.0)

        return best_f, best_x