import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = np.random.RandomState(self.seed)
        scale = np.mean(ub - lb)

        best_x = None
        best_f = np.inf
        evals = 0

        # Initial random sampling
        n_init = min(dim + 1, max(1, int(0.1 * budget)))
        for _ in range(n_init):
            if evals >= budget:
                break
            x = lb + (ub - lb) * rng.rand(dim)
            f = func(x)
            evals += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        if evals >= budget:
            return best_f, best_x

        # Initialize simplex
        n_vertices = dim + 1
        simplex = np.zeros((n_vertices, dim))
        f_simplex = np.full(n_vertices, np.inf)
        simplex[0] = best_x.copy()
        f_simplex[0] = best_f
        for i in range(1, n_vertices):
            if evals >= budget:
                break
            x = rng.uniform(lb, ub, size=dim)
            f = func(x)
            evals += 1
            f_simplex[i] = f
            if f < best_f:
                best_f = f
                best_x = x.copy()
                report_best(best_f, best_x)

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0

        # Pattern search directions
        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        # DE parameters for restart
        F = 0.5
        CR = 0.9

        while evals < budget:
            # Order simplex
            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, lb, ub)
            fr = func(xr)
            evals += 1
            if evals >= budget:
                break
            if fr < best_f:
                best_f = fr
                best_x = xr.copy()
                report_best(best_f, best_x)

            improved = False
            if f_simplex[0] <= fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
                improved = True
            elif fr < f_simplex[0]:
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
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
                improved = True
            else:
                if fr < f_simplex[-1]:
                    xc = centroid + rho * (xr - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, lb, ub)
                fc = func(xc)
                evals += 1
                if evals >= budget:
                    break
                if fc < best_f:
                    best_f = fc
                    best_x = xc.copy()
                    report_best(best_f, best_x)
                if fc < min(fr, f_simplex[-1]):
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                    improved = True
                else:
                    # Shrink
                    for i in range(1, n_vertices):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], lb, ub)
                        fi = func(simplex[i])
                        evals += 1
                        if evals >= budget:
                            break
                        f_simplex[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = simplex[i].copy()
                            report_best(best_f, best_x)

            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Restart condition
            if no_improve_count >= max_no_improve and evals < budget:
                # First try pattern search from best
                step = 0.1 * scale
                pattern_evals = min(50, budget - evals)
                pattern_used = 0
                x_current = best_x.copy()
                f_current = best_f
                improved_locally = False
                while pattern_used < pattern_evals:
                    found = False
                    for d in directions:
                        if pattern_used >= pattern_evals:
                            break
                        candidate = np.clip(x_current + step * d, lb, ub)
                        val = func(candidate)
                        pattern_used += 1
                        evals += 1
                        if val < best_f:
                            best_f = val
                            best_x = candidate.copy()
                            report_best(best_f, best_x)
                            x_current = candidate
                            f_current = val
                            found = True
                            step *= 1.2
                            break
                    if not found:
                        step *= 0.5
                        if step < 1e-12 * scale:
                            break
                    else:
                        improved_locally = True

                # Restart simplex with new points generated by DE
                # Ensure we have enough distinct indices for DE; fallback for dim=1
                if dim == 1:
                    # Use uniform random restarts
                    simplex[0] = best_x.copy()
                    f_simplex[0] = best_f
                    for i in range(1, n_vertices):
                        if evals >= budget:
                            break
                        x = rng.uniform(lb, ub, size=dim)
                        f = func(x)
                        evals += 1
                        f_simplex[i] = f
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                            report_best(best_f, best_x)
                else:
                    # Use DE/rand/1 to generate each new vertex
                    # Store best as vertex 0
                    simplex[0] = best_x.copy()
                    f_simplex[0] = best_f
                    for i in range(1, n_vertices):
                        if evals >= budget:
                            break
                        # Choose 3 distinct random indices from current simplex (excluding the one to be replaced? we have full simplex now)
                        # Use indices in [0, n_vertices-1]
                        a, b, c = rng.choice(n_vertices, size=3, replace=False)
                        mutant = np.clip(simplex[a] + F * (simplex[b] - simplex[c]), lb, ub)
                        # Crossover with best
                        trial = np.where(rng.rand(dim) < CR, mutant, best_x)
                        trial = np.clip(trial, lb, ub)
                        fi = func(trial)
                        evals += 1
                        f_simplex[i] = fi
                        simplex[i] = trial
                        if fi < best_f:
                            best_f = fi
                            best_x = trial.copy()
                            report_best(best_f, best_x)
                no_improve_count = 0

            if evals >= budget:
                break

        return best_f, best_x