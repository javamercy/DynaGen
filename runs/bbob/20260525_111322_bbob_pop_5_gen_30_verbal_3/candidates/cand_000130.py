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

        # Initial parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_no_improve = max(10, dim * 2)
        no_improve_count = 0

        directions = []
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            directions.append(e)
            directions.append(-e)

        # Schedule parameters: decay factors based on budget fraction used
        def get_schedule(t):
            # t = fraction of budget used
            t = max(0, min(1, t))
            # alpha: 1.0 -> 0.5
            alpha_s = 1.0 - 0.5 * t
            # gamma: 2.0 -> 1.0
            gamma_s = 2.0 - 1.0 * t
            # rho: 0.5 -> 0.25
            rho_s = 0.5 - 0.25 * t
            # sigma: 0.5 -> 0.2
            sigma_s = 0.5 - 0.3 * t
            # max_no_improve: larger near beginning, smaller near end
            max_no_imp = max(5, int(max_no_improve * (1 + 0.5 * (1 - t))))
            # pattern step scale: 0.1*scale -> 0.01*scale
            step_scale = (0.1 - 0.09 * t) * scale
            return alpha_s, gamma_s, rho_s, sigma_s, max_no_imp, step_scale

        while evals < budget:
            t = evals / budget
            alpha_curr, gamma_curr, rho_curr, sigma_curr, max_no_imp_curr, step_scale_curr = get_schedule(t)

            order = np.argsort(f_simplex)
            simplex = simplex[order]
            f_simplex = f_simplex[order]

            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha_curr * (centroid - simplex[-1])
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
                xe = centroid + gamma_curr * (xr - centroid)
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
                    xc = centroid + rho_curr * (xr - centroid)
                else:
                    xc = centroid - rho_curr * (centroid - simplex[-1])
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
                    for i in range(1, n_vertices):
                        simplex[i] = simplex[0] + sigma_curr * (simplex[i] - simplex[0])
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

            if no_improve_count >= max_no_imp_curr and evals < budget:
                step = step_scale_curr
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

                # Differential mutation restart with adaptive factor
                mut_factor = 0.3 + 0.4 * (1 - t)  # decays from 0.7 to 0.3
                if improved_locally:
                    new_simplex = np.zeros((n_vertices, dim))
                    new_simplex[0] = best_x.copy()
                    f_new = np.zeros(n_vertices)
                    f_new[0] = best_f
                    for i in range(1, n_vertices):
                        idx = rng.choice(n_vertices, 3, replace=False)
                        a = simplex[idx[0]]
                        b = simplex[idx[1]]
                        c = simplex[idx[2]]
                        mutant = np.clip(a + mut_factor * (b - c), lb, ub)
                        new_simplex[i] = mutant
                        if evals >= budget:
                            break
                        fi = func(mutant)
                        evals += 1
                        f_new[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = mutant.copy()
                            report_best(best_f, best_x)
                    simplex = new_simplex
                    f_simplex = f_new
                else:
                    new_simplex = np.zeros((n_vertices, dim))
                    new_simplex[0] = best_x.copy()
                    f_new = np.zeros(n_vertices)
                    f_new[0] = best_f
                    for i in range(1, n_vertices):
                        a = best_x
                        b = simplex[rng.randint(n_vertices)]
                        c = rng.uniform(lb, ub, size=dim)
                        mutant = np.clip(a + mut_factor * (b - c), lb, ub)
                        new_simplex[i] = mutant
                        if evals >= budget:
                            break
                        fi = func(mutant)
                        evals += 1
                        f_new[i] = fi
                        if fi < best_f:
                            best_f = fi
                            best_x = mutant.copy()
                            report_best(best_f, best_x)
                    simplex = new_simplex
                    f_simplex = f_new
                no_improve_count = 0

            if evals >= budget:
                break

        return best_f, best_x