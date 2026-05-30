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

        # Population size: smaller to give more budget to NM
        pop_size = min(budget // 3, max(2 * dim, 10 + int(dim ** 0.5)))
        pop_size = max(pop_size, 2 * dim)
        pop_size = min(pop_size, budget // 2)

        # Latin Hypercube initial population
        lhs = self._latin_hypercube(pop_size, dim, rng)
        bounds = np.array([lb, ub]).T
        pop = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * lhs

        best_val = np.inf
        best_x = None
        evals = 0

        pop_fitness = np.full(pop_size, np.inf)
        for i in range(pop_size):
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)
            if evals >= budget:
                return best_val, best_x

        # Reserve evaluations for NM local search
        nm_budget = max(2 * dim, budget // 2)
        nm_budget = min(nm_budget, budget - evals)
        max_gen = (budget - evals - nm_budget) // pop_size
        max_gen = max(0, max_gen)

        # Stagnation detection
        stagnation_gen = 0
        stag_limit = max(10, max_gen // 4) if max_gen > 0 else 1
        last_best_val = best_val

        # Adaptive DE
        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                mutant = pop[a] + F * (pop[b] - pop[c])
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                evals += 1
                if val < pop_fitness[i]:
                    pop[i] = trial
                    pop_fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                if evals >= budget - nm_budget:
                    break
            if evals >= budget - nm_budget:
                break

            # Stagnation check
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1

            if stagnation_gen >= stag_limit and evals < budget - nm_budget:
                # Restart worst 20% of population
                worst_idx = np.argsort(pop_fitness)[-max(1, pop_size // 5):]
                for idx in worst_idx:
                    if evals >= budget - nm_budget:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    new_val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = new_val
                    if new_val < best_val:
                        best_val = new_val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation_gen = 0
                last_best_val = best_val

        # Nelder-Mead local search from best point
        if evals < budget:
            # Adaptive step size: decreases as remaining budget is used
            frac_nm = 1.0 - (evals) / budget  # fraction of budget left
            step = (0.15 - 0.10 * frac_nm) * (ub - lb)
            step = np.clip(step, 0.01 * (ub - lb), None)  # minimum step
            # Initialize simplex
            simplex = np.tile(best_x, (dim + 1, 1))
            for i in range(dim):
                simplex[i + 1, i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            for i in range(dim + 1):
                if i == 0:
                    fvals[i] = best_val
                else:
                    if evals >= budget:
                        break
                    x = np.clip(simplex[i], lb, ub)
                    val = func(x)
                    evals += 1
                    fvals[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = x.copy()
                        report_best(best_val, best_x)

            # Nelder-Mead parameters
            rho = 1.0
            chi = 2.0
            psi = 0.5
            sigma = 0.5

            while evals < budget:
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                best_val_local = fvals[0]
                worst_val = fvals[-1]
                second_worst_val = fvals[-2]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget:
                    break
                fr = func(xr)
                evals += 1
                if fr < best_val_local:
                    # Expansion
                    xe = centroid + chi * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget:
                        break
                    fe = func(xe)
                    evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                        if fe < best_val:
                            best_val = fe
                            best_x = xe.copy()
                            report_best(best_val, best_x)
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        if fr < best_val:
                            best_val = fr
                            best_x = xr.copy()
                            report_best(best_val, best_x)
                elif fr < second_worst_val:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                else:
                    # Contraction
                    if fr < worst_val:
                        # Outside contraction
                        xc = centroid + psi * (xr - centroid)
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget:
                            break
                        fc = func(xc)
                        evals += 1
                        if fc < fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_val:
                                best_val = fc
                                best_x = xc.copy()
                                report_best(best_val, best_x)
                        else:
                            # shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                if evals >= budget:
                                    break
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < best_val:
                                    best_val = val_i
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)
                            if evals >= budget:
                                break
                    else:
                        # Inside contraction
                        xc = centroid - psi * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget:
                            break
                        fc = func(xc)
                        evals += 1
                        if fc < worst_val:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_val:
                                best_val = fc
                                best_x = xc.copy()
                                report_best(best_val, best_x)
                        else:
                            # shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                if evals >= budget:
                                    break
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < best_val:
                                    best_val = val_i
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)
                            if evals >= budget:
                                break

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs