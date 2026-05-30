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

        best_val = np.inf
        best_x = None
        evals = 0

        # If budget is very small, just random search
        if budget < 2 * dim + 5:
            for _ in range(budget):
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
                if evals >= budget:
                    break
            return best_val, best_x

        # Population size for DE
        pop_size = max(2 * dim, int(budget ** 0.7))
        pop_size = min(pop_size, budget // 2, 50)
        pop_size = max(pop_size, 3)  # at least 3 for DE

        # Latin hypercube initialization
        lhs = self._latin_hypercube(pop_size, dim, rng)
        pop = lb + (ub - lb) * lhs

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

        # Reserve evaluations for NM: at least 2*dim+5 or 20% of budget
        nm_budget = max(2 * dim + 5, int(0.2 * budget))
        nm_budget = min(nm_budget, budget - evals)
        if nm_budget > 0:
            max_gen = (budget - evals - nm_budget) // pop_size
        else:
            max_gen = 0

        stagnation_gen = 0
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        last_best_val = best_val

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.5
            F = 0.8 + 0.2 * (1 - frac)
            CR = 0.9 - 0.4 * frac

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

            # Stagnation restart: replace worst 20%
            if best_val < last_best_val:
                stagnation_gen = 0
                last_best_val = best_val
            else:
                stagnation_gen += 1

            if stagnation_gen >= stag_limit and evals < budget - nm_budget:
                n_replace = max(1, pop_size // 5)
                worst_idx = np.argsort(pop_fitness)[-n_replace:]
                for idx in worst_idx:
                    if evals >= budget - nm_budget:
                        break
                    new_x = lb + rng.rand(dim) * (ub - lb)
                    val = func(new_x)
                    evals += 1
                    pop[idx] = new_x
                    pop_fitness[idx] = val
                    if val < best_val:
                        best_val = val
                        best_x = new_x.copy()
                        report_best(best_val, best_x)
                stagnation_gen = 0
                last_best_val = best_val

        # Nelder-Mead local search from best
        if evals < budget and nm_budget > 0:
            simplex = np.tile(best_x, (dim + 1, 1))
            step = 0.05 * (ub - lb)
            for i in range(dim):
                simplex[i + 1, i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = best_val
            for i in range(1, dim + 1):
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
                second_worst = fvals[-2]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget: break
                fr = func(xr)
                evals += 1
                if fr < best_val_local:
                    # Expansion
                    xe = centroid + chi * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    if evals >= budget: break
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
                elif fr < second_worst:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                else:
                    # Contraction
                    if fr < worst_val:
                        xc = centroid + psi * (xr - centroid)
                    else:
                        xc = centroid - psi * (centroid - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    if evals >= budget: break
                    fc = func(xc)
                    evals += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                        if fc < best_val:
                            best_val = fc
                            best_x = xc.copy()
                            report_best(best_val, best_x)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if evals >= budget: break
                            val_i = func(simplex[i])
                            evals += 1
                            fvals[i] = val_i
                            if val_i < best_val:
                                best_val = val_i
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                        if evals >= budget: break

        return best_val, best_x

    def _latin_hypercube(self, n, d, rng):
        intervals = np.linspace(0, 1, n + 1)
        lhs = np.zeros((n, d))
        for j in range(d):
            perm = rng.permutation(n)
            for i in range(n):
                lhs[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
        return lhs