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
        best_x = np.empty(dim)
        evals = 0

        # For very small budgets, just random search
        if budget <= 2 * dim + 2:
            for _ in range(budget):
                x = lb + rng.rand(dim) * (ub - lb)
                val = func(x)
                evals += 1
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)
            return best_val, best_x

        # Reserve budget for Nelder-Mead: at least 2*dim, at most budget-1
        nm_budget = max(2 * dim, budget // 3)
        if nm_budget > budget - 1:
            nm_budget = budget - 1
        # Determine population size for DE
        remaining = budget - nm_budget
        pop_size = max(3 * dim, min(15 + int(dim**0.5), remaining // 2))
        if pop_size > remaining:
            pop_size = remaining
        if pop_size < 2:
            pop_size = 2
        # Ensure at least one initial point before DE
        if pop_size > remaining:
            pop_size = remaining
        if pop_size < 1:
            pop_size = 1

        # Latin Hypercube initial population
        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = intervals[perm[i]] + rng.uniform(0, 1/n)
            return samples

        lhs_samples = lhs(pop_size, dim)
        pop = lb + (ub - lb) * lhs_samples
        pop_fitness = np.full(pop_size, np.inf)

        # Evaluate initial population
        for i in range(pop_size):
            if evals >= remaining:
                break
            x = np.clip(pop[i], lb, ub)
            val = func(x)
            evals += 1
            pop_fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # DE generations
        remaining_for_de = remaining - evals
        max_gen = 0
        if pop_size > 0:
            max_gen = remaining_for_de // pop_size
        max_gen = min(max_gen, 100)
        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        stag_counter = 0
        last_best = best_val

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= remaining:
                    break
                strat = rng.randint(3)
                indices = [j for j in range(pop_size) if j != i]
                rng.shuffle(indices)
                a, b, c = indices[:3]
                if strat == 0:
                    mutant = pop[a] + F * (pop[b] - pop[c])
                elif strat == 1:
                    mutant = best_x + F * (pop[b] - pop[c])
                else:
                    mutant = pop[i] + F * (pop[a] - pop[i]) + F * (pop[b] - pop[c])
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
            if evals >= remaining:
                break

            # Stagnation check
            if best_val < last_best:
                stag_counter = 0
                last_best = best_val
            else:
                stag_counter += 1

            if stag_counter >= stag_limit and evals < remaining:
                n_replace = max(1, pop_size // 2)
                worst_idx = np.argsort(pop_fitness)[-n_replace:]
                for idx in worst_idx:
                    if evals >= remaining:
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
                stag_counter = 0
                last_best = best_val

        # Nelder-Mead local search from best point
        nm_evals = 0
        max_nm = budget - evals
        if max_nm >= dim + 1:
            step = 0.1 * (ub - lb)
            simplex = np.tile(best_x, (dim + 1, 1))
            for i in range(dim):
                x_i = np.clip(best_x[i] + step[i], lb[i], ub[i])
                simplex[i+1, i] = x_i
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = best_val
            for i in range(1, dim + 1):
                if nm_evals >= max_nm:
                    break
                x = np.clip(simplex[i], lb, ub)
                val = func(x)
                nm_evals += 1
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
                best_local = fvals[0]
                worst_local = fvals[-1]
                second_worst = fvals[-2]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget: break
                fr = func(xr)
                evals += 1
                if fr < best_local:
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
                    if fr < worst_local:
                        # Outside contraction
                        xc = centroid + psi * (xr - centroid)
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget: break
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
                                if evals >= budget: break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < best_val:
                                    best_val = val_i
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)
                    else:
                        # Inside contraction
                        xc = centroid - psi * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget: break
                        fc = func(xc)
                        evals += 1
                        if fc < worst_local:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_val:
                                best_val = fc
                                best_x = xc.copy()
                                report_best(best_val, best_x)
                        else:
                            # shrink
                            for i in range(1, dim + 1):
                                if evals >= budget: break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                val_i = func(simplex[i])
                                evals += 1
                                fvals[i] = val_i
                                if val_i < best_val:
                                    best_val = val_i
                                    best_x = simplex[i].copy()
                                    report_best(best_val, best_x)
        return best_val, best_x