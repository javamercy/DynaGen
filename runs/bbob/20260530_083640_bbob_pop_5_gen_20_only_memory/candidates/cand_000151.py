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

        nm_budget = max(2 * dim, budget // 3)
        pop_size = max(3 * dim, min(15 + int(dim**0.5), budget // 2))
        pop_size = min(pop_size, budget - nm_budget)
        if pop_size < 2 * dim:
            pop_size = max(2, min(2 * dim, budget - nm_budget))
        if pop_size < 1:
            pop_size = 1

        def lhs(n, d):
            intervals = np.linspace(0, 1, n + 1)
            samples = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                for i in range(n):
                    samples[i, j] = intervals[perm[i]] + rng.uniform(0, 1 / n)
            return samples

        lhs_samples = lhs(pop_size, dim)
        pop = lb + (ub - lb) * lhs_samples

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

        remaining_for_de = budget - evals - nm_budget
        max_gen = max(0, remaining_for_de // pop_size) if pop_size > 0 else 0
        max_gen = min(max_gen, 100)

        stag_limit = max(5, max_gen // 5) if max_gen > 0 else 1
        stag_counter = 0
        last_best = best_val

        for gen in range(max_gen):
            frac = gen / max_gen if max_gen > 0 else 0.0
            F = 0.9 - 0.4 * frac
            CR = 0.5 + 0.4 * frac
            for i in range(pop_size):
                if evals >= budget - nm_budget:
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
                if evals >= budget - nm_budget:
                    break
            if evals >= budget - nm_budget:
                break

            if best_val < last_best:
                stag_counter = 0
                last_best = best_val
            else:
                stag_counter += 1

            if stag_counter >= stag_limit and evals < budget - nm_budget:
                n_replace = max(1, pop_size // 2)  # 40% equivalent, but use half for simplicity
                # Actually we said 40%, but 50% is fine. Let's use 50%.
                worst_idx = np.argsort(pop_fitness)[-n_replace:]
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
                stag_counter = 0
                last_best = best_val

        # Collect top two distinct points for NM
        order = np.argsort(pop_fitness)
        top_indices = order[:2]
        start_points = [pop[idx].copy() for idx in top_indices]
        # Check if second point is sufficiently different from best
        if len(start_points) > 1:
            dist = np.linalg.norm(start_points[0] - start_points[1])
            if dist < 0.1 * np.linalg.norm(ub - lb):
                start_points = [start_points[0]]

        for start_x in start_points:
            if evals >= budget:
                break
            step = 0.1 * (ub - lb)
            simplex = np.tile(start_x, (dim + 1, 1))
            for i in range(dim):
                simplex[i + 1, i] = np.clip(start_x[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = best_val if np.array_equal(start_x, best_x) else func(start_x)  # reuse best_val if same
            # We need to evaluate all simplex points
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
                best_local = fvals[0]
                worst = fvals[-1]
                second_worst = fvals[-2]

                centroid = np.mean(simplex[:-1], axis=0)

                xr = centroid + rho * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                if evals >= budget:
                    break
                fr = func(xr)
                evals += 1
                if fr < best_local:
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
                elif fr < second_worst:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                else:
                    if fr < worst:
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
                            for i in range(1, dim + 1):
                                if evals >= budget:
                                    break
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
                        xc = centroid - psi * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        if evals >= budget:
                            break
                        fc = func(xc)
                        evals += 1
                        if fc < worst:
                            simplex[-1] = xc
                            fvals[-1] = fc
                            if fc < best_val:
                                best_val = fc
                                best_x = xc.copy()
                                report_best(best_val, best_x)
                        else:
                            for i in range(1, dim + 1):
                                if evals >= budget:
                                    break
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
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