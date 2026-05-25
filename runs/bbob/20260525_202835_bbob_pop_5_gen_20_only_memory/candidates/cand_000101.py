import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng
        best_val = np.inf
        best_x = None
        evaluations = 0

        # Phase 1: DE with LHS
        de_budget = max(budget // 2, 2 * (dim + 1))
        if de_budget >= budget:
            de_budget = budget - (dim + 1)
        if de_budget <= 0:
            de_budget = 0
        popsize = max(4, min(4 * dim, de_budget // 2, 20))
        if de_budget < popsize:
            popsize = de_budget
        if popsize < 2:
            popsize = 2

        # Latin hypercube sampling
        samples = np.zeros((popsize, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, popsize + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            rng.shuffle(points)
            samples[:, d] = points
        pop = lb + samples * (ub - lb)
        fitness = np.full(popsize, np.inf)
        for i in range(popsize):
            if evaluations >= de_budget:
                break
            x = pop[i]
            val = func(x)
            evaluations += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = x.copy()
                report_best(best_val, best_x)

        # DE main loop
        F = 0.8
        CR = 0.9
        while evaluations < de_budget:
            for i in range(popsize):
                if evaluations >= de_budget:
                    break
                idx_best = np.argmin(fitness)
                idxs = [j for j in range(popsize) if j != i]
                if len(idxs) < 2:
                    break
                r1, r2 = rng.choice(idxs, 2, replace=False)
                mutant = pop[i] + F * (pop[idx_best] - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)
                cross_points = rng.random(dim) < CR
                if not np.any(cross_points):
                    cross_points[rng.integers(dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                val = func(trial)
                evaluations += 1
                if val < fitness[i]:
                    pop[i] = trial
                    fitness[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

        # Phase 2: Nelder-Mead refinement
        remaining = budget - evaluations
        if remaining > dim + 1 and best_x is not None:
            n = dim + 1
            # Build simplex around best_x
            simplex = np.tile(best_x, (n, 1))
            for i in range(1, n):
                perturb = rng.uniform(-0.1, 0.1, size=dim) * (ub - lb)
                simplex[i] = np.clip(best_x + perturb, lb, ub)
            values = np.full(n, np.inf)
            for i in range(n):
                if evaluations >= budget:
                    break
                x = simplex[i]
                val = func(x)
                evaluations += 1
                values[i] = val
                if val < best_val:
                    best_val = val
                    best_x = x.copy()
                    report_best(best_val, best_x)

            stall_limit = max(5 * dim, remaining // 20)
            last_improvement = evaluations
            while evaluations < budget:
                idx_sorted = np.argsort(values)
                simplex = simplex[idx_sorted]
                values = values[idx_sorted]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evaluations += 1
                if evaluations >= budget:
                    break

                if fr < values[0]:
                    xe = centroid + 2.0 * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evaluations += 1
                    if evaluations >= budget:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        values[-1] = fe
                        if fe < best_val:
                            best_val = fe
                            best_x = xe.copy()
                            report_best(best_val, best_x)
                            last_improvement = evaluations
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                        if fr < best_val:
                            best_val = fr
                            best_x = xr.copy()
                            report_best(best_val, best_x)
                            last_improvement = evaluations
                elif fr < values[-2]:
                    simplex[-1] = xr
                    values[-1] = fr
                    if fr < best_val:
                        best_val = fr
                        best_x = xr.copy()
                        report_best(best_val, best_x)
                        last_improvement = evaluations
                else:
                    if fr < values[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evaluations += 1
                    if evaluations >= budget:
                        break
                    if fc < values[-1]:
                        simplex[-1] = xc
                        values[-1] = fc
                        if fc < best_val:
                            best_val = fc
                            best_x = xc.copy()
                            report_best(best_val, best_x)
                            last_improvement = evaluations
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            val = func(simplex[i])
                            evaluations += 1
                            if evaluations >= budget:
                                break
                            values[i] = val
                            if val < best_val:
                                best_val = val
                                best_x = simplex[i].copy()
                                report_best(best_val, best_x)
                                last_improvement = evaluations
                        if evaluations >= budget:
                            break

                if evaluations - last_improvement > stall_limit and evaluations < budget:
                    # Restart around best_x
                    new_simplex = np.tile(best_x, (n, 1))
                    for i in range(1, n):
                        perturb = rng.uniform(-0.1, 0.1, size=dim) * (ub - lb)
                        new_simplex[i] = np.clip(best_x + perturb, lb, ub)
                        if evaluations >= budget:
                            break
                        val = func(new_simplex[i])
                        evaluations += 1
                        new_simplex[i] = new_simplex[i].copy()
                        values[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_simplex[i].copy()
                            report_best(best_val, best_x)
                            last_improvement = evaluations
                    new_simplex[0] = best_x
                    simplex = new_simplex
                    values[0] = best_val
                    last_improvement = evaluations

        return best_val, best_x