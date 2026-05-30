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

        def evaluate(x):
            nonlocal evals, best_val, best_x
            if evals >= budget:
                return None
            x_clip = np.clip(x, lb, ub)
            val = func(x_clip)
            evals += 1
            if val < best_val:
                best_val = val
                best_x = x_clip.copy()
                report_best(best_val, best_x)
            return val

        # Budget split: 2/3 for DE, 1/3 for NM (adjust if budget small)
        de_budget = max(dim + 2, (budget * 2) // 3)
        nm_budget = budget - de_budget
        if nm_budget < 2 * dim:
            de_budget = budget - 2 * dim
            nm_budget = 2 * dim
        if de_budget < dim + 2:
            de_budget = budget - dim - 2
            nm_budget = dim + 2

        # Population size
        pop_size = max(4 * dim, min(20 + dim, de_budget // 2))
        if pop_size < 2 * dim:
            pop_size = max(2, min(2 * dim, de_budget))
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

        pop = lb + (ub - lb) * lhs(pop_size, dim)
        pop_fit = np.full(pop_size, np.inf)

        for i in range(pop_size):
            if evals >= de_budget:
                break
            val = evaluate(pop[i])
            if val is None:
                break
            pop_fit[i] = val

        # SHADE parameters
        memory_size = dim
        M_F = np.ones(memory_size) * 0.5
        M_CR = np.ones(memory_size) * 0.5
        archive = []
        archive_max = pop_size * 2

        gen = 0
        while evals < de_budget:
            gen += 1
            # Generate F and CR for each individual
            F_arr = np.zeros(pop_size)
            CR_arr = np.zeros(pop_size)
            for i in range(pop_size):
                r = rng.randint(memory_size)
                F_arr[i] = np.clip(M_F[r] + 0.1 * rng.randn(), 0, 1)
                CR_arr[i] = np.clip(M_CR[r] + 0.1 * rng.randn(), 0, 1)

            # Sort population for pbest
            order = np.argsort(pop_fit)
            sorted_pop = pop[order]

            # pbest selection
            p = max(2, int(0.2 * pop_size))

            new_pop = pop.copy()
            new_fit = pop_fit.copy()
            success_F = []
            success_CR = []
            delta_f = []

            for i in range(pop_size):
                if evals >= de_budget:
                    break
                # pbest index
                pbest_idx = rng.randint(p)
                pbest = sorted_pop[pbest_idx]
                # Random individuals a, b (distinct from i and each other)
                indices = list(range(pop_size))
                indices.remove(i)
                a, b = rng.choice(indices, 2, replace=False)
                # Mutation: current-to-pbest/1 with archive
                mutant = pop[i] + F_arr[i] * (pbest - pop[i]) + F_arr[i] * (pop[a] - pop[b])
                # Crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR_arr[i], mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                val = evaluate(trial)
                if val is None:
                    break
                if val < pop_fit[i]:
                    # Success
                    new_pop[i] = trial
                    new_fit[i] = val
                    success_F.append(F_arr[i])
                    success_CR.append(CR_arr[i])
                    delta_f.append(pop_fit[i] - val)
                    # Add replaced individual to archive
                    if len(archive) < archive_max:
                        archive.append(pop[i].copy())
                    else:
                        # Randomly replace
                        idx = rng.randint(archive_max)
                        archive[idx] = pop[i].copy()

            pop = new_pop
            pop_fit = new_fit

            # Update memory if successes
            if len(success_F) > 0:
                # Weighted Lehmer mean
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-10)
                mean_F = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-10)
                mean_CR = np.sum(w * np.array(success_CR)**2) / (np.sum(w * np.array(success_CR)) + 1e-10)
                # Update memory (circular)
                mem_idx = gen % memory_size
                M_F[mem_idx] = mean_F
                M_CR[mem_idx] = mean_CR

            # Stagnation detection: if no improvement for pop_size/2 generations
            if gen % max(1, pop_size // 2) == 0:
                # Check if best has improved recently
                # Simple: replace worst 25% with random points around best
                n_replace = max(1, pop_size // 4)
                worst_idx = np.argsort(pop_fit)[-n_replace:]
                for idx in worst_idx:
                    if evals >= de_budget:
                        break
                    sigma = 0.1 * (ub - lb)
                    new_x = best_x + rng.randn(dim) * sigma
                    new_x = np.clip(new_x, lb, ub)
                    val = evaluate(new_x)
                    if val is None:
                        break
                    pop[idx] = new_x
                    pop_fit[idx] = val

        # Nelder-Mead local search
        if evals < budget and best_x is not None:
            step = 0.1 * (ub - lb)
            simplex = np.tile(best_x, (dim + 1, 1))
            for i in range(dim):
                simplex[i+1, i] = np.clip(best_x[i] + step[i], lb[i], ub[i])
            fvals = np.full(dim + 1, np.inf)
            fvals[0] = best_val
            for i in range(1, dim + 1):
                if evals >= budget:
                    break
                val = evaluate(simplex[i])
                if val is None:
                    break
                fvals[i] = val

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
                            # Shrink
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
                            # Shrink
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