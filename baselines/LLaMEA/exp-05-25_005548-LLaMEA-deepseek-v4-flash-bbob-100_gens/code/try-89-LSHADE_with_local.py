import numpy as np

class LSHADE_with_local:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Budget split: 80% main DE, 20% local search (adjusted)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Halton sequence initialization ----
        def halton(n, d, low, high):
            """Generate n points in d dimensions using Halton sequence."""
            # Generate primes for each dimension
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
            if d > len(primes):
                # fallback to LHS
                result = np.zeros((n, d))
                for i in range(d):
                    perm = np.random.permutation(n)
                    result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
                return result
            result = np.zeros((n, d))
            for i in range(d):
                p = primes[i]
                for k in range(n):
                    # Halton sequence element for prime p
                    x = 0.0
                    f = 1.0 / p
                    kk = k + 1
                    while kk > 0:
                        x += (kk % p) * f
                        kk //= p
                        f /= p
                    result[k, i] = low[i] + x * (high[i] - low[i])
            return result

        NP_init = max(10, 20 * int(np.log(dim)) if dim > 1 else 20)
        NP = NP_init

        pop = halton(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP
        H = 30
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop (jSO style) ----
        generations_since_improvement = 0
        restart_threshold = max(10, int(0.1 * main_budget))  # evaluations without best improvement

        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (jSO)
            ratio = 0.25 - 0.20 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            best_before = self.best_f

            for i in range(NP):
                r = np.random.randint(H)
                # Generate CR from Cauchy
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Generate F from Cauchy, truncated to >0
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # pbest selection
                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # current-to-pbest/1 with archive
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_df.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer for F and weighted arithmetic for CR
            if S_CR:
                w = np.array(S_df) / (np.sum(S_df) + 1e-30)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / (sum_w + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Check for improvement for restart condition
            if self.best_f < best_before:
                generations_since_improvement = 0
            else:
                generations_since_improvement += fevals  # rough count of evaluations

            # Restart if stuck (reinitialize 50% of population, keep best)
            if generations_since_improvement > restart_threshold and fevals < main_budget * 0.8:
                keep = int(NP * 0.5) + 1
                sorted_idx = np.argsort(fitness)
                pop_new = pop[sorted_idx[:keep]].copy()
                fit_new = fitness[sorted_idx[:keep]].copy()
                # generate new individuals
                n_new = NP - keep
                new_points = halton(n_new, dim, lb, ub)
                new_fit = np.array([func(x) for x in new_points])
                fevals += n_new
                pop = np.vstack((pop_new, new_points))
                fitness = np.concatenate((fit_new, new_fit))
                archive = np.empty((0, dim))  # reset archive
                generations_since_improvement = 0
                # reset best if needed
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()
                if fevals >= main_budget:
                    break

        # ---- Enhanced Local Search (golden-section + random directions) ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            # Golden-section search parameters
            phi = (1 + np.sqrt(5)) / 2
            resphi = 2 - phi  # inverse golden ratio

            # Random direction parameters
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)

            while evals < local_budget:
                improved = False
                # Phase 1: Coordinate-wise golden-section search (3 evaluations per dimension per cycle)
                for j in range(dim):
                    if evals + 3 > local_budget:
                        break
                    # Bracket based on current best
                    a = max(lb[j], x_best[j] - step[j])
                    b = min(ub[j], x_best[j] + step[j])
                    # Golden-section search for 5 iterations, using 3 evaluations per iteration? Actually, we'll do a simple two-evaluation check:
                    # Evaluate at two interior points
                    x1 = b - resphi * (b - a)
                    x2 = a + resphi * (b - a)
                    # Ensure within bounds
                    x1 = np.clip(x1, lb[j], ub[j])
                    x2 = np.clip(x2, lb[j], ub[j])
                    # Create candidates
                    cand1 = x_best.copy(); cand1[j] = x1
                    cand2 = x_best.copy(); cand2[j] = x2
                    f1 = func(cand1); evals += 1
                    f2 = func(cand2); evals += 1
                    # Check improvement
                    if f1 < f_best and f1 <= f2:
                        x_best, f_best = cand1, f1
                        improved = True
                        step[j] = min(step[j] * 1.2, max_step[j])
                    elif f2 < f_best and f2 < f1:
                        x_best, f_best = cand2, f2
                        improved = True
                        step[j] = min(step[j] * 1.2, max_step[j])
                    else:
                        # Narrow bracket and try one more point?
                        # Instead, just keep best and reduce step
                        step[j] = max(step[j] * 0.5, min_step[j])
                    # Evaluate third point? Not needed.

                if evals >= local_budget:
                    break

                # Phase 2: Random direction perturbations
                num_random = max(1, int(0.3 * (local_budget - evals)))
                for _ in range(num_random):
                    if evals >= local_budget:
                        break
                    dir = np.random.randn(dim)
                    dir = dir / (np.linalg.norm(dir) + 1e-30)
                    s = np.mean(step)
                    cand = x_best + s * dir
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step = np.minimum(step * 1.2, max_step)
                        improved = True
                    else:
                        step = np.maximum(step * 0.9, min_step)

                if not improved:
                    # Restart step size
                    step = np.minimum(step * 1.5, max_step)

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

                if np.all(step <= min_step * 2):
                    break

        return self.best_f, self.best_x