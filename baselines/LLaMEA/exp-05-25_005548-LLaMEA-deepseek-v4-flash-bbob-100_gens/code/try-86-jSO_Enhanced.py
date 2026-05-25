import numpy as np

class jSO_Enhanced:
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

        # Budget allocation: main DE 80%, local search 20%
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

        # ---- Halton initialization (low-discrepancy sequence) ----
        NP_init = max(10, 20 * int(np.log(dim)) if dim > 1 else 20)
        NP = NP_init

        # Generate Halton sequence for first NP_init points
        def halton(n, d):
            # bases: first d primes
            primes = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,
                      73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,
                      157,163,167,173,179,181,191,193,197,199,211,223,227,229]
            pts = np.zeros((n, d))
            for i in range(d):
                base = primes[i]
                for j in range(n):
                    jj = j + 1
                    f = 1.0
                    x = 0.0
                    while jj > 0:
                        f /= base
                        x += (jj % base) * f
                        jj //= base
                    pts[j, i] = x
            return pts

        halton_pts = halton(NP, dim)
        pop = lb + (ub - lb) * halton_pts
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

        # ---- Main jSO-inspired DE loop with improvements ----
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
                    # Keep only NP archive members furthest from best (or random)
                    # Use distance-based deletion: keep members far from best
                    if len(archive) > NP:
                        dists = np.linalg.norm(archive - self.best_x, axis=1)
                        # retain the ones with largest distance
                        idx_keep = np.argsort(-dists)[:NP]
                        archive = archive[idx_keep]
                max_archive = NP

            # Quadratic pbest ratio (jSO style: more exploration early)
            p = 0.25 * (remaining_evals / main_budget) ** 2
            p = max(0.02, min(0.25, p))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from population or archive (with equal probability)
                if np.random.rand() < 0.5 and len(archive) > 0:
                    idx = np.random.randint(len(archive))
                    r2_vec = archive[idx]
                else:
                    idx = np.random.randint(NP)
                    while idx == i or idx == r1:
                        idx = np.random.randint(NP)
                    r2_vec = pop[idx]

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
                u[still_high] = np.random.uniform(ub[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_df.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # Archive management: keep the replaced parent but delete farthest from best if full
                    if len(archive) < max_archive:
                        archive = np.vstack((archive, pop[i]))
                    else:
                        # replace the archive member that is closest to best
                        dists = np.linalg.norm(archive - self.best_x, axis=1)
                        closest_idx = np.argmin(dists)
                        archive[closest_idx] = pop[i]
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update memory with weighted means
            if S_CR:
                w = np.array(S_df) / np.sum(S_df)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Enhanced Local Search with restart ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)
            no_improve_count = 0
            max_no_improve = max(5, int(0.1 * local_budget))

            while evals < local_budget:
                improved = False
                # Phase 1: Coordinate descent
                dim_order = np.random.permutation(dim)
                for j in dim_order:
                    if evals >= local_budget:
                        break
                    # positive direction
                    cand = x_best.copy()
                    cand[j] += step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.2, max_step[j])
                        improved = True
                        no_improve_count = 0
                        continue
                    # negative direction
                    cand = x_best.copy()
                    cand[j] -= step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.2, max_step[j])
                        improved = True
                        no_improve_count = 0
                    else:
                        step[j] = max(step[j] * 0.5, min_step[j])

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
                        no_improve_count = 0
                    else:
                        step = np.maximum(step * 0.9, min_step)

                if not improved:
                    no_improve_count += 1
                    # Restart if stagnation
                    if no_improve_count >= max_no_improve:
                        x_best = np.random.uniform(lb, ub)
                        f_best = func(x_best)
                        evals += 1
                        step = 0.05 * (ub - lb)
                        no_improve_count = 0
                        if f_best < self.best_f:
                            self.best_f = f_best
                            self.best_x = x_best.copy()
                else:
                    no_improve_count = 0

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

                if np.all(step <= min_step * 2):
                    break

        return self.best_f, self.best_x