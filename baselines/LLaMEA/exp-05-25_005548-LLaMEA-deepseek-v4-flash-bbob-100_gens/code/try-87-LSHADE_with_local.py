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

        # Budget allocation: main DE and local pattern search
        local_budget = max(20 * dim, int(0.2 * budget))
        main_budget = budget - local_budget
        if main_budget < 10 * dim:
            # not enough budget for proper DE, fallback to simple random search
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization ----
        NP_init = max(10, 6 * dim)
        NP = NP_init

        def lhs(n, d, low, high):
            res = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                res[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return res

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP
        H = 50
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop (jSO style) ----
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
                    idx_keep = np.random.choice(len(archive), min(NP, len(archive)), replace=False)
                    archive = archive[idx_keep]
                max_archive = NP

            # pbest ratio (decreases from 0.2 to 0.05)
            progress = 1.0 - remaining_evals / main_budget
            p = 0.2 - 0.15 * progress
            p = max(0.05, min(0.2, p))
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
                # Generate CR and F from Cauchy distributions
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
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

                # Mutation: current-to-pbest/1 with archive
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
                u = np.clip(u, lb, ub)  # final clamp for safety

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

            if fevals >= main_budget:
                break

            # Update memory with weighted Lehmer mean for F, arithmetic for CR
            if S_CR:
                w = np.array(S_df) / np.sum(S_df)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                wF = w * F_arr
                sum_w = np.sum(wF)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Enhanced Pattern Search Local Optimizer ----
        # Uses coordinate descent and random orthogonal directions with adaptive step sizes
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            # initial step per dimension
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)

            while evals < local_budget:
                improved_flag = False
                # Phase 1: Coordinate descent along each dimension
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
                        improved_flag = True
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
                        improved_flag = True
                    else:
                        step[j] = max(step[j] * 0.85, min_step[j])

                if evals >= local_budget:
                    break

                # Phase 2: Random orthogonal directions (to handle non-separability)
                # Generate a set of 2*dim random unit directions (orthogonalized via QR)
                ndir = min(2 * dim, (local_budget - evals) // 2)
                if ndir < 1:
                    break
                # random matrix and orthogonalize
                R = np.random.randn(dim, dim)
                Q, _ = np.linalg.qr(R)
                # Take first ndir directions (as column vectors)
                dirs = Q[:, :ndir].T  # shape (ndir, dim)
                scale = np.mean(step)
                for dvec in dirs:
                    if evals >= local_budget:
                        break
                    cand = x_best + scale * dvec
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        scale = min(scale * 1.2, np.mean(max_step))
                        improved_flag = True
                    else:
                        # also try negative direction
                        cand = x_best - scale * dvec
                        cand = np.clip(cand, lb, ub)
                        f_cand = func(cand)
                        evals += 1
                        if f_cand < f_best:
                            x_best, f_best = cand, f_cand
                            scale = min(scale * 1.2, np.mean(max_step))
                            improved_flag = True
                        else:
                            scale = max(scale * 0.85, np.mean(min_step))
                    # update all step sizes to reflect the scale change
                    step = np.minimum(step * (scale / np.mean(step) if np.mean(step) > 1e-30 else 1.0), max_step)
                    step = np.maximum(step, min_step)

                if not improved_flag:
                    # if no improvement in full cycle, restart with larger steps to escape
                    step = np.minimum(step * 1.5, max_step)

                # update global best
                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

                # termination: steps too small everywhere
                if np.all(step <= min_step * 2):
                    break

        return self.best_f, self.best_x