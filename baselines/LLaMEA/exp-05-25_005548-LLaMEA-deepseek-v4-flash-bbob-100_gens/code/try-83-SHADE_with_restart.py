import numpy as np

class SHADE_with_restart:
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

        # Budget split: majority for DE, remainder for local search and restarts
        local_budget = max(10 * dim, int(0.12 * budget))
        main_budget = budget - local_budget
        # Reserve a small fraction for potential restarts within DE
        restart_budget_frac = 0.08
        de_budget = int(main_budget * (1 - restart_budget_frac))
        restart_budget = main_budget - de_budget

        if de_budget < 20 * dim:
            # Fallback: random search if budget too small
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Opposition-based Latin hypercube initialization ----
        NP_init = max(10, 20 * int(np.log(dim)) if dim > 1 else 20)
        NP = NP_init

        def lhs(n, d, low, high):
            pts = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                pts[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return pts

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        # Opposition: generate opposite points and keep best N
        opp_pop = lb + ub - pop
        opp_fitness = np.array([func(x) for x in opp_pop])
        fevals += NP
        all_pop = np.vstack((pop, opp_pop))
        all_fit = np.concatenate((fitness, opp_fitness))
        idx = np.argsort(all_fit)[:NP]
        pop = all_pop[idx]
        fitness = all_fit[idx]

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP
        H = 30
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation tracking
        stall_evals = 0
        last_improve_evals = fevals
        best_f_previous = self.best_f

        # ---- Main DE loop with linear population reduction and restart ----
        while fevals < de_budget:
            remaining_evals = de_budget - fevals
            # Linear population size reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / de_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # pbest ratio: decreasing to focus exploitation
            ratio = 0.25 - 0.20 * (1 - remaining_evals / de_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Stagnation detection: if no improvement for 0.05*de_budget evals, restart
            if fevals - last_improve_evals > int(0.05 * de_budget) and fevals < de_budget - 100:
                # Restart: reinitialize part of population around best
                num_restart = NP // 2
                new_pts = lhs(num_restart, dim, lb, ub) * 0.2 + self.best_x * 0.8  # local perturbation
                new_pts = np.clip(new_pts, lb, ub)
                new_fit = np.array([func(x) for x in new_pts])
                fevals += num_restart
                # Keep best current individuals
                keep = NP - num_restart
                idx_keep = np.argsort(fitness)[:keep]
                pop = np.vstack((pop[idx_keep], new_pts))
                fitness = np.concatenate((fitness[idx_keep], new_fit))
                # Reset memory
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                archive = np.empty((0, dim))
                last_improve_evals = fevals
                continue

            S_CR = []
            S_F = []
            S_df = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy for CR and F
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

                combined = np.vstack((pop, archive))
                while True:
                    idx_ar = np.random.randint(len(combined))
                    if idx_ar == i or idx_ar == r1:
                        continue
                    break
                r2_vec = combined[idx_ar]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Boundary handling: reflection + random if still out
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
                        last_improve_evals = fevals

                if fevals >= de_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= de_budget:
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

            # Update best if no improvement in this generation
            if self.best_f >= best_f_previous:
                stall_evals += NP
            else:
                stall_evals = 0
                best_f_previous = self.best_f

        # ---- Local search: Adaptive pattern search (coordinate + random) ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)

            dim_order = list(range(dim))

            while evals < local_budget:
                improved = False
                # Coordinate descent
                np.random.shuffle(dim_order)
                for j in dim_order:
                    if evals >= local_budget:
                        break
                    cand = x_best.copy()
                    cand[j] += step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.15, max_step[j])
                        improved = True
                        continue
                    cand = x_best.copy()
                    cand[j] -= step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.15, max_step[j])
                        improved = True
                    else:
                        step[j] = max(step[j] * 0.5, min_step[j])

                if evals >= local_budget:
                    break

                # Random directions
                num_random = max(1, int(0.4 * (local_budget - evals)))
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
                        step = np.minimum(step * 1.15, max_step)
                        improved = True
                    else:
                        step = np.maximum(step * 0.85, min_step)

                if not improved:
                    step = np.minimum(step * 1.5, max_step)

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

                if np.all(step <= min_step * 2):
                    break

        return self.best_f, self.best_x