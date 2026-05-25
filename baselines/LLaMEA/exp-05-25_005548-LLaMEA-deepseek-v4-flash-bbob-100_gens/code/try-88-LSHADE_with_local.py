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

        # Allocate budget: main DE and local search (20% for local)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            x = np.random.uniform(lb, ub)
            self.best_f = func(x)
            self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization (or Sobol-like) ----
        NP_init = max(10, min(100, 5 * dim))
        NP = NP_init

        def lhs(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = 2 * NP          # larger archive for diversity
        H = 60                        # increased memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main jSO-inspired DE loop ----
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
                max_archive = 2 * NP

            # Adaptive pbest ratio (quadratic schedule)
            ratio = 0.25 * (1 - (1 - remaining_evals / main_budget)**2) + 0.05
            p = max(0.05, min(0.25, ratio))
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

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

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

            if fevals >= main_budget:
                break

            # Update memory (Lehmer mean for F, arithmetic for CR)
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

            # Restart if population diversity is low
            pop_std = np.std(pop, axis=0).mean()
            range_len = np.mean(ub - lb)
            if pop_std < 0.05 * range_len and remaining_evals > 0.2 * main_budget:
                # Replace worst half with LHS around best
                n_replace = max(1, NP // 2)
                worst_idx = np.argsort(fitness)[-n_replace:]
                new_pts = lhs(n_replace, dim, lb, ub)
                for k, idx in enumerate(worst_idx):
                    cand = new_pts[k]
                    f_cand = func(cand)
                    fevals += 1
                    pop[idx] = cand
                    fitness[idx] = f_cand
                    if f_cand < self.best_f:
                        self.best_f = f_cand
                        self.best_x = cand.copy()
                    if fevals >= main_budget:
                        break

        # ---- Local Search: Nelder-Mead with restarts ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            # Initial simplex size
            step = 0.1 * (ub - lb)
            while evals < local_budget:
                # Build simplex
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_best
                for i in range(dim):
                    simplex[i+1] = x_best.copy()
                    simplex[i+1][i] += step[i]
                    simplex[i+1] = np.clip(simplex[i+1], lb, ub)
                f_simplex = np.array([f_best] + [func(simplex[i]) for i in range(1, dim+1)])
                evals += dim
                # Evaluate first point again? To be consistent: we count only new evals.
                f_simplex[0] = f_best  # already known

                nm_evals = 0
                while evals < local_budget and nm_evals < 50 * dim:
                    # Ensure simplex sorted (worst first for Nelder-Mead standard)
                    idx = np.argsort(f_simplex)
                    simplex = simplex[idx]
                    f_simplex = f_simplex[idx]
                    x_bar = np.mean(simplex[:-1], axis=0)  # centroid of best points
                    x_r = x_bar + (x_bar - simplex[-1])    # reflection
                    x_r = np.clip(x_r, lb, ub)
                    f_r = func(x_r)
                    evals += 1
                    nm_evals += 1
                    if f_r < f_simplex[-1] and f_r >= f_simplex[0]:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                    elif f_r < f_simplex[0]:
                        # expansion
                        x_e = x_bar + 2.0 * (x_r - x_bar)
                        x_e = np.clip(x_e, lb, ub)
                        f_e = func(x_e)
                        evals += 1
                        if f_e < f_r:
                            simplex[-1] = x_e
                            f_simplex[-1] = f_e
                        else:
                            simplex[-1] = x_r
                            f_simplex[-1] = f_r
                    else:
                        # contraction
                        x_c = x_bar + 0.5 * (simplex[-1] - x_bar)
                        x_c = np.clip(x_c, lb, ub)
                        f_c = func(x_c)
                        evals += 1
                        if f_c < f_simplex[-1]:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # shrink
                            for i in range(1, dim+1):
                                simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                            for i in range(1, dim+1):
                                f_simplex[i] = func(simplex[i])
                                evals += 1
                                if evals >= local_budget:
                                    break
                    # update best
                    best_local_idx = np.argmin(f_simplex)
                    if f_simplex[best_local_idx] < f_best:
                        f_best = f_simplex[best_local_idx]
                        x_best = simplex[best_local_idx].copy()
                        # restart if significant improvement
                        step = np.minimum(step * 1.2, 0.3 * (ub - lb))
                    if evals >= local_budget:
                        break
                # Restart Nelder-Mead with smaller step near best
                step = np.maximum(step * 0.5, 1e-5 * (ub - lb))
                # Update global best
                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()
                # Break if step too small or budget exhausted
                if np.all(step < 1e-6 * (ub - lb)) or evals >= local_budget:
                    break

        return self.best_f, self.best_x