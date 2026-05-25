import numpy as np

class iLSHADE_improved:
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

        # Budget split: main DE + local search + restart reserve
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = int(0.75 * budget) - local_budget
        restart_reserve = budget - main_budget - local_budget
        if restart_reserve < 0:
            restart_reserve = 0
            main_budget = int(0.85 * budget)
            local_budget = budget - main_budget

        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization ----
        NP_init = max(10, min(int(10 * dim), 50))
        NP = NP_init
        NP_min = 4

        def lhs(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = lhs(NP_init, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP_init

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP_init
        H = 15  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0
        stagnation_counter = 0
        best_f_prev = self.best_f

        # ---- Main DE loop ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Exponential population reduction
            ratio = remaining_evals / main_budget
            NP_new = max(NP_min, int(NP_min + (NP_init - NP_min) * (ratio ** 1.5)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (jSO style)
            p_ratio = 0.2 + 0.2 * (1 - ratio)
            p = max(0.05, min(0.3, p_ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Adaptation of CR and F per individual
                r = np.random.randint(H)
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
                r2_vec = combined[idx]

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

            # Update memory with weighted Lehmer mean for F, weighted arithmetic mean for CR
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

            # Stagnation detection: if no improvement in 5% of main budget, inject diversity
            if self.best_f < best_f_prev - 1e-12:
                best_f_prev = self.best_f
                stagnation_counter = 0
            else:
                stagnation_counter += NP
            if stagnation_counter > 0.05 * main_budget:
                # Replace worst 30% with random points
                num_replace = max(1, int(0.3 * NP))
                worst_idx = np.argsort(fitness)[-num_replace:]
                for idx in worst_idx:
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                stagnation_counter = 0

        # ---- Two-phase Local Search ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0

            # Phase 1: Adaptive Coordinate Descent with per-dimension step sizes
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)
            dim_order = np.arange(dim)
            success_rate = 0.5

            while evals < local_budget // 2:
                improved = False
                np.random.shuffle(dim_order)
                for j in dim_order:
                    if evals >= local_budget // 2:
                        break
                    # Positive direction
                    cand = x_best.copy()
                    cand[j] += step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.2, max_step[j])
                        improved = True
                        continue
                    # Negative direction
                    cand = x_best.copy()
                    cand[j] -= step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.2, max_step[j])
                        improved = True
                    else:
                        step[j] = max(step[j] * 0.5, min_step[j])

                if not improved:
                    break

            # Phase 2: DE-based local search (small population) for non-separable refinement
            if evals < local_budget:
                ls_pop_size = max(3, min(10, dim))
                ls_pop = np.array([x_best + 0.01 * np.random.randn(dim) * (ub - lb) for _ in range(ls_pop_size)])
                ls_pop = np.clip(ls_pop, lb, ub)
                ls_fit = np.array([func(ind) for ind in ls_pop])
                evals += ls_pop_size
                # Update best
                best_ls_idx = np.argmin(ls_fit)
                if ls_fit[best_ls_idx] < f_best:
                    x_best, f_best = ls_pop[best_ls_idx].copy(), ls_fit[best_ls_idx]

                # Run a few generations of DE/best/1 with small population
                while evals < local_budget:
                    for i in range(ls_pop_size):
                        if evals >= local_budget:
                            break
                        # Mutation: DE/best/1
                        r1, r2 = np.random.choice(ls_pop_size, 2, replace=False)
                        F_local = 0.6 + 0.2 * np.random.rand()
                        v = x_best + F_local * (ls_pop[r1] - ls_pop[r2])
                        # Crossover with probability CR_local = 0.9
                        u = ls_pop[i].copy()
                        j_rand = np.random.randint(dim)
                        for j in range(dim):
                            if np.random.rand() < 0.9 or j == j_rand:
                                u[j] = v[j]
                        u = np.clip(u, lb, ub)
                        f_u = func(u)
                        evals += 1
                        if f_u < ls_fit[i]:
                            ls_pop[i] = u
                            ls_fit[i] = f_u
                            if f_u < f_best:
                                x_best, f_best = u.copy(), f_u
                    # Shrink population if no improvement
                    if evals >= local_budget:
                        break

            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best.copy()

        # ---- Restart with improved initialization if budget remains ----
        if restart_reserve > 0:
            # Use a simple random search around best with gradual shrinking
            for _ in range(restart_reserve):
                alpha = np.random.uniform(0, 0.1)
                cand = self.best_x + alpha * np.random.randn(dim) * (ub - lb)
                cand = np.clip(cand, lb, ub)
                f_cand = func(cand)
                if f_cand < self.best_f:
                    self.best_f = f_cand
                    self.best_x = cand.copy()

        return self.best_f, self.best_x