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

        # Reserve budget for local search
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube sampling
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
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

        # Archive and memories
        archive = np.empty((0, dim))
        max_archive = 2 * NP  # increased from NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart tracking
        last_improvement_eval = 0
        restart_threshold = int(0.15 * main_budget)
        initial_NP = NP

        while fevals < main_budget:
            # Remaining evaluations
            remaining_evals = main_budget - fevals

            # Linear population reduction
            NP_new = max(4, int(4 + (initial_NP - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > 2 * NP:
                    np.random.shuffle(archive)
                    archive = archive[:2 * NP]
                max_archive = 2 * NP

            # Adaptive pbest ratio: 0.2 -> 0.05
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

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

                # pbest with power‑law selection (give more weight to top)
                idx_rank = np.random.randint(0, pbest_num)
                # convert rank to index using exponential distribution
                pbest_idx = pbest_pool[idx_rank]
                pbest = pop[pbest_idx]

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
                    delta_fitness.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improvement_eval = fevals

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            # Restart if stagnation
            if fevals - last_improvement_eval > restart_threshold and fevals + local_budget < budget:
                # Keep best 20% and replace the rest with random points
                sorted_idx = np.argsort(fitness)
                keep_num = max(1, int(0.2 * NP))
                kept_pop = pop[sorted_idx[:keep_num]]
                kept_fit = fitness[sorted_idx[:keep_num]]
                # Generate new individuals (excluding best)
                new_inds = lhs(NP - keep_num, dim, lb, ub)
                new_fit = np.array([func(x) for x in new_inds])
                fevals += (NP - keep_num)
                # Combine
                pop = np.vstack((kept_pop, new_inds))
                fitness = np.concatenate((kept_fit, new_fit))
                # Reset archive and memory
                archive = np.empty((0, dim))
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                # Update best
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()
                last_improvement_eval = fevals

            # Update memory for successful parameters
            if S_CR and len(S_CR) > 0:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            if fevals >= main_budget:
                break

        # ---- Hooke-Jeeves pattern search local search ----
        if local_budget > 0:
            x0 = self.best_x.copy()
            f0 = self.best_f
            step = max(0.001, 0.02 * (ub - lb).mean())
            step_min = 1e-7

            evals = 0
            x_best = x0.copy()
            f_best = f0

            while evals < local_budget:
                improved = False
                # Exploratory moves: positive and negative directions
                for i in range(dim):
                    # positive direction
                    x_pos = x_best.copy()
                    x_pos[i] += step
                    x_pos = np.clip(x_pos, lb, ub)
                    f_pos = func(x_pos)
                    evals += 1
                    if f_pos < f_best:
                        f_best = f_pos
                        x_best = x_pos.copy()
                        improved = True
                        continue
                    # negative direction
                    x_neg = x_best.copy()
                    x_neg[i] -= step
                    x_neg = np.clip(x_neg, lb, ub)
                    f_neg = func(x_neg)
                    evals += 1
                    if f_neg < f_best:
                        f_best = f_neg
                        x_best = x_neg.copy()
                        improved = True

                if improved:
                    # pattern move: accelerate along the direction of improvement
                    x_pattern = 2 * x_best - x0
                    x_pattern = np.clip(x_pattern, lb, ub)
                    f_pattern = func(x_pattern)
                    evals += 1
                    if f_pattern < f_best:
                        f_best = f_pattern
                        x_best = x_pattern.copy()
                    else:
                        # shrink step size
                        step *= 0.5
                else:
                    # no improvement: shrink step size
                    step *= 0.5

                # Update starting point for next cycle
                x0 = x_best.copy()

                if step < step_min:
                    break

                if evals >= local_budget:
                    break

            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best.copy()

        return self.best_f, self.best_x