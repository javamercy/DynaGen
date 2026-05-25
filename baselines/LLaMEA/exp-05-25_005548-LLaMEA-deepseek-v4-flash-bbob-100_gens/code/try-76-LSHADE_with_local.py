import numpy as np
from scipy.stats import qmc

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

        # Budget split: main DE + local search
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

        # ---- Sobol Initialization ----
        NP_init = max(10, 20 * int(np.log(dim)) if dim > 1 else 20)
        NP = NP_init

        # Sobol sequence generator
        sobol = qmc.Sobol(d, scramble=True)
        # Generate points in [0,1]^d and scale to bounds
        pop = qmc.scale(sobol.random(n=NP), lb, ub)
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

        # ---- Main jSO-inspired DE loop with adaptive pbest ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
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

            # Adaptive pbest ratio (jSO style, but more aggressive near end)
            ratio = 0.25 - 0.22 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []  # improvement (delta f) for weighting

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Generate CR from Cauchy with mean M_CR[r] truncated
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Generate F from Cauchy with mean M_F[r], truncated to >0
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

            # Update memory with Lehmer mean for F, arithmetic mean for CR (weighted)
            if S_CR:
                w = np.array(S_df) / np.sum(S_df)
                # CR: weighted arithmetic
                mean_CR = np.sum(w * np.array(S_CR))
                # F: weighted Lehmer mean
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Enhanced Local Search (two-phase) ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)

            dim_order = list(range(dim))
            # Phase 1: Coordinate descent with adaptive step
            while evals < local_budget * 0.6:
                improved = False
                np.random.shuffle(dim_order)
                for j in dim_order:
                    if evals >= local_budget:
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
                    break  # if no improvement in full cycle, move to phase 2

            # Phase 2: Covariance-adapted random perturbations
            if evals < local_budget:
                # Use a simple adaptation of direction distribution (like 1+1-CMA with limited budget)
                # Initialize covariance as identity
                C = np.eye(dim)
                sigma = np.mean(step)
                # Evolution path
                pc = np.zeros(dim)
                cc = 0.1
                c_sigma = 0.1
                # To limit evaluations, we run a fixed number of iterations
                max_iter = min(5, local_budget - evals)  # each iteration uses 2 evals (mut+recomb)
                for _ in range(max_iter):
                    if evals >= local_budget:
                        break
                    # Generate candidate with adapted covariance
                    z = np.random.randn(dim)
                    cand = x_best + sigma * C @ z
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        # Update evolution path and covariance
                        pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc)) * z
                        C = (1 - c_sigma) * C + c_sigma * np.outer(pc, pc)
                        sigma = sigma * 1.2
                        x_best, f_best = cand, f_cand
                    else:
                        sigma = sigma * 0.8
                    # Ensure sigma bounds
                    sigma = max(np.min(min_step), min(np.mean(max_step), sigma))

                # Final coordinate descent refinement if budget left
                while evals < local_budget:
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
                            step[j] = min(step[j] * 1.2, max_step[j])
                        else:
                            step[j] = max(step[j] * 0.5, min_step[j])

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

        return self.best_f, self.best_x