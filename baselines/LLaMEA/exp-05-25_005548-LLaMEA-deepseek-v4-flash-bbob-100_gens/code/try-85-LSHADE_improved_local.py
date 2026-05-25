import numpy as np

class LSHADE_improved_local:
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

        # budget split: 80% for DE, 20% for local search
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube initialization
        NP_init = max(10, 20 * (int(np.log(dim)) if dim > 1 else 20))
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

        # archive initially empty
        archive = np.empty((0, dim))
        H = 50  # increased history size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0
        stall_counter = 0  # for soft restart

        # ---- Main DE loop (improved jSO) ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Linear population reduction down to 4 (jSO style)
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # shrink archive to at most NP
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]

            # pbest ratio (exponential decay based on remaining budget)
            ratio = 0.25 - 0.20 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.30, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Rank-based weights for selection (exponential ranking)
            ranks = np.arange(NP) + 1
            weights = np.exp(-0.1 * ranks)  # bias towards better
            weights /= weights.sum()

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

                # pbest selection with exponential weighting (better individuals more likely)
                pbest_idx = np.random.choice(pbest_pool, p=weights[:pbest_num]/weights[:pbest_num].sum())
                pbest = pop[pbest_idx]

                # r1: randomly from population (excluding i)
                indices = list(range(NP))
                indices.remove(i)
                r1 = np.random.choice(indices)

                # r2: from union of population and archive (excluding i and r1, weighted)
                combined = np.vstack((pop, archive))
                comb_weights = np.hstack((weights, np.full(len(archive), 0.5/len(archive) if len(archive)>0 else 0)))
                # ensure i and r1 not selected
                comb_weights[i] = 0.0
                comb_weights[NP + r1] = 0.0 if r1 < len(archive) else comb_weights[r1] = 0.0? Wait careful.
                # Correct: compute combined weights and zero out indices of i and r1
                comb_weights = np.ones(len(combined))
                # zero out indices corresponding to i (in pop) and r1 (could be in pop or archive)
                comb_weights[i] = 0.0
                if r1 < NP:
                    comb_weights[r1] = 0.0
                else:
                    # r1 is in archive? Actually r1 is index in pop, archive indices start at NP
                    pass
                # if r1 is from pop, it's index r1; if archive, it's NP + (r1 - NP?) but r1 was chosen from pop only.
                # So r1 always < NP.
                comb_weights[r1] = 0.0
                if np.sum(comb_weights) == 0:
                    comb_weights[:] = 1.0
                comb_weights /= np.sum(comb_weights)
                r2_idx = np.random.choice(len(combined), p=comb_weights)
                r2_vec = combined[r2_idx]

                # current-to-pbest/1
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
                u[still_high] = np.random.uniform(ub[still_high], lb[still_high])  # correct?

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_df.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # archive insertion: store the replaced individual
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > NP:
                        # remove worst from archive based on fitness? But we don't have fitness for archive. Random.
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stall_counter = 0
                    else:
                        stall_counter += 1

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update memory with weighted Lehmer mean for F and weighted arithmetic mean for CR
            if len(S_CR) > 0:
                w = np.array(S_df) / np.sum(S_df)
                mean_CR = np.dot(w, np.array(S_CR))
                F_arr = np.array(S_F)
                sum_wF = np.dot(w, F_arr)
                sum_wF2 = np.dot(w, F_arr ** 2)
                mean_F = sum_wF2 / sum_wF if sum_wF > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Soft restart if stagnation (no improvement for many generations)
            if stall_counter > 50 * NP_init / np.sqrt(dim):
                # reinitialize 30% of worst individuals with random in hypercube
                num_reinit = max(1, int(0.3 * NP))
                worst_idx = np.argsort(fitness)[-num_reinit:]
                for idx in worst_idx:
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                stall_counter = 0

        # ---- Enhanced Local Search (adaptive coordinate and random pattern) ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = 0.05 * (ub - lb)
            step_min = 1e-7 * (ub - lb)
            step_max = 0.2 * (ub - lb)
            dim_order = list(range(dim))

            while evals < local_budget:
                improved = False
                # Coordinate descent
                np.random.shuffle(dim_order)
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
                        step[j] = min(step[j] * 1.2, step_max[j])
                        improved = True
                        continue
                    # negative direction
                    cand = x_best.copy()
                    cand[j] -= step[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.2, step_max[j])
                        improved = True
                    else:
                        step[j] = max(step[j] * 0.5, step_min[j])

                if evals >= local_budget:
                    break

                # Random direction perturbation (multiple attempts)
                num_rand = max(1, int(0.4 * (local_budget - evals)))
                for _ in range(num_rand):
                    if evals >= local_budget:
                        break
                    dir = np.random.randn(dim)
                    dir /= (np.linalg.norm(dir) + 1e-30)
                    s = np.mean(step)
                    cand = x_best + s * dir
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step = np.minimum(step * 1.2, step_max)
                        improved = True
                    else:
                        step = np.maximum(step * 0.9, step_min)

                if not improved:
                    step = np.minimum(step * 1.5, step_max)

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

                if np.all(step <= step_min * 2):
                    break

        return self.best_f, self.best_x