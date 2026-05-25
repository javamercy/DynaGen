import numpy as np

class LSHADE_enhanced_local:
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

        # allocate budget: main DE with restarts and final local search
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

        total_de_evals = main_budget
        used_evals = 0
        restart_budget = total_de_evals
        restart_count = 0

        # track best ever
        best_ever_f = np.inf
        best_ever_x = None

        while restart_budget > 0 and restart_count < 6:
            # each restart uses a fraction of the remaining budget
            cur_budget = int(min(restart_budget, max(200*dim, total_de_evals // (restart_count+1))))
            if cur_budget < 30:
                break
            restart_budget -= cur_budget
            restart_count += 1

            # Latin Hypercube Initialization
            NP_init = max(10, min(200, 20 * int(np.log(dim)) if dim > 1 else 20))
            NP = NP_init

            def lhs(n, d, low, high):
                result = np.zeros((n, d))
                for i in range(d):
                    perm = np.random.permutation(n)
                    result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
                return result

            pop = lhs(NP, dim, lb, ub)
            # if not first restart, inject best ever if found
            if best_ever_x is not None:
                pop[0] = best_ever_x.copy()
            fitness = np.array([func(x) for x in pop])
            used_evals += NP

            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_ever_f:
                best_ever_f = fitness[best_idx]
                best_ever_x = pop[best_idx].copy()
                if best_ever_f < self.best_f:
                    self.best_f = best_ever_f
                    self.best_x = best_ever_x.copy()

            archive = np.empty((0, dim))
            max_archive = NP
            H = 30
            M_CR = 0.5 * np.ones(H)
            M_F = 0.5 * np.ones(H)
            mem_idx = 0
            fevals = NP
            stall_evals = 0
            best_f_old = np.inf

            # DE phase for this restart
            while fevals < cur_budget:
                remaining_evals = cur_budget - fevals
                # linear population reduction
                NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / cur_budget)))
                if NP_new < NP:
                    sorted_idx = np.argsort(fitness)
                    pop = pop[sorted_idx[:NP_new]]
                    fitness = fitness[sorted_idx[:NP_new]]
                    NP = NP_new
                    if len(archive) > NP:
                        np.random.shuffle(archive)
                        archive = archive[:NP]
                    max_archive = NP

                # adaptive pbest ratio (jSO style)
                ratio = 0.25 - 0.20 * (1 - remaining_evals / cur_budget)
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
                        if idx < NP:
                            if idx != i and idx != r1:
                                break
                        else:
                            break
                    r2_vec = combined[idx]

                    # mutation: current-to-pbest/1 with archive
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                    u = pop[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]

                    # reflected boundary handling
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
                        if f_u < best_ever_f:
                            best_ever_f = f_u
                            best_ever_x = u.copy()
                            if best_ever_f < self.best_f:
                                self.best_f = best_ever_f
                                self.best_x = best_ever_x.copy()

                    if fevals >= cur_budget:
                        break

                pop = new_pop
                fitness = new_fitness

                if fevals >= cur_budget:
                    break

                # update memory
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

                # check stagnation for restart within restart (optional)
                if fevals % (10*dim) == 0:
                    if abs(best_ever_f - best_f_old) < 1e-12:
                        stall_evals += fevals
                    else:
                        stall_evals = 0
                    best_f_old = best_ever_f
                    if stall_evals > 0.1 * cur_budget:
                        break  # early restart trigger

            # after DE phase of this restart, if remaining budget in the overall local budget, do a quick local search on best
            if used_evals < main_budget * 0.8:
                # use a small portion for local refinement here (optional)
                pass

        # ---- Final Enhanced Local Search (adaptive random basis pattern search with Cauchy perturbations) ----
        if local_budget > 0:
            x_best = best_ever_x.copy() if best_ever_x is not None else np.random.uniform(lb, ub)
            f_best = self.best_f
            evals = 0
            step = 0.1 * (ub - lb)
            min_step = 1e-8 * (ub - lb)
            max_step = 0.25 * (ub - lb)
            basis = np.eye(dim)

            while evals < local_budget:
                improved = False
                # coordinate descent along current basis
                for j in range(dim):
                    if evals >= local_budget:
                        break
                    cand = x_best + step[j] * basis[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.3, max_step[j])
                        improved = True
                        cand2 = x_best + step[j] * basis[j]
                        cand2 = np.clip(cand2, lb, ub)
                        if evals < local_budget:
                            f2 = func(cand2)
                            evals += 1
                            if f2 < f_best:
                                x_best, f_best = cand2, f2
                                step[j] = min(step[j] * 1.3, max_step[j])
                        continue
                    cand = x_best - step[j] * basis[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[j] = min(step[j] * 1.3, max_step[j])
                        improved = True
                        cand2 = x_best - step[j] * basis[j]
                        cand2 = np.clip(cand2, lb, ub)
                        if evals < local_budget:
                            f2 = func(cand2)
                            evals += 1
                            if f2 < f_best:
                                x_best, f_best = cand2, f2
                                step[j] = min(step[j] * 1.3, max_step[j])
                    else:
                        step[j] = max(step[j] * 0.5, min_step[j])

                if evals >= local_budget:
                    break

                # random direction sampling with basis rotation
                if np.random.rand() < 0.3:
                    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                    basis = Q.T

                num_rand = max(1, min(int(0.2 * (local_budget - evals)), 5))
                for _ in range(num_rand):
                    if evals >= local_budget:
                        break
                    idx_dir = np.random.randint(dim)
                    s = np.mean(step)
                    cand = x_best + s * basis[idx_dir]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step = np.minimum(step * 1.3, max_step)
                        improved = True
                    else:
                        step = np.maximum(step * 0.9, min_step)

                if not improved:
                    step = np.maximum(step * 0.8, min_step)
                    if np.all(step <= min_step * 2):
                        break
                else:
                    step = np.minimum(step * 1.1, max_step)

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

            # remaining budget: Cauchy random perturbations
            if evals < local_budget:
                while evals < local_budget:
                    scale = np.max(step) * (1 - evals / local_budget)
                    if scale < 1e-8:
                        break
                    # heavy-tailed Cauchy
                    noise = np.random.standard_cauchy(size=dim) * scale * 0.5
                    noise = np.clip(noise, -5*(ub-lb), 5*(ub-lb))
                    cand = x_best + noise
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step = np.minimum(step * 1.2, max_step)
                    else:
                        step = np.maximum(step * 0.9, min_step)
                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

        # final check
        if best_ever_f < self.best_f:
            self.best_f = best_ever_f
            self.best_x = best_ever_x.copy()

        return self.best_f, self.best_x