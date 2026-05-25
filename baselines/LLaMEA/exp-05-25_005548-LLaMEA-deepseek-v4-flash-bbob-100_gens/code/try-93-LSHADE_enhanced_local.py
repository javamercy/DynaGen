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

        # allocate budget: main DE and local search
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization ----
        NP_init = max(10, min(200, int(20 * np.sqrt(dim)) if dim > 1 else 20))
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
        max_archive = NP
        H = 30
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Multi-strategy success memory ----
        # strategy 0: current-to-pbest/1 (with archive)
        # strategy 1: current-to-rand/1 (diversity)
        strat_success = [1.0, 1.0]  # smoothed success rates
        strat_counts = [1, 1]
        strat_prob = 0.5  # probability of using strategy 0

        # ---- Stagnation tracking ----
        no_improve_gens = 0
        best_ever = self.best_f

        # ---- Main jSO-inspired DE loop with multi-strategy ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # linear population reduction
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

            # adaptive pbest ratio (jSO style)
            ratio = 0.25 - 0.20 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = [[], []]  # per strategy
            S_F = [[], []]
            S_df = [[], []]

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            strat_success_this_gen = [0, 0]
            strat_trials_this_gen = [0, 0]

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy for CR
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Cauchy for F
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # choose mutation strategy adaptively
                if np.random.rand() < strat_prob:
                    strat = 0  # current-to-pbest/1
                else:
                    strat = 1  # current-to-rand/1

                if strat == 0:
                    # pbest selection
                    pbest = pop[np.random.choice(pbest_pool)]
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    # archive selection
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx < NP:
                            if idx != i and idx != r1:
                                break
                        else:
                            # archive index always different from pop indices
                            break
                    r2_vec = combined[idx]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                else:
                    # current-to-rand/1 (no archive, purely random distinct indices)
                    r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(NP)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = np.random.randint(NP)
                    v = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

                # binomial crossover
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

                strat_trials_this_gen[strat] += 1
                if f_u <= fitness[i]:
                    S_CR[strat].append(CR)
                    S_F[strat].append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_df[strat].append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                    strat_success_this_gen[strat] += 1

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            # update strategy probabilities using smoothed success rates
            for s in range(2):
                if strat_trials_this_gen[s] > 0:
                    rate = strat_success_this_gen[s] / strat_trials_this_gen[s]
                    # exponential moving average
                    strat_success[s] = 0.8 * strat_success[s] + 0.2 * rate
                    strat_counts[s] += 1
            total = strat_success[0] + strat_success[1]
            if total > 0:
                strat_prob = strat_success[0] / total
                strat_prob = max(0.1, min(0.9, strat_prob))

            if fevals >= main_budget:
                break

            # update memory with Lehmer mean for F, arithmetic for CR
            for s in range(2):
                if S_CR[s]:
                    w = np.array(S_df[s]) / np.sum(S_df[s])
                    mean_CR = np.sum(w * np.array(S_CR[s]))
                    F_arr = np.array(S_F[s])
                    sum_w = np.sum(w * F_arr)
                    sum_w_sq = np.sum(w * F_arr ** 2)
                    mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                    M_CR[mem_idx] = mean_CR
                    M_F[mem_idx] = mean_F
                    mem_idx = (mem_idx + 1) % H

            # ---- Stagnation detection and restart ----
            if self.best_f < best_ever - 1e-12:
                best_ever = self.best_f
                no_improve_gens = 0
            else:
                no_improve_gens += 1

            if no_improve_gens >= max(20, int(0.05 * NP_init)) and fevals < main_budget * 0.8:
                # replace worst 50% of population with new LHS samples (keep best)
                keep_num = max(2, NP // 2)
                sorted_idx = np.argsort(fitness)
                best_kept = pop[sorted_idx[:keep_num]]
                best_fitness_kept = fitness[sorted_idx[:keep_num]]
                new_num = NP - keep_num
                new_individuals = lhs(new_num, dim, lb, ub)
                new_fitness_vals = np.array([func(x) for x in new_individuals])
                fevals += new_num
                pop = np.vstack((best_kept, new_individuals))
                fitness = np.concatenate((best_fitness_kept, new_fitness_vals))
                # Reset memory to avoid old bias
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                # Reset stagnation counter
                no_improve_gens = 0
                # Shrink archive
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                # Reset strategy success rates
                strat_success = [1.0, 1.0]
                strat_counts = [1, 1]
                strat_prob = 0.5
                # After restart, reduce pbest ratio to encourage exploration
                # Keep the best solution global
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()
                if fevals >= main_budget:
                    break

        # ---- Enhanced Local Search (Adaptive Step Size with 1/5 Rule) ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            # Initial step size based on population spread or typical distance
            if len(pop) > 1:
                step = np.std(pop, axis=0) * 0.1
                step = np.clip(step, 1e-4, 0.5*(ub-lb))
            else:
                step = 0.05 * (ub - lb)
            min_step = 1e-6 * (ub - lb)
            max_step = 0.2 * (ub - lb)
            # For 1/5 rule: maintain success history for each dimension
            L = 10  # window length for success fraction
            success_history = np.zeros((dim, L))
            step_index = 0

            basis = np.eye(dim)

            while evals < local_budget:
                improved = False

                # Phase 1: coordinate descent along current basis directions with 1/5 rule
                for j in range(dim):
                    if evals >= local_budget:
                        break
                    # Positive direction
                    cand = x_best + step[j] * basis[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    success = 0
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        improved = True
                        success = 1
                    success_history[j, step_index % L] = success
                    # Also try negative direction if positive not improved? Actually pattern search does both, but here we do them separately.
                    # negative direction
                    cand = x_best - step[j] * basis[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        improved = True
                        success = 1
                    else:
                        success = 0
                    # Update success history for the negative move? We'll treat each move independently: we can just record success for this dimension collectively.
                    # Simpler: record success if any move in that direction gave improvement.
                    # Actually we want the success fraction to decide step size per dimension.
                    # We'll maintain a separate success tracker per dimension, counting successful moves out of total moves (positive+negative) in the window.
                    # Let's define a per-dimension success count.
                    # Instead, we'll update step size after a fixed number of moves (like after each generation of directions).
                # Instead of per-move update, we'll update after each full scan of all dimensions.
                # Use 1/5 rule: if fraction of successful moves in the last L moves per dimension > 0.2, increase step, else decrease.
                for j in range(dim):
                    # For simplicity, count successes in the last L moves across both directions.
                    # But we only have one move per dimension per "scan". Instead, we'll aggregate across scans.
                    # We'll use a global success counter for each dimension that tracks how many times moving in that direction (either sign) improved.
                    # But we need a window. Let's use a simpler method: after each scan, update step size via the global success rate of whole local search (like 1/5 rule for overall step).
                    # That is simpler and works reasonably well.
                    pass

                # We'll use a simpler global step size adaptation: track global success rate over last 10 moves.
                # Since we have many moves, we can just use the overall improvement rate.

                # Actually, the 1/5 rule works well with a single step size. But for per-dimension, we can adjust individually based on success along that dimension.
                # Let's implement a per-dimension exponential smoothing: success_rate = 0.9*success_rate + 0.1*(success_flag)
                # Then if success_rate > 0.2, increase step, else decrease.
                # But we need to compute success_flag for each dimension per move. We'll do that.

                # Reset per-dimension success flags for this scan
                successes_this_scan = np.zeros(dim)
                for j in range(dim):
                    if evals >= local_budget:
                        break
                    # Positive direction
                    cand = x_best + step[j] * basis[j]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        improved = True
                        successes_this_scan[j] += 1
                        # continue to negative? not necessary
                    else:
                        # negative direction
                        cand = x_best - step[j] * basis[j]
                        cand = np.clip(cand, lb, ub)
                        f_cand = func(cand)
                        evals += 1
                        if f_cand < f_best:
                            x_best, f_best = cand, f_cand
                            improved = True
                            successes_this_scan[j] += 1
                # Update step sizes based on success rate (1/5 rule, per dimension)
                for j in range(dim):
                    if successes_this_scan[j] > 0.2 * 2:  # more than 0.4 success out of 2 moves -> increase
                        step[j] = min(step[j] * 1.2, max_step[j])
                    else:
                        step[j] = max(step[j] * 0.8, min_step[j])

                if evals >= local_budget:
                    break

                # Phase 2: random direction sampling with basis rotation
                if np.random.rand() < 0.3:
                    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                    basis = Q.T

                num_rand = max(1, min(int(0.2 * (local_budget - evals)), 5))
                for _ in range(num_rand):
                    if evals >= local_budget:
                        break
                    idx_dir = np.random.randint(dim)
                    s = step[idx_dir]  # use the specific step for that direction
                    cand = x_best + s * basis[idx_dir]
                    cand = np.clip(cand, lb, ub)
                    f_cand = func(cand)
                    evals += 1
                    if f_cand < f_best:
                        x_best, f_best = cand, f_cand
                        step[idx_dir] = min(step[idx_dir] * 1.2, max_step[idx_dir])
                        improved = True
                    else:
                        step[idx_dir] = max(step[idx_dir] * 0.9, min_step[idx_dir])

                if not improved:
                    step = np.maximum(step * 0.9, min_step)
                    if np.all(step <= min_step * 2):
                        break
                else:
                    step = np.minimum(step * 1.1, max_step)

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

            # Final random perturbations with remaining budget
            if evals < local_budget:
                while evals < local_budget:
                    scale = np.max(step) * (1 - evals / local_budget)
                    if scale < 1e-8:
                        break
                    noise = np.random.normal(0, scale, dim)
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

        return self.best_f, self.best_x