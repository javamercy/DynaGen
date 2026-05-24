import numpy as np

class RefinedLSHADE2:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        max_evals = self.budget

        # Population size parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization (simple)
        samples = np.random.uniform(0, 1, (N, dim))
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation (FIFO)
        archive = []
        archive_max = N

        # Success-history memory for F and CR
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Strategy adaptation: prob of using current-to-pbest/1/archive vs current-to-rand/1
        strategy_prob = 0.8
        strategy_success = [0, 0]  # [pbest_success, rand_success]

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.08 * max_evals
        restart_count = 0
        max_restarts = 2

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Pole of diversity measure
        diversity_window = 50
        diversity_history = []

        # Local search using covariance-guided sampling
        def covariance_local_search(best_pos, best_val, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            # Use best 50% of population to estimate covariance
            sorted_idx = np.argsort(fitness)
            top_half = pop[sorted_idx[:max(N//2, 2)]]
            if len(top_half) < 2:
                return pos, val, 0
            cov = np.cov(top_half, rowvar=False) + 1e-12 * np.eye(dim)
            # Step size based on spread of top half
            std = np.std(top_half, axis=0)
            step_size = np.minimum(std, (ub - lb) * 0.25)
            # Generate candidates
            candidates = []
            for _ in range(min(10 * dim, max_local_evals)):
                candidate = np.random.multivariate_normal(pos, cov * 0.25)
                candidate = np.clip(candidate, lb, ub)
                candidates.append(candidate)
            # Evaluate in order, stop if improvement found early
            used = 0
            for candidate in candidates:
                if used >= max_local_evals:
                    break
                f_val = func(candidate)
                used += 1
                if f_val < val:
                    pos = candidate
                    val = f_val
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # Adaptive pbest decay: aggressive quadratic decay with diversity factor
            # Compute diversity: mean pairwise distance in population
            if N > 1:
                mean_pos = np.mean(pop, axis=0)
                diversity = np.mean(np.linalg.norm(pop - mean_pos, axis=1))
            else:
                diversity = 0.0
            # Normalize diversity relative to bounds diagonal
            range_diag = np.linalg.norm(ub - lb)
            diversity_ratio = diversity / (range_diag + 1e-12)
            # p: base decay + increase if diversity is low (to promote exploration)
            base_p = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05
            p = base_p * (1.0 + 0.5 * (1.0 - diversity_ratio))
            p = np.clip(p, 0.05, 0.5)

            # Strategy adaptation: update probability based on recent success
            if n_evals > 10:
                total_success = max(strategy_success[0] + strategy_success[1], 1)
                strategy_prob = strategy_success[0] / total_success
                strategy_prob = np.clip(strategy_prob, 0.2, 0.9)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            strategy_success_tmp = [0, 0]

            # Generate offspring
            for i in range(N):
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                if archive:
                    union = np.vstack((pop, np.array(archive)))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest index
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # Decide mutation strategy
                use_pbest = np.random.rand() < strategy_prob
                base = pop[i]
                if use_pbest:
                    # current-to-pbest/1/archive
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    # current-to-rand/1
                    diff1 = pop[r1] - base
                    diff2 = union[r2] - base
                    mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: reflection and clamping
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive (FIFO)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)
                    if use_pbest:
                        strategy_success_tmp[0] += 1
                    else:
                        strategy_success_tmp[1] += 1
                else:
                    if use_pbest:
                        strategy_success_tmp[0] += 0.2  # small reward for trying
                    else:
                        strategy_success_tmp[1] += 0.2

            # Update strategy success rates (exponential smoothing)
            if n_evals > 10:
                alpha = 0.3
                strategy_success[0] = alpha * strategy_success_tmp[0] + (1 - alpha) * strategy_success[0]
                strategy_success[1] = alpha * strategy_success_tmp[1] + (1 - alpha) * strategy_success[1]

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (faster cubic decay)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if len(archive) > archive_max:
                    archive = archive[-archive_max:]
                N = N_new

            # Periodic local refinement using covariance-guided search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 5)
                new_pos, new_val, used = covariance_local_search(best_pos, best_val, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation (allow up to 2 restarts)
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8 and restart_count < max_restarts):
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Quasi-random Latin hypercube using Sobol-like (simple uniform)
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                else:
                    samples = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                # Insert best individual and evaluate
                pop = np.zeros((new_N, dim))
                pop[0] = best_ind
                fitness = np.full(new_N, np.inf)
                fitness[0] = best_fit
                for j in range(1, new_N):
                    pop[j] = samples[j]
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory with slight perturbations
                MF[:] = 0.5 + 0.2 * np.random.rand(H)
                MCR[:] = 0.8 + 0.2 * np.random.rand(H)
                memory_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt