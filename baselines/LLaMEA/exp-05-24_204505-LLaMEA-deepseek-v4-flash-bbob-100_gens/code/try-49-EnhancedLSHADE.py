import numpy as np

class EnhancedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        max_evals = self.budget

        # Population size parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization
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

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0

        # Covariance estimation for CMA-like sampling
        n_top = max(int(0.3 * N), 2)

        def update_covariance(pop, fitness, n_top):
            idx = np.argsort(fitness)[:n_top]
            top = pop[idx] - pop[idx].mean(axis=0)
            cov = top.T @ top / max(1, n_top - 1)
            # Regularize
            cov += 1e-12 * np.eye(dim)
            return cov

        # Calculate initial covariance
        C = update_covariance(pop, fitness, n_top)

        # Pattern search with directional enhancement
        def pattern_search(best_pos, best_val, step, max_local_evals, cov):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative step
            iterations = 0
            used = 0
            # Precompute random directions for later use
            num_rands = min(dim * 2, max_local_evals // 2)
            rand_dirs = np.random.randn(num_rands, dim)
            rand_dirs = rand_dirs / np.linalg.norm(rand_dirs, axis=1, keepdims=True)
            rnd_idx = 0
            while used < max_local_evals and iterations < dim * 6:
                iterations += 1
                improved = False
                # Coordinate directions
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                # Random directions (from covariance eigenvectors or spherical)
                if not improved and used < max_local_evals:
                    for _ in range(min(dim, max_local_evals - used)):
                        if rnd_idx >= num_rands:
                            break
                        dir = rand_dirs[rnd_idx]
                        rnd_idx += 1
                        # Positive step
                        new_pos = np.clip(pos + step_size * dir, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                            break
                        # Negative step
                        new_pos = np.clip(pos - step_size * dir, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                            break
                if improved:
                    # Pattern move
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Update covariance for CMA sampling
            cov = update_covariance(pop, fitness, n_top)

            # Generate offspring
            for i in range(N):
                # Decide whether to use CMA or DE
                use_cma = np.random.rand() < (0.2 * (1 - n_evals / max_evals))  # gradually reduce
                if use_cma:
                    # Sample from multivariate normal around best
                    sigma = 0.1 * (1 - n_evals / max_evals) + 0.01
                    best_idx = np.argmin(fitness)
                    best_vec = pop[best_idx]
                    # Generate trial via Gaussian
                    try:
                        delta = np.random.multivariate_normal(np.zeros(dim), sigma * cov)
                    except:
                        delta = np.random.normal(0, sigma * 0.1, dim)
                    trial = np.clip(best_vec + delta, lb, ub)
                    CR = 0.5  # not used but keep for memory?
                    F = 0.5
                else:
                    # Standard LSHADE DE
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1 = np.random.choice(idxs)
                    if archive.size > 0:
                        union = np.vstack((pop, archive))
                    else:
                        union = pop
                    r2 = np.random.randint(union.shape[0])
                    pbest_size = max(1, int(p * N))
                    sorted_idx = np.argsort(fitness)
                    pbest_candidates = sorted_idx[:pbest_size]
                    pbest_idx = np.random.choice(pbest_candidates)
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                    # Binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                    # Boundary handling
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
                    # For DE, store F/CR; for CMA, don't update memory (or use placeholder)
                    if not use_cma:
                        S_F.append(F)
                        S_CR.append(CR)
                        delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update DE memory (only if DE trials contributed)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (cubic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new
                # Update top count for covariance
                n_top = max(int(0.3 * N), 2)

            # Periodic local refinement
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                # Use current covariance to guide pattern search directions (passed but not used now)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local, cov)
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

            # Restart if stagnation
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Generate new population around best with covariance scaling
                    pop = np.empty((new_N, dim))
                    pop[0] = best_ind
                    # Sample remaining from multivariate normal centered at best
                    sigma_restart = 0.2 * (1 - n_evals / max_evals) + 0.02
                    for j in range(1, new_N):
                        try:
                            pop[j] = np.clip(best_ind + np.random.multivariate_normal(np.zeros(dim), sigma_restart * cov), lb, ub)
                        except:
                            pop[j] = np.clip(best_ind + sigma_restart * np.random.normal(0, 0.1, dim), lb, ub)
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: replace all but best with samples from best + noise
                    pop[0] = best_ind
                    sigma_restart = 0.2 * (1 - n_evals / max_evals) + 0.02
                    for j in range(1, N):
                        try:
                            pop[j] = np.clip(best_ind + np.random.multivariate_normal(np.zeros(dim), sigma_restart * cov), lb, ub)
                        except:
                            pop[j] = np.clip(best_ind + sigma_restart * np.random.normal(0, 0.1, dim), lb, ub)
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Update covariance
                cov = update_covariance(pop, fitness, max(int(0.3 * N), 2))

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt