import numpy as np

class RefinedLSHADE:
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

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Diversity metric
        last_diversity_restart = -1e9
        diversity_restart_interval = max_evals // 5

        def random_direction_pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * np.mean(ub - lb)  # scalar step
            used = 0
            # Generate a set of orthogonal directions (random rotation)
            # Use Gram-Schmidt on random vectors
            dirs = np.random.randn(dim, dim)
            dirs, _ = np.linalg.qr(dirs)  # orthonormal columns
            dir_idx = 0
            while used < max_local_evals and dir_idx < dim * 3:
                # Choose a direction from the set
                d = dirs[:, dir_idx % dim]
                # Try positive and negative
                improved = False
                for sign in [1, -1]:
                    if used >= max_local_evals:
                        break
                    new_pos = np.clip(pos + sign * step_size * d, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        # Pattern move: continue along this direction
                        step_size *= 1.2
                        step_size = min(step_size, 0.5 * np.mean(ub - lb))
                        # Try an extrapolation step
                        if used < max_local_evals:
                            ext_pos = np.clip(pos + sign * step_size * d, lb, ub)
                            ext_val = func(ext_pos)
                            used += 1
                            if ext_val < val:
                                pos = ext_pos
                                val = ext_val
                        break  # break inner loop after success
                if improved:
                    dir_idx = 0  # reset direction index to explore all directions again
                else:
                    dir_idx += 1
                    step_size *= 0.85  # shrink step on failure
                if step_size < 1e-12 * np.mean(ub - lb):
                    break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Choose mutation strategy: occasionally use rand/1 for diversity
                if np.random.rand() < 0.1 and n_evals < 0.7 * max_evals:
                    # rand/1/bin
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    if archive.size > 0:
                        union = np.vstack((pop, archive))
                    else:
                        union = pop
                    r_arc = np.random.randint(union.shape[0])
                    # Sample F and CR
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                else:
                    # current-to-pbest/1/archive
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
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: reflect and clamp
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
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        # Remove the archive individual most similar to current population
                        # Compute distances to nearest neighbor in pop for all archive members
                        if archive.shape[0] > 0:
                            min_dists = np.min([np.linalg.norm(archive - p, axis=1) for p in pop], axis=0)
                            remove_idx = np.argmin(min_dists)  # remove most similar (least diversity)
                            archive = np.delete(archive, remove_idx, axis=0)
                else:
                    # If trial is better than some worst in archive, we could add it? Not needed.
                    pass

            # Update population and fitness
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

            # Population size reduction (quadratic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    # Remove archive members that are most similar to the retained population
                    if archive.shape[0] > 0:
                        min_dists = np.min([np.linalg.norm(archive - p, axis=1) for p in pop], axis=0)
                        # Keep those with largest min distance (most diverse)
                        keep_idx = np.argsort(min_dists)[-archive_max:]
                        archive = archive[keep_idx]
                N = N_new

            # Periodic local refinement using random-direction pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Adaptive step size based on remaining evals and dimensionality
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = random_direction_pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity check: if population has collapsed, restart even without stagnation
            if n_evals - last_diversity_restart > diversity_restart_interval and n_evals < max_evals * 0.7:
                # Compute average distance from best
                best_idx = np.argmin(fitness)
                avg_dist = np.mean([np.linalg.norm(p - pop[best_idx]) for p in pop])
                if avg_dist < 0.01 * np.linalg.norm(ub - lb):
                    # Population too concentrated, trigger restart
                    last_diversity_restart = n_evals
                    evals_no_improve = restart_threshold  # force restart condition
                    # We'll let the restart block handle it

            # Restart if stagnation detected or diversity triggered
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8) or \
               (n_evals - last_diversity_restart < diversity_restart_interval and evals_no_improve > 0.5 * restart_threshold):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population: 20% from Gaussian around best, 80% uniform random
                new_pop = np.empty((new_N, dim))
                new_fitness = np.full(new_N, np.inf)
                # Place best at position 0
                new_pop[0] = best_ind
                new_fitness[0] = best_fit
                # Gaussian samples around best with covariance from current population
                if N >= dim:
                    cov = np.cov(pop.T)
                    # Regularize covariance
                    cov += 1e-10 * np.eye(dim)
                else:
                    cov = 0.1 * np.eye(dim)
                num_gauss = max(1, int(new_N * 0.2))
                gauss_samples = np.random.multivariate_normal(best_ind, cov, size=num_gauss)
                gauss_samples = np.clip(gauss_samples, lb, ub)
                for j in range(min(num_gauss, new_N)):
                    if j == 0:
                        continue
                    new_pop[j] = gauss_samples[j]
                # Uniform random for rest
                uniform_samples = lb + np.random.uniform(0, 1, (new_N - num_gauss, dim)) * (ub - lb)
                for j in range(num_gauss, new_N):
                    new_pop[j] = uniform_samples[j - num_gauss]
                # Evaluate new population
                for j in range(1, new_N):
                    new_fitness[j] = func(new_pop[j])
                    n_evals += 1
                    if new_fitness[j] < self.f_opt:
                        self.f_opt = new_fitness[j]
                        self.x_opt = new_pop[j].copy()
                pop = new_pop
                fitness = new_fitness
                N = new_N
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                last_diversity_restart = n_evals

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt