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

        # Latin hypercube initialization (shuffled for better coverage)
        samples = np.zeros((N, dim))
        for d in range(dim):
            perm = np.random.permutation(N)
            samples[:, d] = (perm + np.random.uniform(0, 1, N)) / N
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = 2 * N

        # Success-history memory for F and CR (circular, larger)
        H = 12
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0
        # Track success counts per memory slot for adaptive sampling
        success_counts = np.ones(H)
        use_counts = np.ones(H)

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals  # earlier restart

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0
        # Direction memory for pattern search (gradient-like)
        direction_memory = np.zeros(dim)

        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative per dimension
            # Use a vector of step sizes that can adapt per dimension
            dim_step = step_size.copy()
            iterations = 0
            used = 0
            while used < max_local_evals and iterations < dim * 5:
                iterations += 1
                improved = False
                # Try the remembered direction first (exploit gradient)
                if np.any(np.abs(direction_memory) > 1e-12):
                    new_pos = np.clip(pos + direction_memory * 0.5, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                # Coordinate search with adaptive per-dimension steps
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + dim_step[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        # Expand step for this dimension
                        dim_step[d] = min(dim_step[d] * 1.5, (ub[d] - lb[d]) * 0.4)
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - dim_step[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        dim_step[d] = min(dim_step[d] * 1.5, (ub[d] - lb[d]) * 0.4)
                    else:
                        # contract step for this dimension
                        dim_step[d] = max(dim_step[d] * 0.7, 1e-12 * (ub[d] - lb[d]))
                if improved:
                    # pattern move: accelerate along net improvement direction
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        # Re-evaluate the best position after coordinate search
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                        # Update direction memory
                        direction_memory = delta
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # No improvement: try a random expansion
                    if iterations % 3 == 0:
                        rand_dir = np.random.uniform(-1, 1, dim)
                        rand_dir = rand_dir / np.linalg.norm(rand_dir) * np.mean(dim_step)
                        new_pos = np.clip(pos + rand_dir, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                    if not improved:
                        # shrink all steps if no success
                        dim_step *= 0.5
                        if np.max(dim_step) < 1e-10 * np.max(ub - lb):
                            break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05 (more aggressive)
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            # Store indices of successful generations for memory update weighting
            mem_weights = []

            # Generate offspring
            for i in range(N):
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest index
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # Sample F and CR with memory slot selection based on success rate
                # Weight selection probability by success counts
                prob = success_counts / (use_counts + 1e-30)
                prob = prob / prob.sum()
                mem = np.random.choice(H, p=prob)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # Mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
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
                    mem_weights.append(mem)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                # Update use_counts for the selected memory slot
                use_counts[mem] += 1

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means, also track success counts per slot
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                # Update the specific memory slot used by each successful individual
                # Actually update the whole memory? Original LSHADE updates one slot.
                # We'll update memory_idx as before but also increment success_counts for used slots
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H
                # Increment success counts for each memory slot that contributed
                unique_mems, counts = np.unique(mem_weights, return_counts=True)
                for m, c in zip(unique_mems, counts):
                    success_counts[m] += c

            # Population size reduction using sigmoid schedule (smoother)
            frac = n_evals / max_evals
            # Sigmoid from 0.5 to 0.95 to allow slower reduction early
            sig = 1 / (1 + np.exp(-10 * (frac - 0.5)))
            N_new = N_min + (N_init - N_min) * (1 - sig)
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = int(2 * N_new)  # adjust archive size
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using pattern search with direction memory
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals (more refined later)
                step = 0.15 * (1 - n_evals / max_evals) ** 0.5 + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
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

            # Restart if stagnation detected or if diversity is low
            # Compute diversity (average distance to best)
            best_idx = np.argmin(fitness)
            dists = np.linalg.norm(pop - pop[best_idx], axis=1)
            diversity = np.mean(dists) / np.linalg.norm(ub - lb) if np.linalg.norm(ub - lb) > 0 else 0
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.85) or diversity < 1e-6:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Latin hypercube with Sobol-like shuffling (better space-filling)
                    samples = np.zeros((new_N, dim))
                    for d in range(dim):
                        perm = np.random.permutation(new_N)
                        samples[:, d] = (perm + np.random.uniform(0, 1, new_N)) / new_N
                    pop = lb + samples * (ub - lb)
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    # Fill rest with random evaluations
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: randomize all but best
                    samples = np.zeros((N, dim))
                    for d in range(dim):
                        perm = np.random.permutation(N)
                        samples[:, d] = (perm + np.random.uniform(0, 1, N)) / N
                    pop = lb + samples * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory parameters
                MF[:] = 0.5
                MCR[:] = 0.5
                success_counts[:] = 1
                use_counts[:] = 1
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0
                direction_memory = np.zeros(dim)

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt