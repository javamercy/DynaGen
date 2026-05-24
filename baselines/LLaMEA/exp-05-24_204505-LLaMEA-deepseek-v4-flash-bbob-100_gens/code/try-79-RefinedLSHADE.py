import numpy as np

class RefinedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dim = self.dim
        max_evals = self.budget
        budget_ratio = max_evals / 10000  # scale parameters for different budgets

        # Population size parameters (scaled for low-budget)
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization (improved)
        segments = np.linspace(0, 1, N + 1)
        intervals = np.random.uniform(segments[:-1], segments[1:], (N, dim))
        perm = np.array([np.random.permutation(N) for _ in range(dim)]).T
        samples = np.zeros((N, dim))
        for i in range(N):
            for j in range(dim):
                samples[i, j] = intervals[perm[i, j], j]
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation (larger size)
        archive = np.empty((0, dim))
        archive_max = int(2.5 * N)

        # Success-history memory for F and CR (H=10, as original)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection: track best fitness history
        best_fitness_hist = [self.f_opt]
        stagnation_counter = 0
        stagnation_threshold = max(50 * dim, int(0.1 * max_evals))

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0
        pattern_step = 0.1  # initial relative step

        # Local pattern search with random directions (adaptive step)
        def pattern_search(best_pos, best_val, step_size, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            step = step_size * (ub - lb)  # relative step
            used = 0
            while used < max_evals_local:
                improved = False
                # Random direction pattern
                d = np.random.randn(dim)
                d = d / (np.linalg.norm(d) + 1e-30)
                # Positive direction
                new_pos = np.clip(pos + step * d, lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    improved = True
                else:
                    # Negative direction
                    new_pos = np.clip(pos - step * d, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: continue in the same direction
                    for _ in range(2):  # at most 2 extra steps
                        new_pos = np.clip(pos + step * d, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                        else:
                            break
                    # Expand step on success
                    step *= 1.2
                    step = np.minimum(step, (ub - lb) * 0.5)
                else:
                    # Contract step on failure
                    step *= 0.85
                    if np.max(step) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05 (faster decay)
            p = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            success_count = 0

            # Generate offspring
            for i in range(N):
                # Decide mutation strategy: 80% current-to-pbest/1, 20% current-to-rand/1 (no crossover)
                use_rand = np.random.rand() < 0.2

                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0.0, 1.0)

                if not use_rand:
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
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                    # Binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                else:
                    # current-to-rand/1 (no crossover, rotation invariant)
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[r2] - pop[i]   # different donor
                    mutant = base + F * diff1 + F * diff2
                    trial = mutant  # no crossover, full replacement

                # Boundary handling: reflect + clamp
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
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    success_count += 1
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (if any successes)
            if len(S_F) > 0:
                # Sort by delta_f (largest improvement first)
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order]
                w = w / (np.sum(w) + 1e-30)
                # Compute weighted Lehmer mean for F
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                # For CR, use weighted arithmetic mean (more stable)
                MCR[memory_idx] = np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction (like jSO)
            N_new = int(round(N_init - (N_init - N_min) * (n_evals / max_evals)))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                # Adjust archive max
                archive_max = int(2.5 * N_new)
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using pattern search (only when near optimum?)
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = pattern_step * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        stagnation_counter = 0
                # Replace worst individual
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation detected (no improvement for long time)
            if (stagnation_counter > stagnation_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Generate new population via Latin hypercube
                    segments = np.linspace(0, 1, new_N + 1)
                    intervals = np.random.uniform(segments[:-1], segments[1:], (new_N, dim))
                    perm = np.array([np.random.permutation(new_N) for _ in range(dim)]).T
                    samples = np.zeros((new_N, dim))
                    for i in range(new_N):
                        for j in range(dim):
                            samples[i, j] = intervals[perm[i, j], j]
                    pop = lb + samples * (ub - lb)
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: randomize all but the best
                    for i in range(1, N):
                        pop[i] = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                        fitness[i] = func(pop[i])
                        n_evals += 1
                        if fitness[i] < self.f_opt:
                            self.f_opt = fitness[i]
                            self.x_opt = pop[i].copy()
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = int(2.5 * N)
                stagnation_counter = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt