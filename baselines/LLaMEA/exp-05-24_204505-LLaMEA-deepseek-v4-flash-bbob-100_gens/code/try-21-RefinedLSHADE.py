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

        # Latin hypercube initialization (improved version with shuffling)
        samples = np.random.uniform(0, 1, (N, dim))
        for j in range(dim):
            samples[:, j] = (np.random.permutation(N) + np.random.uniform(0, 1, N)) / N
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation (larger capacity)
        archive = np.empty((0, dim))
        archive_max = 2 * N  # increased archive size

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.1 * max_evals  # reduced threshold

        # Local search parameters (adaptive frequency)
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0

        # Helper: diversity measure (average distance to best)
        def diversity(population, best):
            if len(population) < 2:
                return 0.0
            dists = np.sqrt(np.sum((population - best) ** 2, axis=1))
            return float(np.mean(dists))

        # Pattern search with adaptive step (improved direction exploitation)
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            iterations = 0
            used = 0
            last_improved_direction = np.zeros(dim)
            while used < max_local_evals and iterations < dim * 6:
                iterations += 1
                improved = False
                # Coordinate search with pattern move
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
                        last_improved_direction[d] = 1.0
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
                        last_improved_direction[d] = -1.0
                        improved = True
                if improved:
                    # Pattern move along the accumulated direction
                    if np.any(np.abs(last_improved_direction) > 0):
                        delta = last_improved_direction * step_size
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size on success
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size on failure
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
                # Reset direction vector if no improvement in last full cycle
                if not improved:
                    last_improved_direction.fill(0.0)
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: smoother decay from 0.2 to 0.05
            r = n_evals / max_evals
            p = 0.2 * (1 - r ** 1.5) + 0.05 * (r ** 2)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

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
                # Sample F and CR
                mem = np.random.randint(H)
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
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (improved with sorted deltas)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (exponential-like schedule)
            r = (max_evals - n_evals) / max_evals
            N_new = N_min + (N_init - N_min) * (r ** 2)  # quadratic schedule (kept for compatibility)
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = 2 * N_new  # maintain archive proportion
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using pattern search (adaptive interval)
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = 0.2 * (1 - n_evals / max_evals) + 0.01
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

            # Restart based on stagnation OR low diversity
            diversity_val = diversity(pop, pop[np.argmin(fitness)])
            diversity_threshold = 0.02 * np.linalg.norm(ub - lb)
            restart_condition = (evals_no_improve > restart_threshold) or \
                                (diversity_val < diversity_threshold and n_evals < max_evals * 0.7)
            if restart_condition and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Improved restart: Latin hypercube around best with scaling
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    for j in range(dim):
                        samples[:, j] = (np.random.permutation(new_N) + np.random.uniform(0, 1, new_N)) / new_N
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
                    # Partial restart: randomize all but best
                    samples = np.random.uniform(0, 1, (N, dim))
                    for j in range(dim):
                        samples[:, j] = (np.random.permutation(N) + np.random.uniform(0, 1, N)) / N
                    pop = lb + samples * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory parameters with a mix of old and new
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt