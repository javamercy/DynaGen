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
        N_min = max(4, dim // 5)
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
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Diversity threshold for restart
        diversity_threshold = 0.01 * np.mean(ub - lb)

        # Covariance-adaptive pattern search (local search)
        def covariance_pattern_search(best_pos, best_val, step_sizes, evo_path, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step = step_sizes.copy()
            path = evo_path.copy()
            n_used = 0
            success_count = 0
            for iteration in range(dim * 5):
                if n_used >= max_local_evals:
                    break
                improved = False
                # Try positive and negative directions for each dimension, but scale by step
                for d in range(dim):
                    if n_used >= max_local_evals:
                        break
                    # Positive
                    new_pos = pos.copy()
                    delta = step[d] * (ub[d] - lb[d])
                    new_pos[d] = np.clip(pos[d] + delta, lb[d], ub[d])
                    new_val = func(new_pos)
                    n_used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        path[d] = 0.9 * path[d] + 0.1 * delta  # update evolution path
                        step[d] = step[d] * (1.0 + 0.2)  # expand
                        step[d] = min(step[d], 0.5)
                        success_count += 1
                        continue
                    # Negative
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - delta, lb[d], ub[d])
                    new_val = func(new_pos)
                    n_used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        path[d] = 0.9 * path[d] - 0.1 * delta
                        step[d] = step[d] * (1.0 + 0.2)
                        step[d] = min(step[d], 0.5)
                        success_count += 1
                    else:
                        step[d] = step[d] * 0.9
                        step[d] = max(step[d], 1e-12)
                if not improved:
                    break
                # Pattern move along cumulative path
                move = np.clip(pos - best_pos + path, lb - pos, ub - pos)
                if np.any(np.abs(move) > 1e-12):
                    new_pos = np.clip(pos + move, lb, ub)
                    new_val = func(new_pos)
                    n_used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        path *= 0.9
            return pos, val, n_used, step, path

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
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using covariance-adaptive pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Initialize step sizes (relative) and evolution path
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                step_sizes = np.ones(dim) * step
                evo_path = np.zeros(dim)
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used, step_sizes, evo_path = covariance_pattern_search(
                    best_pos, best_val, step_sizes, evo_path, max_local)
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

            # Diversity-based partial restart
            if n_evals < max_evals * 0.9:
                # Compute average pairwise distance (sample)
                if N > 1:
                    sample_idx = np.random.choice(N, min(N, 50), replace=False)
                    if len(sample_idx) > 1:
                        distances = np.linalg.norm(pop[sample_idx] - pop[sample_idx[0]], axis=1)
                        mean_dist = np.mean(distances)
                        if mean_dist < diversity_threshold:
                            # Restart 50% of population (except best)
                            num_replace = max(1, N // 2)
                            best_idx = np.argmin(fitness)
                            worst_indices = np.argsort(fitness)[-num_replace:]
                            for idx in worst_indices:
                                if idx == best_idx:
                                    continue
                                pop[idx] = lb + np.random.rand(dim) * (ub - lb)
                                fitness[idx] = func(pop[idx])
                                n_evals += 1
                                if fitness[idx] < self.f_opt:
                                    self.f_opt = fitness[idx]
                                    self.x_opt = pop[idx].copy()
                                    evals_no_improve = 0

            # Restart if stagnation detected
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Quasi-random Latin hypercube
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
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
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
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
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt