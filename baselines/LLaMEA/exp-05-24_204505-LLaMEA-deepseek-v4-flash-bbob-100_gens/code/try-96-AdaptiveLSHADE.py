import numpy as np

class AdaptiveLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        max_evals = self.budget

        # Population size: start with 10*dim, minimum 4
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, dim // 5)
        N = N_init

        # Latin hypercube initialization
        rng = np.random.RandomState(42)  # fixed seed for reproducibility (can be removed)
        samples = rng.uniform(0, 1, (N, dim))
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive (FIFO) for DE mutation
        archive = []
        archive_max = N

        # Success-history memory for F and CR
        H = 30
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals
        restart_count = 0
        max_restarts = 3

        # Local search parameters
        local_search_interval = max(25, int(0.012 * max_evals))
        last_local_search = 0

        # Rotated pattern search with adaptive step size
        def rotated_pattern_search(best_pos, best_val, step_scale, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step = step_scale * (ub - lb)
            used = 0
            # Generate random orthonormal directions (Gram-Schmidt)
            dirs = rng.randn(dim, dim)
            for k in range(dim):
                for l in range(k):
                    dirs[:, k] -= np.dot(dirs[:, k], dirs[:, l]) * dirs[:, l]
                dirs[:, k] /= np.linalg.norm(dirs[:, k])
            iter_count = 0
            while used < max_local_evals and iter_count < dim * 8:
                iter_count += 1
                improved = False
                # Test all directions (positive and negative) in random order
                order = rng.permutation(dim)
                for d in order:
                    if used >= max_local_evals:
                        break
                    # Positive direction
                    trial = np.clip(pos + step * dirs[:, d], lb, ub)
                    ft = func(trial)
                    used += 1
                    if ft < val:
                        pos = trial
                        val = ft
                        improved = True
                        continue
                    # Negative direction
                    trial = np.clip(pos - step * dirs[:, d], lb, ub)
                    ft = func(trial)
                    used += 1
                    if ft < val:
                        pos = trial
                        val = ft
                        improved = True
                if improved:
                    # Pattern move: extrapolate along successful direction difference
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        trial = np.clip(pos + delta, lb, ub)
                        ft = func(trial)
                        used += 1
                        if ft < val:
                            pos = trial
                            val = ft
                    step *= 1.15  # expand on success
                    step = np.clip(step, 1e-10 * (ub - lb), 0.3 * (ub - lb))
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step *= 0.75  # contract on failure
                    if np.max(step) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: exponential decay
            p = 0.2 * np.exp(-4.0 * n_evals / max_evals) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Select r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = rng.choice(idxs)
                # r2 from union of population and archive
                union_pop = pop
                if archive:
                    union = np.vstack((pop, np.array(archive)))
                else:
                    union = pop
                r2 = rng.randint(union.shape[0])
                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = rng.choice(pbest_candidates)
                # Sample F and CR from memory
                mem = rng.randint(H)
                F = np.clip(MF[mem] + 0.1 * rng.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * rng.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * rng.randn(), 0, 1)
                # Mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = rng.randint(dim)
                trial = np.where(rng.rand(dim) < CR, mutant, base)
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
                    # Add parent to archive (FIFO)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (inverse order: larger improvement first)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction: quadratic schedule
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
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

            # Local refinement using rotated pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01  # adaptive step scale
                max_local = min(dim * 4, max_evals - n_evals - 10)
                new_pos, new_val, used = rotated_pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual if improved
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation (up to 3 restarts)
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8 and restart_count < max_restarts):
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Reinitialize with Latin hypercube, keep best
                    samples = rng.uniform(0, 1, (new_N, dim))
                    pop = lb + samples * (ub - lb)
                    pop[0] = best_ind
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
                    # Partial restart: randomize all but best
                    pop = lb + rng.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory with slight variation
                MF[:] = 0.5 + 0.2 * rng.rand(H)
                MCR[:] = 0.8 + 0.2 * rng.rand(H)
                memory_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt