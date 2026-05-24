import numpy as np

class LSHADE_ASLR:
    """
    Adaptive Strategy Selection and Local Refinement for LSHADE:
    - Dual mutation: current-to-pbest/1 with archive and rand/1 (selected by success rates)
    - Quasi-random Sobol sequences for initialization and restarts
    - Enhanced pattern search with per-dimension adaptive step sizes
    - Diversity-driven restart based on mean pairwise distance
    """
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def _sobol(self, n, d):
        """Generate Sobol quasi-random sequence scaled to [0,1]."""
        # Minimal Sobol generator using predefined direction numbers (d <= 5)
        if d > 5:
            return np.random.uniform(0, 1, (n, d))  # fallback
        # Use a simple base-2 Sobol implementation for small d
        max_n = 2**12
        if n > max_n:
            return np.random.uniform(0, 1, (n, d))
        sobol = np.zeros((max_n, d))
        for j in range(d):
            v = np.zeros(60, dtype=np.uint64)
            # Direction numbers from Joe and Kuo (2008)
            dir_nums = {
                1: [1, 3, 5, 15, 17, 51, 85, 255, 257, 771, 1285, 3855, 4369, 13107, 21845, 65535],
                2: [1, 1, 7, 11, 13, 61, 67, 79, 465, 721, 823, 4091, 4433, 5227, 30269, 32609],
                3: [1, 3, 7, 5, 7, 43, 49, 147, 441, 1339, 4009, 4011, 5327, 30269, 49053, 49237],
                4: [1, 1, 5, 3, 15, 51, 125, 189, 369, 2759, 4077, 4639, 6605, 12119, 32033, 40649],
                5: [1, 1, 1, 15, 7, 49, 101, 191, 139, 745, 2999, 5639, 6313, 16211, 23983, 41019]
            }
            vals = dir_nums.get(j+1, [1] + [0]*15)
            for i in range(16):
                v[i] = vals[i]
            # Initialize first values
            X = 0
            for i in range(1, n+1):
                # find index of rightmost zero bit
                k = (i & -i).bit_length()  # lowest zero bit index
                X ^= v[k-1]  # flip bit
                sobol[i-1, j] = X / (2**32)  # approximate
        return sobol[:n]

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        max_evals = self.budget

        # Population size parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Sobol initialization
        samples = self._sobol(N, dim)
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive
        archive = np.empty((0, dim))
        archive_max = N

        # Strategy success memory (0: current-to-pbest, 1: rand/1)
        strategy_success = [0.0, 0.0]
        strategy_count = [1, 1]
        strategy_prob = [0.5, 0.5]

        # Memory for F, CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters
        local_search_interval = max(40, int(0.03 * max_evals))
        last_local_search = 0

        # Pattern search with per-dimension adaptive step
        def enhanced_pattern_search(best_pos, best_val, steps, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            used = 0
            max_iter = min(dim * 4, max_local_evals // 2)
            for _ in range(max_iter):
                if used >= max_local_evals:
                    break
                improved = False
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # try both directions
                    for sign in [1, -1]:
                        new_pos = pos.copy()
                        new_pos[d] = np.clip(pos[d] + sign * steps[d], lb[d], ub[d])
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                            steps[d] *= 1.2  # expand on success
                            break
                    else:
                        steps[d] *= 0.5  # contract on failure
                if improved:
                    # pattern move: accelerate along direction of net change
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            # also expand steps along delta direction
                            steps = np.maximum(steps, np.abs(delta)*0.1)
                    best_pos = pos
                    best_val = val
                # clamp steps
                steps = np.clip(steps, 1e-12, (ub-lb)*0.2)
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            S_F = []
            S_CR = []
            delta_f = []
            strat_used = []

            # Generate offspring
            for i in range(N):
                # Choose strategy based on success probabilities
                strategy = np.random.choice(2, p=strategy_prob)
                strat_used.append(strategy)

                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                base = pop[i]

                if strategy == 0:  # current-to-pbest/1 with archive
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
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:  # rand/1
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: random reinitialization for out-of-bounds coordinates
                out = (trial < lb) | (trial > ub)
                if np.any(out):
                    # reinitialize out-of-bounds dimensions with random values in [lb, ub]
                    trial[out] = lb[out] + np.random.uniform(0, 1, size=np.sum(out)) * (ub - ub)[out]  # fix: ub-lb
                    trial[out] = lb[out] + np.random.uniform(0, 1, size=np.sum(out)) * (ub[out] - lb[out])
                # Also clip to ensure numeric safety
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
                    fitness[i] = trial_f
                    pop[i] = trial
                    # add parent to archive
                    archive = np.vstack((archive, base.reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                    # update strategy success
                    strategy_success[strategy] += 1
                strategy_count[strategy] += 1

            # Update strategy probabilities (softmax of success rates)
            if np.sum(strategy_count) > 0:
                success_rate = np.array(strategy_success) / np.maximum(np.array(strategy_count), 1)
                # use exp to amplify differences
                strategy_prob = np.exp(success_rate) / np.sum(np.exp(success_rate))
                # avoid extremes
                strategy_prob = np.clip(strategy_prob, 0.1, 0.9)
                strategy_prob[0] = 1 - strategy_prob[1]  # normalize

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (quadratic)
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

            # Local refinement (pattern search) on best solution
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # per-dimension steps proportional to range and remaining budget
                step_scale = 0.1 * (1 - n_evals / max_evals) + 0.02
                steps = (ub - lb) * step_scale * np.ones(dim)
                max_local = min(dim * 3, max_evals - n_evals - 10)
                new_pos, new_val, used = enhanced_pattern_search(best_pos, best_val, steps, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual if improvement
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity-driven restart
            if n_evals < max_evals * 0.8:
                # compute mean pairwise distance (sample-based)
                if N > 1:
                    sample = pop[np.random.choice(N, min(50, N), replace=False)]
                    mean_dist = np.mean([np.linalg.norm(sample[i]-sample[j]) for i in range(len(sample)) for j in range(i+1, min(5, len(sample)))])
                else:
                    mean_dist = 0
                diversity_low = mean_dist < 0.01 * np.linalg.norm(ub-lb)
                if (evals_no_improve > restart_threshold) or (diversity_low and evals_no_improve > 0.05 * max_evals):
                    # Restart: keep best, reinitialize others with Sobol
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    remaining = max_evals - n_evals
                    new_N = min(N_init * 2, N * 2, remaining // 2)
                    new_N = max(new_N, N_min)
                    if new_N > N:
                        samples = self._sobol(new_N, dim)
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
                        # partial restart: keep best, rest random
                        pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                        pop[0] = best_ind
                        for j in range(1, N):
                            fitness[j] = func(pop[j])
                            n_evals += 1
                            if fitness[j] < self.f_opt:
                                self.f_opt = fitness[j]
                                self.x_opt = pop[j].copy()
                    # reset memory and archive
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    archive = np.empty((0, dim))
                    archive_max = N
                    evals_no_improve = 0
                    # reset strategy tracking
                    strategy_success = [0.0, 0.0]
                    strategy_count = [1, 1]
                    strategy_prob = [0.5, 0.5]

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt