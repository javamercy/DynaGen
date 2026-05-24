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

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Adaptive mutation strategy probabilities (strategy1: current-to-pbest/1; strategy2: current-to-rand/1)
        success_strategies = [0, 0]  # count of successful offspring for each strategy
        attempts_strategies = [0, 0]
        strategy_prob = 0.8  # probability to use strategy1

        # Per-dimension step size history for pattern search
        step_history = np.ones(dim) * 0.15  # relative step

        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # absolute step
            iterations = 0
            used = 0
            while used < max_local_evals and iterations < dim * 4:
                iterations += 1
                improved = False
                # Coordinate search
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
                        # adapt step size per dimension
                        step_size[d] *= 1.2
                        step_size[d] = min(step_size[d], (ub[d] - lb[d]) * 0.5)
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
                        step_size[d] *= 1.2
                        step_size[d] = min(step_size[d], (ub[d] - lb[d]) * 0.5)
                    else:
                        step_size[d] *= 0.9  # shrink slightly if no improvement in that dimension
                if improved:
                    # Pattern move: accelerate along direction of improvement
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        # Line search along delta with increasing step
                        scale = 1.0
                        for _ in range(3):  # try multiples
                            if used >= max_local_evals:
                                break
                            new_pos = np.clip(best_pos + scale * delta, lb, ub)
                            new_val = func(new_pos)
                            used += 1
                            if new_val < val:
                                pos = new_pos
                                val = new_val
                                scale *= 1.5
                            else:
                                break
                    # Expand step size on success
                    step_size *= 1.1
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size on failure (global)
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[], []]  # for each strategy
            S_CR = [[], []]
            delta_f = [[], []]

            # Generate offspring
            for i in range(N):
                # Decide which strategy to use
                if np.random.rand() < strategy_prob:
                    strat = 0  # strategy1: current-to-pbest/1 with archive
                else:
                    strat = 1  # strategy2: current-to-rand/1 (rotation-invariant)
                attempts_strategies[strat] += 1

                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                if strat == 0:
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
                    # Mutation: current-to-pbest/1
                    diff1 = pop[pbest_idx] - pop[i]
                    diff2 = pop[r1] - union[r2]
                    mutant = pop[i] + F * diff1 + F * diff2
                    # Binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                    trial[j_rand] = mutant[j_rand]
                else:  # strategy2: current-to-rand/1 (rotation-invariant, no crossover)
                    r2 = np.random.randint(N)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(N)
                    # Sample F (no CR needed)
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    # Mutation: current-to-rand/1 (no crossover)
                    diff1 = pop[r1] - pop[i]
                    diff2 = pop[r2] - pop[i]
                    mutant = pop[i] + F * diff1 + F * diff2
                    # Rotation-invariant, so just use mutant directly
                    trial = mutant
                    # For simplicity, treat CR as 1 (always use mutant)
                    CR = 1.0

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
                    S_F[strat].append(F)
                    S_CR[strat].append(CR)
                    delta_f[strat].append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    if strat == 0:
                        # Add parent to archive
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                        if archive.shape[0] > archive_max:
                            remove_idx = np.random.randint(archive.shape[0])
                            archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means per strategy (only use strategy 0 for MF/MCR)
            if len(S_F[0]) > 0:
                sorted_order = np.argsort(delta_f[0])[::-1]
                S_F0 = np.array(S_F[0])[sorted_order]
                S_CR0 = np.array(S_CR[0])[sorted_order]
                w = np.array(delta_f[0])[sorted_order] / (np.sum(delta_f[0]) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F0 ** 2) / (np.sum(w * S_F0) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR0 ** 2) / (np.sum(w * S_CR0) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Update strategy probability based on success rates (over a sliding window)
            total_attempts = sum(attempts_strategies)
            if total_attempts > 20:
                total_success = sum(success_strategies)
                if total_success > 0:
                    # Adaptive probability: more weight to successful strategy
                    strat_success_rate = [success_strategies[0] / max(attempts_strategies[0], 1),
                                          success_strategies[1] / max(attempts_strategies[1], 1)]
                    # Adjust probability to be between 0.4 and 0.9
                    strategy_prob = 0.5 + 0.4 * (strat_success_rate[0] - strat_success_rate[1])
                    strategy_prob = np.clip(strategy_prob, 0.2, 0.9)
                # Reset counters
                success_strategies = [0, 0]
                attempts_strategies = [0, 0]

            # Population size reduction (linear schedule)
            N_new = N_min + (N_init - N_min) * (1 - (n_evals / max_evals))
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

            # Periodic local refinement using pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
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

            # Restart if stagnation detected
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.75):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Generate new orthogonal Latin hypercube
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    # Keep best
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    # Random rest for the rest
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: keep top 30% individuals
                    sorted_idx = np.argsort(fitness)
                    keep_count = max(1, int(N * 0.3))
                    kept_pop = pop[sorted_idx[:keep_count]]
                    kept_fit = fitness[sorted_idx[:keep_count]]
                    # Fill with new random points
                    num_new = N - keep_count
                    new_pts = lb + np.random.uniform(0, 1, (num_new, dim)) * (ub - lb)
                    pop = np.vstack((kept_pop, new_pts))
                    for j in range(keep_count, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    # Also re-evaluate kept individuals? Not needed as already evaluated.
                # Reset memory but keep a fraction
                MF[:] = np.clip(0.3 + 0.4 * np.random.rand(H), 0, 1)
                MCR[:] = np.clip(0.3 + 0.5 * np.random.rand(H), 0, 1)
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Boost population size for diversity
                N = min(N * 2, N_init * 3, max_evals - n_evals)
                N = max(N, N_min)

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt