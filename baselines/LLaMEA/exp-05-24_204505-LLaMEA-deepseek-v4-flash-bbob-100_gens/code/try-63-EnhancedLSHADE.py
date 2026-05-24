import numpy as np

class EnhancedLSHADE:
    """Enhanced LSHADE with multi-strategy DE, improved memory adaptation, and hybrid local search."""
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

        # Population size parameters (larger initial)
        N_init = min(max(10 * dim, 80), max_evals // 2)
        N_min = max(4, int(dim / 4))
        N = N_init

        # Sobol-like initialization using Latin hypercube
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

        # Success-history memory for F and CR (two strategy groups: current-to-pbest and rand-to-pbest)
        H = 10
        MF1 = np.ones(H) * 0.5   # for current-to-pbest/1
        MCR1 = np.ones(H) * 0.8
        MF2 = np.ones(H) * 0.5   # for rand-to-pbest/1
        MCR2 = np.ones(H) * 0.8
        memory_idx1 = 0
        memory_idx2 = 0
        # Strategy probabilities
        prob_strat = np.array([0.5, 0.5])  # start equal
        success_strat = np.array([0.0, 0.0])
        fail_strat = np.array([0.0, 0.0])
        strat_counts = np.array([0, 0])

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals
        diversity_record = []

        # Local search parameters
        local_search_interval = max(40, int(0.02 * max_evals))
        last_local_search = 0

        # Pattern search with adaptive step (Hooke-Jeeves style)
        def hooke_jeeves(best_pos, best_val, step, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            while used < max_evals_local:
                improved = False
                # Coordinate search
                for d in range(dim):
                    if used >= max_evals_local:
                        break
                    # positive
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if not improved:
                    # reduce step
                    step_size *= 0.5
                    if np.max(step_size) < 1e-12:
                        break
                else:
                    # pattern move
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # expand step
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                # contract if no improvement in full sweep
                # (already handled)
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio decreasing
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F1, S_CR1, S_delta1 = [], [], []
            S_F2, S_CR2, S_delta2 = [], [], []
            strat_chosen = []

            # Generate offspring
            for i in range(N):
                # Select strategy based on success probabilities
                if np.random.rand() < prob_strat[0] / (prob_strat.sum() + 1e-30):
                    strat = 0  # current-to-pbest/1 with archive
                else:
                    strat = 1  # rand-to-pbest/1
                strat_chosen.append(strat)
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

                # Sample F and CR according to strategy
                if strat == 0:
                    mem = np.random.randint(H)
                    F = np.clip(MF1[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF1[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR1[mem] + 0.1 * np.random.randn(), 0, 1)
                    # current-to-pbest/1
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    mem = np.random.randint(H)
                    F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR2[mem] + 0.1 * np.random.randn(), 0, 1)
                    # rand-to-pbest/1 (no archive difference)
                    base = pop[r1]
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
                    if strat == 0:
                        S_F1.append(F)
                        S_CR1.append(CR)
                        S_delta1.append(fitness[i] - trial_f)
                        success_strat[0] += 1
                    else:
                        S_F2.append(F)
                        S_CR2.append(CR)
                        S_delta2.append(fitness[i] - trial_f)
                        success_strat[1] += 1
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        # Remove random (maintain diversity)
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                else:
                    if strat == 0:
                        fail_strat[0] += 1
                    else:
                        fail_strat[1] += 1

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory for each strategy
            def update_memory(MF, MCR, memory_idx, S_F, S_CR, delta_f):
                if len(S_F) > 0:
                    sorted_order = np.argsort(delta_f)[::-1]
                    S_F = np.array(S_F)[sorted_order]
                    S_CR = np.array(S_CR)[sorted_order]
                    w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                    MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                    MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                    memory_idx = (memory_idx + 1) % H
                return MF, MCR, memory_idx

            MF1, MCR1, memory_idx1 = update_memory(MF1, MCR1, memory_idx1, S_F1, S_CR1, S_delta1)
            MF2, MCR2, memory_idx2 = update_memory(MF2, MCR2, memory_idx2, S_F2, S_CR2, S_delta2)

            # Update strategy probabilities (exponential moving average)
            success_strat += 1e-30
            fail_strat += 1e-30
            for s in range(2):
                prob_strat[s] = 0.9 * prob_strat[s] + 0.1 * (success_strat[s] / (success_strat[s] + fail_strat[s]))
            # Normalize
            prob_strat /= prob_strat.sum()
            # Reset counters periodically
            if n_evals % (max_evals // 10) == 0:
                success_strat[:] = 0
                fail_strat[:] = 0

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

            # Periodic local refinement using Hooke-Jeeves
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # dynamic step size
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = hooke_jeeves(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    # Replace worst individual
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = new_pos
                    fitness[worst_idx] = new_val

            # Stagnation detection with diversity check
            if n_evals > max_evals * 0.1:
                # Compute diversity as average pairwise distance in best 50%
                sorted_idx = np.argsort(fitness)
                halfl = max(2, N // 2)
                best_half = pop[sorted_idx[:halfl]]
                if len(best_half) > 1:
                    centroid = np.mean(best_half, axis=0)
                    diversity = np.mean(np.linalg.norm(best_half - centroid, axis=1))
                    diversity_record.append(diversity)
                else:
                    diversity_record.append(0.0)
                # if diversity is too low and no improvement for too long, restart partially
                if len(diversity_record) > 10:
                    recent_div = np.mean(diversity_record[-10:])
                    norm_div = recent_div / (np.mean(ub - lb) * np.sqrt(dim))
                    if norm_div < 0.005 and evals_no_improve > restart_threshold * 0.6:
                        # Partial restart: keep best 20% and regenerate rest
                        sorted_idx = np.argsort(fitness)
                        kept = max(2, int(N * 0.2))
                        keep_pop = pop[sorted_idx[:kept]].copy()
                        keep_fit = fitness[sorted_idx[:kept]].copy()
                        new_N = max(N, kept)
                        pop = lb + np.random.uniform(0, 1, (new_N, dim)) * (ub - lb)
                        pop[:kept] = keep_pop
                        fitness = np.full(new_N, np.inf)
                        fitness[:kept] = keep_fit
                        # evaluate new individuals
                        for j in range(kept, new_N):
                            fitness[j] = func(pop[j])
                            n_evals += 1
                            if fitness[j] < self.f_opt:
                                self.f_opt = fitness[j]
                                self.x_opt = pop[j].copy()
                        N = new_N
                        archive = np.empty((0, dim))
                        archive_max = N
                        evals_no_improve = 0
                        # reset memory partially
                        MF1[:] = 0.5
                        MCR1[:] = 0.5
                        MF2[:] = 0.5
                        MCR2[:] = 0.5
                        prob_strat[:] = 0.5
                        success_strat[:] = 0
                        fail_strat[:] = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt