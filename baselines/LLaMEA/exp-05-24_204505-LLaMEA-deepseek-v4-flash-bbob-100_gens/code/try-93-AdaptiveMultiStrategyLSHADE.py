import numpy as np

class AdaptiveMultiStrategyLSHADE:
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

        # Latin hypercube initialization with space-filling improvement
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

        # Archive (FIFO)
        archive = []
        archive_max = N

        # Memory for two strategies: 0 = current-to-pbest/1/archive, 1 = current-to-rand/1
        H = 30
        MF = np.ones((2, H)) * 0.5
        MCR = np.ones((2, H)) * 0.8
        memory_idx = np.zeros(2, dtype=int)

        # Success counters for strategy selection
        success1 = 0
        success2 = 0
        total1 = 0
        total2 = 0
        p_strat = 0.5  # probability to use strategy 1

        # Stagnation detection and diversity
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.05 * max_evals
        restart_count = 0
        max_restarts = 3

        # Compute initial population diversity (mean std normalized)
        def compute_diversity(pop):
            stds = np.std(pop, axis=0) / (ub - lb)
            return np.mean(stds)

        initial_diversity = compute_diversity(pop)
        diversity_min = 0.02 * initial_diversity

        # Local search parameters
        local_search_interval = max(30, int(0.01 * max_evals))
        last_local_search = 0

        # Pattern search with refined step adaptation
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            iterations = 0
            while used < max_local_evals and iterations < dim * 6:
                iterations += 1
                improved = False
                # Random permutation of coordinates for robustness
                perm = np.random.permutation(dim)
                for d in perm:
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
                    step_size *= 1.2  # expansion
                    step_size = np.minimum(step_size, (ub - lb) * 0.4)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.6  # contraction
                    if np.max(step_size) < 1e-12 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: exponential decay
            p = 0.2 * np.exp(-2.0 * (n_evals / max_evals)) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[], []]   # for strategy 0 and 1
            S_CR = [[], []]
            delta_f = [[], []]
            gen_success1 = 0
            gen_success2 = 0
            gen_total1 = 0
            gen_total2 = 0

            for i in range(N):
                # Strategy selection
                if np.random.rand() < p_strat:
                    sid = 0  # current-to-pbest/1/archive
                    gen_total1 += 1
                else:
                    sid = 1  # current-to-rand/1
                    gen_total2 += 1

                # Choose individuals for mutation
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # For strategy 0: use archive; for strategy 1: no archive
                if sid == 0:
                    # Current-to-pbest/1/archive
                    union = np.vstack((pop, np.array(archive))) if archive else pop
                    r2 = np.random.randint(union.shape[0])
                    pbest_size = max(1, int(p * N))
                    sorted_idx = np.argsort(fitness)
                    pbest_candidates = sorted_idx[:pbest_size]
                    pbest_idx = np.random.choice(pbest_candidates)
                    # Other individuals for strategy 1
                    # For strategy 1 we need two more random distinct indices
                    remaining = [j for j in idxs if j != r1]
                    r2_s1 = np.random.choice(remaining)
                    remaining.remove(r2_s1)
                    r3 = np.random.choice(remaining)
                else:
                    # Strategy 1: current-to-rand/1 (uses three random distinct)
                    remaining = [j for j in idxs if j != r1]
                    r2 = np.random.choice(remaining)
                    remaining.remove(r2)
                    r3 = np.random.choice(remaining)

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[sid, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[sid, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[sid, mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                base = pop[i]
                if sid == 0:
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    diff1 = pop[r1] - base
                    diff2 = pop[r2] - pop[r3]
                    mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Reflective boundary handling (up to 5 reflections)
                for _ in range(5):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)

                # Evaluation
                trial_f = func(trial)
                n_evals += 1

                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F[sid].append(F)
                    S_CR[sid].append(CR)
                    delta_f[sid].append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    if sid == 0:
                        # Archive only for strategy 0
                        archive.append(pop[i].copy())
                        if len(archive) > archive_max:
                            archive.pop(0)
                    # Update success counters for strategy selection
                    if sid == 0:
                        gen_success1 += 1
                    else:
                        gen_success2 += 1

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory for each strategy that had successes
            for sid in [0, 1]:
                if len(S_F[sid]) > 0:
                    w = np.array(delta_f[sid]) / (np.sum(delta_f[sid]) + 1e-30)
                    w = w / np.sum(w)  # normalize
                    S_F_arr = np.array(S_F[sid])
                    S_CR_arr = np.array(S_CR[sid])
                    MF[sid, memory_idx[sid]] = np.sum(w * S_F_arr**2) / (np.sum(w * S_F_arr) + 1e-30)
                    MCR[sid, memory_idx[sid]] = np.sum(w * S_CR_arr**2) / (np.sum(w * S_CR_arr) + 1e-30)
                    memory_idx[sid] = (memory_idx[sid] + 1) % H

            # Update strategy probability using moving average
            success1 += gen_success1
            success2 += gen_success2
            total1 += gen_total1
            total2 += gen_total2
            if total1 + total2 > 0:
                p_strat = (success1 + 1) / (total1 + total2 + 2)
                p_strat = np.clip(p_strat, 0.1, 0.9)  # avoid extreme

            # Population size reduction (cubic)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
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

            # Periodic local refinement
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
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
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity-based restart
            current_div = compute_diversity(pop)
            if (current_div < diversity_min and evals_no_improve > 0.01 * max_evals
                    and n_evals < max_evals * 0.8 and restart_count < max_restarts):
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
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
                    # Partial restart
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    fitness = np.full(N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory and counters
                MF[:] = 0.5 + 0.2 * np.random.rand(2, H)
                MCR[:] = 0.8 + 0.2 * np.random.rand(2, H)
                memory_idx[:] = 0
                archive = []
                archive_max = N
                evals_no_improve = 0
                success1 = 0
                success2 = 0
                total1 = 0
                total2 = 0
                p_strat = 0.5
                # Recompute diversity
                initial_diversity = compute_diversity(pop)

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt