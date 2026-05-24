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

        # Population size parameters (linear reduction from L-SHADE)
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

        # Success-history memory for F and CR (separate per strategy)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Strategy memory: probabilities for 3 strategies
        # 0: current-to-pbest/1/archive, 1: current-to-rand/1, 2: best/1
        strat_prob = np.array([0.5, 0.3, 0.2])
        strat_F = np.ones(H) * 0.5   # per strategy memory
        strat_CR = np.ones(H) * 0.5
        strat_idx = 0
        success_strat = []

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Diversity tracking
        pop_diversity = np.std(pop, axis=0).mean()
        diversity_threshold = 0.01 * (ub.mean() - lb.mean())

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Pattern search with adaptive step
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            success_count = 0
            for iteration in range(dim * 4):
                if used >= max_local_evals:
                    break
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
                        success_count += 1
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
                        success_count += 1
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

            # For strategy adaptation
            strat_success = np.zeros(3)
            strat_total = np.zeros(3)

            # Generate offspring
            for i in range(N):
                # Choose strategy based on probabilities
                strat = np.random.choice(3, p=strat_prob)
                strat_total[strat] += 1

                if strat == 0:  # current-to-pbest/1/archive
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

                elif strat == 1:  # current-to-rand/1
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, size=2, replace=False)
                    mem = np.random.randint(H)
                    F = np.clip(strat_F[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(strat_F[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(strat_CR[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[r2] - base
                    mutant = base + F * diff1 + F * diff2

                else:  # best/1
                    best_idx = np.argmin(fitness)
                    idxs = list(range(N))
                    idxs.remove(best_idx)
                    r1, r2 = np.random.choice(idxs, size=2, replace=False)
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff1 = pop[best_idx] - base
                    diff2 = pop[r1] - pop[r2]
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
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                    strat_success[strat] += 1
                    success_strat.append(strat)

            # Update population
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

            # Update strategy probabilities based on success rates
            if np.sum(strat_total) > 0:
                for s in range(3):
                    if strat_total[s] > 0:
                        strat_prob[s] = strat_prob[s] * 0.9 + 0.1 * (strat_success[s] / strat_total[s])
                strat_prob /= strat_prob.sum()

            # Population size reduction (linear)
            N_new = int(N_init - (N_init - N_min) * (n_evals / max_evals))
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
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity check for restart
            pop_diversity = np.std(pop, axis=0).mean()
            if pop_diversity < diversity_threshold and n_evals < max_evals * 0.8:
                restart_trigger = True
            else:
                restart_trigger = False

            # Restart if stagnation or low diversity
            if (evals_no_improve > restart_threshold or restart_trigger) and (n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Latin hypercube around best
                    cov = 0.1 * (ub - lb)**2 * np.eye(dim)
                    samples = np.random.multivariate_normal(best_ind, cov, size=new_N-1)
                    samples = np.clip(samples, lb, ub)
                    pop = np.vstack([best_ind.reshape(1,-1), samples])
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                else:
                    # Partial restart: randomize all but best
                    cov = 0.3 * (ub - lb)**2 * np.eye(dim)
                    samples = np.random.multivariate_normal(best_ind, cov, size=N-1)
                    samples = np.clip(samples, lb, ub)
                    pop = np.vstack([best_ind.reshape(1,-1), samples])
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    fitness[0] = best_fit
                # Reset memories
                MF[:] = 0.5
                MCR[:] = 0.5
                strat_F[:] = 0.5
                strat_CR[:] = 0.5
                strat_prob = np.array([0.5, 0.3, 0.2])
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Reset diversity threshold
                diversity_threshold = 0.01 * (ub.mean() - lb.mean()) * (1 - n_evals/max_evals)

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt