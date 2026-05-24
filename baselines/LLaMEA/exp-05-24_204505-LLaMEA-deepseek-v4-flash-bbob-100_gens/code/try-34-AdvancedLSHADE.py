import numpy as np

class AdvancedLSHADE:
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

        # Archive
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR (two strategies)
        H = 10
        MF = np.ones((2, H)) * 0.5     # strategy 0: current-to-pbest/1, 1: current-to-rand/1
        MCR = np.ones((2, H)) * 0.8
        memory_idx = np.zeros(2, dtype=int)

        # Strategy selection probabilities
        strat_probs = np.array([0.5, 0.5])
        strat_success = np.zeros(2)
        strat_total = np.zeros(2)
        learning_period = 20
        gen_counter = 0

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.05 * (ub - lb).mean()

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Helper: random direction local search with adaptive step size
        def random_direction_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb).mean()  # initial step
            success_count = 0
            used = 0
            while used < max_local_evals:
                # Generate random direction
                direction = np.random.randn(dim)
                direction = direction / (np.linalg.norm(direction) + 1e-30)
                # Try positive direction
                new_pos = np.clip(pos + step_size * direction, lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    success_count += 1
                else:
                    # Try negative direction
                    new_pos = np.clip(pos - step_size * direction, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        success_count += 1
                # Adjust step size every few attempts
                if used % (dim * 2) == 0:
                    if success_count > used * 0.25:
                        step_size *= 1.2
                    else:
                        step_size *= 0.8
                    step_size = np.clip(step_size, 1e-8 * (ub - lb).mean(), 0.5 * (ub - lb).mean())
                    success_count = 0
                if used >= max_local_evals:
                    break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[], []]
            S_CR = [[], []]
            delta_f = [[], []]

            # Generate offspring
            for i in range(N):
                # Decide which strategy to use
                rnd = np.random.rand()
                if rnd < strat_probs[0]:
                    strat_idx = 0
                else:
                    strat_idx = 1
                # Record selection
                strat_total[strat_idx] += 1

                if strat_idx == 0:
                    # current-to-pbest/1 with archive
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
                else:
                    # current-to-rand/1 (no archive)
                    idxs = [j for j in range(N) if j != i]
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    pbest_idx = None  # not used

                # Sample F and CR from memory of chosen strategy
                mem = np.random.randint(H)
                F = np.clip(MF[strat_idx, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[strat_idx, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[strat_idx, mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                if strat_idx == 0:
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    # current-to-rand/1
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[r2] - base
                    mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Boundary handling (reflection)
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
                    S_F[strat_idx].append(F)
                    S_CR[strat_idx].append(CR)
                    delta_f[strat_idx].append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial.copy()
                    strat_success[strat_idx] += 1
                    if strat_idx == 0:
                        # Add parent to archive
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                        if archive.shape[0] > archive_max:
                            remove_idx = np.random.randint(archive.shape[0])
                            archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means for each strategy
            for s in range(2):
                if len(S_F[s]) > 0:
                    sorted_order = np.argsort(delta_f[s])[::-1]
                    S_F_s = np.array(S_F[s])[sorted_order]
                    S_CR_s = np.array(S_CR[s])[sorted_order]
                    w = np.array(delta_f[s])[sorted_order] / (np.sum(delta_f[s]) + 1e-30)
                    MF[s, memory_idx[s]] = np.sum(w * S_F_s ** 2) / (np.sum(w * S_F_s) + 1e-30)
                    MCR[s, memory_idx[s]] = np.sum(w * S_CR_s ** 2) / (np.sum(w * S_CR_s) + 1e-30)
                    memory_idx[s] = (memory_idx[s] + 1) % H

            # Update strategy probabilities every learning_period generations
            gen_counter += 1
            if gen_counter >= learning_period:
                # Update prob using success rates (smoothed)
                total_success = strat_success.sum()
                if total_success > 0:
                    for s in range(2):
                        strat_probs[s] = strat_success[s] / (strat_total[s] + 1e-30)
                # Reset counters
                strat_success[:] = 0
                strat_total[:] = 0
                gen_counter = 0
                # Ensure both strategies have non-zero probability
                strat_probs = np.clip(strat_probs, 0.1, 0.9)
                strat_probs = strat_probs / strat_probs.sum()

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

            # Periodic local refinement using random direction search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = random_direction_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity measure (mean distance from best)
            if N > 1:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx]
                distances = np.sqrt(((pop - best_ind) ** 2).sum(axis=1))
                mean_dist = distances.mean()
            else:
                mean_dist = (ub - lb).mean() * 0.5

            # Restart if stagnation or low diversity
            if (evals_no_improve > restart_threshold or mean_dist < diversity_threshold) and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Keep top 2 individuals to preserve information
                sorted_idx = np.argsort(fitness)
                kept_inds = [sorted_idx[0]]
                if N > 1:
                    kept_inds.append(sorted_idx[1])
                kept_pop = pop[kept_inds].copy()
                kept_fit = fitness[kept_inds].copy()
                if new_N > N:
                    # Reinitialize with Latin hypercube but include best individuals
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    for k, idx in enumerate(kept_inds[:new_N]):
                        pop[k] = kept_pop[k]
                        fitness[k] = kept_fit[k]
                    for j in range(len(kept_inds), new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: keep best individuals, randomize rest
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    for k, idx in enumerate(kept_inds[:N]):
                        pop[k] = kept_pop[k]
                        fitness[k] = kept_fit[k]
                    for j in range(len(kept_inds), N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memories and archives
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx[:] = 0
                strat_probs[:] = 0.5
                strat_success[:] = 0
                strat_total[:] = 0
                gen_counter = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt