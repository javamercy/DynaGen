import numpy as np
from scipy.optimize import minimize

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

        # Sobol-like Latin hypercube initialization
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

        # Memory for F and CR (two strategies)
        H = 10
        MF1 = np.ones(H) * 0.5  # current-to-pbest/1
        MCR1 = np.ones(H) * 0.8
        MF2 = np.ones(H) * 0.5  # current-to-rand/1
        MCR2 = np.ones(H) * 0.3
        memory_idx = 0

        # Strategy probabilities (adaptive)
        P_strat = 0.5  # initial probability for strat1; strat2 uses 1-P_strat

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.2 * max_evals
        diversity_threshold = 1e-3 * np.mean(ub - lb)

        # Track best so far
        best_f = self.f_opt
        best_x = self.x_opt.copy()

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0

        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            # Also try a random direction pattern
            direction = np.random.randn(dim)
            direction /= np.linalg.norm(direction) + 1e-30
            for _ in range(min(dim * 3, max_local_evals // 2)):
                if used >= max_local_evals:
                    break
                improved = False
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive step
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative step
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if not improved:
                    # Try random direction
                    rnd_dir = np.random.randn(dim)
                    rnd_dir /= np.linalg.norm(rnd_dir) + 1e-30
                    new_pos = np.clip(pos + step_size * rnd_dir, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        # Change step direction
                        direction = rnd_dir
                if improved:
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.4)
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-8 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            # Update strategy probability based on recent success
            if n_evals > max_evals * 0.1:
                # estimate success proportion from last generation? simplified: keep adaptive
                pass  # we use adaptive later with collected scores

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F1, S_CR1, S_delta1 = [], [], []
            S_F2, S_CR2, S_delta2 = [], [], []
            n_success1 = 0
            n_success2 = 0

            for i in range(N):
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

                # Choose strategy: 0 for current-to-pbest/1, 1 for current-to-rand/1
                if np.random.rand() < P_strat:
                    # Strategy 1: current-to-pbest/1 with archive
                    mem = np.random.randint(H)
                    F = np.clip(MF1[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF1[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR1[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                    # Binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                    strat_used = 1
                else:
                    # Strategy 2: current-to-rand/1 (no archive, rotation invariant)
                    mem = np.random.randint(H)
                    F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR2[mem] + 0.1 * np.random.randn(), 0, 1)
                    # Use two random distinct indices
                    idx_pool = list(range(N))
                    idx_pool.remove(i)
                    np.random.shuffle(idx_pool)
                    r2_ = idx_pool[0]
                    r3_ = idx_pool[1]
                    base = pop[i]
                    diff1 = pop[r1] - pop[r2_]
                    diff2 = pop[pbest_idx] - pop[r3_]
                    mutant = base + F * diff1 + F * diff2
                    # Binomial crossover with j_rand
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                    strat_used = 2

                # Boundary handling: reflection and clamp
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
                    # Success
                    if strat_used == 1:
                        S_F1.append(F)
                        S_CR1.append(CR)
                        S_delta1.append(fitness[i] - trial_f)
                        n_success1 += 1
                    else:
                        S_F2.append(F)
                        S_CR2.append(CR)
                        S_delta2.append(fitness[i] - trial_f)
                        n_success2 += 1
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update strategy probability (simple: use proportion of successes)
            total_success = n_success1 + n_success2
            if total_success > 0:
                P_strat = 0.9 * P_strat + 0.1 * (n_success1 / total_success)
                P_strat = np.clip(P_strat, 0.2, 0.8)

            # Update memories for both strategies
            def update_memory(S_F, S_CR, S_delta, MF, MCR):
                if len(S_F) > 0:
                    sorted_order = np.argsort(S_delta)[::-1]
                    S_F = np.array(S_F)[sorted_order]
                    S_CR = np.array(S_CR)[sorted_order]
                    w = np.array(S_delta)[sorted_order] / (np.sum(S_delta) + 1e-30)
                    MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                    MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                    return (memory_idx + 1) % H
                return memory_idx

            memory_idx = update_memory(S_F1, S_CR1, S_delta1, MF1, MCR1)
            memory_idx = update_memory(S_F2, S_CR2, S_delta2, MF2, MCR2)

            # Population size reduction (exponential-like)
            remaining_ratio = (max_evals - n_evals) / max_evals
            N_new = N_min + (N_init - N_min) * (remaining_ratio ** 1.5)
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

            # Local search every interval, but only if population diversity is sufficient
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.9):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Compute diversity
                centroid = np.mean(pop, axis=0)
                diversity = np.mean(np.linalg.norm(pop - centroid, axis=1))
                if diversity > diversity_threshold or n_evals < max_evals * 0.5:
                    step = 0.15 * (1 - n_evals / max_evals) + 0.01
                    max_local = min(dim * 3, max_evals - n_evals - 5)
                    new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
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

            # Restart if stagnation or low diversity
            diversity = np.mean(np.linalg.norm(pop - np.mean(pop, axis=0), axis=1))
            if (evals_no_improve > restart_threshold) or (diversity < 1e-4 * np.mean(ub - lb) and n_evals < max_evals * 0.75):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                # Generate new population with mixture: 50% around best, 50% uniform
                new_N = max(N, int(remaining * 0.4))
                new_N = min(new_N, N_init * 2)
                new_N = max(new_N, N_min)
                pop_new = np.empty((new_N, dim))
                # Best individual
                pop_new[0] = best_ind
                # Generate around best using Cauchy distribution
                scale = 0.2 * (ub - lb)
                n_around = new_N // 2
                for j in range(1, n_around):
                    noise = np.random.standard_cauchy(dim)
                    # clip scale to avoid extreme jumps
                    step = scale * np.clip(noise, -10, 10)
                    candidate = np.clip(best_ind + step, lb, ub)
                    pop_new[j] = candidate
                # Uniform rest
                for j in range(n_around, new_N):
                    pop_new[j] = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                # Evaluate
                fitness_new = np.full(new_N, np.inf)
                fitness_new[0] = best_fit
                for j in range(1, new_N):
                    fitness_new[j] = func(pop_new[j])
                    n_evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = pop_new[j].copy()
                pop = pop_new
                fitness = fitness_new
                N = new_N
                # Reset memories and archive
                MF1[:] = 0.5
                MCR1[:] = 0.8
                MF2[:] = 0.5
                MCR2[:] = 0.3
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            # Early termination if global optimum known? Not needed.

        # Final local refinement from best solution
        if n_evals < max_evals - 10:
            step = 0.01
            max_local = max_evals - n_evals
            _, best_x, used = pattern_search(self.x_opt, self.f_opt, step, max_local)
            self.f_opt = best_f if best_f < self.f_opt else self.f_opt
            n_evals += used

        return self.f_opt, self.x_opt