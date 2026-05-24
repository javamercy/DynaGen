import numpy as np

class EnhancedLSHADE:
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

        # Population size parameters (linear reduction)
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization (simple implementation)
        def lhs(n, d, lb, ub):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = (perm + np.random.uniform(size=n)) / n
            return lb + samples * (ub - lb)

        pop = lhs(N, dim, lb, ub)
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

        # Memory for F and CR
        H = 20  # increased memory size
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        no_improve_steps = 0
        restart_threshold = 0.12 * max_evals  # slightly lower threshold

        # Local search parameters
        local_search_interval = max(40, int(0.015 * max_evals))
        last_local_search = 0
        pattern_failures = 0  # for adaptive local search scheduling

        # Pattern search with adaptive expansion/contraction
        def pattern_search(best_pos, best_val, step_frac, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step_frac * (ub - lb)
            used = 0
            consecutive_success = 0
            while used < max_local_evals:
                improved = False
                # Coordinate search in random order
                order = np.random.permutation(dim)
                for d in range(order.size):
                    if used >= max_local_evals:
                        break
                    d_idx = order[d]
                    # positive direction
                    new_pos = pos.copy()
                    new_pos[d_idx] = np.clip(pos[d_idx] + step_size[d_idx], lb[d_idx], ub[d_idx])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d_idx] = np.clip(pos[d_idx] - step_size[d_idx], lb[d_idx], ub[d_idx])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    consecutive_success += 1
                    # Pattern move (accelerate along direction of change)
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size more aggressively on success
                    step_size *= min(1.5, 1.0 + 0.1 * consecutive_success)
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    consecutive_success = 0
                    # Contract step size on failure
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used, consecutive_success  # return success count

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: linear schedule from 0.2 to 0.05
            frac = n_evals / max_evals
            p = 0.2 * (1.0 - frac) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Ensure at least two distinct indices
                candidates = list(range(N))
                candidates.remove(i)
                if len(candidates) < 2:
                    r1 = 0 if 0 != i else 1
                else:
                    r1 = np.random.choice(candidates)
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
                # Sample F and CR from memory
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
                # Binomial crossover with j_rand
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Sophisticated boundary repair: reflect then push to bound if still out
                for _ in range(5):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, lb + (lb - trial), trial)
                    trial = np.where(out_high, ub - (trial - ub), trial)
                trial = np.clip(trial, lb, ub)
                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive (if space)
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (sorted by improvement)
            if len(S_F) > 0:
                # Convert to arrays and sort by delta_f descending
                arr_F = np.array(S_F)
                arr_CR = np.array(S_CR)
                arr_delta = np.array(delta_f)
                order = np.argsort(arr_delta)[::-1]
                arr_F = arr_F[order]
                arr_CR = arr_CR[order]
                arr_delta = arr_delta[order]
                w = arr_delta / (np.sum(arr_delta) + 1e-30)
                MF[memory_idx] = np.sum(w * arr_F**2) / (np.sum(w * arr_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * arr_CR**2) / (np.sum(w * arr_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction (L-SHADE style)
            N_new = N_min + (N_init - N_min) * (1.0 - n_evals / max_evals)
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

            # Periodic local refinement using pattern search (every local_search_interval evals)
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step fraction decreases linearly with remaining budget
                step_frac = 0.15 * (1.0 - n_evals / max_evals) + 0.02
                max_local = min(dim * 4, max_evals - n_evals - 10)
                new_pos, new_val, used, success_cnt = pattern_search(best_pos, best_val, step_frac, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        no_improve_steps = 0
                # Replace worst individual with improved candidate
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val
                # If pattern search had many consecutive failures, increase interval temporarily
                if success_cnt == 0:
                    pattern_failures += 1
                else:
                    pattern_failures = max(0, pattern_failures - 1)
                local_search_interval = max(30, int(0.015 * max_evals) + pattern_failures * 5)

            # Restart if stagnation (no improvement over threshold or low diversity)
            diversity = np.mean(np.std(pop, axis=0)) / np.mean(ub - lb)
            if (no_improve_steps > restart_threshold or diversity < 1e-4) and n_evals < max_evals * 0.85:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Keep best individual and fill rest with Latin hypercube
                pop = lhs(new_N, dim, lb, ub)
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
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                no_improve_steps = 0
                pattern_failures = 0

            if n_evals >= max_evals:
                break

        # Final call to ensure best solution is returned
        return self.f_opt, self.x_opt