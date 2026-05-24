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

        # Population size parameters (larger initial for high dimensions)
        N_init = min(max(10 * dim, 60), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Quasi-opposition based initialization
        def quasi_opposition(x):
            mid = (lb + ub) / 2
            return np.where(x < mid, 2 * mid - x, 2 * mid - x)  # symmetrical opposite
            # Actually, quasi-opposition: random point in [mid, lb+ub-x] for each coordinate
        samples_lhs = np.random.uniform(0, 1, (N, dim))
        samples_lhs = lb + samples_lhs * (ub - lb)
        opp = np.zeros_like(samples_lhs)
        for i in range(N):
            for d in range(dim):
                mid = (lb[d] + ub[d]) / 2
                if samples_lhs[i, d] < mid:
                    opp[i, d] = np.random.uniform(mid, lb[d] + ub[d] - samples_lhs[i, d])
                else:
                    opp[i, d] = np.random.uniform(lb[d] + ub[d] - samples_lhs[i, d], mid)
        pop_full = np.vstack((samples_lhs, opp))
        fitness_full = np.full(2*N, np.inf)
        for i in range(2*N):
            fitness_full[i] = func(pop_full[i])
            if fitness_full[i] < self.f_opt:
                self.f_opt = fitness_full[i]
                self.x_opt = pop_full[i].copy()
        n_evals = 2 * N
        # Select best N individuals
        idx = np.argsort(fitness_full)[:N]
        pop = pop_full[idx].copy()
        fitness = fitness_full[idx].copy()
        # Re-evaluate? No, already evaluated.

        # Archive for DE mutation (larger size)
        archive = np.empty((0, dim))
        archive_max = 2 * N

        # Success-history memory for F and CR
        H = 10
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

        # Enhanced Hooke-Jeeves pattern search with adaptive step
        def hooke_jeeves(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative step
            used = 0
            # Pattern move using previous improvement direction
            prev_delta = np.zeros(dim)
            while used < max_local_evals:
                improved = False
                # Exploratory moves in each coordinate (both directions)
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
                        # negative direction unnecessary if positive improved
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
                    # Pattern move: accelerate along aggregated improvement direction
                    delta = pos - best_pos
                    # If delta is non-zero and step size not too small
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            # update best for next pattern iteration
                            best_pos = pos.copy()
                            best_val = val
                            prev_delta = delta.copy()
                        # else keep pos unchanged
                    # Expand step size on success
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                else:
                    # Contract step size on failure
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
                # After pattern move, try direction from previous successful pattern
                if np.any(np.abs(prev_delta) > 1e-12) and improved:
                    new_pos = np.clip(pos + prev_delta, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: exponential decrease
            ratio = n_evals / max_evals
            p = 0.2 * np.exp(-2 * ratio) + 0.05

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
                # Sample F and CR (Cauchy, Gaussian)
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
                archive_max = 2 * N_new  # keep archive proportional
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using enhanced Hooke-Jeeves
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = 0.15 * (1 - ratio) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = hooke_jeeves(best_pos, best_val, step, max_local)
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

            # Diversity measure (normalized standard deviation)
            pop_std = np.std(pop, axis=0)
            diversity = np.mean(pop_std / (ub - lb))

            # Restart if stagnation or low diversity
            if (evals_no_improve > restart_threshold or diversity < 1e-5) and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population: quasi-random with Sobol-like (Latin hypercube)
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
                # Reset memory parameters
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt