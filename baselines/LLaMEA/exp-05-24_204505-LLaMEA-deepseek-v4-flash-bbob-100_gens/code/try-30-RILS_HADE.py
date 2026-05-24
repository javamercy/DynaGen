import numpy as np

class RILS_HADE:
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

        # Stagnation and diversity tracking
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        # Initial diversity
        centroid = np.mean(pop, axis=0)
        distances = np.sqrt(np.sum((pop - centroid)**2, axis=1))
        initial_diversity = np.median(distances) + 1e-12
        # For restart trigger
        no_improve_gen = 0

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Helper: generate random orthogonal rotation matrix
        def random_rotation(d):
            Q, _ = np.linalg.qr(np.random.randn(d, d))
            return Q

        # Rotationally invariant pattern search
        def rotated_pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            Q = random_rotation(dim)
            step_size = step * np.ones(dim)  # scalar step per direction
            used = 0
            iterations = 0
            while used < max_local_evals and iterations < dim * 3:
                iterations += 1
                improved = False
                # Search along each rotated direction
                for d_idx in np.random.permutation(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = np.clip(pos + step_size[d_idx] * Q[:, d_idx], lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction (skip if already improved)
                    if used >= max_local_evals:
                        break
                    new_pos = np.clip(pos - step_size[d_idx] * Q[:, d_idx], lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: accelerate
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        if used < max_local_evals:
                            new_val = func(new_pos)
                            used += 1
                            if new_val < val:
                                pos = new_pos
                                val = new_val
                    step_size *= 1.1
                    step_size = np.minimum(step_size, 0.5 * (ub - lb))
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-8 * np.max(ub - lb):
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

            # Generate offspring
            for i in range(N):
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                if archive.shape[0] > 0:
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
                # Mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # Occasionally add Gaussian noise
                if np.random.rand() < 0.1:
                    sigma = 0.1 * (1 - n_evals/max_evals) * (ub - lb)
                    mutant += sigma * np.random.randn(dim)
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
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using rotated pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = rotated_pattern_search(best_pos, best_val, step, max_local)
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

            # Diversity tracking and restart trigger
            centroid = np.mean(pop, axis=0)
            distances = np.sqrt(np.sum((pop - centroid)**2, axis=1))
            current_diversity = np.median(distances) + 1e-12
            # Trigger restart if diversity too low or stagnation
            low_diversity = current_diversity < 0.1 * initial_diversity
            stagnation = evals_no_improve > restart_threshold
            if (low_diversity or stagnation) and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Mixed restart: 50% uniform, 50% Gaussian around best
                pop_new = np.empty((new_N, dim))
                half = new_N // 2
                # Uniform part
                pop_new[:half] = lb + np.random.uniform(0, 1, (half, dim)) * (ub - lb)
                # Gaussian part around best with spread from current diversity
                scale = max(current_diversity, 0.2 * np.max(ub - lb))
                pop_new[half:] = best_ind + scale * np.random.randn(new_N - half, dim)
                pop_new = np.clip(pop_new, lb, ub)
                # Evaluate all except ensure best is included
                fitness_new = np.full(new_N, np.inf)
                pop_new[0] = best_ind
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
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Recompute diversity baseline
                centroid = np.mean(pop, axis=0)
                distances = np.sqrt(np.sum((pop - centroid)**2, axis=1))
                initial_diversity = np.median(distances) + 1e-12

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt