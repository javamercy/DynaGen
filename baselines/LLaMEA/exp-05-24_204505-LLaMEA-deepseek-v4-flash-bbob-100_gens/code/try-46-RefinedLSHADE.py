import numpy as np

class RefinedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        # Assume bounds [-5,5] (BBOB standard)
        lb = np.full(self.dim, -5.0)
        ub = np.full(self.dim, 5.0)
        dim = self.dim
        max_evals = self.budget

        # Population parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization + opposition
        samples = np.random.uniform(0, 1, (N, dim))
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        # Generate opposite of half the points
        half_idx = np.random.choice(N, N//2, replace=False)
        opp_pop = lb + ub - pop[half_idx]
        # Combine and trim to N
        all_candidates = np.vstack((pop, opp_pop))
        if all_candidates.shape[0] > N:
            idx = np.random.choice(all_candidates.shape[0], N, replace=False)
            pop = all_candidates[idx]
        else:
            pop = all_candidates[:N]
        N = pop.shape[0]

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
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity tracking
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.1 * max_evals
        diversity_threshold = 0.05 * (ub[0] - lb[0])  # 0.05 * range

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0
        final_local_search = False

        # Helper: generate random rotation matrix (orthonormal)
        def random_rotation(d):
            A = np.random.randn(d, d)
            Q, _ = np.linalg.qr(A)
            return Q

        # Helper: rotated pattern search (coordinate search in rotated basis)
        def rotated_pattern_search(best_pos, best_val, step, max_local_evals, rotation_matrix):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative step in original coordinates
            # In rotated space, step sizes are same but directions are columns of rotation_matrix
            directions = rotation_matrix  # each column is a direction
            iterations = 0
            used = 0
            while used < max_local_evals and iterations < dim * 3:
                iterations += 1
                improved = False
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = np.clip(pos + step_size * directions[:, d], lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = np.clip(pos - step_size * directions[:, d], lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # pattern move along accumulated delta
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring with DE/current-to-pbest/1/archive
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
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling (reflect)
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
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

            pop = new_pop
            fitness = new_fitness

            # Update memory (weighted Lehmer mean)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (exponential schedule)
            alpha = 5.0
            ratio = n_evals / max_evals
            N_new = N_min + (N_init - N_min) * np.exp(-alpha * ratio)
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

            # Periodic local refinement with rotated pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                # Use a random rotation matrix for this session
                rot = random_rotation(dim)
                new_pos, new_val, used = rotated_pattern_search(best_pos, best_val, step, max_local, rot)
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
            if n_evals < max_evals * 0.8:
                # Compute diversity: average pairwise distance from best
                best_idx = np.argmin(fitness)
                dists = np.linalg.norm(pop - pop[best_idx], axis=1)
                diversity = np.mean(dists) if len(dists) > 1 else 0.0
                if diversity < diversity_threshold and evals_no_improve > restart_threshold:
                    # Restart: keep best, random rest
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    remaining = max_evals - n_evals
                    new_N = min(N_init, remaining // 2)
                    new_N = max(new_N, N_min)
                    # Generate new population with opposition (some from best)
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    # Opposition of some random points
                    opp_idx = np.random.choice(new_N, new_N//2, replace=False)
                    samples[opp_idx] = lb + ub - samples[opp_idx]
                    pop = np.vstack((best_ind.reshape(1,-1), samples[:new_N-1]))
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                    # Reset memory
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    archive = np.empty((0, dim))
                    archive_max = N
                    evals_no_improve = 0

            # Final local search near budget end
            if not final_local_search and n_evals >= max_evals * 0.95 and n_evals < max_evals:
                final_local_search = True
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.005  # fine search
                max_local = max_evals - n_evals - 2
                rot = random_rotation(dim)
                new_pos, new_val, used = rotated_pattern_search(best_pos, best_val, step, max_local, rot)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = new_pos.copy()

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt