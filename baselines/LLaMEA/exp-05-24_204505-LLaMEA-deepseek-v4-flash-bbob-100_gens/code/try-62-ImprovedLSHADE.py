import numpy as np

class ImprovedLSHADE:
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

        # Sobol sequence for quasi-random initialization
        sobol = np.random.RandomState(42)
        # simple Sobol approximation using Latin hypercube
        def sobol_samples(n, d):
            samples = np.zeros((n, d))
            for i in range(d):
                u = np.random.rand(n)
                # scramble
                samples[:, i] = (np.arange(n) + u) / n
            return samples

        # Population size parameters (more aggressive reduction)
        N_init = min(max(10 * dim, 60), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Initialize with Sobol Latin hypercube
        samples = sobol_samples(N, dim)
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation (increased capacity)
        archive = np.empty((0, dim))
        archive_max = 2 * N

        # Success-history memory for F and CR (larger memory)
        H = 15
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.9
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # Local search parameters
        local_search_interval = max(30, int(0.015 * max_evals))
        last_local_search = 0

        # Pattern search with finite-difference gradient approximation
        def pattern_search_gradient(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb) * 0.1
            used = 0
            while used < max_local_evals:
                # Finite difference gradient
                grad = np.zeros(dim)
                for d in range(dim):
                    if used >= max_local_evals - 1:
                        break
                    h = step_size[d] * 0.5
                    if h < 1e-12:
                        continue
                    pos_plus = pos.copy()
                    pos_plus[d] = np.clip(pos[d] + h, lb[d], ub[d])
                    f_plus = func(pos_plus)
                    used += 1
                    pos_minus = pos.copy()
                    pos_minus[d] = np.clip(pos[d] - h, lb[d], ub[d])
                    f_minus = func(pos_minus)
                    used += 1
                    grad[d] = (f_plus - f_minus) / (2 * h)
                if used >= max_local_evals:
                    break
                # Line search along steepest descent direction (with random perturbation)
                direction = -grad
                if np.linalg.norm(direction) < 1e-12:
                    direction = np.random.uniform(-1, 1, dim)
                    direction = direction / (np.linalg.norm(direction) + 1e-30)
                else:
                    direction = direction / (np.linalg.norm(direction) + 1e-30)
                # Try multiple step lengths
                best_step = 0
                best_new_val = val
                for mult in [1.0, 0.5, 0.25, 2.0, 4.0]:
                    if used >= max_local_evals:
                        break
                    new_pos = np.clip(pos + mult * step_size * direction, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < best_new_val:
                        best_new_val = new_val
                        best_step = mult
                if best_new_val < val:
                    pos = np.clip(pos + best_step * step_size * direction, lb, ub)
                    val = best_new_val
                    step_size *= 1.2
                else:
                    step_size *= 0.5
                if np.max(step_size) < 1e-10 * np.max(ub - lb):
                    break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: smooth decay from 0.25 to 0.03
            p = 0.25 * (1 - (n_evals / max_evals) ** 1.2) + 0.03

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
                # Sample F and CR from Cauchy/normal with more variation
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.15 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.15 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.15 * np.random.randn(), 0, 1)
                # Mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: reflect with probability, else random restart
                for _ in range(5):
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
                    # Add parent to archive (only if different)
                    if not np.any(np.all(pop[i] == archive, axis=1)):
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                        if archive.shape[0] > archive_max:
                            remove_idx = np.random.randint(archive.shape[0])
                            archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (exponential smoothing)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF_new = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR_new = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                # Exponential smoothing (0.8 old, 0.2 new)
                MF[memory_idx] = 0.8 * MF[memory_idx] + 0.2 * MF_new
                MCR[memory_idx] = 0.8 * MCR[memory_idx] + 0.2 * MCR_new
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (cubic schedule, more aggressive early)
            frac = (max_evals - n_evals) / max_evals
            N_new = N_min + (N_init - N_min) * frac ** 3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = 2 * N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using gradient pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 10)
                new_pos, new_val, used = pattern_search_gradient(best_pos, best_val, step, max_local)
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

            # Restart if stagnation detected (with diversity reinitialization)
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate quasi-random population via Sobol
                samples_sobol = sobol_samples(new_N, dim)
                new_pop = lb + samples_sobol * (ub - lb)
                new_pop[0] = best_ind
                # Evaluate new individuals
                for j in range(new_N):
                    if j == 0:
                        new_fitness = np.full(new_N, np.inf)
                        new_fitness[0] = best_fit
                        continue
                    new_fitness[j] = func(new_pop[j])
                    n_evals += 1
                    if new_fitness[j] < self.f_opt:
                        self.f_opt = new_fitness[j]
                        self.x_opt = new_pop[j].copy()
                pop = new_pop
                fitness = new_fitness
                N = new_N
                # Reset memory with high exploration values
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt