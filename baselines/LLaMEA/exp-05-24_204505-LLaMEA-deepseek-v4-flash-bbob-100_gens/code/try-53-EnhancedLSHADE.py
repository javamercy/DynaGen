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

        # Population initialisation
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization
        samples = np.random.uniform(0, 1, (N, dim))
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive
        archive = np.empty((0, dim))
        archive_max = int(2.0 * N_init)

        # Memory for parameters
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        mem_idx = 0

        # Stagnation handling
        best_fitness_hist = [self.f_opt]
        no_improve_evals = 0
        restart_threshold = 0.12 * max_evals
        diversity_threshold = 0.05 * (ub[0] - lb[0])  # scaled

        # Local search parameters (dimension-wise step)
        ls_interval = max(30, int(0.02 * max_evals))
        last_ls = 0

        # ------------------------------------------------------------
        # Local search with per-dimension adaptive step
        # ------------------------------------------------------------
        def localized_pattern_search(best_pos, best_val, step_total, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            # Per-dimension step sizes
            step_sizes = step_total * (ub - lb) * np.ones(dim)
            used = 0
            iteration = 0
            while used < max_local_evals and iteration < dim * 4:
                iteration += 1
                improved_any = False
                # Random order of dimensions
                order = np.random.permutation(dim)
                for d in order:
                    if used >= max_local_evals:
                        break
                    step = step_sizes[d]
                    # Positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step, lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        step_sizes[d] *= 1.2
                        step_sizes[d] = min(step_sizes[d], (ub[d]-lb[d])*0.5)
                        improved_any = True
                        continue
                    # Negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step, lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        step_sizes[d] *= 1.2
                        step_sizes[d] = min(step_sizes[d], (ub[d]-lb[d])*0.5)
                        improved_any = True
                    else:
                        step_sizes[d] *= 0.5
                        if step_sizes[d] < 1e-12 * (ub[d]-lb[d]):
                            step_sizes[d] = 1e-12 * (ub[d]-lb[d])
                if improved_any:
                    # Pattern step: extrapolate along direction of improvement
                    delta = pos - best_pos
                    if np.max(np.abs(delta)) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    best_pos = pos.copy()
                    best_val = val
            return pos, val, used

        # ------------------------------------------------------------
        # Helper: generate trial using current-to-pbest/1 with archive
        # ------------------------------------------------------------
        def create_trial(pop, fitness, i, memory_idx):
            # parameters sampling
            mem = np.random.randint(H)
            F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0.0, 1.0)
            while F <= 0:
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
            CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0.0, 1.0)
            # indices
            idxs = list(range(N))
            idxs.remove(i)
            r1 = np.random.choice(idxs)
            if archive.size > 0:
                union = np.vstack((pop, archive))
            else:
                union = pop
            r2 = np.random.randint(union.shape[0])
            # pbest
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.02
            pbest_size = max(1, int(p * N))
            sorted_idx = np.argsort(fitness)
            pbest_candidates = sorted_idx[:pbest_size]
            pbest = pop[np.random.choice(pbest_candidates)]
            # mutation
            base = pop[i]
            diff1 = pbest - base
            diff2 = pop[r1] - union[r2]
            mutant = base + F * diff1 + F * diff2
            # binomial crossover
            j_rand = np.random.randint(dim)
            trial = np.where(np.random.rand(dim) < CR, mutant, base)
            trial[j_rand] = mutant[j_rand]
            # boundary reflection
            for _ in range(8):
                below = trial < lb
                above = trial > ub
                if not (np.any(below) or np.any(above)):
                    break
                trial = np.where(below, 2 * lb - trial, trial)
                trial = np.where(above, 2 * ub - trial, trial)
            trial = np.clip(trial, lb, ub)
            return trial, F, CR

        # ------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------
        while n_evals < max_evals:
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generation of offspring
            for i in range(N):
                if n_evals >= max_evals:
                    break
                trial, F, CR = create_trial(pop, fitness, i, mem_idx)
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    no_improve_evals = 0
                else:
                    no_improve_evals += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[order]
                S_CR = np.array(S_CR)[order]
                w = np.array(delta_f)[order] / (np.sum(delta_f) + 1e-30)
                MF[mem_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[mem_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                mem_idx = (mem_idx + 1) % H

            # Reduce population size (quadratic)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:N_new]]
                fitness = fitness[idx_sorted[:N_new]]
                archive_max = int(2.0 * N_new)
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement
            if (n_evals - last_ls >= ls_interval) and (n_evals < max_evals * 0.95):
                last_ls = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = localized_pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        no_improve_evals = 0
                # Replace worst individual if better
                if best_val < fitness[np.argmax(fitness)]:
                    worst = np.argmax(fitness)
                    pop[worst] = best_pos
                    fitness[worst] = best_val

            # Stagnation detection and restart (including diversity check)
            diversity = np.mean(np.std(pop, axis=0))
            if (no_improve_evals > restart_threshold or diversity < diversity_threshold) and n_evals < max_evals * 0.8:
                # Restart with Cauchy distributed points around best
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, int(N * 1.5), remaining // 2)
                new_N = max(new_N, N_min)
                # Reinitialize population with heavy-tailed perturbations
                pop = np.zeros((new_N, dim))
                fitness = np.full(new_N, np.inf)
                # Keep best individual
                pop[0] = best_ind
                fitness[0] = best_fit
                for j in range(1, new_N):
                    # Cauchy perturbations
                    scale = 0.5 * (ub - lb) * (1 + np.random.rand())  # larger scale
                    offset = np.random.standard_cauchy(dim) * scale
                    candidate = np.clip(best_ind + offset, lb, ub)
                    pop[j] = candidate
                    fitness[j] = func(candidate)
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory to exploration-friendly values
                MF[:] = 0.5 + 0.2 * np.random.rand(H)
                MCR[:] = 0.9
                mem_idx = 0
                archive = np.empty((0, dim))
                archive_max = int(2.0 * N)
                no_improve_evals = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt