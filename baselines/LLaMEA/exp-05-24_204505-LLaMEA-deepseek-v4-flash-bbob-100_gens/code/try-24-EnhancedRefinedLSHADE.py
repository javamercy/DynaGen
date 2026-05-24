import numpy as np

class EnhancedRefinedLSHADE:
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

        # Increased history size for better adaptation
        H = 30
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

        # Archive: larger capacity (2.5*N)
        archive = np.empty((0, dim))
        archive_max = int(2.5 * N)

        # Success-history memory
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.01 * (ub - lb).mean() * np.sqrt(dim)

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0
        step_global = 0.1  # initial step for pattern search

        # Pattern search with probabilistic acceptance (simulated annealing style)
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            iterations = 0
            used = 0
            temp = 0.05 * (val + 1e-10)  # initial temperature based on objective value
            while used < max_local_evals and iterations < dim * 4:
                iterations += 1
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
                    if new_val < val or np.random.rand() < np.exp((val - new_val) / (temp + 1e-30)):
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                        else:
                            pos = new_pos
                            val = new_val
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val or np.random.rand() < np.exp((val - new_val) / (temp + 1e-30)):
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                        else:
                            pos = new_pos
                            val = new_val
                if improved:
                    # Pattern move
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val or np.random.rand() < np.exp((val - new_val) / (temp + 1e-30)):
                            if new_val < val:
                                pos = new_pos
                                val = new_val
                            else:
                                pos = new_pos
                                val = new_val
                    # Expand step
                    step_size *= 1.1
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                else:
                    step_size *= 0.5
                    temp *= 0.9  # cool down
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
                best_pos = pos.copy()
                best_val = val
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreases from 0.25 to 0.05
            p = 0.25 * (1 - (n_evals / max_evals) ** 1.2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Weighted pbest selection (rank-based)
                sorted_idx = np.argsort(fitness)
                pbest_size = max(1, int(p * N))
                # Use ranks to compute probabilities (inverse rank)
                ranks = np.arange(1, pbest_size + 1)
                prob = 1.0 / ranks
                prob /= prob.sum()
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates, p=prob)
                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.2 * np.random.standard_cauchy(), 0, 1)  # larger scale
                while F <= 0:
                    F = np.clip(MF[mem] + 0.2 * np.random.standard_cauchy(), 0, 1)
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
                    if archive.shape[0] < archive_max:
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    else:
                        # randomly replace one archive member
                        idx = np.random.randint(archive.shape[0])
                        archive[idx] = pop[i]

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (use all successful delta_f)
            if len(S_F) > 0:
                # sort by improvement
                order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[order]
                S_CR = np.array(S_CR)[order]
                w = np.array(delta_f)[order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (quadratic)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                # Truncate archive to maintain size
                archive_max = int(2.5 * N_new)
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement
            if n_evals - last_local_search >= local_search_interval and n_evals < max_evals * 0.95:
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Adaptive step size
                step = 0.15 * (1 - n_evals / max_evals) ** 0.5 + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    # replace worst individual
                    worst_idx = np.argmax(fitness)
                    if new_val < fitness[worst_idx]:
                        pop[worst_idx] = new_pos
                        fitness[worst_idx] = new_val

            # Restart if stagnation (no improvement or low diversity)
            diversity = np.mean(np.std(pop, axis=0))
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8) or \
               (diversity < diversity_threshold and n_evals < max_evals * 0.9):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Use Sobol-like quasi-random for better coverage
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
                # Reset memory with increased average
                MF[:] = 0.7
                MCR[:] = 0.6
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = int(2.5 * N)
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt