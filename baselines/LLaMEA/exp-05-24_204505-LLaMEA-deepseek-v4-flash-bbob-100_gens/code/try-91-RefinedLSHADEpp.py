import numpy as np

class RefinedLSHADEpp:
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

        # FIFO archive
        archive = []
        archive_max = N

        # Success-history memory for F and CR
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection via diversity
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals
        restart_count = 0
        max_restarts = 2

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0

        # Diversity measure for restart
        diversity_threshold = 0.05  # fraction of domain range

        def bounce_back(x, lb, ub):
            """Bounce-back boundary handling."""
            for _ in range(10):
                out_low = x < lb
                out_high = x > ub
                if not (np.any(out_low) or np.any(out_high)):
                    break
                x = np.where(out_low, lb + (lb - x), x)
                x = np.where(out_high, ub - (x - ub), x)
            return np.clip(x, lb, ub)

        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            iterations = 0
            while used < max_local_evals and iterations < dim * 6:
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
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
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
                    # Pattern move: accelerate along direction of improvement
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size on success (gentler)
                    step_size *= 1.1
                    step_size = np.minimum(step_size, (ub - lb) * 0.4)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size on failure (faster)
                    step_size *= 0.7
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        def population_diversity(pop, lb, ub):
            """Average distance to centroid, normalized by domain range."""
            centroid = np.mean(pop, axis=0)
            dists = np.sqrt(np.sum((pop - centroid)**2, axis=1))
            avg_dist = np.mean(dists)
            norm = np.sqrt(dim) * (ub - lb).mean()
            return avg_dist / norm if norm > 1e-12 else 0.0

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: adaptive with quadratic decay, plus random jitter
            p_base = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05
            p = np.clip(p_base + 0.05 * np.random.randn(), 0.01, 0.5)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive (FIFO)
                if archive:
                    union = np.vstack((pop, np.array(archive)))
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
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: bounce-back
                trial = bounce_back(trial, lb, ub)
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
                    # Add parent to archive (FIFO)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (improved stability)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.clip(np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30), 0.2, 0.9)
                MCR[memory_idx] = np.clip(np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30), 0, 1)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (cubic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if len(archive) > archive_max:
                    archive = archive[-archive_max:]
                N = N_new

            # Periodic local refinement using pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size: larger initial step in later phase to escape local optima
                step = 0.2 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
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

            # Diversity-aware restart (instead of stagnation count)
            div = population_diversity(pop, lb, ub)
            need_restart = (div < diversity_threshold) or (evals_no_improve > restart_threshold)
            if need_restart and n_evals < max_evals * 0.8 and restart_count < max_restarts:
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 4)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Generate new population with Latin hypercube, keep best
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
                else:
                    # Partial restart: randomize all but best
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory to promote exploration
                MF[:] = 0.5 + 0.2 * np.random.rand(H)
                MCR[:] = 0.8 + 0.2 * np.random.rand(H)
                memory_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt