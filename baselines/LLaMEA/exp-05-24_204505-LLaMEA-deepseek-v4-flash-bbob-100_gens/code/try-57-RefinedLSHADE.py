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

        # Population size parameters (quadratic reduction)
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init
        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = N
        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0
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
        # Stagnation and diversity tracking
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals
        # Local search parameters
        last_local_search = 0
        local_search_interval = max(30, int(0.02 * max_evals))
        # Mutation strategy probabilities (adaptive selection)
        strategy_probs = np.array([0.8, 0.2])  # current-to-pbest/1, current-to-rand/1
        strategy_success = np.zeros(2)
        strategy_trials = np.ones(2)

        # Random-direction line search (efficient for high dimensions)
        def line_search(best_x, best_val, dir_vec, max_evals_local):
            dir_norm = np.linalg.norm(dir_vec)
            if dir_norm < 1e-12:
                return best_x, best_val, 0
            dir_unit = dir_vec / dir_norm
            step = 0.5 * (ub - lb).mean() * min(1.0, dir_norm)  # adaptive initial step
            # Expand step on success, contract on failure (max 3 steps)
            improved = True
            used = 0
            while used < max_evals_local and step > 1e-10 * (ub-lb).mean():
                # positive direction
                x_candidate = np.clip(best_x + step * dir_unit, lb, ub)
                val = func(x_candidate)
                used += 1
                if val < best_val:
                    best_x = x_candidate
                    best_val = val
                    step *= 2.0
                    improved = True
                    continue
                # negative direction
                x_candidate = np.clip(best_x - step * dir_unit, lb, ub)
                val = func(x_candidate)
                used += 1
                if val < best_val:
                    best_x = x_candidate
                    best_val = val
                    step *= 2.0
                    improved = True
                    continue
                # no improvement, contract step
                step *= 0.5
                improved = False
            return best_x, best_val, used

        # Main evolutionary loop
        while n_evals < max_evals:
            # Compute pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05
            # Diversity measure: average distance to centroid
            centroid = pop.mean(axis=0)
            diversity = np.mean(np.sqrt(((pop - centroid) ** 2).sum(axis=1)))
            div_thresh = 0.01 * (ub - lb).mean() * dim ** 0.5
            # Trigger restart if very low diversity and no improvement recently
            if diversity < div_thresh and evals_no_improve > 0.1 * restart_threshold:
                # preserve best
                best_idx = np.argmin(fitness)
                best_x = pop[best_idx].copy()
                best_f = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population around best with Cauchy perturbation
                pop = best_x + 0.1 * np.random.standard_cauchy((new_N, dim)) * (ub - lb)
                pop = np.clip(pop, lb, ub)
                pop[0] = best_x
                fitness = np.full(new_N, np.inf)
                fitness[0] = best_f
                for j in range(1, new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                archive = np.empty((0, dim))
                archive_max = N
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                evals_no_improve = 0
                continue

            # Generate offspring
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            # Choose mutation strategy adaptively
            if np.random.rand() < strategy_probs[0] / strategy_probs.sum():
                strategy_choice = 0  # current-to-pbest/1
            else:
                strategy_choice = 1  # current-to-rand/1
            for i in range(N):
                # Select indices
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest index (only for strategy 0)
                if strategy_choice == 0:
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
                # Mutation
                if strategy_choice == 0:
                    # current-to-pbest/1/archive
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    # current-to-rand/1 (diversity enhancement)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = np.random.uniform(-1, 1, dim) * (ub - lb) * 0.25
                    mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: bounce-back
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, lb + np.random.uniform(0, 0.1) * (ub - lb), trial)
                    trial = np.where(out_high, ub - np.random.uniform(0, 0.1) * (ub - lb), trial)
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
                # Selection
                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial.copy()
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                    # Update strategy success (only for this individual)
                    strategy_success[strategy_choice] += 1
                strategy_trials[strategy_choice] += 1

            # Update population
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

            # Update adaptation of mutation strategy probabilities
            for s in range(2):
                if strategy_trials[s] > 0:
                    strategy_probs[s] = strategy_success[s] / strategy_trials[s] + 1e-6
            # Normalize
            strategy_probs /= strategy_probs.sum()

            # Quadratic population size reduction
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

            # Periodic local search with random-direction line search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_x = pop[best_idx].copy()
                best_f = fitness[best_idx]
                # Use current best and random direction for line search
                for _ in range(min(3, dim)):
                    if n_evals >= max_evals - 5:
                        break
                    dir_vec = np.random.randn(dim)
                    max_local = min(6, max_evals - n_evals - 5)
                    new_x, new_f, used = line_search(best_x, best_f, dir_vec, max_local)
                    n_evals += used
                    if new_f < best_f:
                        best_x = new_x
                        best_f = new_f
                        if best_f < self.f_opt:
                            self.f_opt = best_f
                            self.x_opt = best_x.copy()
                            evals_no_improve = 0
                # Replace worst individual in population if improvement
                if best_f < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_x
                    fitness[worst_idx] = best_f

            # Simple stagnation restart (keep as safety net)
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_x = pop[best_idx].copy()
                best_f = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Quasi-random Latin hypercube
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    pop = lb + samples * (ub - lb)
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_x
                    fitness[0] = best_f
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
                    pop[0] = best_x
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                strategy_probs[:] = [0.8, 0.2]
                strategy_success[:] = 0
                strategy_trials[:] = 1

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt