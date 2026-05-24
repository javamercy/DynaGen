import numpy as np

class RefinedLSHADE_v2:
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
        N_init = min(max(10 * dim, 60), max_evals // 2)
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
        H = 12
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_stagnation = 0.12 * max_evals
        # Diversity restart
        min_diversity_threshold = 0.05 * (ub - lb).mean()
        restart_diversity = 0

        # Local search memory for step sizes (pattern search)
        ls_step = 0.15 * (ub - lb).mean()
        ls_step_memory = [ls_step]
        ls_success = True

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing with accelerated decay
            t = n_evals / max_evals
            p = 0.2 * (1 - t ** 1.8) + 0.05

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
                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # Mutation: current-to-pbest/1/archive with occasional mean-based direction
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                # Use weighted difference with random scaling
                if np.random.rand() < 0.15:  # occasionally use population mean
                    mean_pop = np.mean(pop, axis=0)
                    diff1 = 0.5 * diff1 + 0.5 * (mean_pop - base)
                mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Boundary handling: reflection and clamping
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

            # Population size reduction (quadratic schedule with earlier reduction)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 1.5
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

            # Adaptive local search (pattern search with memory-based step)
            if n_evals < max_evals * 0.95:
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]

                # Use step from memory, adapt based on success
                step = ls_step_memory[-1] if ls_step_memory else 0.15 * (ub - lb).mean()
                # Decay step as evaluations progress
                step *= (1 - 0.5 * (n_evals / max_evals))
                step = max(step, 1e-6 * (ub - lb).mean())

                max_local_evals = min(dim * 3, max_evals - n_evals - 5)
                local_evals_used = 0
                improved = False

                # Pattern search with random coordinate order
                order = np.random.permutation(dim)
                for _ in range(2):  # repeat twice
                    for d in order:
                        if local_evals_used >= max_local_evals:
                            break
                        # positive direction
                        delta = np.zeros(dim)
                        delta[d] = step
                        new_pos = np.clip(best_pos + delta, lb, ub)
                        new_val = func(new_pos)
                        local_evals_used += 1
                        if new_val < best_val:
                            best_pos = new_pos
                            best_val = new_val
                            improved = True
                            continue
                        # negative direction
                        new_pos = np.clip(best_pos - delta, lb, ub)
                        new_val = func(new_pos)
                        local_evals_used += 1
                        if new_val < best_val:
                            best_pos = new_pos
                            best_val = new_val
                            improved = True
                    # Pattern move: if any improvement, accelerate along success direction
                    if improved:
                        # attempt a larger step in the aggregated direction
                        shift = best_pos - pop[best_idx]
                        if np.linalg.norm(shift) > 1e-12:
                            new_pos = np.clip(best_pos + shift, lb, ub)
                            new_val = func(new_pos)
                            local_evals_used += 1
                            if new_val < best_val:
                                best_pos = new_pos
                                best_val = new_val
                                improved = True
                        # Expand step on success
                        step *= 1.3
                        step = min(step, 0.5 * (ub - lb).mean())
                    else:
                        # Contract step on failure
                        step *= 0.6
                    if step < 1e-10 * (ub - lb).mean():
                        break
                n_evals += local_evals_used
                # Update local search step memory
                ls_step_memory.append(step)
                if len(ls_step_memory) > 20:
                    ls_step_memory.pop(0)

                if best_val < fitness[best_idx]:
                    # Replace worst individual with improved best
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0

            # Diversity check for restart
            pop_std = np.std(pop, axis=0).mean()
            diversity_low = pop_std < min_diversity_threshold
            stagnation = evals_no_improve > restart_stagnation
            if (stagnation or diversity_low) and n_evals < max_evals * 0.8:
                # Restart: keep best individual, generate new population
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Use Latin hypercube with some perturbation around best
                samples = np.random.uniform(0, 1, (new_N, dim))
                samples = lb + samples * (ub - lb)
                # Mix best into population
                pop = samples.copy()
                fitness = np.full(new_N, np.inf)
                pop[0] = best_ind
                fitness[0] = best_fit
                # Evaluate new individuals (random order to improve diversity)
                indices = list(range(1, new_N))
                np.random.shuffle(indices)
                for j in indices:
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset parameters
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                ls_step_memory = [0.15 * (ub - lb).mean()]

            if n_evals >= max_evals:
                break

        # Final local search on best (if budget allows)
        if n_evals < max_evals:
            best_idx = np.argmin(fitness)
            best_pos = pop[best_idx].copy()
            best_val = fitness[best_idx]
            remaining = max_evals - n_evals
            local_evals = 0
            step = max(0.01 * (ub - lb).mean(), 1e-8)
            for _ in range(min(remaining, dim * 4)):
                improved = False
                order = np.random.permutation(dim)
                for d in order:
                    if local_evals >= remaining:
                        break
                    for direction in [1, -1]:
                        if local_evals >= remaining:
                            break
                        delta = np.zeros(dim)
                        delta[d] = step * direction
                        new_pos = np.clip(best_pos + delta, lb, ub)
                        new_val = func(new_pos)
                        local_evals += 1
                        if new_val < best_val:
                            best_pos = new_pos
                            best_val = new_val
                            improved = True
                if improved:
                    step *= 1.2
                else:
                    step *= 0.5
                if step < 1e-12 * (ub - lb).mean():
                    break
            n_evals += local_evals
            if best_val < self.f_opt:
                self.f_opt = best_val
                self.x_opt = best_pos.copy()

        return self.f_opt, self.x_opt