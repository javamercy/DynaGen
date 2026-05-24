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
        range_ = ub - lb

        # Population size parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization (improved)
        samples = np.random.uniform(0, 1, (N, dim))
        samples = lb + samples * range_
        pop = samples.copy()
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

        # Success-history memory (increased size)
        H = 15
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = int(0.1 * max_evals)  # stricter restart
        diversity_threshold = 0.01 * np.linalg.norm(range_)  # small diversity triggers restart

        # Local search parameters
        last_local_search = 0
        local_search_interval = max(30, int(0.015 * max_evals))

        # Success-rate adaptive pattern search
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            # initial step size relative to range
            step_size = step * range_
            successes = 0
            trials_local = 0
            used = 0
            while used < max_local_evals and trials_local < dim * 8:
                improved = False
                trials_local += 1
                # Coordinate search (randomized order)
                order = np.random.permutation(dim)
                for d in order:
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
                        successes += 1
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
                        successes += 1
                if improved:
                    # pattern move
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            successes += 1
                    # expand step based on success rate
                    success_rate = successes / max(1, trials_local)
                    if success_rate > 0.5:
                        step_size *= 1.2
                    else:
                        step_size *= 0.9
                    step_size = np.clip(step_size, 1e-8 * range_, 0.5 * range_)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # contract
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(range_):
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
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # Sample F (Cauchy, positive half)
                mem = np.random.randint(H)
                F = MF[mem] + 0.1 * np.random.standard_cauchy()
                F = np.clip(F, 0, 1)
                while F <= 0:
                    F = MF[mem] + 0.1 * np.random.standard_cauchy()
                    F = np.clip(F, 0, 1)
                # Sample CR (truncated normal)
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
                # Improved boundary handling: reflect and clamp
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
                    # Add parent to archive (random removal when full)
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

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

            # Linear population size reduction (finer control)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals)
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    # Keep archive solutions with best fitness or random? Use random.
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
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

            # Diversity calculation for restart
            if N > 1:
                pop_center = np.mean(pop, axis=0)
                avg_dist = np.mean(np.linalg.norm(pop - pop_center, axis=1))
            else:
                avg_dist = 0.0

            # Restart if stagnation or low diversity
            restart_condition = (evals_no_improve > restart_threshold) or \
                                (avg_dist < diversity_threshold and n_evals > 0.2 * max_evals)
            if restart_condition and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population using Latin hypercube with best point preserved
                pop_new = lb + np.random.uniform(0, 1, (new_N, dim)) * range_
                pop_new[0] = best_ind
                fitness_new = np.full(new_N, np.inf)
                fitness_new[0] = best_fit
                # Also add some Cauchy-perturbed best individuals for diversity
                num_perturb = min(int(new_N * 0.3), new_N - 1)
                for j in range(num_perturb):
                    pert = best_ind + np.random.standard_cauchy(dim) * 0.1 * range_
                    pert = np.clip(pert, lb, ub)
                    pop_new[1 + j] = pert
                    fitness_new[1 + j] = func(pop_new[1 + j])
                    n_evals += 1
                    if fitness_new[1 + j] < self.f_opt:
                        self.f_opt = fitness_new[1 + j]
                        self.x_opt = pop_new[1 + j].copy()
                        evals_no_improve = 0
                # Evaluate remaining random points
                for j in range(num_perturb + 1, new_N):
                    fitness_new[j] = func(pop_new[j])
                    n_evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = pop_new[j].copy()
                        evals_no_improve = 0
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

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt