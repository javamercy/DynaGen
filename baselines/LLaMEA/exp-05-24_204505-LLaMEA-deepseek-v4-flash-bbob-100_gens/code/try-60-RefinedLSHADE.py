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

        # Population size
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

        # Archive (larger size for better diversity)
        archive_max = 2 * N
        archive = np.empty((0, dim))
        archive_fitness = np.array([])

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # Local search state
        pattern_step = 0.1 * (ub - lb)
        pattern_success = True
        pattern_evals_since_reset = 0

        # Helper: remove worst from archive when full
        def archive_add(ind, fit):
            nonlocal archive, archive_fitness
            archive = np.vstack((archive, ind.reshape(1, -1)))
            archive_fitness = np.append(archive_fitness, fit)
            if archive.shape[0] > archive_max:
                # remove the one with highest (worst) fitness
                worst_idx = np.argmax(archive_fitness)
                archive = np.delete(archive, worst_idx, axis=0)
                archive_fitness = np.delete(archive_fitness, worst_idx)

        # Pattern search (rotationally invariant, uses random directions)
        def local_search(base_pos, base_val, step_vec, max_evals_local):
            pos = base_pos.copy()
            val = base_val
            step = step_vec.copy()
            used = 0
            improved = True
            cycle = 0
            while used < max_evals_local and np.max(step) > 1e-10 * np.max(ub - lb):
                if improved:
                    # expand step on success
                    step *= 1.4
                    step = np.minimum(step, 0.4*(ub-lb))
                else:
                    step *= 0.6
                improved = False
                # try a few random directions (dim is small, use dim*2)
                for _ in range(min(dim*2, max_evals_local - used)):
                    dir = np.random.randn(dim)
                    dir = dir / (np.linalg.norm(dir)+1e-30)
                    # positive direction
                    new_pos = np.clip(pos + step * dir, lb, ub)
                    new_val = func(new_pos); used += 1
                    if new_val < val:
                        val = new_val; pos = new_pos; improved = True; break
                    # negative direction
                    new_pos = np.clip(pos - step * dir, lb, ub)
                    new_val = func(new_pos); used += 1
                    if new_val < val:
                        val = new_val; pos = new_pos; improved = True; break
            return pos, val, used

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_pos = pop[best_idx].copy()

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
                    # choose random with probability proportional to inverse fitness? no.
                    r2 = np.random.randint(len(archive)) if archive.shape[0]>0 else np.random.randint(N)
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2_idx = np.random.randint(union.shape[0])
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
                diff2 = pop[r1] - union[r2_idx]
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
                    archive_add(pop[i], fitness[i])

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

            # Population size reduction (quadratic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = 2 * N_new
                if archive.shape[0] > archive_max:
                    # keep best archived
                    keep_order = np.argsort(archive_fitness)[:archive_max]
                    archive = archive[keep_order]
                    archive_fitness = archive_fitness[keep_order]
                N = N_new

            # Periodic local refinement using pattern search
            local_interval = max(30, int(0.02 * max_evals))
            if (n_evals % local_interval == 0) and (n_evals < max_evals * 0.9):
                best_idx = np.argmin(fitness)
                base_pos = pop[best_idx].copy()
                base_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 5)
                step = pattern_step * (1 - n_evals / max_evals) + 0.01 * (ub - lb)
                new_pos, new_val, used = local_search(base_pos, base_val, step, max_local)
                n_evals += used
                if new_val < base_val:
                    pop[best_idx] = new_pos
                    fitness[best_idx] = new_val
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    pattern_step = step * 1.2
                else:
                    pattern_step = step * 0.5

            # Restart if stagnation or low diversity
            diversity = np.mean(np.std(pop, axis=0))
            if (evals_no_improve > restart_threshold or diversity < 0.01*(ub-lb).mean()) and n_evals < max_evals*0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # generate new population around best
                    pop = best_ind + 0.2 * np.random.randn(new_N, dim) * (ub-lb)
                    pop = np.clip(pop, lb, ub)
                    pop[0] = best_ind
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j]); n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]; self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # partial restart: keep best, replace rest with random
                    pop = lb + np.random.uniform(0,1,(N,dim))*(ub-lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j]); n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]; self.x_opt = pop[j].copy()
                    fitness[0] = best_fit
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_fitness = np.array([])
                archive_max = 2 * N
                evals_no_improve = 0
                pattern_step = 0.1 * (ub - lb)

            if n_evals >= max_evals:
                break

        # Final local search around best
        best_idx = np.argmin(fitness)
        base_pos = pop[best_idx].copy()
        base_val = fitness[best_idx]
        remaining = max_evals - n_evals
        if remaining > 10:
            step = 0.05 * (ub - lb)
            new_pos, new_val, used = local_search(base_pos, base_val, step, remaining)
            n_evals += used
            if new_val < base_val and new_val < self.f_opt:
                self.f_opt = new_val
                self.x_opt = new_pos.copy()

        return self.f_opt, self.x_opt