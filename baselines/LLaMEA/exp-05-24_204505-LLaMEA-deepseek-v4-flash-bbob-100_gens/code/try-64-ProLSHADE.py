import numpy as np

class ProLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        max_evals = self.budget

        # Initial population size (adaptive)
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Quasi‑Latin hypercube initialization (improved diversity)
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

        # DE archives and success memories
        archive = np.empty((0, dim))
        archive_max = N
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        mem_idx = 0

        # Stagnation and diversity tracking
        best_hist = [self.f_opt]
        stall_evals = 0
        stall_threshold = 0.15 * max_evals

        # Local pattern-search parameters
        last_ls_evals = 0
        ls_interval = max(30, int(0.02 * max_evals))

        # Helper: dynamic pattern search with success‑rate step adaptation
        def pattern_search(best_pos, best_val, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step = 0.15 * (ub - lb) * (1 - n_evals / max_evals) ** 0.5 + 0.01
            used = 0
            success_steps = 0
            attempts = 0
            while used < max_local_evals and attempts < dim * 6:
                attempts += 1
                improved = False
                # Coordinate search
                for d in np.random.permutation(dim):
                    if used >= max_local_evals:
                        break
                    delta = step[d]
                    for sign in [1, -1]:
                        new_pos = pos.copy()
                        new_pos[d] = np.clip(pos[d] + sign * delta, lb[d], ub[d])
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved = True
                            break
                # Pattern move (accelerate along successful direction)
                if improved:
                    delta2 = pos - best_pos
                    if np.linalg.norm(delta2) > 1e-12:
                        new_pos2 = np.clip(pos + delta2, lb, ub)
                        new_val2 = func(new_pos2)
                        used += 1
                        if new_val2 < val:
                            pos = new_pos2
                            val = new_val2
                    step *= 1.2
                    step = np.minimum(step, 0.5 * (ub - lb))
                    success_steps += 1
                else:
                    # Random direction perturbation (increase diversity)
                    if np.random.rand() < 0.3 and used < max_local_evals:
                        rand_dir = np.random.uniform(-1, 1, dim)
                        rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-12)
                        stepsize = np.linalg.norm(step)
                        new_pos3 = np.clip(pos + rand_dir * stepsize, lb, ub)
                        new_val3 = func(new_pos3)
                        used += 1
                        if new_val3 < val:
                            pos = new_pos3
                            val = new_val3
                            improved = True
                    step *= 0.5
                    if np.max(step) < 1e-8 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05 with sin modulation
            ratio = 0.2 * np.sin(np.pi * (0.5 + 0.5 * n_evals / max_evals)) + 0.05
            p = max(ratio, 0.05)

            new_pop = pop.copy()
            new_fit = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Mutation: current-to-pbest/1 with archive
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from population ∪ archive
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

                # Sample F and CR from success memories
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: reflection (up to 10 tries)
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
                    stall_evals = 0
                else:
                    stall_evals += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fit[i] = trial_f
                    new_pop[i] = trial.copy()
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove, axis=0)

            pop, fitness = new_pop, new_fit

            # Update success memories with weighted Lehmer mean
            if S_F:
                order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[order]
                S_CR = np.array(S_CR)[order]
                w = np.array(delta_f)[order] / (np.sum(delta_f) + 1e-30)
                MF[mem_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[mem_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                mem_idx = (mem_idx + 1) % H

            # Population size reduction (sigmoidal schedule)
            t = n_evals / max_evals
            N_new = N_min + (N_init - N_min) * (1 / (1 + np.exp(10 * (t - 0.5))))
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

            # Periodic local refinement with pattern search
            if (n_evals - last_ls_evals >= ls_interval) and (n_evals < 0.95 * max_evals):
                last_ls_evals = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 4, max_evals - n_evals - 10)
                new_pos, new_val, used = pattern_search(best_pos, best_val, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        stall_evals = 0
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart: stagnation or low diversity (median distance to best < 1% of range)
            if stall_evals > stall_threshold and n_evals < 0.8 * max_evals:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                # Quasi‑oppositional initialization
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population via opposition around current best
                rnd = np.random.uniform(0, 1, (new_N, dim))
                new_pts = lb + rnd * (ub - lb)
                oppo = lb + ub - new_pts
                # Mix original and opposition points
                pop_new = np.vstack((new_pts, oppo))
                fit_new = np.full(2 * new_N, np.inf)
                pop_new[0] = best_ind
                fit_new[0] = best_fit
                for j in range(1, 2 * new_N):
                    fit_new[j] = func(pop_new[j])
                    n_evals += 1
                    if fit_new[j] < self.f_opt:
                        self.f_opt = fit_new[j]
                        self.x_opt = pop_new[j].copy()
                # Select best new_N individuals
                order = np.argsort(fit_new)[:new_N]
                pop = pop_new[order]
                fitness = fit_new[order]
                N = new_N
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                mem_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                stall_evals = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt