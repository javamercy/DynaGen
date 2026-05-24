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

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # --- Nelder-Mead simplex local search (bounded) ---
        def simplex_search(best_pos, best_val, step, max_local_evals):
            # Build initial simplex around best_pos with size = step * (ub-lb)
            n = dim
            sigma = step * (ub - lb)  # array of step sizes per dimension
            simplex = np.zeros((n + 1, n))
            simplex[0] = best_pos
            for i in range(n):
                p = best_pos.copy()
                p[i] = np.clip(p[i] + sigma[i], lb[i], ub[i])
                simplex[i+1] = p
            simplex_vals = np.full(n+1, np.nan)
            simplex_vals[0] = best_val
            used = 0
            # Evaluate the other points
            for i in range(1, n+1):
                if used >= max_local_evals:
                    break
                simplex_vals[i] = func(simplex[i])
                used += 1
                if simplex_vals[i] < self.f_opt:
                    self.f_opt = simplex_vals[i]
                    self.x_opt = simplex[i].copy()
                    evals_no_improve = 0

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_shrink = 0.5  # shrink

            while used < max_local_evals:
                # Order simplex by fitness
                idx = np.argsort(simplex_vals)
                simplex = simplex[idx]
                simplex_vals = simplex_vals[idx]

                if used >= max_local_evals:
                    break
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), lb, ub)
                fr = func(xr)
                used += 1
                if fr < self.f_opt:
                    self.f_opt = fr
                    self.x_opt = xr.copy()
                    evals_no_improve = 0

                if simplex_vals[0] <= fr < simplex_vals[-2]:
                    sim=simplex.copy(); sim_vals=simplex_vals.copy()
                    sim[-1]=xr; sim_vals[-1]=fr
                elif fr < simplex_vals[0]:
                    # Expansion
                    xe = np.clip(centroid + gamma * (xr - centroid), lb, ub)
                    fe = func(xe)
                    used += 1
                    if fe < self.f_opt:
                        self.f_opt = fe
                        self.x_opt = xe.copy()
                        evals_no_improve = 0
                    if fe < fr:
                        simplex[-1]=xe; simplex_vals[-1]=fe
                    else:
                        simplex[-1]=xr; simplex_vals[-1]=fr
                else:
                    # Contraction
                    if fr < simplex_vals[-1]:
                        xc = np.clip(centroid + rho * (xr - centroid), lb, ub)
                        fc = func(xc)
                        used += 1
                        if fc < self.f_opt:
                            self.f_opt = fc
                            self.x_opt = xc.copy()
                            evals_no_improve = 0
                        if fc < fr:
                            simplex[-1]=xc; simplex_vals[-1]=fc
                        else:
                            # Shrink
                            for i in range(1,n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), lb, ub)
                                simplex_vals[i] = func(simplex[i])
                                used += 1
                                if simplex_vals[i] < self.f_opt:
                                    self.f_opt = simplex_vals[i]
                                    self.x_opt = simplex[i].copy()
                                    evals_no_improve = 0
                    else:
                        # Contract around best
                        xc = np.clip(centroid + rho * (simplex[-1] - centroid), lb, ub)
                        fc = func(xc)
                        used += 1
                        if fc < self.f_opt:
                            self.f_opt = fc
                            self.x_opt = xc.copy()
                            evals_no_improve = 0
                        if fc < simplex_vals[-1]:
                            simplex[-1]=xc; simplex_vals[-1]=fc
                        else:
                            # Shrink
                            for i in range(1,n+1):
                                simplex[i] = np.clip(simplex[0] + sigma_shrink * (simplex[i] - simplex[0]), lb, ub)
                                simplex_vals[i] = func(simplex[i])
                                used += 1
                                if simplex_vals[i] < self.f_opt:
                                    self.f_opt = simplex_vals[i]
                                    self.x_opt = simplex[i].copy()
                                    evals_no_improve = 0
            # Return best from simplex
            best_idx = np.argmin(simplex_vals)
            return simplex[best_idx], simplex_vals[best_idx], used

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

            # Population size reduction (quadratic schedule)
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

            # Periodic local refinement using simplex
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Step size inversely proportional to remaining evals
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = simplex_search(best_pos, best_val, step, max_local)
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

            # Restart if stagnation detected
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                # Keep top 3 individuals for diversity
                sorted_idx = np.argsort(fitness)
                top_inds = sorted_idx[:min(3, N)]
                top_positions = pop[top_inds].copy()
                top_fitness = fitness[top_inds].copy()
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Quasi-random Latin hypercube
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    # Place top individuals
                    for j, (pos, fit_val) in enumerate(zip(top_positions, top_fitness)):
                        pop[j] = pos
                        fitness[j] = fit_val
                    for j in range(len(top_inds), new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: randomize all but best and some top
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    for j, (pos, fit_val) in enumerate(zip(top_positions, top_fitness)):
                        pop[j] = pos
                        fitness[j] = fit_val
                    for j in range(len(top_inds), N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory parameters with a mix of old and new
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt