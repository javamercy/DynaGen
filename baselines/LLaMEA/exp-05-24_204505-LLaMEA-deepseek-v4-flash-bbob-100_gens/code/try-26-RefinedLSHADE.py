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

        # Quasi-random initialization (Sobol-like) using random latin hypercube
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

        # Mutation strategy success tracking
        # strategies: 0 = current-to-pbest/1, 1 = current-to-rand/1, 2 = w/rank-based
        strategy_probs = np.ones(3) / 3
        strategy_success = np.zeros(3)
        strategy_attempts = np.zeros(3)

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # Local search parameters (Nelder-Mead)
        local_search_interval = max(50, int(0.02 * max_evals))
        last_local_search = 0

        # Covariance estimation for restart
        cov = np.eye(dim) * (ub - lb).mean() * 0.1

        def nelder_mead(x0, f0, max_evals_local):
            """Nelder-Mead simplex local search with bounds."""
            n = dim
            # Build initial simplex
            simplex = np.zeros((n+1, n))
            simplex[0] = x0
            vals = np.zeros(n+1)
            vals[0] = f0
            used = 0
            # Generate other points
            delta = (ub - lb) * 0.05
            for i in range(n):
                new_x = x0.copy()
                new_x[i] += delta[i]
                new_x = np.clip(new_x, lb, ub)
                simplex[i+1] = new_x
                vals[i+1] = func(new_x)
                used += 1
                if vals[i+1] < self.f_opt:
                    self.f_opt = vals[i+1]
                    self.x_opt = simplex[i+1].copy()
                if used >= max_evals_local:
                    return simplex[0], vals[0], used

            alpha, beta, gamma = 1.0, 0.5, 2.0  # reflection, contraction, expansion
            while used < max_evals_local:
                # Order
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                used += 1
                if fr < self.f_opt:
                    self.f_opt = fr
                    self.x_opt = xr.copy()
                if vals[0] <= fr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fr
                elif fr < vals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    used += 1
                    if fe < self.f_opt:
                        self.f_opt = fe
                        self.x_opt = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                else:
                    # Contraction
                    xc = centroid + beta * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    used += 1
                    if fc < self.f_opt:
                        self.f_opt = fc
                        self.x_opt = xc.copy()
                    if fc < vals[-1]:
                        simplex[-1] = xc
                        vals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            vals[i] = func(simplex[i])
                            used += 1
                            if vals[i] < self.f_opt:
                                self.f_opt = vals[i]
                                self.x_opt = simplex[i].copy()
                            if used >= max_evals_local:
                                break
                if used >= max_evals_local:
                    break
            best_idx = np.argmin(vals)
            return simplex[best_idx], vals[best_idx], used

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
                # Choose mutation strategy adaptively
                if strategy_attempts.sum() > 10:
                    strategy_probs = strategy_success / (strategy_attempts + 1e-30)
                    strategy_probs /= strategy_probs.sum()
                strategy = np.random.choice(3, p=strategy_probs)
                strategy_attempts[strategy] += 1

                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Generate mutant based on strategy
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                if strategy == 0:  # current-to-pbest/1/archive
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                elif strategy == 1:  # current-to-rand/1
                    base = pop[i]
                    diff1 = pop[r1] - base
                    r3 = np.random.randint(N)
                    while r3 == i or r3 == r1:
                        r3 = np.random.randint(N)
                    diff2 = pop[r3] - base
                    mutant = base + F * diff1 + F * diff2
                else:  # strategy 2: rank-based differential
                    # Use difference of two random individuals, weighted by fitness rank
                    r3 = np.random.randint(N)
                    while r3 == i or r3 == r1:
                        r3 = np.random.randint(N)
                    rank1 = np.where(sorted_idx == r1)[0][0] / N
                    rank3 = np.where(sorted_idx == r3)[0][0] / N
                    weight = 1.0 - rank1 + rank3  # favor better solutions
                    base = pop[i]
                    mutant = base + F * (pop[r1] - pop[r3]) * weight + F * (pop[pbest_idx] - base) * (1 - weight)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
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
                    strategy_success[strategy] += 1
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

            # Periodic local refinement using Nelder-Mead
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 10)
                new_pos, new_val, used = nelder_mead(best_pos, best_val, max_local)
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

            # Restart if stagnation detected: use covariance-based resampling
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)

                # Estimate covariance from best solutions in archive and population
                top_k = min(10, len(pop))
                top_idx = np.argsort(fitness)[:top_k]
                top_points = pop[top_idx]
                cov = np.cov(top_points, rowvar=False) + 1e-10 * np.eye(dim)
                # Ensure symmetric positive definite
                L = np.linalg.cholesky(cov)
                # Generate new population centered at best
                new_pop = best_ind + 0.5 * (L @ np.random.randn(dim, new_N)).T
                new_pop = np.clip(new_pop, lb, ub)
                new_pop[0] = best_ind  # keep best
                pop = new_pop
                fitness = np.full(new_N, np.inf)
                for j in range(new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                strategy_success[:] = 0
                strategy_attempts[:] = 1  # avoid zero division

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt