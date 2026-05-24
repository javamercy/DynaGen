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
        H = 15
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals

        # Local search parameters
        last_local_search = 0

        # For focused restart
        original_lb = lb.copy()
        original_ub = ub.copy()
        shrink_factor = 0.2  # factor to shrink domain around best

        # ---------- Local search implementations ----------
        def rotated_pattern_search(best_pos, best_val, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            # Generate random orthonormal basis (rotation matrix)
            # Use QR decomposition of random matrix to get random orthonormal columns
            R = np.random.randn(dim, dim)
            Q, _ = np.linalg.qr(R)
            # initial step size relative to domain range
            step0 = 0.1 * (ub - lb)
            step = step0.copy()
            used = 0
            iters = 0
            max_iters = min(50, max_evals_local // (2*dim))
            while used < max_evals_local and iters < max_iters:
                improved = False
                # Search along each direction in rotated basis
                for d in range(dim):
                    if used >= max_evals_local:
                        break
                    direction = Q[:, d]
                    # positive
                    new_pos = np.clip(pos + step[d] * direction, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative
                    new_pos = np.clip(pos - step[d] * direction, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: accelerate along net improvement direction
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    step *= 1.2
                    step = np.minimum(step, 0.5 * (ub - lb))
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step *= 0.5
                    if np.max(step) < 1e-10 * np.max(ub - lb):
                        break
                iters += 1
            return pos, val, used

        def bounded_nelder_mead(best_pos, best_val, max_evals_local):
            # Simple Nelder-Mead with bound handling via clamping
            # Build initial simplex around best
            n = dim
            sigma = 0.05 * (ub - lb)
            simplex = np.zeros((n+1, n))
            simplex[0] = best_pos.copy()
            for i in range(n):
                p = best_pos.copy()
                p[i] += sigma[i]
                p = np.clip(p, lb, ub)
                simplex[i+1] = p
            vals = np.full(n+1, np.inf)
            vals[0] = best_val
            for i in range(1, n+1):
                vals[i] = func(simplex[i])
            used = 1 + n  # we already evaluated best earlier, but count new ones
            # iteration
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma_s = 0.5
            while used < max_evals_local:
                # order
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]
                centroid = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                used += 1
                if fr < vals[-2]:
                    if fr < vals[0]:
                        # expansion
                        xe = centroid + gamma * (xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        used += 1
                        if fe < fr:
                            simplex[-1] = xe
                            vals[-1] = fe
                        else:
                            simplex[-1] = xr
                            vals[-1] = fr
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                else:
                    # contraction
                    if fr < vals[-1]:
                        xc = centroid + rho * (xr - centroid)
                    else:
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    used += 1
                    if fc < vals[-1]:
                        simplex[-1] = xc
                        vals[-1] = fc
                    else:
                        # shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            if i > 0:  # re-evaluate all but best
                                vals[i] = func(simplex[i])
                                used += 1
                # check if best improved
                if vals[0] < best_val:
                    best_val = vals[0]
                    best_pos = simplex[0].copy()
            return best_pos, best_val, used

        # Choose local search based on dimensionality
        if dim <= 10:
            local_search_fn = bounded_nelder_mead
        else:
            local_search_fn = rotated_pattern_search

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

            # Periodic local refinement
            local_search_interval = max(30, int(0.015 * max_evals * (1 - n_evals/max_evals) + 0.005*max_evals))
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 5)
                new_pos, new_val, used = local_search_fn(best_pos, best_val, max_local)
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

            # Restart based on stagnation and diversity
            diversity = np.mean(np.std(pop, axis=0))
            if (evals_no_improve > restart_threshold or diversity < 0.01*(np.max(ub-lb))) and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]

                # Focused restart: shrink bounds to 20% around best
                new_lb = np.clip(best_ind - shrink_factor * (original_ub - original_lb), original_lb, original_ub)
                new_ub = np.clip(best_ind + shrink_factor * (original_ub - original_lb), original_lb, original_ub)
                lb = new_lb
                ub = new_ub

                remaining = max_evals - n_evals
                new_N = min(N_init, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population within shrunken bounds
                if new_N > N:
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
                    # Partial restart
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()

                # Reset memories
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt