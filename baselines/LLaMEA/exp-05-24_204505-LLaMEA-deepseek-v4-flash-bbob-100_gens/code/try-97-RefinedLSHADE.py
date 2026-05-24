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
        N_min = max(4, dim // 5)
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
        evals = N

        # Archive
        archive = []
        archive_max = N

        # Success-history memory
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        mem_idx = 0
        # Stagnation
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals
        restart_count = 0
        max_restarts = 2
        # diversity measure
        diversity = 0.0

        # --- Nelder-Mead local search (simplex) ---
        def nelder_mead(best_pos, best_val, max_fe):
            # Build initial simplex around best_pos
            n = dim
            simplex = np.zeros((n+1, n))
            values = np.zeros(n+1)
            # best point as first vertex
            simplex[0] = best_pos.copy()
            values[0] = best_val
            # generate others by perturbing each coordinate
            for i in range(n):
                perturb = 0.05 * (ub[i]-lb[i]) * (2*np.random.rand() - 1)
                simplex[i+1] = np.clip(best_pos.copy() + (ub[i]-lb[i])*0.05*(i==np.arange(n)).astype(float) + perturb*0.5, lb[i], ub[i])
                # only 1 extra evaluation per new vertex? we'll evaluate later
            # Actually evaluate new vertices
            for i in range(1, n+1):
                values[i] = func(simplex[i])
                # track improvements
            fe = 1 + n  # we already counted best_val as given, but need to count these new evaluations
            # Now perform Nelder-Mead steps
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            while fe < max_fe:
                # order
                idx = np.argsort(values)
                simplex = simplex[idx]
                values = values[idx]
                # centroid of best n points
                centroid = np.mean(simplex[:-1], axis=0)
                # Reflect
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                fe += 1
                if values[0] <= fr < values[-2]:
                    simplex[-1] = xr
                    values[-1] = fr
                elif fr < values[0]:
                    # Expand
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, lb, ub)
                    fe_val = func(xe)
                    fe += 1
                    if fe_val < fr:
                        simplex[-1] = xe
                        values[-1] = fe_val
                    else:
                        simplex[-1] = xr
                        values[-1] = fr
                else:
                    # Contract
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    fe += 1
                    if fc < values[-1]:
                        simplex[-1] = xc
                        values[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n+1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            values[i] = func(simplex[i])
                            fe += 1
                # early break if converged
                if np.std(values) < 1e-12 * (np.max(values) + 1e-30):
                    break
            # Return best vertex found
            best_idx = np.argmin(values)
            return simplex[best_idx], values[best_idx], fe

        # --- Main loop ---
        while evals < max_evals:
            # pbest ratio with quadratic decay (aggressive)
            p = 0.2 * (1 - (evals / max_evals) ** 2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # choose distinct indices
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                union = pop
                if archive:
                    union = np.vstack((pop, np.array(archive)))
                r2 = np.random.randint(union.shape[0])
                # pbest index
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # F and CR
                mem = np.random.randint(H)
                F = MF[mem] + 0.1 * np.random.standard_cauchy()
                F = np.clip(F, 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = MCR[mem] + 0.1 * np.random.randn()
                CR = np.clip(CR, 0, 1)
                # mutation current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # boundary handling: reflect and clamp
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2*lb - trial, trial)
                    trial = np.where(out_high, 2*ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_f = func(trial)
                evals += 1
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
                    # add parent to archive (FIFO)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[order]
                S_CR = np.array(S_CR)[order]
                w = np.array(delta_f)[order] / (np.sum(delta_f) + 1e-30)
                MF[mem_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[mem_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                mem_idx = (mem_idx + 1) % H

            # Population size reduction (cubic)
            N_new = N_min + (N_init - N_min) * ((max_evals - evals) / max_evals)**3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                idx_sort = np.argsort(fitness)
                pop = pop[idx_sort[:N_new]]
                fitness = fitness[idx_sort[:N_new]]
                archive_max = N_new
                if len(archive) > archive_max:
                    archive = archive[-archive_max:]
                N = N_new

            # Diversity measure: average distance to best
            best_idx = np.argmin(fitness)
            best_pos = pop[best_idx].copy()
            diversity = np.mean([np.linalg.norm(p - best_pos) for p in pop])
            # Periodic local search using Nelder-Mead (simplex)
            if evals < max_evals * 0.95 and (evals % max(20, int(0.015*max_evals)) == 0 or (diversity < 0.01 * np.max(ub-lb) and evals > max_evals*0.3)):
                best_val = fitness[best_idx]
                max_fe = min(dim*8, max_evals - evals - 5)
                if max_fe >= dim+1:
                    new_pos, new_val, used = nelder_mead(best_pos, best_val, max_fe)
                    evals += used
                    if new_val < best_val:
                        best_val = new_val
                        best_pos = new_pos
                        if best_val < self.f_opt:
                            self.f_opt = best_val
                            self.x_opt = best_pos.copy()
                            evals_no_improve = 0
                        # replace worst individual
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = best_pos
                        fitness[worst_idx] = best_val

            # Restart if stagnation (diversity or no improvement)
            if (evals_no_improve > restart_threshold or diversity < 1e-8 * np.max(ub-lb)) and evals < max_evals*0.8 and restart_count < max_restarts:
                restart_count += 1
                best_ind = pop[np.argmin(fitness)].copy()
                best_fit = fitness[np.argmin(fitness)]
                remaining = max_evals - evals
                new_N = min(N_init*2, N*2, remaining//2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    samples = np.random.uniform(0,1,(new_N,dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    pop = lb + np.random.uniform(0,1,(N,dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1,N):
                        fitness[j] = func(pop[j])
                        evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory with variation
                MF[:] = 0.5 + 0.2*np.random.rand(H)
                MCR[:] = 0.8 + 0.2*np.random.rand(H)
                mem_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0

            if evals >= max_evals:
                break

        return self.f_opt, self.x_opt