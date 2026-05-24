import numpy as np
from scipy.stats import qmc

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

        # Sobol initialization for better coverage
        sampler = qmc.Sobol(d=dim, scramble=True, seed=np.random.randint(0, 2**31))
        samples = sampler.random(n=N)
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = int(2.5 * N_init)  # larger archive

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals  # slightly more aggressive

        # Diversity tracking for restart
        prev_pop = pop.copy()
        diversity_hist = []

        # Local search parameters
        local_search_interval = max(30, int(0.015 * max_evals))
        last_local_search = 0

        # Rotational pattern search with adaptive step (handles ill-conditioned cases)
        def rotational_pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative step
            # generate a random orthonormal basis for rotation
            Q, _ = np.linalg.qr(np.random.randn(dim, dim))
            used = 0
            while used < max_local_evals:
                improved = False
                # Explore along rotated axes
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    dir_vec = Q[:, d] * step_size[d]
                    new_pos = np.clip(pos + dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = np.clip(pos - dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # pattern move: accelerate along net direction
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # expand step size
                    step_size *= 1.15
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    # regenerated Q to avoid axis stagnation
                    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # contract and rotate
                    step_size *= 0.6
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
                    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
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

            # Generate offspring with strategy pool: sometimes use current-to-rand/1 for diversity
            for i in range(N):
                # decide mutation strategy: 80% current-to-pbest, 20% current-to-rand
                use_rand = np.random.rand() < 0.2 and n_evals < 0.6 * max_evals
                if use_rand:
                    # current-to-rand/1 (no crossover, rotation-invariant)
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
                    F = np.clip(0.5 + 0.1 * np.random.randn(), 0.2, 0.9)
                    mutant = pop[i] + F * (pop[r1] - pop[r2]) + F * (pop[r3] - pop[i])
                    trial = mutant  # no crossover
                else:
                    # standard current-to-pbest/1/archive
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
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
                    CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                    # binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                # boundary handling: reflection + clamp
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                # evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    if not use_rand:  # only store params from pbest strategy
                        S_F.append(F)
                        S_CR.append(CR)
                        delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive (fitness-based pruning later)
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        # remove a random solution from archive
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means (only from pbest strategy)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (quadratic schedule)
            frac = max(0, (max_evals - n_evals) / max_evals)
            N_new = N_min + (N_init - N_min) * frac ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = int(2.5 * N_new)
                if archive.shape[0] > archive_max:
                    # keep a diverse subset: choose random
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Diversity measure (average pairwise distance normalized)
            if n_evals > 0 and np.mod(n_evals, 100) == 0:
                # compute diversity in current population
                dists = 0.0
                cnt = 0
                for i in range(min(N, 50)):
                    for j in range(i+1, min(N, 50)):
                        dists += np.linalg.norm(pop[i] - pop[j])
                        cnt += 1
                if cnt > 0:
                    div = dists / cnt
                    diversity_hist.append(div)
                else:
                    diversity_hist.append(0.0)

            # Periodic local refinement using rotational pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = rotational_pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # replace worst individual
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation detected (no improvement or diversity collapse)
            restart_flag = False
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                restart_flag = True
            if len(diversity_hist) >= 10:
                # if diversity drops below 20% of initial diversity, restart
                initial_div = diversity_hist[0] if diversity_hist[0] > 0 else 1.0
                if diversity_hist[-1] < 0.2 * initial_div:
                    restart_flag = True
            if restart_flag:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Sobol restart for better coverage
                    sampler = qmc.Sobol(d=dim, scramble=True, seed=np.random.randint(0, 2**31))
                    samples = sampler.random(n=new_N)
                    pop = lb + samples * (ub - lb)
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
                    # partial restart: Sobol for new individuals
                    sampler = qmc.Sobol(d=dim, scramble=True, seed=np.random.randint(0, 2**31))
                    samples = sampler.random(n=N)
                    pop_new = lb + samples * (ub - lb)
                    pop_new[0] = best_ind
                    fitness_new = np.full(N, np.inf)
                    fitness_new[0] = best_fit
                    for j in range(1, N):
                        fitness_new[j] = func(pop_new[j])
                        n_evals += 1
                        if fitness_new[j] < self.f_opt:
                            self.f_opt = fitness_new[j]
                            self.x_opt = pop_new[j].copy()
                    pop = pop_new
                    fitness = fitness_new
                # Reset memory parameters with a mix of old and new
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = int(2.5 * N)
                evals_no_improve = 0
                diversity_hist = []

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt