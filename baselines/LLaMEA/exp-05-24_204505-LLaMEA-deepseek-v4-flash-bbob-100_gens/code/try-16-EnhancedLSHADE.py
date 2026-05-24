import numpy as np

class EnhancedLSHADE:
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
        restart_threshold = 0.12 * max_evals

        # Diversity measure
        min_diversity = 1e-6 * np.max(ub - lb)

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0
        local_search_budget_frac = 0.2

        # Pattern search with orthonormal directions and adaptive step
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used_evals = 0
            basis = np.random.randn(dim, dim)
            basis, _ = np.linalg.qr(basis)
            scale = np.ones(dim)
            iters = 0
            while used_evals < max_local_evals and iters < dim * 5:
                iters += 1
                improved = False
                for d in range(dim):
                    if used_evals >= max_local_evals:
                        break
                    dir_vec = basis[:, d] * scale[d] * step_size
                    new_pos = np.clip(pos + dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used_evals += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        scale[d] *= 1.2
                        continue
                    new_pos = np.clip(pos - dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used_evals += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        scale[d] *= 1.2
                    else:
                        scale[d] *= 0.85
                if improved:
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used_evals += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used_evals

        # Main loop
        while n_evals < max_evals:
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.02

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
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
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                for _ in range(5):
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
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            pop = new_pop
            fitness = new_fitness

            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

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

            remaining_evals = max_evals - n_evals
            if (n_evals - last_local_search >= local_search_interval) and (remaining_evals > dim * 5):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.005
                max_local = min(int(local_search_budget_frac * remaining_evals), remaining_evals - 10)
                max_local = max(max_local, dim * 2)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                if new_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity maintenance
            center = np.mean(pop, axis=0)
            diversity = np.mean(np.sqrt(np.sum((pop - center)**2, axis=1)))
            if diversity < min_diversity and n_evals < max_evals * 0.9 and N > N_min:
                n_replace = max(1, int(0.2 * N))
                worst_idx = np.argsort(fitness)[-n_replace:]
                for idx in worst_idx:
                    new_sample = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                    new_f = func(new_sample)
                    n_evals += 1
                    if new_f < self.f_opt:
                        self.f_opt = new_f
                        self.x_opt = new_sample.copy()
                    pop[idx] = new_sample
                    fitness[idx] = new_f

            # Restart if stagnation
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.75) or (evals_no_improve > 0.25 * max_evals and n_evals < max_evals * 0.5):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    sigma = 0.2 * (ub - lb)
                    for j in range(1, new_N):
                        if np.random.rand() < 0.3:
                            perturb = np.random.normal(0, sigma)
                            pop[j] = np.clip(best_ind + perturb, lb, ub)
                        else:
                            pop[j] = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    sig = 0.3 * (ub - lb)
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        if np.random.rand() < 0.4:
                            perturb = np.random.normal(0, sig)
                            pop[j] = np.clip(best_ind + perturb, lb, ub)
                        else:
                            pop[j] = lb + np.random.uniform(0, 1, dim) * (ub - lb)
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt