import numpy as np

class EnhancedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        max_evals = self.budget

        # --- population parameters ---
        N_init = int(min(max(10 * dim, 50), max_evals // 2))
        N_min = max(4, int(dim / 5))
        N = N_init

        # --- Latin hypercube initialization ---
        samples = np.random.uniform(0, 1, (N, dim))
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # --- archive ---
        archive = np.empty((0, dim))
        archive_max = N

        # --- success-history memory (F and CR) ---
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # --- stagnation detection ---
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # --- local search control ---
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0
        local_success_rate = 0.5   # smoothed success rate for step adaptation
        ls_step = 0.15

        # ------------------------------------------------------------
        # rotated directional local search (RDL)
        # ------------------------------------------------------------
        def rotated_search(best_pos, best_val, step, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            used = 0
            # generate a random orthonormal basis (direct rotation)
            R = np.random.randn(dim, dim)
            R, _ = np.linalg.qr(R)   # orthogonal matrix
            step_vec = step * (ub - lb)   # initial step per dimension
            step_size = np.linalg.norm(step_vec) / np.sqrt(dim)  # scalar step
            iterations = 0
            while used < max_evals_local and iterations < dim * 5:
                iterations += 1
                improved = False
                # coordinate search in rotated space
                for d in range(dim):
                    if used >= max_evals_local:
                        break
                    dir = R[:, d]
                    # positive direction
                    new_pos = pos + step_size * dir
                    new_pos = np.clip(new_pos, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        step_size *= 1.2  # expand
                        continue
                    # negative direction
                    new_pos = pos - step_size * dir
                    new_pos = np.clip(new_pos, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        step_size *= 1.2
                    else:
                        step_size *= 0.9  # contract
                if improved:
                    # pattern move: extrapolate along net improvement direction
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = pos + delta
                        new_pos = np.clip(new_pos, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                # reset step if too small
                if step_size < 1e-12 * np.max(ub - lb):
                    step_size = 0.01 * np.max(ub - lb)
                if step_size > 0.5 * np.max(ub - lb):
                    step_size = 0.5 * np.max(ub - lb)
            return pos, val, used

        # ------------------------------------------------------------
        # main loop
        # ------------------------------------------------------------
        while n_evals < max_evals:
            # ---- pbest ratio: hyperbolic decreasing from 0.2 to 0.02 ----
            p = 0.2 / (1.0 + 10.0 * (n_evals / max_evals)) + 0.02

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # --- generate offspring ---
            for i in range(N):
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of pop and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # pbest index – weighted by fitness rank (more diversity)
                sorted_idx = np.argsort(fitness)
                ranks = np.argsort(sorted_idx)  # rank 0 best, N-1 worst
                weights = np.exp(-2.0 * ranks / N)   # exponential weighting
                pbest_size = max(1, int(p * N))
                candidates = sorted_idx[:pbest_size]
                # select pbest probabilistically from the top p*N
                w_sub = weights[candidates]
                w_sub /= np.sum(w_sub)
                pbest_idx = np.random.choice(candidates, p=w_sub)

                # sample F and CR from memory (Cauchy for F, normal for CR)
                mem = np.random.randint(H)
                F = MF[mem] + 0.1 * np.random.standard_cauchy()
                while F <= 0:
                    F = MF[mem] + 0.1 * np.random.standard_cauchy()
                F = np.clip(F, 0, 1)
                CR = MCR[mem] + 0.1 * np.random.randn()
                CR = np.clip(CR, 0, 1)

                # mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # boundary handling (reflect multiple times)
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
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # ---- update population ----
            pop = new_pop
            fitness = new_fitness

            # ---- update memory (weighted Lehmer mean) ----
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # ---- population size reduction (quadratic) ----
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

            # ---- rotated directional local search ----
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 4, max_evals - n_evals - 5)
                # adjust step based on local success history
                step = ls_step * (0.5 + 0.5 * local_success_rate)
                new_pos, new_val, used = rotated_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    improvement = best_val - new_val
                    local_success_rate = 0.9 * local_success_rate + 0.1   # success
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                else:
                    local_success_rate = 0.9 * local_success_rate  # failure
                # replace worst individual if improved
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # ---- restart on stagnation ----
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # generate new population around best and LHS
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    # mix: 30% near best, rest LHS
                    near = max(1, int(new_N * 0.3))
                    for j in range(1, near):
                        pop[j] = best_ind + 0.1 * np.random.randn(dim) * (ub - lb)
                        pop[j] = np.clip(pop[j], lb, ub)
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    for j in range(near, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # partial restart: keep best, randomize others
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # reset memory
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                local_success_rate = 0.5

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt