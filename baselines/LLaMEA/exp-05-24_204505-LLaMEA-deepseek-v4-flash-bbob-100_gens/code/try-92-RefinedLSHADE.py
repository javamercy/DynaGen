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

        # Archive for DE mutation (FIFO queue)
        archive = []
        archive_max = N

        # Success-history memory for F and CR
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals
        restart_count = 0
        max_restarts = 2

        # Local search parameters (adaptive interval)
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0
        last_best = self.f_opt
        local_search_failures = 0

        # --- Pattern search with rotated basis ---
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * np.max(ub - lb)  # scalar step for simplicity, but we will scale per dimension later
            # initial basis: identity matrix
            basis = np.eye(dim)
            used = 0
            iterations = 0
            no_improve_count = 0
            while used < max_local_evals and iterations < dim * 8:
                iterations += 1
                improved = False
                # For each direction in current basis
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    dir_vec = basis[d] * step_size * (ub - lb)  # scale step per dimension
                    # positive direction
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
                    # Pattern move: accelerate along net improvement
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12 * np.max(ub - lb)):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size (gentle)
                    step_size = min(step_size * 1.2, 0.4 * np.max(ub - lb))
                    no_improve_count = 0
                    best_pos = pos.copy()
                    best_val = val
                else:
                    no_improve_count += 1
                    # Contract step size
                    step_size *= 0.6
                    if step_size < 1e-12 * np.max(ub - lb):
                        break
                    # If many failures, rotate basis randomly to explore new directions
                    if no_improve_count >= dim*2:
                        # random orthonormal rotation
                        Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                        basis = Q
                        no_improve_count = 0
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: logistic decay (starts 0.2, decays to 0.05)
            t = n_evals / max_evals
            p = 0.05 + 0.15 / (1 + np.exp(5 * (t - 0.45)))

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                union_pop = pop
                if archive:
                    union = np.vstack((pop, np.array(archive)))
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

                # Boundary handling: reflect (with limited iterations)
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
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

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

            # Population size reduction (quadratic, slower reduction early)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if len(archive) > archive_max:
                    archive = archive[-archive_max:]
                N = N_new

            # Adaptive local search: trigger based on improvement rate
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                current_best = fitness.min()
                rel_improvement = (last_best - current_best) / (abs(last_best) + 1e-30)
                # Trigger if improvement is small or if no improvement for a while
                if (rel_improvement < 1e-6 and n_evals > max_evals*0.2) or (n_evals - last_local_search > local_search_interval*2):
                    last_local_search = n_evals
                    last_best = current_best
                    best_idx = np.argmin(fitness)
                    best_pos = pop[best_idx].copy()
                    best_val = fitness[best_idx]
                    # Adaptive step: larger early, smaller later
                    step = 0.2 * (1 - t) + 0.01
                    max_local = min(dim * 5, max_evals - n_evals - 5)
                    new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                    n_evals += used
                    if new_val < best_val:
                        best_val = new_val
                        best_pos = new_pos
                        if best_val < self.f_opt:
                            self.f_opt = best_val
                            self.x_opt = best_pos.copy()
                            evals_no_improve = 0
                        local_search_failures = 0
                    else:
                        local_search_failures += 1
                    # Replace worst individual if improved
                    if best_val < fitness.max():
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = best_pos
                        fitness[worst_idx] = best_val
                    # Increase interval if failures accumulate
                    if local_search_failures >= 3:
                        local_search_interval = int(local_search_interval * 1.5)
                        local_search_failures = 0

            # Restart if stagnation detected (allow up to 2 restarts)
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8 and restart_count < max_restarts):
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Cauchy-based diversified restart around best
                    pop = np.tile(best_ind, (new_N, 1)) + 0.1 * np.random.standard_cauchy(size=(new_N, dim)) * (ub - lb)
                    pop = np.clip(pop, lb, ub)
                    pop[0] = best_ind
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: randomize all but best with Cauchy noise
                    pop = np.tile(best_ind, (N, 1)) + 0.2 * np.random.standard_cauchy(size=(N, dim)) * (ub - lb)
                    pop = np.clip(pop, lb, ub)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory with exploration-friendly values
                MF[:] = 0.6 + 0.2 * np.random.rand(H)
                MCR[:] = 0.8 + 0.2 * np.random.rand(H)
                memory_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0
                local_search_interval = max(30, int(0.02 * max_evals))  # reset interval
                local_search_failures = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt