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

        # Archive
        archive = []
        archive_max = N

        # Memory for F and CR (larger H = 50)
        H = 50
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals
        restart_count = 0
        max_restarts = 2

        # Local search interval
        local_search_interval = max(20, int(0.02 * max_evals))
        last_local_search = 0

        # Pattern search with adaptive step (Hooke‑Jeeves style)
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            iteration = 0
            while used < max_local_evals and iteration < dim * 6:
                iteration += 1
                improved = False
                # Coordinate search
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: move further in the direction of improvement
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size (aggressive)
                    step_size *= 1.5
                    step_size = np.minimum(step_size, (ub - lb) * 0.4)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size (aggressive)
                    step_size *= 0.6
                    if np.max(step_size) < 1e-12 * np.max(ub - lb):
                        break
            return pos, val, used

        def opposition_point(p, lb, ub):
            return lb + ub - p

        def latin_hypercube(N, dim, lb, ub):
            samples = np.random.uniform(0, 1, (N, dim))
            return lb + samples * (ub - lb)

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: quadratic aggressive decay
            p = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Choose r1
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # Union of population and archive
                if archive:
                    union = np.vstack((pop, np.array(archive)))
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
                    # Add parent to archive (FIFO)
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

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

            # Population size reduction (cubic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
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

            # Periodic local refinement
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.2 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual if improved
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8 and restart_count < max_restarts:
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Reinitialize population with Latin hypercube and keep best
                pop = latin_hypercube(new_N, dim, lb, ub)
                fitness = np.full(new_N, np.inf)
                pop[0] = best_ind
                fitness[0] = best_fit
                # Add opposition points around best (exploitation)
                for k in range(1, min(5, new_N)):
                    opp = opposition_point(best_ind, lb, ub)
                    opp = np.clip(opp, lb, ub)
                    pop[k] = opp
                    fitness[k] = func(opp)
                    n_evals += 1
                    if fitness[k] < self.f_opt:
                        self.f_opt = fitness[k]
                        self.x_opt = pop[k].copy()
                # Evaluate remaining
                for j in range(5 if new_N > 5 else 1, new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory
                MF[:] = 0.5 + 0.2 * np.random.rand(H)
                MCR[:] = 0.8 + 0.2 * np.random.rand(H)
                memory_idx = 0
                archive = []
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt