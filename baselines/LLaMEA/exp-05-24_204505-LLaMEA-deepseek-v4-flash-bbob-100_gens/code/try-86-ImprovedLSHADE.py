import numpy as np

class ImprovedLSHADE:
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
        N_init = min(max(10*dim, 50), max_evals//2)
        N_min = max(4, int(dim/5))
        N = N_init

        # Sobol-like low-discrepancy initialization (using Latin hypercube as fallback)
        n_samples = N
        samples = np.random.uniform(0, 1, (n_samples, dim))
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F, CR, and strategy
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Strategy memory (0: current-to-pbest/1, 1: current-to-rand/1, 2: rand/1)
        strategy_success = np.zeros(3)
        strategy_counts = np.ones(3) * 1e-10
        strategy_prob = np.ones(3) / 3

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0
        # Pattern search step memory for adaptive step
        ps_step = 0.15  # initial relative step

        # Helper: generate Sobol-like sequence (crude, using Latin hypercube)
        def sobol_like(n, d):
            # Simple randomized orthogonal array – for performance, just use LHS
            return np.random.uniform(0, 1, (n, d))

        # Randomized pattern search with adaptive step and random direction order
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)
            used = 0
            # Random permutation of coordinates to avoid bias
            coords = np.arange(dim)
            while used < max_local_evals:
                improved = False
                np.random.shuffle(coords)
                for d in coords:
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
                    # Pattern move: accelerate along direction
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb)*0.4)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-8 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals)**1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring using multi-strategy
            for i in range(N):
                # Choose strategy based on adaptive probabilities
                s = np.random.choice(3, p=strategy_prob)
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of pop and archive
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
                # Mutation based on strategy
                base = pop[i]
                if s == 0:  # current-to-pbest/1
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                elif s == 1:  # current-to-rand/1 (better exploration)
                    mutant = base + F * (pop[r1] - base) + F * (union[r2] - pop[r1])
                else:  # rand/1
                    # choose two more random indices different
                    r3 = np.random.choice(idxs)
                    r4 = np.random.choice(idxs)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3]) + F * (union[r4] - pop[r1])
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
                    trial = np.where(out_low, 2*lb - trial, trial)
                    trial = np.where(out_high, 2*ub - trial, trial)
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
                    strategy_success[s] += 1
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1,-1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                else:
                    strategy_counts[s] += 1

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update strategy probabilities (softmax-like)
            success_rates = strategy_success / (strategy_counts + 1e-30)
            # Exponentially decaying window (simplified)
            strategy_prob = success_rates / (success_rates.sum() + 1e-30)
            # Reset counts periodically to avoid lock-in
            if n_evals % max(100, dim*20) < N:
                strategy_success *= 0.9
                strategy_counts = np.maximum(strategy_counts * 0.9, 1)

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (quadratic)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals)/max_evals)**2
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

            # Periodic local refinement on best and top 10% individuals
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals*0.95):
                last_local_search = n_evals
                # Apply local search on top 10% (but limited to avoid overhead)
                top_k = max(1, int(0.1 * N))
                best_indices = np.argsort(fitness)[:top_k]
                for idx in best_indices:
                    if n_evals >= max_evals - 5:
                        break
                    best_pos = pop[idx].copy()
                    best_val = fitness[idx]
                    # Adaptive step based on remaining evaluations and function landscape
                    step = ps_step * (1 - n_evals/max_evals) + 0.01
                    max_local = min(dim*2, max_evals - n_evals - 5)
                    new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                    n_evals += used
                    if new_val < best_val:
                        pop[idx] = new_pos
                        fitness[idx] = new_val
                        if new_val < self.f_opt:
                            self.f_opt = new_val
                            self.x_opt = new_pos.copy()
                            evals_no_improve = 0
                # Update global pattern search step based on success of last local search
                if n_evals > last_local_search:
                    ps_step *= 0.95  # gradually reduce step size

            # Restart if stagnation
            if evals_no_improve > restart_threshold and n_evals < max_evals*0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init*2, N*2, remaining//2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Low-discrepancy new population, inject best
                    samples = sobol_like(new_N, dim)
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
                    # Partial restart around best with Cauchy perturbation
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    # Perturb some individuals around best
                    for j in range(1, min(1+int(N/4), N)):
                        pert = np.random.standard_cauchy(dim) * (ub - lb) * 0.1
                        pop[j] = np.clip(best_ind + pert, lb, ub)
                    for j in range(N):
                        if j == 0:
                            continue
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory parameters
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Reset strategy memory
                strategy_success = np.zeros(3)
                strategy_counts = np.ones(3) * 1e-10
                strategy_prob = np.ones(3) / 3

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt