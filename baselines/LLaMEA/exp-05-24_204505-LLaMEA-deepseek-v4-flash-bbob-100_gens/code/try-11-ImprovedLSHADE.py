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

        # Population size: initial and minimum
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))   # slightly larger min for high dim
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
        archive = np.empty((0, dim))
        archive_max = N

        # Memory for success-history
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation detection & local search parameters
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        restart_threshold = 0.2 * max_evals
        n_restarts = 0
        local_search_interval = max(10, int(0.02 * max_evals))  # do local search every 2% of budget
        last_local_search = 0

        # For p-ratio adaptation
        def p_ratio(evals):
            # non-linear decay: starts at 0.2, ends at 0.05
            return 0.2 * (1 - (evals / max_evals) ** 1.5) + 0.05

        # Main loop
        while n_evals < max_evals:
            p = p_ratio(n_evals)
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # r2 from union of pop and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # pbest selection (top p*N)
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.standard_normal(), 0, 1)

                # Mutation current-to-pbest/1 with archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Improved bound handling: reflect repeatedly until inside
                for _ in range(10):  # prevent infinite loop
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                # If still outside after max reflects, clamp and add small jitter
                mask_low = trial < lb
                mask_high = trial > ub
                trial[mask_low] = lb[mask_low] + 1e-10 * (ub[mask_low] - lb[mask_low]) * np.random.rand(np.sum(mask_low))
                trial[mask_high] = ub[mask_high] - 1e-10 * (ub[mask_high] - lb[mask_high]) * np.random.rand(np.sum(mask_high))

                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_since_last_improvement = 0
                else:
                    evals_since_last_improvement += 1

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

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with successful parameters
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Local refinement on best solution (periodic)
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                # Perform a small local search around the best solution
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.2 * (1 - n_evals / max_evals) * (ub - lb)  # shrinking step size
                for _ in range(min(dim, 5)):  # at most 5 evaluations per local search
                    perturbation = np.random.randn(dim) * step
                    new_pos = np.clip(best_pos + perturbation, lb, ub)
                    if n_evals >= max_evals:
                        break
                    new_val = func(new_pos)
                    n_evals += 1
                    if new_val < best_val:
                        best_val = new_val
                        best_pos = new_pos.copy()
                        if new_val < self.f_opt:
                            self.f_opt = new_val
                            self.x_opt = new_pos.copy()
                        evals_since_last_improvement = 0
                    else:
                        evals_since_last_improvement += 1
                # Optionally insert improved best into population (if better than current worst)
                if best_val < fitness[best_idx]:
                    pop[best_idx] = best_pos
                    fitness[best_idx] = best_val

            # Restart if stagnation and budget left
            if evals_since_last_improvement > restart_threshold and n_evals < max_evals * 0.75:
                # keep best solution, reinitialize rest with larger population if possible
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                # New population size: increase by 20% if enough budget (max 2*N_init)
                new_N = min(N * 2, N_init * 2, max_evals - n_evals - 10)
                new_N = max(new_N, N_min)
                # But we cannot evaluate more than remaining budget - a small margin
                remaining = max_evals - n_evals
                if new_N > remaining - 5:
                    new_N = remaining - 5
                if new_N > N:
                    # Expand
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
                    # Keep current size but reinitialize random half
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset archive, memory, and stagnation counter
                archive = np.empty((0, dim))
                archive_max = N
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                evals_since_last_improvement = 0
                n_restarts += 1

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt