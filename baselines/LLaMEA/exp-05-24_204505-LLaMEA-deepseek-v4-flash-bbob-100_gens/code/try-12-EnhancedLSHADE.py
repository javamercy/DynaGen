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
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive (size = current population size)
        archive = np.empty((0, dim))

        # Success-history memory
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation detection
        evals_since_last_improvement = 0
        restart_threshold = 0.2 * max_evals
        last_local_search = 0
        local_search_interval = max(10, int(0.02 * max_evals))

        # Diversity monitoring
        diversity_threshold = 0.05 * (ub - lb).mean()

        def weighted_lehmer_mean(values, weights):
            # Weighted Lehmer mean: sum(w * v^2) / sum(w * v)
            v = np.array(values)
            w = np.array(weights)
            return np.sum(w * v ** 2) / (np.sum(w * v) + 1e-30)

        def p_ratio(evals):
            # Non-linear decay from 0.2 to 0.04
            return 0.2 * (1 - (evals / max_evals) ** 0.8) + 0.04

        def generate_trial(base, pbest, r1, r2, F, CR, pop_union):
            mutant = base + F * (pbest - base) + F * (r1 - r2)
            j_rand = np.random.randint(dim)
            trial = np.where(np.random.rand(dim) < CR, mutant, base)
            trial[j_rand] = mutant[j_rand]
            # Reflective bound handling
            for _ in range(10):
                out_low = trial < lb
                out_high = trial > ub
                if not (np.any(out_low) or np.any(out_high)):
                    break
                trial = np.where(out_low, 2 * lb - trial, trial)
                trial = np.where(out_high, 2 * ub - trial, trial)
            # Clamp if still out-of-bounds
            trial = np.clip(trial, lb, ub)
            return trial

        def local_search_best(best_pos, best_val, step_size, n_evals_left):
            # Random walk with step size
            for _ in range(min(dim, 8)):
                if n_evals_left <= 0:
                    break
                pert = np.random.randn(dim) * step_size
                new_pos = np.clip(best_pos + pert, lb, ub)
                new_val = func(new_pos)
                n_evals_left -= 1
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos.copy()
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                # If no improvement, try opposite direction
                else:
                    new_pos = np.clip(best_pos - pert, lb, ub)
                    new_val = func(new_pos)
                    n_evals_left -= 1
                    if new_val < best_val:
                        best_val = new_val
                        best_pos = new_pos.copy()
                        if best_val < self.f_opt:
                            self.f_opt = best_val
                            self.x_opt = best_pos.copy()
            return best_pos, best_val, n_evals_left

        # Main loop
        while n_evals < max_evals:
            p = p_ratio(n_evals)
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Selection indices
                ids = list(range(N))
                ids.remove(i)
                r1 = np.random.choice(ids)
                union = np.vstack((pop, archive)) if archive.size > 0 else pop
                r2 = np.random.randint(union.shape[0])

                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory (Cauchy and normal)
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.standard_normal(), 0, 1)

                # Generate trial
                trial = generate_trial(pop[i], pop[pbest_idx], pop[r1], union[r2], F, CR, union)

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
                    if archive.shape[0] > N:  # archive size = N
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with successful parameters (weighted Lehmer mean)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = weighted_lehmer_mean(S_F, w)
                MCR[memory_idx] = weighted_lehmer_mean(S_CR, w)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                N = N_new
                # Shrink archive accordingly
                if archive.shape[0] > N:
                    perm = np.random.permutation(archive.shape[0])[:N]
                    archive = archive[perm]

            # Periodic local refinement on best solution
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                step = 0.2 * (1 - n_evals / max_evals) * (ub - lb).mean()
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                best_pos, best_val, _ = local_search_best(best_pos, best_val, step, max_evals - n_evals)
                if best_val < fitness[best_idx]:
                    pop[best_idx] = best_pos
                    fitness[best_idx] = best_val

            # Diversity-based restart (if population too concentrated)
            if n_evals < max_evals * 0.75:
                sorted_idx = np.argsort(fitness)
                best_ind = pop[sorted_idx[0]].copy()
                best_fit = fitness[sorted_idx[0]]
                # Compute average distance from each individual to best
                dists = np.sqrt(np.sum((pop - best_ind) ** 2, axis=1))
                mean_dist = np.mean(dists)
                if mean_dist < diversity_threshold * (1 - n_evals / max_evals) and evals_since_last_improvement > max(50, 0.05 * max_evals):
                    # Trigger restart: keep best, replace others with random opposition-based
                    new_N = min(N * 2, N_init * 2, max_evals - n_evals - 10)
                    new_N = max(new_N, N_min)
                    remaining = max_evals - n_evals
                    if new_N > remaining - 5:
                        new_N = remaining - 5
                    if new_N > N:
                        pop = np.empty((new_N, dim))
                        fitness = np.full(new_N, np.inf)
                        pop[0] = best_ind
                        fitness[0] = best_fit
                        for j in range(1, new_N):
                            if j % 2 == 0:
                                # Latin hypercube sample
                                sample = np.random.uniform(0, 1, dim)
                                pop[j] = lb + sample * (ub - lb)
                            else:
                                # Opposition-based: reflect best around center
                                center = (lb + ub) / 2
                                opp = 2 * center - best_ind + np.random.uniform(-0.5, 0.5, dim) * (ub - lb)
                                pop[j] = np.clip(opp, lb, ub)
                            fitness[j] = func(pop[j])
                            n_evals += 1
                            if fitness[j] < self.f_opt:
                                self.f_opt = fitness[j]
                                self.x_opt = pop[j].copy()
                        N = new_N
                    else:
                        # keep N and reinitialize half
                        idx = np.random.choice(N, size=N//2, replace=False)
                        for j in idx:
                            if j == sorted_idx[0]:
                                continue
                            sample = np.random.uniform(0, 1, dim)
                            pop[j] = lb + sample * (ub - lb)
                            fitness[j] = func(pop[j])
                            n_evals += 1
                            if fitness[j] < self.f_opt:
                                self.f_opt = fitness[j]
                                self.x_opt = pop[j].copy()
                    # Reset archive and memory
                    archive = np.empty((0, dim))
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    evals_since_last_improvement = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt