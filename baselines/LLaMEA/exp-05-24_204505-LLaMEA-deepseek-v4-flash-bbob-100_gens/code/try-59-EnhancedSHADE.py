import numpy as np

class EnhancedSHADE:
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

        # Population size – start larger for better coverage
        N_init = min(max(15 * dim, 50), max_evals // 2)  # increased from 10*dim
        N_min = max(4, int(dim / 6))                     # slightly smaller min
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

        # Archive – keep up to 2*N parents
        archive = np.empty((0, dim))
        archive_max = 2 * N

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.10 * max_evals  # more sensitive

        # Local search parameters
        local_search_interval = max(20, int(0.01 * max_evals))
        last_local_search = 0

        # Diversity measure (pairwise average distance)
        def population_diversity(pop):
            if len(pop) <= 1:
                return 0.0
            mean = np.mean(pop, axis=0)
            return np.mean(np.sqrt(np.sum((pop - mean)**2, axis=1)))

        # Improved pattern search with random directions
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            scale = ub - lb
            step_size = step * scale
            used = 0
            improved_global = False

            # coordinate search
            for _ in range(dim):  # multiple iterations to allow pattern move
                improved = False
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    improved_global = True
                    # pattern move (accelerate)
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # expand step
                    step_size *= 1.2
                    step_size = np.minimum(step_size, scale * 0.4)
                else:
                    break

            # random direction search (if budget left)
            if used < max_local_evals - 2:
                for _ in range(min(2, max_local_evals - used)):
                    dir = np.random.randn(dim)
                    dir = dir / (np.linalg.norm(dir) + 1e-30)
                    for sgn in [1.0, -1.0]:
                        new_pos = np.clip(pos + sgn * step_size * dir, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                            improved_global = True
                            break

            # if no improvement, contract step for next activation
            if not improved_global:
                step_size *= 0.5
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: more aggressive early
            p = 0.25 * (1 - (n_evals / max_evals) ** 0.8) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # choose r1
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # choose r2 from union
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
                # sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # boundary handling (reflect + clamp)
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
                    # add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        idx_remove = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, idx_remove, axis=0)

            # update population
            pop = new_pop
            fitness = new_fitness

            # update memory with weighted Lehmer mean
            if len(S_F) > 0:
                order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[order]
                S_CR = np.array(S_CR)[order]
                w = np.array(delta_f)[order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # population reduction (quadratic, but keep diversity longer)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 1.5  # higher exponent slows reduction
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                idx = np.argsort(fitness)
                pop = pop[idx[:N_new]]
                fitness = fitness[idx[:N_new]]
                archive_max = 2 * N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # periodic local search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                improved_pos, improved_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if improved_val < best_val:
                    if improved_val < self.f_opt:
                        self.f_opt = improved_val
                        self.x_opt = improved_pos.copy()
                        evals_no_improve = 0
                    # replace worst in population
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = improved_pos
                    fitness[worst_idx] = improved_val

            # diversity check – restart if too converged
            diversity = population_diversity(pop)
            if diversity < 0.01 * np.mean(ub-lb) and n_evals < max_evals * 0.7:
                evals_no_improve = max(evals_no_improve, int(0.9 * restart_threshold))  # force restart soon

            # restart on stagnation
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, max(N * 2, N_min), remaining // 2)
                new_N = max(new_N, N_min)
                # create new population: 50% from best perturbation, 50% random
                pop_new = np.empty((new_N, dim))
                n_best = new_N // 2
                n_rand = new_N - n_best
                # perturb best with Cauchy-like steps
                scale = 0.2 * (ub - lb) * (1 - n_evals / max_evals) + 0.05 * (ub - lb)
                for k in range(n_best):
                    pert = np.random.standard_cauchy(size=dim)
                    # clamp perturbation to avoid huge outliers
                    pert = np.clip(pert, -3, 3)
                    candidate = best_ind + scale * pert
                    candidate = np.clip(candidate, lb, ub)
                    pop_new[k] = candidate
                # random points
                pop_new[n_best:] = lb + np.random.uniform(0, 1, (n_rand, dim)) * (ub - lb)
                # evaluate all except the best (already evaluated)
                pop_new[0] = best_ind
                fitness_new = np.full(new_N, np.inf)
                fitness_new[0] = best_fit
                for j in range(1, new_N):
                    fitness_new[j] = func(pop_new[j])
                    n_evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = pop_new[j].copy()
                pop = pop_new
                fitness = fitness_new
                N = new_N
                # reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt