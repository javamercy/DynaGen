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

        # Parameters
        N_init = min(max(10 * dim, 60), max_evals // 2)
        N_min = max(6, int(dim / 4))
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

        # Archive
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory (F and CR)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.9
        memory_idx = 0

        # Restart / diversity tracking
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        last_local_evals = 0
        local_interval = max(50, int(0.02 * max_evals))

        # Spherical pattern search function
        def spherical_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)      # relative step per dimension
            used = 0
            n_dirs = min(dim + 2, 5)          # number of random directions per iteration
            while used < max_local_evals:
                improved = False
                for _ in range(n_dirs):
                    if used >= max_local_evals:
                        break
                    # random unit vector
                    dir_vec = np.random.randn(dim)
                    dir_vec /= (np.linalg.norm(dir_vec) + 1e-30)
                    # positive direction
                    new_pos = np.clip(pos + step_size * dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = np.clip(pos - step_size * dir_vec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # expand step on success
                    step_size = np.clip(step_size * 1.3, 1e-8 * (ub - lb), 0.5 * (ub - lb))
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # contract on failure
                    step_size = np.clip(step_size * 0.5, 1e-10 * (ub - lb), 0.5 * (ub - lb))
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
                # if step becomes very small, stop
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Offspring generation
            for i in range(N):
                # choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                union = np.vstack((pop, archive)) if archive.size > 0 else pop
                r2 = np.random.randint(union.shape[0])
                # pbest
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # boundary reflection
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
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory (weighted Lehmer mean)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F_arr = np.array(S_F)[sorted_order]
                S_CR_arr = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F_arr ** 2) / (np.sum(w * S_F_arr) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR_arr ** 2) / (np.sum(w * S_CR_arr) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population reduction (quadratic)
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

            # Spherical local search on best solution
            if n_evals - last_local_evals >= local_interval and n_evals < max_evals * 0.95:
                last_local_evals = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 5, max_evals - n_evals - 5)
                new_pos, new_val, used = spherical_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart on stagnation or low diversity
            # Compute diversity: average pairwise distance from best
            best_idx = np.argmin(fitness)
            dists = np.sqrt(np.sum((pop - pop[best_idx]) ** 2, axis=1))
            diversity = np.mean(dists) / np.mean(ub - lb) if np.mean(ub - lb) > 0 else 0

            need_restart = False
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                need_restart = True
            if diversity < 1e-5 and n_evals < max_evals * 0.9:
                need_restart = True

            if need_restart:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population using opposition-based learning
                # Keep best, for others generate random and opposition
                pop_new = np.empty((new_N, dim))
                pop_new[0] = best_ind
                idx = 1
                for _ in range(new_N - 1):
                    if np.random.rand() < 0.5:
                        # random point
                        point = np.random.uniform(lb, ub, dim)
                    else:
                        # opposition point relative to best
                        point = lb + ub - best_ind + np.random.uniform(-0.1, 0.1, dim) * (ub - lb)
                        point = np.clip(point, lb, ub)
                    pop_new[idx] = point
                    idx += 1
                pop = pop_new
                fitness = np.full(new_N, np.inf)
                fitness[0] = best_fit
                for j in range(1, new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt