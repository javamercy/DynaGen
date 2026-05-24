import numpy as np

class HADEC_LSR:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        max_evals = self.budget

        # Quasi-random sequence (Sobol) for initial population
        try:
            from scipy.stats.qmc import Sobol
            sobol = Sobol(dim, scramble=True)
            pop = sobol.random(n=int(min(1000, max_evals//2)))
        except ImportError:
            pop = np.random.uniform(0, 1, (1000, dim))
        pop = lb + pop * (ub - lb)
        N_init = min(max(10*dim, 50), max_evals//2)
        N = N_init
        pop = pop[:N]

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

        # Success-history memory for F and CR
        H = 5
        MF = np.full(H, 0.5)
        MCR = np.full(H, 0.8)
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        diversity_threshold = 0.01 * dim  # scaled diversity
        last_restart_eval = 0

        # Local search parameters
        ls_step = 0.1 * (ub - lb)
        ls_success_rate = 0.5

        # Main loop
        while n_evals < max_evals:
            # Population size reduction (sigmoidal schedule)
            remaining = max_evals - n_evals
            total = max_evals
            ratio = remaining / total
            N_new = N_min = max(4, int(dim/5))
            # sigmoidal: smoother reduction at end
            N_new = int(N_min + (N_init - N_min) / (1 + np.exp(10*(ratio-0.5))))
            N_new = max(N_min, min(N_new, N))
            if N_new < N:
                idx = np.argsort(fitness)[:N_new]
                pop = pop[idx].copy()
                fitness = fitness[idx].copy()
                N = N_new
                # reduce archive
                if archive.shape[0] > N:
                    archive = archive[np.random.choice(archive.shape[0], N, replace=False)]
                archive_max = N

            # pbest ratio: reduces from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals)**1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Choose mutation strategy adaptively based on success history
                # Use current-to-pbest/1/bin with probability 0.8, otherwise rand/1/exp
                if np.random.rand() < 0.8:
                    # current-to-pbest/1/bin with archive
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
                    pbest_idx = np.random.choice(sorted_idx[:pbest_size])
                    mem = np.random.randint(H)
                    F = MF[mem] + 0.1 * np.random.standard_cauchy()
                    while F <= 0:
                        F = MF[mem] + 0.1 * np.random.standard_cauchy()
                    F = np.clip(F, 0, 1)
                    CR = MCR[mem] + 0.1 * np.random.randn()
                    CR = np.clip(CR, 0, 1)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                    # binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                else:
                    # rand/1/exp with adaptive F
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
                    F = 0.5 + 0.5 * np.random.rand()
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])
                    # exponential crossover
                    CR = 0.5 + 0.5 * np.random.rand()
                    trial = pop[i].copy()
                    start = np.random.randint(dim)
                    L = 0
                    while np.random.rand() < CR and L < dim:
                        idx = (start + L) % dim
                        trial[idx] = mutant[idx]
                        L += 1

                # Boundary handling: reflect and clamp
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2*lb - trial, trial)
                    trial = np.where(out_high, 2*ub - trial, trial)
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
                    # add parent to archive
                    if archive.shape[0] < archive_max:
                        archive = np.vstack((archive, pop[i].reshape(1,-1)))
                    else:
                        # FIFO replacement
                        archive = np.vstack((archive[1:], pop[i].reshape(1,-1)))
                else:
                    # optional: keep trial if close to best?
                    pass

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Local search: adaptive coordinate descent
            if n_evals < max_evals * 0.9 and np.random.rand() < 0.1:
                # select best individual
                best_idx = np.argmin(fitness)
                x_best = pop[best_idx].copy()
                f_best = fitness[best_idx]
                # adaptive step size based on local success rate
                step = ls_step * max(0.01, (1 - n_evals/max_evals))
                improved = True
                used = 0
                max_local = min(dim*3, max_evals - n_evals - 5)
                while improved and used < max_local:
                    improved = False
                    order = np.random.permutation(dim)
                    for d in order:
                        if used >= max_local:
                            break
                        # try positive
                        cand = x_best.copy()
                        cand[d] = np.clip(x_best[d] + step[d], lb[d], ub[d])
                        f_cand = func(cand)
                        used += 1
                        n_evals += 1
                        if f_cand < f_best:
                            x_best = cand
                            f_best = f_cand
                            improved = True
                            step[d] *= 1.2
                            if f_cand < self.f_opt:
                                self.f_opt = f_cand
                                self.x_opt = cand.copy()
                                evals_no_improve = 0
                            continue
                        # try negative
                        cand = x_best.copy()
                        cand[d] = np.clip(x_best[d] - step[d], lb[d], ub[d])
                        f_cand = func(cand)
                        used += 1
                        n_evals += 1
                        if f_cand < f_best:
                            x_best = cand
                            f_best = f_cand
                            improved = True
                            step[d] *= 1.2
                            if f_cand < self.f_opt:
                                self.f_opt = f_cand
                                self.x_opt = cand.copy()
                                evals_no_improve = 0
                        else:
                            step[d] *= 0.85
                # incorporate improved individual into population (replace worst)
                if f_best < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = x_best
                    fitness[worst_idx] = f_best
                ls_step = step

            # Restart if stagnation or low diversity
            # Compute diversity: average distance to best
            if N > 1:
                center = pop.mean(axis=0)
                diversity = np.mean(np.sqrt(np.sum((pop - center)**2, axis=1)))
            else:
                diversity = 0
            need_restart = False
            if evals_no_improve > 0.1 * max_evals and n_evals < max_evals * 0.8:
                need_restart = True
            if diversity < 0.001 * np.mean(ub-lb) and n_evals < max_evals * 0.8:
                need_restart = True

            if need_restart and (n_evals - last_restart_eval) > 0.05*max_evals:
                last_restart_eval = n_evals
                # keep best individual
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init, N*2, remaining//2)
                new_N = max(N_min, new_N)
                # generate new population with Latin hypercube around best and globally
                # mix: 30% near best, 70% global
                n_near = int(0.3*new_N)
                n_global = new_N - n_near
                # near population: around best with Gaussian
                near_pop = best_ind + np.random.randn(n_near, dim) * 0.1 * (ub-lb)
                near_pop = np.clip(near_pop, lb, ub)
                # global: quasi-random
                try:
                    from scipy.stats.qmc import Sobol
                    sobol = Sobol(dim, scramble=True)
                    global_samples = sobol.random(n=n_global)
                except ImportError:
                    global_samples = np.random.uniform(0, 1, (n_global, dim))
                global_pop = lb + global_samples * (ub - lb)
                new_pop = np.vstack((best_ind.reshape(1,-1), near_pop, global_pop))
                new_pop = new_pop[:new_N]
                new_fitness = np.full(new_N, np.inf)
                new_fitness[0] = best_fit
                for j in range(1, new_N):
                    new_fitness[j] = func(new_pop[j])
                    n_evals += 1
                    if new_fitness[j] < self.f_opt:
                        self.f_opt = new_fitness[j]
                        self.x_opt = new_pop[j].copy()
                pop = new_pop
                fitness = new_fitness
                N = new_N
                # reset archive and memory
                archive = np.empty((0, dim))
                archive_max = N
                MF[:] = 0.5
                MCR[:] = 0.8
                memory_idx = 0
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt