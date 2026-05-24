import numpy as np

class AdaptiveLSHADE:
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

        # population size setup
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # latin hypercube initialization
        samples = np.random.uniform(0, 1, (N, dim))
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # archive
        archive = np.empty((0, dim))
        archive_max = N

        # success-history memory (H=10)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # second mutation strategy probability (adaptable)
        prob_rand_mut = 0.2
        prob_rand_mut_success = []

        # covariance learning: cov matrix from top 50% individuals (only when dim <= 30)
        use_cov = (dim <= 30)
        cov = np.eye(dim) * 0.1
        cov_lr = 0.1

        # stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.1 * max_evals

        # local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # line search helper (quadratic interpolation)
        def quad_line_search(center, direction, step, max_evals_local):
            """Quadratic interpolation along a direction."""
            # evaluate three points: center, center+step, center+2*step (if allowed)
            f0 = func(center)
            evals = 1
            # step1 = step * (ub - lb) normalized direction
            d = direction / (np.linalg.norm(direction) + 1e-30)
            s = step * np.min(ub - lb)
            x1 = np.clip(center + s * d, lb, ub)
            f1 = func(x1); evals += 1
            if f1 < f0:
                x2 = np.clip(center + 2 * s * d, lb, ub)
                f2 = func(x2); evals += 1
                # quadratic fit for minimum point
                # a = (f0 - 2*f1 + f2)/2, b = (f2 - f0)/2
                a = (f0 - 2*f1 + f2) / 2.0
                b = (f2 - f0) / 2.0
                if abs(a) > 1e-15:
                    t_opt = -b / (2*a)  # normalized step
                    t_opt = np.clip(t_opt, 0.1, 1.9)  # keep within interval
                    x_opt = np.clip(center + t_opt * s * d, lb, ub)
                    f_opt = func(x_opt); evals += 1
                    if f_opt < f1:
                        return x_opt, f_opt, evals
                return x1, f1, evals
            else:
                # try negative direction
                x1n = np.clip(center - s * d, lb, ub)
                f1n = func(x1n); evals += 1
                if f1n < f0:
                    x2n = np.clip(center - 2 * s * d, lb, ub)
                    f2n = func(x2n); evals += 1
                    a = (f0 - 2*f1n + f2n) / 2.0
                    b = (f2n - f0) / 2.0
                    if abs(a) > 1e-15:
                        t_opt = -b / (2*a)
                        t_opt = np.clip(t_opt, -1.9, -0.1)
                        x_opt = np.clip(center + t_opt * s * d, lb, ub)
                        f_opt = func(x_opt); evals += 1
                        if f_opt < f1n:
                            return x_opt, f_opt, evals
                    return x1n, f1n, evals
                else:
                    return center, f0, evals

        # main loop
        while n_evals < max_evals:
            # pbest ratio: dynamic
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            rand_mut_success_count = 0
            rand_mut_attempts = 0

            for i in range(N):
                # choose strategy: with prob prob_rand_mut use de/rand/1, else current-to-pbest/1
                use_rand = np.random.rand() < prob_rand_mut
                if use_rand:
                    rand_mut_attempts += 1
                    # DE/rand/1/bin
                    idxs = list(range(N))
                    idxs.remove(i)
                    r1, r2, r3 = np.random.choice(idxs, size=3, replace=False)
                    F = np.clip(np.random.standard_cauchy()*0.1 + 0.5, 0, 1)
                    CR = np.clip(np.random.randn()*0.1 + 0.8, 0, 1)
                    base = pop[r1]
                    mutant = base + F * (pop[r2] - pop[r3])
                else:
                    # current-to-pbest/1/archive
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

                # optionally add covariance-guided perturbation (only for current-to-pbest)
                if not use_rand and use_cov and np.random.rand() < 0.3:
                    # sample from Gaussian with cov
                    if np.any(np.linalg.eigvalsh(cov) < 0):
                        cov = 0.5*(cov + cov.T) + 1e-8*np.eye(dim)
                    perturbation = np.random.multivariate_normal(np.zeros(dim), cov)
                    mutant = mutant + 0.1 * perturbation * (ub - lb)

                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # boundary handling: reflect
                for _ in range(10):
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
                    if use_rand:
                        rand_mut_success_count += 1
                    else:
                        S_F.append(F)
                        S_CR.append(CR)
                        delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # update success probability for rand mutation
            if rand_mut_attempts > 0:
                success_rate = rand_mut_success_count / rand_mut_attempts
                prob_rand_mut = 0.8 * prob_rand_mut + 0.2 * success_rate
                prob_rand_mut = np.clip(prob_rand_mut, 0.05, 0.5)

            pop = new_pop
            fitness = new_fitness

            # update memory (only for current-to-pbest)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # update covariance matrix from top 50% individuals
            if use_cov and n_evals % (2*dim) == 0:
                sorted_idx = np.argsort(fitness)
                best_half = pop[sorted_idx[:max(N//2, 2)]]
                mean = np.mean(best_half, axis=0)
                diff = best_half - mean
                cov_new = (diff.T @ diff) / (best_half.shape[0] - 1) + 1e-8 * np.eye(dim)
                cov = (1 - cov_lr) * cov + cov_lr * cov_new

            # population size reduction (linear schedule)
            N_new = int(N_min + (N_init - N_min) * (1 - n_evals / max_evals))
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

            # local refinement using quadratic line search on best direction
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]

                # compute direction to second best (if distinct)
                if N >= 2:
                    second_idx = np.argsort(fitness)[1]
                    direction = pop[best_idx] - pop[second_idx]
                else:
                    direction = np.random.randn(dim)
                if np.linalg.norm(direction) < 1e-12:
                    direction = np.random.randn(dim)

                max_local_evals = min(3*dim, max_evals - n_evals - 5)
                new_pos, new_val, used = quad_line_search(best_pos, direction, 0.1, max_local_evals)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                    # replace worst
                    if best_val < fitness[np.argmax(fitness)]:
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = best_pos
                        fitness[worst_idx] = best_val

            # restart on stagnation or low diversity
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8):
                # measure diversity: mean distance to best
                best_pos = pop[np.argmin(fitness)]
                distances = np.linalg.norm(pop - best_pos, axis=1)
                mean_dist = np.mean(distances)
                if mean_dist < 0.05 * np.linalg.norm(ub - lb):
                    # partial restart: resample all but the best
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    remaining = max_evals - n_evals
                    new_N = min(N_init * 2, N * 2, remaining // 2)
                    new_N = max(new_N, N_min)
                    if new_N > N:
                        samples = lb + np.random.uniform(0, 1, (new_N, dim)) * (ub - lb)
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
                        pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                        pop[0] = best_ind
                        fitness[0] = best_fit
                        for j in range(1, N):
                            fitness[j] = func(pop[j])
                            n_evals += 1
                            if fitness[j] < self.f_opt:
                                self.f_opt = fitness[j]
                                self.x_opt = pop[j].copy()
                    # reset memory
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    prob_rand_mut = 0.2
                    memory_idx = 0
                    archive = np.empty((0, dim))
                    archive_max = N
                    evals_no_improve = 0
                else:
                    evals_no_improve = int(0.5 * evals_no_improve)  # reduce threshold

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt