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

        # initial population size
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, dim // 5)
        N = N_init

        # latin hypercube initialisation
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

        # archive
        archive = np.empty((0, dim))
        archive_max = N

        # success history for F and CR (two memories: one for each strategy)
        H = 10
        MF = np.ones((2, H)) * 0.5
        MCR = np.ones((2, H)) * 0.8
        memory_idx = np.zeros(2, dtype=int)  # one per strategy

        # strategy probabilities, initially 0.5 each
        strategy_prob = np.array([0.5, 0.5])
        strategy_successes = 0.0
        strategy_failures = 0.0

        # stagnation detection
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals

        # local search parameters
        local_interval = max(30, int(0.02 * max_evals))
        last_local = 0

        # helper: simplex local search (Nelder-Mead variant)
        def simplex_local(best_pos, best_val, step0, max_evals_local):
            # generate initial simplex around best
            n = dim
            step = step0 * (ub - lb)
            sigma = np.minimum(step, 0.5 * (ub - lb))
            simplex = np.tile(best_pos, (n + 1, 1)) + np.eye(n + 1, n) * sigma
            simplex = np.clip(simplex, lb, ub)
            vals = np.full(n + 1, best_val)
            for i in range(n):
                vals[i] = func(simplex[i])
            vals[n] = best_val
            simplex[n] = best_pos
            evals_used = n
            # standard Nelder-Mead parameters
            alpha, gamma, rho, sigma_ = 1.0, 2.0, 0.5, 0.5
            for _ in range(max_evals_local):
                if evals_used >= max_evals_local:
                    break
                # order
                order = np.argsort(vals)
                simplex = simplex[order]
                vals = vals[order]
                # centroid
                x0 = np.mean(simplex[:-1], axis=0)
                # reflection
                xr = x0 + alpha * (x0 - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals_used += 1
                if vals[0] <= fr < vals[-2]:
                    # accept reflection
                    simplex[-1] = xr
                    vals[-1] = fr
                elif fr < vals[0]:
                    # expansion
                    xe = x0 + gamma * (xr - x0)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals_used += 1
                    if fe < fr:
                        simplex[-1] = xe
                        vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fr
                else:
                    # contraction
                    xc = x0 + rho * (x0 - simplex[-1])
                    xc = np.clip(xc, lb, ub)
                    fc = func(xc)
                    evals_used += 1
                    if fc < vals[-1]:
                        simplex[-1] = xc
                        vals[-1] = fc
                    else:
                        # shrink simplex towards best
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_ * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lb, ub)
                            vals[i] = func(simplex[i])
                            evals_used += 1
                            if evals_used >= max_evals_local:
                                break
                if np.max(np.abs(simplex[0] - simplex[1:]).max(axis=1)) < 1e-8 * (ub - lb).max():
                    break
            best_idx = np.argmin(vals)
            return simplex[best_idx], vals[best_idx], evals_used

        # main loop
        while n_evals < max_evals:
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[] for _ in range(2)]
            S_CR = [[] for _ in range(2)]
            delta_f = [[] for _ in range(2)]

            for i in range(N):
                # select strategy
                strat = 0 if np.random.rand() < strategy_prob[0] else 1

                # sample F, CR from respective memory
                mem = np.random.randint(H)
                F = np.clip(MF[strat, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[strat, mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[strat, mem] + 0.1 * np.random.randn(), 0, 1)

                # choose indices
                idxs = [j for j in range(N) if j != i]
                r1 = np.random.choice(idxs)
                union = np.vstack((pop, archive)) if archive.size > 0 else pop
                r2 = np.random.randint(union.shape[0])
                sorted_idx = np.argsort(fitness)
                pbest_size = max(1, int(p * N))
                pbest = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest)

                # mutation
                if strat == 0:  # current-to-pbest/1/archive
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:  # current-to-rand/1 (no archive)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[np.random.choice(N)] - pop[np.random.choice(N)]
                    mutant = base + F * diff1 + 0.5 * F * diff2

                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # boundary handling
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
                    S_F[strat].append(F)
                    S_CR[strat].append(CR)
                    delta_f[strat].append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    strategy_successes += 1
                    # add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)
                else:
                    strategy_failures += 1

            # update strategy probabilities
            total = strategy_successes + strategy_failures
            if total > 0:
                success_rate = strategy_successes / total if total > 0 else 0.5
                # keep probabilities flexible: mix with previous
                strategy_prob[0] = 0.5 * strategy_prob[0] + 0.5 * success_rate
                strategy_prob[1] = 1 - strategy_prob[0]

            # update memory for each strategy
            for strat in range(2):
                if len(S_F[strat]) > 0:
                    sorted_order = np.argsort(delta_f[strat])[::-1]
                    F_arr = np.array(S_F[strat])[sorted_order]
                    CR_arr = np.array(S_CR[strat])[sorted_order]
                    w = np.array(delta_f[strat])[sorted_order] / (np.sum(delta_f[strat]) + 1e-30)
                    MF[strat, memory_idx[strat]] = np.sum(w * F_arr ** 2) / (np.sum(w * F_arr) + 1e-30)
                    MCR[strat, memory_idx[strat]] = np.sum(w * CR_arr ** 2) / (np.sum(w * CR_arr) + 1e-30)
                    memory_idx[strat] = (memory_idx[strat] + 1) % H

            # update population
            pop = new_pop
            fitness = new_fitness

            # population reduction (linear schedule)
            N_new = N_min + (N_init - N_min) * (1 - n_evals / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                # keep best and maintain diversity via crowding distance
                order = np.argsort(fitness)
                best_idx = order[0]
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                # select remaining with crowding distance
                remaining = [x for x in order if x != best_idx]
                selected = [best_idx]
                while len(selected) < N_new and remaining:
                    # compute crowding distance for remaining
                    dists = []
                    for idx in remaining:
                        # distance to nearest selected in objective space
                        d = np.min([np.linalg.norm(fitness[idx] - fitness[s]) for s in selected])
                        dists.append(d)
                    # select the one with largest distance
                    i_max = np.argmax(dists)
                    selected.append(remaining[i_max])
                    remaining.pop(i_max)
                pop = pop[selected]
                fitness = fitness[selected]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # local search (simplex) every 'local_interval' evals
            if (n_evals - last_local >= local_interval) and (n_evals < max_evals * 0.95):
                last_local = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 5, max_evals - n_evals - 5)
                new_pos, new_val, used = simplex_local(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # replace worst individual
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # restart if stagnation
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # compute covariance of top solutions for better resampling
                top_k = min(2 * dim, N)
                top_idx = np.argsort(fitness)[:top_k]
                top_pos = pop[top_idx]
                mean = np.mean(top_pos, axis=0)
                cov = np.cov(top_pos, rowvar=False) + 1e-8 * np.eye(dim)
                # sample from multivariate normal bounded
                samples = np.random.multivariate_normal(mean, cov, size=new_N)
                samples = np.clip(samples, lb, ub)
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
                # reset memories with small noise
                MF[:] = 0.5 + 0.1 * np.random.randn(2, H)
                MCR[:] = 0.5 + 0.1 * np.random.randn(2, H)
                memory_idx[:] = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt