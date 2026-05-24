import numpy as np

class RefinedLSHADE:
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

        # --- Population sizes ---
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # --- Latin hypercube initialization ---
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

        # --- Archive ---
        archive = np.empty((0, dim))
        archive_max = N

        # --- Success-history for F and CR ---
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # --- Adaptive mutation strategy pool ---
        # two strategies: 0: current-to-pbest/1/archive, 1: rand-to-pbest/1/archive
        strategy_rates = np.array([0.5, 0.5])  # selection probability
        strategy_success = np.zeros(2)
        strategy_total = np.zeros(2)

        # --- Stagnation & diversity ---
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold_eval = 0.15 * max_evals
        diversity_threshold = 0.05 * (ub - lb).mean()

        # --- Self-adaptive local search (SALS) parameters ---
        ls_cov = np.eye(dim) * 0.1
        ls_step = 0.1 * (ub - lb).mean()
        ls_success_rate = 0.0
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # ------------------------------------------------------------
        # Helper: Self-adaptive local search (inspired by simple CMA)
        # ------------------------------------------------------------
        def self_adaptive_local_search(best_x, best_f, remaining_budget):
            nonlocal ls_cov, ls_step, ls_success_rate
            if remaining_budget < dim + 2:
                return best_x, best_f, 0

            x = best_x.copy()
            f = best_f
            used = 0
            # Try a few random directions scaled by covariance
            n_trials = min(dim * 2, remaining_budget // 2)
            success_count = 0
            for _ in range(n_trials):
                if used >= remaining_budget - dim:
                    break
                # sample direction from N(0, cov)
                d = np.random.multivariate_normal(np.zeros(dim), ls_cov)
                d = d / (np.linalg.norm(d) + 1e-30)
                # left and right steps
                step = ls_step
                for sign in [1, -1]:
                    trial = np.clip(x + sign * step * d, lb, ub)
                    val = func(trial)
                    used += 1
                    if used >= remaining_budget:
                        break
                    if val < f:
                        x = trial.copy()
                        f = val
                        # update covariance with success direction
                        # exponential smoothing
                        alpha = 0.1
                        ls_cov = (1 - alpha) * ls_cov + alpha * np.outer(d, d)
                        success_count += 1
                        break  # accept first improvement

            # Update step size based on success rate
            if n_trials > 0:
                sr = success_count / n_trials
                ls_success_rate = 0.9 * ls_success_rate + 0.1 * sr
                if ls_success_rate > 0.2:
                    ls_step *= 1.1
                else:
                    ls_step *= 0.9
                ls_step = min(ls_step, (ub - lb).mean() * 0.5)
                ls_step = max(ls_step, (ub - lb).mean() * 1e-6)

            return x, f, used

        # ------------------------------------------------------------
        # Main loop
        # ------------------------------------------------------------
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            strat_used = [0, 0]
            strat_success = [0, 0]

            # --- Generate offspring ---
            for i in range(N):
                # Choose mutation strategy adaptively
                if np.random.rand() < strategy_rates[0]:
                    strat = 0  # current-to-pbest/1/archive
                else:
                    strat = 1  # rand-to-pbest/1/archive

                # Build vectors
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                sorted_idx = np.argsort(fitness)
                pbest_size = max(1, int(p * N))
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F, CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                if strat == 0:
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    base = pop[r1]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[np.random.choice(idxs)] - union[r2]  # use another random from pop
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

                strat_used[strat] += 1
                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    strat_success[strat] += 1
                    # Archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # --- Update population & archive ---
            pop = new_pop
            fitness = new_fitness

            # --- Update strategy selection rates ---
            for s in range(2):
                if strat_used[s] > 0:
                    strategy_success[s] = 0.8 * strategy_success[s] + 0.2 * (strat_success[s] / strat_used[s])
                strategy_total[s] = 0.8 * strategy_total[s] + 0.2 * (strat_used[s] / (N + 1e-30))
            # Normalize rates
            total = strategy_success.sum() + 1e-30
            if total > 0:
                strategy_rates = strategy_success / total
            else:
                strategy_rates = np.array([0.5, 0.5])

            # --- Update F and CR memory ---
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # --- Population size reduction (quadratic) ---
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

            # --- Local search (self-adaptive) ---
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                remaining = max_evals - n_evals
                max_local = min(dim * 3, remaining - 5)
                new_pos, new_val, used = self_adaptive_local_search(best_pos, best_val, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # --- Restart based on stagnation and diversity ---
            diversity = np.std(pop, axis=0).mean()
            if (evals_no_improve > restart_threshold_eval or diversity < diversity_threshold) and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Quasi-random Latin hypercube
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
                    # Partial restart: randomize all but best
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memories
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Also reset local search covariance
                ls_cov = np.eye(dim) * 0.1
                ls_step = 0.1 * (ub - lb).mean()
                ls_success_rate = 0.0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt