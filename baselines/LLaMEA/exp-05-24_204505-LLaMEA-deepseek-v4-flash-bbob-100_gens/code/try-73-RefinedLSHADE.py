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

        # Population size parameters
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

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR (two strategies)
        H = 10
        # Strategy 1: current-to-pbest/1 with archive (exploitation)
        MF1 = np.ones(H) * 0.5
        MCR1 = np.ones(H) * 0.8
        # Strategy 2: current-to-rand/1 (exploration, no archive)
        MF2 = np.ones(H) * 0.5
        MCR2 = np.ones(H) * 0.8
        memory_idx = 0

        # Adaptive strategy selection
        strategy_probs = [0.5, 0.5]  # initially equal
        strategy_success = [0, 0]
        strategy_attempts = [0, 0]
        strategy_window = 50
        strategy_win_count = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # Local search parameters
        local_search_interval = max(20, int(0.015 * max_evals))
        last_local_search = 0

        # Local search: random perturbation around best with adaptive step size
        def local_search(best_pos, best_val, step_size, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            step = step_size * (ub - lb) * 0.1
            used = 0
            while used < max_evals_local:
                # Random direction
                direction = np.random.randn(dim)
                direction /= np.linalg.norm(direction) + 1e-30
                new_pos = np.clip(pos + step * direction, lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    step *= 1.1  # expand on success
                else:
                    step *= 0.9  # contract on failure
                if step < 1e-10:
                    break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05 using cubic
            p = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05
            # Adaptive strategy selection: update probabilities every window
            if strategy_win_count >= strategy_window:
                total_success = max(1, strategy_success[0] + strategy_success[1])
                total_attempts = max(1, strategy_attempts[0] + strategy_attempts[1])
                # Update probabilities based on success rates
                rate1 = strategy_success[0] / max(1, strategy_attempts[0])
                rate2 = strategy_success[1] / max(1, strategy_attempts[1])
                sum_rate = rate1 + rate2 + 1e-30
                strategy_probs[0] = 0.9 * strategy_probs[0] + 0.1 * (rate1 / sum_rate)
                strategy_probs[1] = 0.9 * strategy_probs[1] + 0.1 * (rate2 / sum_rate)
                # Reset counters
                strategy_success = [0, 0]
                strategy_attempts = [0, 0]
                strategy_win_count = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F1 = []; S_CR1 = []; delta_f1 = []
            S_F2 = []; S_CR2 = []; delta_f2 = []

            for i in range(N):
                # Choose strategy according to probabilities
                strat = 0 if np.random.rand() < strategy_probs[0] else 1
                strategy_attempts[strat] += 1

                # Select memory based on strategy
                if strat == 0:
                    MF = MF1; MCR = MCR1
                else:
                    MF = MF2; MCR = MCR2

                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                if strat == 0:
                    # current-to-pbest/1 with archive
                    if archive.size > 0:
                        union = np.vstack((pop, archive))
                    else:
                        union = pop
                    r2 = np.random.randint(union.shape[0])
                    pbest_size = max(1, int(p * N))
                    sorted_idx = np.argsort(fitness)
                    pbest_candidates = sorted_idx[:pbest_size]
                    pbest_idx = np.random.choice(pbest_candidates)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                else:
                    # current-to-rand/1 (no archive)
                    r2 = np.random.randint(N)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(N)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[r2] - base
                    F_rand = 0.5 * (1 + np.random.rand())  # random F for exploration

                # Sample F and CR from memory (cauchy/normal)
                mem = np.random.randint(H)
                if strat == 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                else:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    F = 0.5 * (F + F_rand)  # combine with random for exploration
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                if strat == 0:
                    mutant = base + F * diff1 + F * diff2
                else:
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

                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    if strat == 0:
                        S_F1.append(F); S_CR1.append(CR); delta_f1.append(fitness[i] - trial_f)
                        # Add parent to archive (only for strategy 0)
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                        if archive.shape[0] > archive_max:
                            remove_idx = np.random.randint(archive.shape[0])
                            archive = np.delete(archive, remove_idx, axis=0)
                    else:
                        S_F2.append(F); S_CR2.append(CR); delta_f2.append(fitness[i] - trial_f)
                    strategy_success[strat] += 1
                    new_fitness[i] = trial_f
                    new_pop[i] = trial

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory for each strategy
            if len(S_F1) > 0:
                sorted_order = np.argsort(delta_f1)[::-1]
                S_F1_arr = np.array(S_F1)[sorted_order]
                S_CR1_arr = np.array(S_CR1)[sorted_order]
                w = np.array(delta_f1)[sorted_order] / (np.sum(delta_f1) + 1e-30)
                MF1[memory_idx] = np.sum(w * S_F1_arr ** 2) / (np.sum(w * S_F1_arr) + 1e-30)
                MCR1[memory_idx] = np.sum(w * S_CR1_arr ** 2) / (np.sum(w * S_CR1_arr) + 1e-30)
            if len(S_F2) > 0:
                sorted_order = np.argsort(delta_f2)[::-1]
                S_F2_arr = np.array(S_F2)[sorted_order]
                S_CR2_arr = np.array(S_CR2)[sorted_order]
                w = np.array(delta_f2)[sorted_order] / (np.sum(delta_f2) + 1e-30)
                MF2[memory_idx] = np.sum(w * S_F2_arr ** 2) / (np.sum(w * S_F2_arr) + 1e-30)
                MCR2[memory_idx] = np.sum(w * S_CR2_arr ** 2) / (np.sum(w * S_CR2_arr) + 1e-30)
            memory_idx = (memory_idx + 1) % H
            strategy_win_count += 1

            # Population size reduction (quadratic)
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

            # Local search around best
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step_size = 0.2 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 2, max_evals - n_evals - 5)
                new_pos, new_val, used = local_search(best_pos, best_val, step_size, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart if stagnation detected
            if (evals_no_improve > restart_threshold) and (n_evals < max_evals * 0.8):
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
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
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memories
                MF1[:] = 0.5; MCR1[:] = 0.8
                MF2[:] = 0.5; MCR2[:] = 0.8
                memory_idx = 0
                strategy_probs = [0.5, 0.5]
                strategy_success = [0, 0]
                strategy_attempts = [0, 0]
                strategy_win_count = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt