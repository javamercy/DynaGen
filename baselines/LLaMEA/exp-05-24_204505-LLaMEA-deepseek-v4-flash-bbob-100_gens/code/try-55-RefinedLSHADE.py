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

        # Latin hypercube initialisation
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
        archive_max = 2 * N  # standard LSHADE

        # Success‑history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Mutation strategy memory (0: current-to-pbest, 1: rand/1)
        strategy_mem = np.ones(H) * 0.1  # probability of using rand/1
        strategy_idx = 0
        last_strategy = 0

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = max(0.2 * max_evals, 100 * dim)
        diversity_threshold = 0.01 * np.linalg.norm(ub - lb)  # per dimension avg

        # Local search parameters
        local_search_interval = max(50, int(0.03 * max_evals))
        last_local_search = 0

        # Helper: number of local search evaluations (small)
        def local_search(best_pos, best_val, budget_local):
            pos = best_pos.copy()
            val = best_val
            step = 0.1 * (1 - n_evals / max_evals) + 0.01  # shrinking step
            used = 0
            for _ in range(budget_local // (dim + 1)):
                if used >= budget_local:
                    break
                # Random direction (normalized)
                d = np.random.randn(dim)
                d = d / (np.linalg.norm(d) + 1e-30)
                # Positive step
                new_pos = np.clip(pos + step * d * (ub - lb), lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    # Expand step on success
                    step *= 1.1
                    continue
                # Negative step
                new_pos = np.clip(pos - step * d * (ub - lb), lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    step *= 1.1
                else:
                    step *= 0.9  # contract on failure
                step = np.clip(step, 1e-8, 0.5)
            return pos, val, used

        # Diversity measure (mean pairwise distance among top 20%)
        def compute_diversity(pop, fitness):
            sorted_idx = np.argsort(fitness)
            top = min(len(pop), max(5, int(0.2 * len(pop))))
            top_pop = pop[sorted_idx[:top]]
            if top <= 1:
                return 0.0
            dist = np.zeros((top, top))
            for i in range(top):
                for j in range(i+1, top):
                    dist[i,j] = np.linalg.norm(top_pop[i] - top_pop[j])
            return np.mean(dist) if top > 1 else 0.0

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: exponential decay
            p = 0.2 * np.exp(-5 * (n_evals / max_evals)) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            S_strategy = []  # track which strategy was used for successful updates

            # Generate offspring
            for i in range(N):
                # Decide mutation strategy (current-to-pbest or rand/1)
                if np.random.rand() < strategy_mem[strategy_idx]:
                    use_rand1 = True
                else:
                    use_rand1 = False

                # Choose distinct random indices
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                if use_rand1:
                    # DE/rand/1
                    mutant = pop[r1] + F * (pop[np.random.choice(idxs)] - pop[np.random.choice(idxs)])
                else:
                    # current-to-pbest/1/archive
                    pbest_size = max(1, int(p * N))
                    sorted_idx = np.argsort(fitness)
                    pbest_candidates = sorted_idx[:pbest_size]
                    pbest_idx = np.random.choice(pbest_candidates)
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: clamp with reflection
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

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    S_strategy.append(1 if use_rand1 else 0)  # 1 = rand/1 used
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

                # Update strategy memory
                succ_strategies = np.array(S_strategy)[sorted_order]
                w_strat = w
                if len(succ_strategies) > 0:
                    prob_rand1 = np.sum(w_strat * succ_strategies) / (np.sum(w_strat) + 1e-30)
                    strategy_mem[strategy_idx] = 0.9 * strategy_mem[strategy_idx] + 0.1 * prob_rand1
                    strategy_idx = (strategy_idx + 1) % H
                last_strategy = 0

            # Population size reduction (linear)
            N_new = N_min + (N_init - N_min) * (1 - n_evals / max_evals)
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = 2 * N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using random directional search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 10)
                new_pos, new_val, used = local_search(best_pos, best_val, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst
                worst_idx = np.argmax(fitness)
                if best_val < fitness[worst_idx]:
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Restart conditions: stagnation + low diversity
            diversity = compute_diversity(pop, fitness)
            stagnation = evals_no_improve > restart_threshold
            low_diversity = diversity < diversity_threshold and evals_no_improve > 50 * dim
            need_restart = stagnation or low_diversity

            if need_restart and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new individuals using opposition-based learning
                half = new_N // 2
                # Quasi-random Latin hypercube
                samples = np.random.uniform(0, 1, (half, dim))
                samples = lb + samples * (ub - lb)
                # Opposition of best
                opp_best = lb + ub - best_ind
                opp_samples = np.random.uniform(0, 1, (half, dim))
                opp_samples = lb + opp_samples * (ub - lb)
                # Combine
                pop = np.vstack((best_ind.reshape(1, -1), samples, opp_samples))
                if pop.shape[0] < new_N:
                    extra = new_N - pop.shape[0]
                    extra_samples = np.random.uniform(0, 1, (extra, dim))
                    extra_samples = lb + extra_samples * (ub - lb)
                    pop = np.vstack((pop, extra_samples))
                pop = pop[:new_N]
                # Evaluate new individuals
                fitness = np.full(new_N, np.inf)
                fitness[0] = best_fit
                for j in range(1, new_N):
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset memory and archive
                MF[:] = 0.5
                MCR[:] = 0.8
                strategy_mem[:] = 0.1
                memory_idx = 0
                strategy_idx = 0
                archive = np.empty((0, dim))
                archive_max = 2 * N
                evals_no_improve = 0
                # Reset local search timer
                last_local_search = n_evals

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt