import numpy as np
from scipy.stats import qmc

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

        # Latin hypercube initialization with Halton sequence
        sampler = qmc.LatinHypercube(d=dim)
        samples = sampler.random(n=N)
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation (FIFO-like)
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Ensemble mutation strategy probabilities (initial equally weighted)
        strat_probs = np.array([0.33, 0.33, 0.34])  # three strategies
        strat_success = np.zeros(3, dtype=np.float64)

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.01 * np.mean(ub - lb)

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Helper: exponential crossover
        def exponential_crossover(target, mutant, cr):
            trial = target.copy()
            start = np.random.randint(dim)
            L = 0
            while L < dim and np.random.rand() < cr:
                L += 1
            for i in range(L):
                trial[(start + i) % dim] = mutant[(start + i) % dim]
            return trial

        # Pattern search with adaptive step (expanding/contracting)
        def pattern_search(best_pos, best_val, step, max_local_evals, shrink_factor=1.0):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb) * shrink_factor
            used = 0
            iterations = 0
            while used < max_local_evals and iterations < dim * 4:
                iterations += 1
                improved = False
                # Coordinate search
                for d in range(dim):
                    if used >= max_local_evals:
                        break
                    # positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step_size[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        def compute_diversity(pop):
            mean = np.mean(pop, axis=0)
            return np.mean(np.sqrt(np.sum((pop - mean)**2, axis=1)))

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            strat_counts = np.zeros(3, dtype=np.int64)
            strat_success_acc = np.zeros(3, dtype=np.float64)

            # Generate offspring
            for i in range(N):
                # Select strategy based on probabilities
                strat_idx = np.random.choice(3, p=strat_probs)
                strat_counts[strat_idx] += 1

                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
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
                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                base = pop[i]
                # Mutation according to strategy
                if strat_idx == 0:  # current-to-pbest/1/archive
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                elif strat_idx == 1:  # current-to-rand/1
                    r3 = np.random.choice(idxs)
                    r4 = np.random.choice(idxs)
                    mutant = base + F * (pop[r1] - pop[i]) + F * (pop[r3] - pop[r4])
                else:  # DE/rand/1/bin
                    mutant = pop[r1] + F * (pop[np.random.choice(idxs)] - pop[np.random.choice(idxs)])
                    CR = np.clip(CR * 1.2, 0.0, 1.0)  # slightly higher CR for this strategy

                # Crossover: binomial or exponential with some randomness
                if np.random.rand() < 0.5:
                    # binomial
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]
                else:
                    # exponential
                    trial = exponential_crossover(base, mutant, CR)

                # Boundary handling: reflection and clamping
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
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    strat_success_acc[strat_idx] += 1
                    # Add parent to archive (FIFO-like)
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        archive = archive[1:]  # remove oldest
                else:
                    # Record failure for strategy? optional
                    pass

            # Update strategy probabilities based on success rates
            if strat_counts.sum() > 0:
                success_rates = np.where(strat_counts > 0, strat_success_acc / strat_counts, 0)
                # Smooth update
                strat_probs = 0.9 * strat_probs + 0.1 * (success_rates / (success_rates.sum() + 1e-30))
                strat_probs = np.clip(strat_probs, 0.05, None)  # minimum probability
                strat_probs /= strat_probs.sum()

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

            # Population size reduction (quadratic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    archive = archive[:archive_max]  # keep newest (already FIFO)
                N = N_new

            # Periodic local refinement using pattern search (with adaptive shrink factor)
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                # Use a shrink factor that decreases with remaining budget to focus exploitation
                shrink = 1.0 - 0.5 * (n_evals / max_evals)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local, shrink_factor=shrink)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual if improved
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Check diversity and stagnation for restart
            diversity = compute_diversity(pop)
            need_restart = False
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                need_restart = True
            if diversity < diversity_threshold and n_evals > 0.1 * max_evals:
                need_restart = True

            if need_restart:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min, 10)
                # Partially reinitialize: keep best individual, generate others from Latin hypercube
                sampler = qmc.LatinHypercube(d=dim)
                samples = sampler.random(n=new_N)
                new_pop = lb + samples * (ub - lb)
                new_fitness = np.full(new_N, np.inf)
                new_pop[0] = best_ind
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
                # Reset memory but keep some diversity
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                # Reset strategy probabilities
                strat_probs = np.array([0.33, 0.33, 0.34])
                strat_success_acc[:] = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt