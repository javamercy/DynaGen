import numpy as np
from scipy.stats import cauchy

class RefinedLSHADE_enhanced:
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

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Memory for mutation strategy selection (0: pbest, 1: rand)
        strategy_memory = np.ones(H) * 0.5  # probability of using pbest
        strategy_idx = 0

        # Stagnation and diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.01 * (ub - lb).mean()
        last_diversity_check = 0

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Pattern search with adaptive step and orthogonal directions
        def pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            # Generate a set of orthogonal directions via Gram-Schmidt
            directions = np.eye(dim)
            # Random rotation for better coverage
            Q, _ = np.linalg.qr(np.random.randn(dim, dim))
            directions = (Q @ directions.T).T  # rotate each direction
            step_size = step * (ub - lb) * 0.1  # base step per dimension
            used = 0
            while used < max_local_evals:
                improved = False
                # Pattern move: try all directions once
                for d_idx in range(dim):
                    if used >= max_local_evals:
                        break
                    d = directions[d_idx]
                    # Positive direction
                    new_pos = np.clip(pos + step_size * d, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                        continue
                    # Negative direction
                    new_pos = np.clip(pos - step_size * d, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: double step in the overall improvement direction
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step size
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step size
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreases non-linearly
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []
            success_strategy = []  # 0 for pbest, 1 for rand

            # Generate offspring
            for i in range(N):
                # Choose mutation strategy adaptively
                if np.random.rand() < strategy_memory[strategy_idx]:
                    strategy = 0  # current-to-pbest/1
                else:
                    strategy = 1  # current-to-rand/1

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * cauchy.rvs(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * cauchy.rvs(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Select indices
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

                if strategy == 0:
                    # current-to-pbest/1
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    # current-to-rand/1 (generates trial directly, no crossover)
                    K = np.random.uniform(0, 1)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = pop[pbest_idx] - union[r2]
                    trial = base + K * diff1 + F * diff2
                    # Crossover not needed for pure rand/1, but we apply binomial for consistency
                    # Actually we can apply binomial crossover with CR
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, trial, base)
                    trial[j_rand] = mutant[j_rand] if strategy == 0 else trial[j_rand]

                if strategy == 0:
                    # Binomial crossover for pbest mutation
                    j_rand = np.random.randint(dim)
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                    trial[j_rand] = mutant[j_rand]

                # Boundary handling: reflection + random mutation if still outside
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
                    success_strategy.append(strategy)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive (only for pbest mutation)
                    if strategy == 0:
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
                # Update strategy memory: fraction of pbest successes
                if len(success_strategy) > 0:
                    pbest_success = sum(success_strategy) / len(success_strategy)
                    strategy_memory[strategy_idx] = 0.9 * strategy_memory[strategy_idx] + 0.1 * pbest_success
                memory_idx = (memory_idx + 1) % H
                strategy_idx = (strategy_idx + 1) % H

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
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement using pattern search
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 3, max_evals - n_evals - 5)
                new_pos, new_val, used = pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst individual if improvement
                if best_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Diversity check for restart
            if n_evals - last_diversity_check > max_evals * 0.05:
                last_diversity_check = n_evals
                if N > 1:
                    distances = np.linalg.norm(pop - pop[0], axis=1)
                    avg_dist = np.mean(distances[1:])  # exclude best itself
                    if avg_dist < diversity_threshold:
                        # low diversity → restart
                        evals_no_improve = max_evals  # force restart

            # Restart if stagnation or low diversity
            if (evals_no_improve > restart_threshold and n_evals < max_evals * 0.8) or \
               (n_evals - last_diversity_check < 0.05*max_evals and np.mean(np.linalg.norm(pop - pop[0], axis=1)) < diversity_threshold and N>1):
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
                    # Partial restart: randomize all but best, but add scaled noise
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind + np.random.normal(0, 0.1*(ub-lb), dim)
                    pop[0] = np.clip(pop[0], lb, ub)
                    for j in range(N):
                        if j == 0:
                            continue
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memories
                MF[:] = 0.5
                MCR[:] = 0.5
                strategy_memory[:] = 0.5
                memory_idx = 0
                strategy_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0
                diversity_threshold = 0.01 * (ub - lb).mean() * (1 + 0.1 * np.random.randn())

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt