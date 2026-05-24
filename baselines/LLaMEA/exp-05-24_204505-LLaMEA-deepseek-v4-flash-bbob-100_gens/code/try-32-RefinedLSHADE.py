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

        # Initial population size (inspired by iLSHADE)
        N_init = min(max(10 * dim, 100), max_evals // 2)
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

        # Archive
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F, CR, and strategy selection
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        # Strategy memory: 0 = current-to-pbest/1, 1 = current-to-rand/1
        Mstr = np.ones(H) * 0.5  # probability of using strategy 0 (pbest)
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        diversity_threshold = 0.001 * np.sum(ub - lb)  # scaled

        # Local search parameters
        local_search_interval = max(30, int(0.02 * max_evals))
        last_local_search = 0

        # Randomized pattern search (Hooke-Jeeves style with random directions)
        def randomized_pattern_search(best_pos, best_val, step, max_local_evals):
            pos = best_pos.copy()
            val = best_val
            step_size = step * (ub - lb)  # relative step per dimension
            used = 0
            while used < max_local_evals:
                improved = False
                # Exploratory moves: coordinate directions + additional random directions
                directions = []
                # axes
                for d in range(dim):
                    dir = np.zeros(dim)
                    dir[d] = 1.0
                    directions.append(dir)
                    directions.append(-dir)
                # random directions (sample uniformly on sphere)
                num_random = min(dim, max(3, max_local_evals - used - 10))
                for _ in range(num_random):
                    r = np.random.randn(dim)
                    r /= np.linalg.norm(r) + 1e-30
                    directions.append(r)
                    directions.append(-r)
                # Evaluate in each direction
                for dvec in directions:
                    if used >= max_local_evals:
                        break
                    new_pos = np.clip(pos + step_size * dvec, lb, ub)
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos
                        val = new_val
                        improved = True
                if improved:
                    # Pattern move: accelerate along net direction
                    delta = pos - best_pos
                    if np.any(np.abs(delta) > 1e-12):
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos
                            val = new_val
                    # Expand step
                    step_size *= 1.2
                    step_size = np.minimum(step_size, (ub - lb) * 0.5)
                    best_pos = pos.copy()
                    best_val = val
                else:
                    # Contract step
                    step_size *= 0.5
                    if np.max(step_size) < 1e-10 * np.max(ub - lb):
                        break
            return pos, val, used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            S_str = []  # store which strategy succeeded (0 for pbest, 1 for rand)
            delta_f = []

            for i in range(N):
                # Decide strategy based on success history
                mem = np.random.randint(H)
                use_pbest = np.random.rand() < Mstr[mem]
                # Sample F and CR
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                r1, r2 = i, i
                while r1 == i:
                    r1 = np.random.randint(N)
                if use_pbest:
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
                    mutant = base + F * diff1 + F * diff2
                else:
                    # current-to-rand/1 (no archive, better diversity)
                    r2 = np.random.randint(N)
                    while r2 == i or r2 == r1:
                        r2 = np.random.randint(N)
                    base = pop[i]
                    diff1 = pop[r1] - pop[r2]
                    # also use another individual from population
                    r3 = np.random.randint(N)
                    while r3 == i or r3 == r1 or r3 == r2:
                        r3 = np.random.randint(N)
                    mutant = base + F * diff1 + 0.5 * (pop[r3] - pop[i])  # slight blend

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # Boundary handling (reflect)
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
                    S_str.append(1 if use_pbest else 0)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memories with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                S_str = np.array(S_str)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                # Update strategy memory: fraction of successful pbest strategies (weighted)
                w_str = w * S_str
                Mstr[memory_idx] = np.sum(w_str) / (np.sum(w) + 1e-30)
                Mstr[memory_idx] = np.clip(Mstr[memory_idx], 0.1, 0.9)
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
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = randomized_pattern_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    # Insert replacing worst individual
                    worst_idx = np.argmax(fitness)
                    if new_val < fitness[worst_idx]:
                        pop[worst_idx] = new_pos
                        fitness[worst_idx] = new_val

            # Restart based on stagnation and diversity
            # Compute diversity: average pairwise distance normalized by range
            if n_evals < max_evals * 0.8:
                # Diversity check using sampling
                if N > 1:
                    idxs = np.random.choice(N, min(100, N), replace=False)
                    sampled_pop = pop[idxs]
                    centroid = np.mean(sampled_pop, axis=0)
                    distances = np.mean(np.abs(sampled_pop - centroid), axis=1)  # mean absolute deviation
                    diversity = np.mean(distances)
                else:
                    diversity = 1.0
                if (evals_no_improve > restart_threshold or diversity < diversity_threshold) and n_evals < max_evals * 0.8:
                    # Keep top individuals, generate rest with Latin hypercube around them
                    top_n = max(2, int(0.1 * N))
                    sorted_idx = np.argsort(fitness)
                    top_pos = pop[sorted_idx[:top_n]].copy()
                    top_fit = fitness[sorted_idx[:top_n]].copy()
                    remaining = N - top_n
                    # Generate Latin hypercube around centroid of top
                    centroid = np.mean(top_pos, axis=0)
                    stds = np.std(top_pos, axis=0) + 1e-30
                    new_samples = np.random.uniform(0, 1, (remaining, dim))
                    # Scale by stds to create diversity
                    new_samples = centroid + (new_samples - 0.5) * 4.0 * stds
                    new_samples = np.clip(new_samples, lb, ub)
                    pop = np.vstack((top_pos, new_samples))
                    fitness = np.full(N, np.inf)
                    fitness[:top_n] = top_fit
                    for j in range(top_n, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    # Reset memories partially
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    Mstr[:] = 0.5
                    memory_idx = 0
                    archive = np.empty((0, dim))
                    archive_max = N
                    evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt