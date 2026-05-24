import numpy as np

class ImprovedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        dim = self.dim
        max_evals = self.budget

        # Initial population size
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = 4
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

        # Archive for inferior solutions
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory (H=10)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation detection parameters
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        stagnation_threshold = 0.12 * max_evals  # slightly lower than before

        # Diversity-based restart parameters
        diversity_threshold = 0.5 * (ub - lb).mean()  # adapt to dimension
        diversity_check_interval = max(10, max_evals // 100)
        generation = 0

        while n_evals < max_evals:
            generation += 1
            p = 0.2 - 0.1 * (n_evals / max_evals)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # For each individual
            for i in range(N):
                # Select r1 != i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # Select r2 from union of population and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])

                # pbest selection
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR from memory using Cauchy and normal
                mem = np.random.randint(H)
                # Use truncated Cauchy for F
                F = MF[mem] + 0.1 * np.random.standard_cauchy()
                F = np.clip(F, 0, 1) if F > 0 else np.random.uniform(0.1, 0.5)
                CR = np.clip(MCR[mem] + 0.1 * np.random.standard_normal(), 0, 1)

                # Mutation: current-to-pbest/1 with archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Bound handling: mirror and random reinit if still out
                trial = np.where(trial < lb, 2*lb - trial, trial)
                trial = np.where(trial > ub, 2*ub - trial, trial)
                mask = (trial < lb) | (trial > ub)
                if np.any(mask):
                    trial[mask] = lb[mask] + (ub[mask] - lb[mask]) * np.random.rand(np.sum(mask))

                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_since_last_improvement = 0
                else:
                    evals_since_last_improvement += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial

                    # Add replaced individual to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory using weighted Lehmer mean (as in SHADE)
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / np.sum(delta_f)
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F) if np.sum(w * S_F) != 0 else 0.5
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR) if np.sum(w * S_CR) != 0 else 0.5
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = round(N_init - (N_init - N_min) * (n_evals / max_evals))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Diversity-based restart check (every few generations)
            if generation % diversity_check_interval == 0 and n_evals < max_evals * 0.8:
                # Compute average distance from centroid
                centroid = np.mean(pop, axis=0)
                distances = np.sqrt(np.sum((pop - centroid)**2, axis=1))
                mean_dist = np.mean(distances)
                if mean_dist < diversity_threshold * (1 - 0.5 * n_evals / max_evals):  # decay threshold
                    # Keep best individual
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    # Reinitialize population: 30% around best, 70% uniform
                    N_restart = N
                    pop_new = np.empty((N_restart, dim))
                    pop_new[0] = best_ind
                    # Generate around best using Cauchy
                    n_around = max(1, int(0.3 * N_restart))
                    for j in range(1, n_around):
                        step = 0.5 * (ub - lb) * np.random.standard_cauchy(dim)  # Cauchy step
                        trial = best_ind + step
                        trial = np.clip(trial, lb, ub)
                        pop_new[j] = trial
                    # Remaining uniform
                    for j in range(n_around, N_restart):
                        pop_new[j] = lb + (ub - lb) * np.random.rand(dim)
                    # Evaluate new individuals (except best)
                    new_fitness = np.full(N_restart, np.inf)
                    new_fitness[0] = best_fit
                    for j in range(1, N_restart):
                        new_fitness[j] = func(pop_new[j])
                        n_evals += 1
                        if new_fitness[j] < self.f_opt:
                            self.f_opt = new_fitness[j]
                            self.x_opt = pop_new[j].copy()
                    pop = pop_new
                    fitness = new_fitness
                    # Reset archive and memory
                    archive = np.empty((0, dim))
                    archive_max = N_restart
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    evals_since_last_improvement = 0
            # Stagnation-based restart (as before but with lower threshold)
            if evals_since_last_improvement > stagnation_threshold and n_evals < max_evals * 0.8:
                restarts_remaining = max_evals - n_evals
                if restarts_remaining > N_init * 0.5:
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    N_restart = N
                    pop_new = np.empty((N_restart, dim))
                    pop_new[0] = best_ind
                    # Uniform for rest
                    for j in range(1, N_restart):
                        pop_new[j] = lb + (ub - lb) * np.random.rand(dim)
                    new_fitness = np.full(N_restart, np.inf)
                    new_fitness[0] = best_fit
                    for j in range(1, N_restart):
                        new_fitness[j] = func(pop_new[j])
                        n_evals += 1
                        if new_fitness[j] < self.f_opt:
                            self.f_opt = new_fitness[j]
                            self.x_opt = pop_new[j].copy()
                    pop = pop_new
                    fitness = new_fitness
                    archive = np.empty((0, dim))
                    archive_max = N_restart
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    evals_since_last_improvement = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt