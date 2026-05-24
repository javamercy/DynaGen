import numpy as np

class ImprovedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)  # ensure array
        ub = np.array(func.bounds.ub)
        dim = self.dim
        max_evals = self.budget

        # Population size
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

        # Archive
        archive = np.empty((0, dim))
        archive_max = N

        # Memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation and diversity tracking
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        restart_threshold = 0.15 * max_evals
        # Initial diversity: average distance to best
        best_idx = np.argmin(fitness)
        initial_diversity = np.mean(np.linalg.norm(pop - pop[best_idx], axis=1))
        diversity_threshold = 0.01 * initial_diversity  # adaptive threshold

        # Main loop
        while n_evals < max_evals:
            p = 0.2 - 0.1 * (n_evals / max_evals)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
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
                # Occasionally use extreme F for exploration
                if np.random.rand() < 0.1:
                    F = 0.9 if np.random.rand() < 0.5 else 0.1
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

                # Bound handling: mirror then random
                trial = np.where(trial < lb, 2*lb - trial, trial)
                trial = np.where(trial > ub, 2*ub - trial, trial)
                mask = (trial < lb) | (trial > ub)
                if np.any(mask):
                    trial[mask] = lb[mask] + (ub[mask] - lb[mask]) * np.random.rand(np.sum(mask))

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
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Update memory
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / np.sum(delta_f)
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F)  # Lehmer
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H

            # Linear population size reduction
            N_new = max(N_min, round(N_init - (N_init - N_min) * (n_evals / max_evals)))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Diversity measure and local injection
            best_idx = np.argmin(fitness)
            dist_to_best = np.linalg.norm(pop - pop[best_idx], axis=1)
            current_diversity = np.mean(dist_to_best)
            # If diversity too low, perform a local restart (replace worst 25% with points near best)
            if current_diversity < diversity_threshold and n_evals < max_evals * 0.9:
                n_replace = max(1, int(0.25 * N))
                indices = np.argsort(fitness)[::-1][:n_replace]  # worst
                for idx in indices:
                    # Gaussian perturbation around best
                    sigma = 0.05 * (ub - lb) * np.random.rand()
                    new_individual = pop[best_idx] + np.random.randn(dim) * sigma
                    new_individual = np.clip(new_individual, lb, ub)
                    pop[idx] = new_individual
                    fitness[idx] = func(new_individual)
                    n_evals += 1
                    if fitness[idx] < self.f_opt:
                        self.f_opt = fitness[idx]
                        self.x_opt = pop[idx].copy()
                # Reset diversity threshold to avoid churning
                diversity_threshold = max(diversity_threshold * 0.9, 1e-6)

            # Restart on stagnation (global)
            if evals_since_last_improvement > restart_threshold and n_evals < max_evals * 0.8:
                restarts_remaining = max_evals - n_evals
                if restarts_remaining > N_init * 0.5:
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    new_N = N
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    pop = lb + samples * (ub - lb)
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    archive = np.empty((0, dim))
                    archive_max = new_N
                    MF[:] = 0.5
                    MCR[:] = 0.8
                    memory_idx = 0
                    evals_since_last_improvement = 0
                    # Reset diversity threshold
                    current_best = np.argmin(fitness)
                    initial_diversity = np.mean(np.linalg.norm(pop - pop[current_best], axis=1))
                    diversity_threshold = 0.01 * initial_diversity

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt