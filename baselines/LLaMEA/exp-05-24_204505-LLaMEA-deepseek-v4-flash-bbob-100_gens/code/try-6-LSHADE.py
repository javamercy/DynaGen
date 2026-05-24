import numpy as np

class LSHADE:
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

        # Population size: initial and minimum
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = 4
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

        # Archive for inferior solutions
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory parameters
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation detection
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        restart_threshold = 0.12 * max_evals
        n_restarts = 0

        # Diversity measure parameters
        diversity_threshold = 0.05 * (ub - lb).mean()  # relative to bound range
        centroid = np.mean(pop, axis=0)

        # Success rates for strategy selection
        strategy_weights = [0.5, 0.5]  # [current-to-pbest, current-to-rand]
        n_success = [0, 0]
        n_trials = [0, 0]

        # Main loop
        while n_evals < max_evals:
            # Adaptive pbest ratio - decreases over time
            p = 0.2 - 0.1 * (n_evals / max_evals)
            p = max(p, 0.05)

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            for i in range(N):
                # Strategy selection based on success rates (adaptive)
                if np.random.rand() < strategy_weights[0]:
                    use_pbest = True
                    n_trials[0] += 1
                else:
                    use_pbest = False
                    n_trials[1] += 1

                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0.01, 1.0)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
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
                    # current-to-rand/1 (no archive, no pbest)
                    idxs2 = list(range(N))
                    idxs2.remove(i)
                    idxs2.remove(r1)
                    r2 = np.random.choice(idxs2)
                    base = pop[i]
                    diff1 = pop[r1] - base
                    diff2 = np.random.rand(dim) * (ub - lb)  # random vector
                    mutant = base + F * diff1 + 0.5 * F * diff2

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # Bound handling: mirror then random reinitialization
                trial = np.where(trial < lb, 2*lb - trial, trial)
                trial = np.where(trial > ub, 2*ub - trial, trial)
                mask = (trial < lb) | (trial > ub)
                if np.any(mask):
                    trial[mask] = lb[mask] + (ub[mask] - lb[mask]) * np.random.rand(np.sum(mask))

                # Evaluate
                trial_f = func(trial)
                n_evals += 1

                # Update global best
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_since_last_improvement = 0
                else:
                    evals_since_last_improvement += 1

                # Selection
                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Update strategy success count
                    if use_pbest:
                        n_success[0] += 1
                    else:
                        n_success[1] += 1

                    # Archive
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
                MF[memory_idx] = np.sum(w * S_F**2) / np.sum(w * S_F)
                MCR[memory_idx] = np.sum(w * S_CR**2) / np.sum(w * S_CR)
                memory_idx = (memory_idx + 1) % H

            # Update strategy weights using success rates (smoothed)
            total_success = n_success[0] + n_success[1]
            if total_success > 0:
                strategy_weights[0] = n_success[0] / total_success
                strategy_weights[1] = 1 - strategy_weights[0]
                # Keep some exploration: weights between 0.2 and 0.8
                strategy_weights = np.clip(strategy_weights, 0.2, 0.8)
                # Reset counters periodically
                if n_evals > 1000:
                    n_success = [0, 0]
                    n_trials = [0, 0]

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

            # Diversity measure: mean distance from centroid
            centroid = np.mean(pop, axis=0)
            distances = np.sqrt(np.sum((pop - centroid)**2, axis=1))
            mean_dist = np.mean(distances)

            # Restart if stagnation or low diversity
            restart_now = (evals_since_last_improvement > restart_threshold) or \
                          (mean_dist < diversity_threshold and n_evals < max_evals * 0.9)
            if restart_now and n_evals < max_evals * 0.85:
                restarts_remaining = max_evals - n_evals
                if restarts_remaining > N_init * 0.3:
                    # Keep best individual
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    # Reinitialize 70% of population with Latin hypercube, 30% as mutations of best
                    new_N = N
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    # Place best at index 0
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    # Fill rest: first half from Latin hypercube, second half perturbed best
                    perturbed = best_ind + 0.1 * (ub - lb) * np.random.randn(new_N - 1, dim)
                    perturbed = np.clip(perturbed, lb, ub)
                    pop[1:] = perturbed
                    # Evaluate new individuals (except best idx)
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    # Reset archive and memory
                    archive = np.empty((0, dim))
                    archive_max = new_N
                    MF[:] = 0.5
                    MCR[:] = 0.5
                    memory_idx = 0
                    evals_since_last_improvement = 0
                    n_restarts += 1

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt