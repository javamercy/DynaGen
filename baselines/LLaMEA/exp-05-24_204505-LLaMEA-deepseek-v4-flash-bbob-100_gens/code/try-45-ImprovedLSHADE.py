import numpy as np
from scipy.stats import qmc

class ImprovedLSHADE:
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

        # Population size parameters
        N_init = min(max(10 * dim, 60), max_evals // 2)
        N_min = max(4, int(dim / 4))
        N = N_init
        archive_max = N

        # Sobol initialisation
        sampler = qmc.Sobol(dim, scramble=True)
        samples = sampler.random(N)
        pop = lb + samples * (ub - lb)
        fitness = np.full(N, np.inf)
        n_evals = 0
        for i in range(N):
            fitness[i] = func(pop[i])
            n_evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive
        archive = np.empty((0, dim))

        # Success-history memory for F and CR (two sets for two mutation strategies)
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0
        # Second strategy parameters
        MF2 = np.ones(H) * 0.7
        MCR2 = np.ones(H) * 0.6
        memory_idx2 = 0

        # Ensemble of mutation strategies: 0 = current-to-pbest/1, 1 = current-to-rand/1
        strategy_probs = np.array([0.5, 0.5])
        strategy_rates = np.array([0.0, 0.0])
        strategy_counts = np.array([1, 1])

        # Stagnation detection: track median fitness over last generations
        median_history = []
        stagnation_window = max(20, int(0.02 * max_evals))
        last_restart_eval = 0

        # Local search parameters
        step_sizes = np.ones(dim) * 0.2 * (ub - lb)

        def pattern_search(best_pos, best_val, step_sizes_cur, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            step = step_sizes_cur.copy()
            used = 0
            while used < max_evals_local:
                improved = False
                # Perturbation order shuffled per dimension
                d_order = np.random.permutation(dim)
                for d in d_order:
                    if used >= max_evals_local:
                        break
                    # Positive direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] + step[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos.copy()
                        val = new_val
                        improved = True
                        step[d] *= 1.2
                        continue
                    # Negative direction
                    new_pos = pos.copy()
                    new_pos[d] = np.clip(pos[d] - step[d], lb[d], ub[d])
                    new_val = func(new_pos)
                    used += 1
                    if new_val < val:
                        pos = new_pos.copy()
                        val = new_val
                        improved = True
                        step[d] *= 1.2
                    else:
                        step[d] *= 0.8  # contract if no improvement in that dimension
                if improved and used < max_evals_local:
                    # Pattern move: vector from best to new
                    delta = pos - best_pos
                    if np.linalg.norm(delta) > 1e-12:
                        new_pos = np.clip(pos + delta, lb, ub)
                        new_val = func(new_pos)
                        used += 1
                        if new_val < val:
                            pos = new_pos.copy()
                            val = new_val
                # Keep step sizes within reasonable bounds
                step = np.clip(step, (ub - lb) * 1e-6, (ub - lb) * 0.5)
                if not improved:
                    break  # stop local search when no improvement across all dims
            return pos, val, used, step

        # Main loop
        gen = 0
        evals_no_improve = 0
        while n_evals < max_evals:
            gen += 1
            # Adaptive pbest ratio based on progress
            progress = 1.0 - (max_evals - n_evals) / max_evals
            p = 0.2 * (1.0 - progress**1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[] for _ in range(2)]
            S_CR = [[] for _ in range(2)]
            delta_f = [[] for _ in range(2)]

            # Generate offspring
            for i in range(N):
                # Select mutation strategy using roulette wheel
                strategy = np.random.choice(2, p=strategy_probs / strategy_probs.sum())
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
                # pbest index for strategy 0
                if strategy == 0:
                    pbest_size = max(1, int(p * N))
                    sorted_idx = np.argsort(fitness)
                    pbest_candidates = sorted_idx[:pbest_size]
                    pbest_idx = np.random.choice(pbest_candidates)
                    # Sample F and CR from memory0
                    mem = np.random.randint(H)
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                    # Mutation: current-to-pbest/1/archive
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                else:
                    # Strategy 1: current-to-rand/1 (no archive, no pbest)
                    mem = np.random.randint(H)
                    F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    while F <= 0:
                        F = np.clip(MF2[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                    # CR not used in arithmetic recombination, but still sample for memory update
                    CR = np.clip(MCR2[mem] + 0.1 * np.random.randn(), 0, 1)
                    base = pop[i]
                    diff = pop[r1] - union[r2]
                    # arithmetic crossover (no binomial)
                    mutant = base + F * diff
                    # apply a small j_rand to satisfy binomial condition? Actually we want arithmetic.
                    # We'll use a binomial style but with CR=1 effectively: trait = mutant
                    # For memory update, we will only record when improvement occurs.
                    # But we need a similar structure: we set CR=1 for crossover.
                    CR = 1.0  # force arithmetic recombination

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                if strategy == 0:
                    trial = np.where(np.random.rand(dim) < CR, mutant, base)
                else:  # arithmetic: always take mutant
                    trial = mutant.copy()
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: reflection with clamping
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)

                # Evaluation
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F[strategy].append(F)
                    S_CR[strategy].append(CR)
                    delta_f[strategy].append(fitness[i] - trial_f)
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

            # Update memories for each strategy
            for s in range(2):
                if len(S_F[s]) > 0:
                    sorted_order = np.argsort(delta_f[s])[::-1]
                    S_F_s = np.array(S_F[s])[sorted_order]
                    S_CR_s = np.array(S_CR[s])[sorted_order]
                    w = np.array(delta_f[s])[sorted_order] / (np.sum(delta_f[s]) + 1e-30)
                    if s == 0:
                        MF[memory_idx] = np.sum(w * S_F_s ** 2) / (np.sum(w * S_F_s) + 1e-30)
                        MCR[memory_idx] = np.sum(w * S_CR_s ** 2) / (np.sum(w * S_CR_s) + 1e-30)
                        memory_idx = (memory_idx + 1) % H
                    else:
                        MF2[memory_idx2] = np.sum(w * S_F_s ** 2) / (np.sum(w * S_F_s) + 1e-30)
                        MCR2[memory_idx2] = np.sum(w * S_CR_s ** 2) / (np.sum(w * S_CR_s) + 1e-30)
                        memory_idx2 = (memory_idx2 + 1) % H

            # Update strategy probabilities using success rates
            for s in range(2):
                if len(S_F[s]) > 0:
                    # success rate = number of successful offspring / population size
                    rate = len(S_F[s]) / N
                else:
                    rate = 0.0
                strategy_rates[s] += rate
                strategy_counts[s] += 1
            if gen % 10 == 0:
                # Compute average rates
                avg_rates = strategy_rates / strategy_counts
                # Apply softmax to get probabilities (inverse temperature 5)
                exp_rates = np.exp(5 * avg_rates)
                strategy_probs = exp_rates / exp_rates.sum()
                # Reset accumulators
                strategy_rates = np.array([0.0, 0.0])
                strategy_counts = np.array([1, 1])

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

            # Local search on best individual when budget permits
            if n_evals < max_evals * 0.9 and gen % max(10, int(0.01 * max_evals)) == 0:
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 5)
                new_pos, new_val, used, step_sizes = pattern_search(best_pos, best_val, step_sizes, max_local)
                n_evals += used
                if new_val < best_val:
                    best_val = new_val
                    best_pos = new_pos
                    if best_val < self.f_opt:
                        self.f_opt = best_val
                        self.x_opt = best_pos.copy()
                        evals_no_improve = 0
                # Replace worst if improved
                if new_val < fitness[np.argmax(fitness)]:
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = best_pos
                    fitness[worst_idx] = best_val

            # Stagnation detection using median fitness improvement
            median_fit = np.median(fitness)
            median_history.append(median_fit)
            if len(median_history) > stagnation_window:
                median_history.pop(0)
            # If median hasn't improved by at least 1e-5 relative in the window, trigger restart
            if len(median_history) >= stagnation_window and n_evals - last_restart_eval > stagnation_window:
                if median_history[-1] >= median_history[0] * (1 - 1e-5) + 1e-10:
                    # Trigger restart
                    best_idx = np.argmin(fitness)
                    best_ind = pop[best_idx].copy()
                    best_fit = fitness[best_idx]
                    remaining = max_evals - n_evals
                    new_N = min(N_init * 2, N * 2, remaining // 2)
                    new_N = max(new_N, N_min)
                    # Generate new population using Sobol around best with perturbation
                    sampler = qmc.Sobol(dim, scramble=True)
                    new_samples = sampler.random(new_N)
                    new_pop = []
                    for j in range(new_N):
                        # Mix best with random perturbation
                        perturb = (np.random.rand(dim) - 0.5) * 0.2 * (ub - lb)
                        candidate = np.clip(best_ind + perturb, lb, ub)
                        if j == 0:
                            candidate = best_ind.copy()
                        new_pop.append(candidate)
                    pop = np.array(new_pop)
                    fitness = np.full(new_N, np.inf)
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                    # Reset memories partially
                    MF[:] = np.clip(MF * 0.5 + 0.25, 0.1, 0.9)
                    MCR[:] = np.clip(MCR * 0.5 + 0.4, 0.2, 0.9)
                    MF2[:] = np.clip(MF2 * 0.5 + 0.35, 0.1, 0.9)
                    MCR2[:] = np.clip(MCR2 * 0.5 + 0.3, 0.2, 0.9)
                    memory_idx = 0
                    memory_idx2 = 0
                    archive = np.empty((0, dim))
                    archive_max = N
                    evals_no_improve = 0
                    last_restart_eval = n_evals
                    median_history = []

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt