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

        # Population size parameters
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

        # Success-history memory
        H = 20
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.5
        memory_idx = 0

        # Stagnation and diversity detection
        best_since_last_restart = self.f_opt
        evals_since_last_improvement = 0
        no_improvement_streak = 0
        restart_threshold = 0.12 * max_evals
        diversity_threshold = 0.02 * (ub - lb).mean()

        # Local search step size
        ls_step = 0.1 * (ub - lb).mean()

        # Main loop
        while n_evals < max_evals:
            p = 0.2 - 0.1 * (n_evals / max_evals)  # pbest fraction

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # r2 from union of population and archive
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

                # Sample F and CR from memory
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.standard_normal(), 0, 1)

                # Mutation (current-to-pbest/1 with archive)
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2

                # Crossover (binomial)
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # Bound handling: mirror then random reinit if out
                trial = np.where(trial < lb, 2 * lb - trial, trial)
                trial = np.where(trial > ub, 2 * ub - trial, trial)
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

                    # Add replaced individual to archive (only if not too close to best)
                    if np.linalg.norm(pop[i] - self.x_opt) > diversity_threshold:
                        archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)

                MF[memory_idx] = np.sum(w * S_F**2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR**2) / (np.sum(w * S_CR) + 1e-30)
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

            # Restart based on stagnation or low diversity
            diversity = np.mean([np.linalg.norm(p - self.x_opt) for p in pop])
            need_restart = (evals_since_last_improvement > restart_threshold or
                            (diversity < diversity_threshold and evals_since_last_improvement > 0.1 * restart_threshold))
            if need_restart and n_evals < max_evals * 0.8:
                restarts_remaining = max_evals - n_evals
                if restarts_remaining > N_init * 0.5:
                    # Keep best and top 10% individuals (up to 3)
                    best_idx = np.argmin(fitness)
                    keep_idx = [best_idx]
                    # Keep a couple of additional diverse good individuals
                    sorted_base = np.argsort(fitness)[:max(2, int(0.1*N))]
                    for idx in sorted_base:
                        if idx != best_idx and len(keep_idx) < 3:
                            keep_idx.append(idx)
                    kept_pop = pop[keep_idx].copy()
                    kept_fit = fitness[keep_idx].copy()
                    # Reinitialize rest
                    new_N = N
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[:len(keep_idx)] = kept_pop
                    fitness[:len(keep_idx)] = kept_fit
                    # Evaluate new individuals
                    for j in range(len(keep_idx), new_N):
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
                    diversity = np.mean([np.linalg.norm(p - self.x_opt) for p in pop])
                    diversity_threshold = max(0.02 * (ub - lb).mean(), diversity * 0.5)

            if n_evals >= max_evals:
                break

        # Post-optimization: adaptive local search around best solution
        if n_evals < max_evals:
            remaining = max_evals - n_evals
            ls_evals = min(remaining, int(0.05 * max_evals))
            best = self.x_opt.copy()
            best_f = self.f_opt
            step = ls_step
            for _ in range(ls_evals):
                candidate = best + step * np.random.randn(dim)
                candidate = np.clip(candidate, lb, ub)
                candidate_f = func(candidate)
                n_evals += 1
                if candidate_f < best_f:
                    best_f = candidate_f
                    best = candidate.copy()
                    step *= 1.1
                else:
                    step *= 0.9
                if step < 1e-8:
                    step = ls_step
                if best_f < self.f_opt:
                    self.f_opt = best_f
                    self.x_opt = best.copy()
                if n_evals >= max_evals:
                    break

        return self.f_opt, self.x_opt