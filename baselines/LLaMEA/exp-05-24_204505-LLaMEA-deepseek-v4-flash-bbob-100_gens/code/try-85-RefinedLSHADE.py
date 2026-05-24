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

        # population size
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # initialization (Latin hypercube)
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

        # archive
        archive = np.empty((0, dim))
        archive_max = N

        # success-history for F, CR and mutation strategy selection
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # multi-strategy probabilities (current-to-pbest, current-to-rand, rand/1)
        strat_probs = np.array([0.5, 0.3, 0.2])
        strat_success = np.array([0.0, 0.0, 0.0])
        strat_trials = np.array([0, 0, 0])

        # stagnation & diversity detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.12 * max_evals

        # local search
        last_local_search = 0
        local_search_interval = max(40, int(0.025 * max_evals))

        # helper: diversity measure
        def diversity(pop):
            if len(pop) <= 1:
                return 0.0
            mean = np.mean(pop, axis=0)
            return np.mean(np.sqrt(np.sum((pop - mean)**2, axis=1)))

        # hybrid local search (coordinate + random directions)
        def hybrid_local_search(best_pos, best_val, step_size, max_evals_local):
            pos = best_pos.copy()
            val = best_val
            step = step_size * (ub - lb)
            used = 0
            # coordinate search
            coord_dirs = np.eye(dim)
            # permutation of directions
            order = np.random.permutation(dim)
            for d in order:
                if used >= max_evals_local:
                    break
                # positive direction
                new_pos = np.clip(pos + step * coord_dirs[d], lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    step *= 1.1  # expand
                    step = np.clip(step, 1e-12, 0.5*(ub-lb))
                    continue
                # negative direction
                new_pos = np.clip(pos - step * coord_dirs[d], lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    step *= 1.1
                else:
                    step *= 0.9  # contract
            # random directions (if budget remains)
            while used < max_evals_local:
                rnd_dir = np.random.randn(dim)
                rnd_dir /= np.linalg.norm(rnd_dir) + 1e-20
                new_pos = np.clip(pos + step * rnd_dir, lb, ub)
                new_val = func(new_pos)
                used += 1
                if new_val < val:
                    pos = new_pos
                    val = new_val
                    step *= 1.1
                else:
                    step *= 0.9
                if step.max() < 1e-12:
                    break
            return pos, val, used

        # main loop
        while n_evals < max_evals:
            p = 0.2 * (1 - (n_evals / max_evals)**1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # update strategy probabilities adaptively
            total_trials = max(1, strat_trials.sum())
            if total_trials > 1:
                # avoid division by zero
                for k in range(3):
                    if strat_trials[k] > 0:
                        strat_probs[k] = 0.9 * strat_probs[k] + 0.1 * (strat_success[k] / strat_trials[k])
                strat_probs /= strat_probs.sum()

            for i in range(N):
                # select mutation strategy
                strat_choice = np.random.choice(3, p=strat_probs)
                strat_trials[strat_choice] += 1

                # choose indices
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union pop+archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)

                base = pop[i]
                if strat_choice == 0:  # current-to-pbest/1/archive
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                elif strat_choice == 1:  # current-to-rand/1 (rotation invariant)
                    mutant = base + F * (pop[r1] - base) + F * (pop[np.random.choice(idxs)] - union[r2])
                else:  # rand/1
                    r3 = np.random.choice(idxs)
                    mutant = pop[r1] + F * (pop[r2] - pop[r3])

                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]

                # boundary handling (reflect)
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
                    strat_success[strat_choice] += 1
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        # remove oldest (FIFO)
                        archive = archive[1:]
                else:
                    # also add mutant if it's close? skip for simplicity
                    pass

            # update population
            pop = new_pop
            fitness = new_fitness

            # update memory
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # population size reduction
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    archive = archive[:archive_max]  # keep oldest
                N = N_new

            # local search with adaptive interval
            if (n_evals - last_local_search >= local_search_interval) and (n_evals < max_evals * 0.95):
                last_local_search = n_evals
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                step = 0.15 * (1 - n_evals / max_evals) + 0.01
                max_local = min(dim * 4, max_evals - n_evals - 5)
                new_pos, new_val, used = hybrid_local_search(best_pos, best_val, step, max_local)
                n_evals += used
                if new_val < best_val:
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = new_pos
                    fitness[worst_idx] = new_val

            # diversity-based restart (in addition to stagnation)
            div = diversity(pop)
            div_threshold = 0.2 * np.mean(ub - lb) if np.mean(ub - lb) > 1e-10 else 0.1
            restart_diversity = (div < div_threshold * 0.1) and (n_evals > 0.2 * max_evals)
            restart_stagnation = (evals_no_improve > restart_threshold) and (n_evals < max_evals * 0.8)

            if restart_diversity or restart_stagnation:
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
                # reset memory
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                strat_probs = np.array([0.5, 0.3, 0.2])
                strat_success = np.array([0.0, 0.0, 0.0])
                strat_trials = np.array([0, 0, 0])
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt