import numpy as np

class Enhanced_SHADE_Plus:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        dim = self.dim
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)

        # Population size (L-SHADE style)
        N_init = max(8, int(15 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # LHS initial population
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive for L-SHADE
        archive = []
        archive_size = pop_size

        # Stagnation detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters (lightweight random direction search)
        ls_freq = max(8, int(0.1 * max_gen))
        ls_step = 0.3 * (ub - lb)  # initial step size
        ls_budget_frac = 0.1

        # Strategy success counters (ensemble: pbest/1 and rand/1)
        success_count = [0, 0]
        attempt_count = [0, 0]
        # mutation strategy selection probabilities
        prob_mut = np.array([0.5, 0.5])

        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate (time-dependent)
            p = 0.2 * (gen / max_gen) ** 2 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Strategy selection based on adaptive probabilities
                strat = np.random.choice(2, p=prob_mut)
                attempt_count[strat] += 1

                if strat == 0:  # current-to-pbest/1
                    pbest_size = max(2, int(p * pop_size))
                    best_indices = np.argsort(fitness)[:pbest_size]
                    pbest_idx = np.random.choice(best_indices)
                    x_pbest = pop[pbest_idx]

                    union = list(range(pop_size)) + list(range(len(archive)))
                    union.remove(i)
                    if len(union) >= 2:
                        r1, r2 = np.random.choice(union, 2, replace=False)
                        def get_ind(idx):
                            if idx < pop_size:
                                return pop[idx]
                            else:
                                return archive[idx - pop_size]
                        x_r1 = get_ind(r1)
                        x_r2 = get_ind(r2)
                    else:
                        indices = [j for j in range(pop_size) if j != i]
                        r1, r2 = np.random.choice(indices, 2, replace=False)
                        x_r1, x_r2 = pop[r1], pop[r2]

                    # Sample F, CR from memory
                    r = np.random.randint(mem_size)
                    F = mem_F[r] + 0.1 * np.random.randn()
                    CR = mem_CR[r] + 0.1 * np.random.randn()
                    F = np.clip(F, 0.1, 1.0)
                    CR = np.clip(CR, 0.0, 1.0)

                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                else:  # current-to-rand/1 (rotation invariant)
                    union = list(range(pop_size))
                    union.remove(i)
                    if len(union) >= 2:
                        r1, r2 = np.random.choice(union, 2, replace=False)
                        x_r1, x_r2 = pop[r1], pop[r2]
                    else:
                        indices = [j for j in range(pop_size) if j != i]
                        r1, r2 = np.random.choice(indices, 2, replace=False)
                        x_r1, x_r2 = pop[r1], pop[r2]

                    # Use a fixed small F and CR (no memory needed)
                    F = 0.6 + 0.2 * np.random.rand()
                    CR = 0.9 * np.random.rand() + 0.1
                    # current-to-rand/1: without crossover, just mutation
                    mutant = pop[i] + np.random.rand() * (x_r1 - pop[i]) + F * (pop[i] - x_r2)

                # Crossover (binomial for strategy 0; for strategy 1 we already did vector mutation -> use binomial or no crossover)
                if strat == 0:
                    j_rand = np.random.randint(dim)
                    mask = np.random.rand(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, mutant, pop[i])
                else:
                    # For rand/1 we use exponential crossover with CR=1?
                    trial = mutant  # fully accept mutation, but clip
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive management
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
                        archive[idx_remove] = pop[i].copy()

                    success_count[strat] += 1
                    if strat == 0:
                        success_F.append(F)
                        success_CR.append(CR)
                        imp = fitness[i] - f_trial
                        weight.append(max(imp, 1e-12))

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update SHADE memory for strategy 0
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Update strategy selection probabilities (adaptive)
            total_success = np.array(success_count) + 1e-10
            total_attempt = np.array(attempt_count) + 1e-10
            success_rate = total_success / total_attempt
            # Use softmax-like update
            prob_mut = success_rate / (success_rate.sum() + 1e-30)
            prob_mut[prob_mut < 0.2] = 0.2  # prevent extinction
            prob_mut = prob_mut / prob_mut.sum()

            # ---------- Lightweight random-direction local search ----------
            if (gen % ls_freq == 0 and
                (self.budget - evals) > dim * 3 + 10 and
                np.std(fitness) < 1.5 and
                stagnation_counter > 2):
                # Use 3 random directions per call
                for _ in range(dim):
                    if evals >= self.budget:
                        break
                    d = np.random.randn(dim)
                    d = d / (np.linalg.norm(d) + 1e-30)
                    step = np.random.uniform(0.2, 1.0) * ls_step
                    # Try positive direction
                    trial = np.clip(self.x_opt + step * d, lb, ub)
                    ft = func(trial)
                    evals += 1
                    if ft < self.f_opt:
                        self.f_opt = ft
                        self.x_opt = trial.copy()
                    # If no improvement, try negative
                    else:
                        trial_neg = np.clip(self.x_opt - step * d, lb, ub)
                        ft_neg = func(trial_neg)
                        evals += 1
                        if ft_neg < self.f_opt:
                            self.f_opt = ft_neg
                            self.x_opt = trial_neg.copy()
                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.06 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                # Keep best solution
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Generate new population around best and uniformly
                for idx in range(pop_size):
                    if idx < n_restart // 2:
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        # LHS-like uniform
                        x = np.random.rand(dim)
                        for j in range(dim):
                            x[j] = (np.argsort(np.random.rand(pop_size))[idx] + 0.5) / pop_size
                        pop[idx] = lb + x * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0
                # Slightly reduce local search frequency to avoid premature convergence
                ls_freq = min(max_gen // 3, ls_freq + 1)

        return self.f_opt, self.x_opt