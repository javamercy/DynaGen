import numpy as np

class Enhanced_SHADE_AdaptiveLS:
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
        domain_range = ub - lb

        # Population sizing – linear reduction from sqrt quadratic
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)  # generous estimate

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initialisation
        n_sob = pop_size
        perm = np.tile(np.arange(1, n_sob + 1), (dim, 1)).T
        for j in range(dim):
            perm[:, j] = np.random.permutation(perm[:, j])
        sobol = (perm - 0.5) / n_sob
        pop = lb + sobol * domain_range

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive
        archive = []
        archive_size = pop_size  # same size as population

        # Stagnation and local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq = 8                # initial frequency (generations between LS calls)
        min_ls_freq = 4
        max_ls_freq = 30
        ls_step_size = 0.1 * domain_range.mean()
        ls_success_rate = 0.0      # moving average of LS success

        # Generation counter for linear reduction
        max_gen_actual = max_gen
        gen_ratio = 0.0

        while evals < self.budget:
            gen += 1
            gen_ratio = gen / max_gen_actual

            # --- Linear population reduction ---
            new_pop_size = max(N_min, int(N_init + (N_min - N_init) * gen_ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # --- pbest rate: linearly from 0.1 to 0.5 ---
            p = 0.1 + 0.4 * gen_ratio
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            # --- Main SHADE loop over population ---
            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # r1, r2 from union of pop and archive (avoid i)
                union = list(range(pop_size)) + list(range(len(archive)))
                try:
                    union.remove(i)
                except ValueError:
                    pass
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_ind(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1 = get_ind(r1)
                    x_r2 = get_ind(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory with small noise
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (70%) or exponential (30%)
                if np.random.rand() < 0.7:
                    j_rand = np.random.randint(dim)
                    mask = np.random.rand(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, mutant, pop[i])
                else:
                    start = np.random.randint(dim)
                    L = 0
                    while L < dim and np.random.rand() < CR:
                        L += 1
                    indices = (np.arange(dim) + start) % dim
                    mask = np.zeros(dim, dtype=bool)
                    mask[indices[:L]] = True
                    trial = np.where(mask, mutant, pop[i])

                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                # Selection
                if f_trial <= fitness[i]:
                    # Archive replacement
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_remove = np.random.randint(len(archive))
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))
                    n_success += 1

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # --- Update SHADE memory ---
            if len(success_F) > 0:
                w = np.array(weight)
                w_sum = np.sum(w)
                if w_sum > 0:
                    w = w / w_sum
                    F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                    CR_mean = np.sum(w * np.array(success_CR))
                    mem_F[mem_idx] = F_lehmer
                    mem_CR[mem_idx] = CR_mean
                    mem_idx = (mem_idx + 1) % mem_size

            # --- Success rate for LS trigger ---
            success_rate = n_success / max(1, pop_size)
            # moving average of last 10 generations
            if not hasattr(self, '_success_rates'):
                self._success_rates = []
            self._success_rates.append(success_rate)
            if len(self._success_rates) > 10:
                self._success_rates.pop(0)

            # --- Adaptive random local search (step-size controlled) ---
            budget_left = self.budget - evals
            low_success = (len(self._success_rates) >= 5 and np.mean(self._success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)

            # Trigger conditions: generation frequency + low success + diversity
            if (gen % ls_freq == 0 and budget_left > 30 and diversity and low_success):
                # Perform local search: adaptive random walk
                x = self.x_opt.copy()
                f = self.f_opt
                step = ls_step_size
                num_iter = max(2, min(10, int(0.03 * budget_left)))  # keep low budget
                improved = False
                for _ in range(num_iter):
                    if evals >= self.budget:
                        break
                    # random direction
                    d = np.random.normal(0, 1, dim)
                    d = d / (np.linalg.norm(d) + 1e-30)
                    x_try = np.clip(x + step * d, lb, ub)
                    f_try = func(x_try)
                    evals += 1
                    if f_try < f:
                        x = x_try
                        f = f_try
                        step *= 1.2      # increase step on success
                        improved = True
                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = x.copy()
                    else:
                        step *= 0.85     # decrease on failure
                    step = max(step, 1e-8 * domain_range.mean())  # keep minimal
                # Update global step size for next LS call
                ls_step_size = step

                # Inject best point into population
                if improved:
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # also inject a perturbed copy for diversity
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.01 * np.random.randn(dim) * domain_range
                        perturbed = np.clip(perturbed, lb, ub)
                        f_pert = func(perturbed)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturbed
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturbed.copy()

                # Adapt LS frequency based on success
                if improved:
                    ls_freq = max(min_ls_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_ls_freq, int(ls_freq * 1.1))

            # --- Stagnation detection and restart ---
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen_actual)):
                n_restart = max(1, int(0.6 * pop_size))
                # Generate half near best with shrinking scale, half Latin hypercube
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen_ratio) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + lhs[idx] * domain_range
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset SHADE memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0
                # Reset local search state
                ls_freq = max(ls_freq, min_ls_freq)
                ls_step_size = 0.1 * domain_range.mean()
                # Reset success rates
                self._success_rates = []

        return self.f_opt, self.x_opt