import numpy as np

class Refined_SHADE_PS:
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

        # Population sizing
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen_est = int(self.budget / pop_size * 2.5)  # rough estimate, used only for scheduling

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
        archive_size = pop_size

        # Stagnation control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search control
        ls_freq = max(8, int(0.05 * max_gen_est))
        min_ls_freq = 4
        max_ls_freq = max(30, int(0.2 * max_gen_est))
        success_rates = []

        # Local search memory: best point before LS
        ls_best_x = self.x_opt.copy()
        ls_best_f = self.f_opt

        # For pattern search: store direction pool once
        pattern_dirs = np.random.randn(20, dim)  # fixed set of random directions
        pattern_dirs /= np.linalg.norm(pattern_dirs, axis=1, keepdims=True) + 1e-12

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction
            ratio_evals = max(0, 1 - (evals / self.budget) ** 1.2)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio_evals))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: decreases linearly from 0.5 to 0.1
            p = 0.5 - 0.4 * (evals / self.budget)
            p = max(0.1, min(0.5, p))

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # r1, r2 from union of pop and archive
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

                # Sample F, CR from memory with adaptive noise
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

            # Update SHADE memory
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

            # Success rate for LS trigger
            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Local Search (Stochastic Pattern Search) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.2)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            # Trigger when generation matches frequency and budget allows enough evals
            if (gen % ls_freq == 0 and budget_left > 20 and diversity and low_success):
                # Use a pattern search around the current best
                x0 = self.x_opt.copy()
                f0 = self.f_opt
                # Number of pattern iterations (each uses ~2 evals)
                max_pattern_iters = max(2, min(8, int(0.02 * budget_left / dim)))
                step_size = 0.1 * domain_range.mean() * (1 - evals / self.budget) + 1e-3
                step_size = max(1e-4, step_size)
                improved = False

                for _ in range(max_pattern_iters):
                    if evals + 2 > self.budget:
                        break
                    # Pick a random direction from the fixed pool
                    d = pattern_dirs[np.random.randint(len(pattern_dirs))]
                    # Positive step
                    x_plus = np.clip(x0 + step_size * d, lb, ub)
                    f_plus = func(x_plus)
                    evals += 1
                    # Negative step
                    x_minus = np.clip(x0 - step_size * d, lb, ub)
                    f_minus = func(x_minus)
                    evals += 1
                    # Check which direction gave improvement
                    if f_plus < f0 and f_plus <= f_minus:
                        x0 = x_plus
                        f0 = f_plus
                        improved = True
                    elif f_minus < f0:
                        x0 = x_minus
                        f0 = f_minus
                        improved = True
                    else:
                        # Reduce step size
                        step_size *= 0.5
                        if step_size < 1e-6:
                            break

                    if f0 < self.f_opt:
                        self.f_opt = f0
                        self.x_opt = x0.copy()
                        # If improved, reset stagnation but not here

                # Inject best found into population (if improved)
                if improved and f0 < fitness.max():
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = x0.copy()
                    fitness[worst_idx] = f0
                    # Also add a slightly perturbed copy for diversity
                    if evals < self.budget:
                        perturbed = np.clip(x0 + 0.01 * np.random.randn(dim) * domain_range, lb, ub)
                        f_pert = func(perturbed)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturbed
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturbed.copy()

                # Adapt local search frequency based on success
                if improved:
                    ls_freq = max(min_ls_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_ls_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen_est)):
                n_restart = max(1, int(0.4 * pop_size))  # reduced from 0.6 to save evals
                # Generate new points: half near best, half Latin hypercube
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - evals / self.budget) + 0.01
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
                ls_freq = max(ls_freq, min_ls_freq)

        return self.f_opt, self.x_opt