import numpy as np

class SHADE_HJ_ARS:
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

        # Population sizing – larger initial population for better coverage
        N_init = max(20, int(18 * np.sqrt(dim)))
        N_min = 6
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 10
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

        archive = []
        archive_size = pop_size

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(6, int(0.04 * max_gen))
        min_freq = 3
        max_freq = max(20, int(0.15 * max_gen))

        # For adaptive restart
        diversity_window = []

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction (slower decay)
            ratio = max(0, 1 - (gen / max_gen) ** 1.5)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: grows with generation
            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

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

                # Sample F, CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 0.9)
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

            # ---------- Hooke-Jeeves Local Search ----------
            budget_left = self.budget - evals
            diversity = np.std(fitness) if pop_size > 1 else 0.0
            diversity_window.append(diversity)
            if len(diversity_window) > 20:
                diversity_window.pop(0)
            low_diversity = (len(diversity_window) >= 5 and np.mean(diversity_window[-5:]) < 0.3 * np.mean(domain_range))
            success_rate = n_success / max(1, pop_size)
            low_success = success_rate < 0.15

            if (gen % ls_freq == 0 and budget_left > 30 and low_diversity and low_success):
                # Use Hooke-Jeeves pattern search on the current best point
                x = self.x_opt.copy()
                f = self.f_opt
                step = 0.2 * domain_range
                step_min = 1e-4 * np.mean(domain_range)
                max_hj_iters = max(2, min(8, int(0.02 * budget_left / dim)))

                for _ in range(max_hj_iters):
                    if evals + 2 * dim + 2 >= self.budget:
                        break
                    improved = False
                    # Exploratory moves (coordinate-wise)
                    for d in range(dim):
                        if step[d] < step_min:
                            continue
                        # Positive direction
                        x_cand = x.copy()
                        x_cand[d] = np.clip(x[d] + step[d], lb[d], ub[d])
                        f_cand = func(x_cand)
                        evals += 1
                        if f_cand < f:
                            x = x_cand
                            f = f_cand
                            improved = True
                            continue
                        # Negative direction
                        x_cand[d] = np.clip(x[d] - step[d], lb[d], ub[d])
                        f_cand = func(x_cand)
                        evals += 1
                        if f_cand < f:
                            x = x_cand
                            f = f_cand
                            improved = True
                    # Pattern move (if improvement found)
                    if improved:
                        # Attempt a big step in the direction of improvement
                        direction = x - self.x_opt
                        if np.linalg.norm(direction) > 1e-12:
                            x_cand = np.clip(x + direction, lb, ub)
                            f_cand = func(x_cand)
                            evals += 1
                            if f_cand < f:
                                x = x_cand
                                f = f_cand
                            else:
                                step = step * 0.5
                        else:
                            step = step * 0.5
                    else:
                        step = step * 0.5
                    # Update best if improved
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                # Adapt local search frequency
                if f < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.15))

            # ---------- Stagnation and adaptive restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if stagnation or extremely low diversity
            need_restart = (stagnation_counter > max(15, int(0.1 * max_gen)) or
                            (len(diversity_window) > 10 and np.mean(diversity_window[-10:]) < 0.1 * np.mean(domain_range)))
            if need_restart:
                n_restart = max(1, int(0.7 * pop_size))
                # Reinitialize with two strategies: near best for exploitation, LHS for exploration
                perm2 = np.tile(np.arange(1, n_restart + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
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
                ls_freq = max(ls_freq, min_freq)
                diversity_window.clear()

        return self.f_opt, self.x_opt