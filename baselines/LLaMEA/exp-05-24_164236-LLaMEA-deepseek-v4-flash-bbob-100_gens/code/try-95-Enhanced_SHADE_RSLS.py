import numpy as np

class Enhanced_SHADE_RSLS:
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
        max_gen = int(self.budget / pop_size * 2.5)

        # SHADE memory
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initialisation
        pop = np.random.uniform(lb, ub, (pop_size, dim))
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
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))
        success_rates = []

        while evals < self.budget:
            gen += 1
            ratio = max(0, 1 - (gen / max_gen) ** 1.2)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)
            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

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

                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                mask = np.random.rand(dim) < CR
                mask[j_rand] = True
                trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
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

            success_rate = n_success / max(1, pop_size)
            success_rates.append(success_rate)
            if len(success_rates) > 10:
                success_rates.pop(0)

            # ---------- Random Subspace Local Search (RSLS) ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 30 and diversity and low_success):
                x = self.x_opt.copy()
                f = self.f_opt
                # subsample dimensions (subspace size = max(1, dim//3))
                sub_d = max(1, dim // 3)
                ls_iters = max(2, min(10, int(0.03 * budget_left / sub_d)))
                for it in range(ls_iters):
                    if evals + 3 >= self.budget:
                        break
                    # Random direction in subspace
                    dir_dims = np.random.choice(dim, sub_d, replace=False)
                    direction = np.zeros(dim)
                    direction[dir_dims] = np.random.randn(sub_d)
                    direction = direction / (np.linalg.norm(direction) + 1e-12)
                    # Golden step line search (requires ~2 evaluations per iteration)
                    step = 0.5 * domain_range.mean()
                    factor = 0.5
                    best_f = f
                    best_x = x.copy()
                    for _ in range(4):  # limited iterations
                        left = np.clip(x - step * direction, lb, ub)
                        right = np.clip(x + step * direction, lb, ub)
                        f_left = func(left)
                        f_right = func(right)
                        evals += 2
                        if f_left < best_f:
                            best_f = f_left
                            best_x = left.copy()
                        if f_right < best_f:
                            best_f = f_right
                            best_x = right.copy()
                        step *= factor
                    if best_f < f:
                        x = best_x.copy()
                        f = best_f
                        if f < self.f_opt:
                            self.f_opt = f
                            self.x_opt = x.copy()
                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                # Adapt frequency
                if f < self.f_opt - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = np.random.uniform(lb, ub, dim)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt