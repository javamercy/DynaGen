import numpy as np

class Enhanced_SHADE_Pro:
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

        # --- Latin Hypercube initialisation ---
        N_init = max(10, int(20 + 10 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        pop = np.empty((pop_size, dim))
        for j in range(dim):
            perm = np.random.permutation(pop_size) + 0.5
            pop[:, j] = lb[j] + (perm / pop_size) * (ub[j] - lb[j])

        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # --- SHADE memory ---
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        archive = []
        archive_size = pop_size

        # Stagnation / restart
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq_init = max(10, int(0.06 * max_gen))
        ls_freq = ls_freq_init
        ls_max_iter = max(3, min(12, int(0.04 * self.budget / (dim + 1))))
        success_rates = []
        ls_success = True

        # Main loop
        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / (1.5 * max_gen)))
            if new_pop_size < pop_size:
                idx = np.argsort(fitness)[:new_pop_size]
                pop = pop[idx].copy()
                fitness = fitness[idx]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate
            p = 0.2 * (gen / max_gen) ** 0.5 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            # --- SHADE mutation & crossover ---
            for i in range(pop_size):
                if evals >= self.budget:
                    break

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

                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover (binomial + exponential mix)
                trial = np.empty(dim)
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
                w = w / (np.sum(w) + 1e-30)
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

            # ---------- Local search (SPSA with pattern step acceptance) ----------
            budget_left = self.budget - evals
            ls_trigger = (gen % ls_freq == 0 and budget_left > 20 and
                          np.std(fitness) < 0.5 and
                          (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15))

            if ls_trigger and ls_success:
                # SPSA local search
                x = self.x_opt.copy()
                f = self.f_opt
                step0 = 0.1 * (ub - lb).mean()
                step = step0
                lr = step0 * 0.5
                for it in range(ls_max_iter):
                    if evals + 2 >= self.budget:
                        break
                    delta = np.random.choice([-1, 1], size=dim)
                    c = step * 0.01
                    x_plus = np.clip(x + c * delta, lb, ub)
                    x_minus = np.clip(x - c * delta, lb, ub)
                    f_plus = func(x_plus)
                    f_minus = func(x_minus)
                    evals += 2

                    # pattern step – accept best of ±
                    improvement = False
                    if f_plus < f:
                        x = x_plus.copy()
                        f = f_plus
                        improvement = True
                    if f_minus < f:
                        x = x_minus.copy()
                        f = f_minus
                        improvement = True

                    # compute SPSA gradient
                    g = (f_plus - f_minus) / (2 * c * delta)
                    g = np.clip(g, -1e10, 1e10)
                    g_norm = np.linalg.norm(g)
                    if g_norm < 1e-12:
                        break

                    # gradient step
                    x_new = np.clip(x - lr * g / (g_norm + 1e-30), lb, ub)
                    f_new = func(x_new)
                    evals += 1

                    if f_new < f:
                        x = x_new.copy()
                        f = f_new
                        step *= 1.2
                        improvement = True
                    else:
                        step *= 0.9

                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

                    if not improvement:
                        lr *= 0.9
                        if lr < 1e-8:
                            break
                    else:
                        lr *= 1.05

                # Inject best point into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    if evals < self.budget:
                        perturbed = self.x_opt + 0.02 * np.random.randn(dim) * (ub - lb)
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

                ls_success = (f < self.f_opt - 1e-12)  # remember if LS improved
            else:
                ls_success = True

            # ---------- Stagnation restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(1, int(0.5 * pop_size))
                # Cauchy restart around best and random LHS
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        cauchy = np.random.standard_cauchy(dim) * scale
                        pop[idx] = self.x_opt + cauchy
                    else:
                        perm = np.random.permutation(n_restart) + 0.5
                        pop[idx] = lb + (perm[idx] / n_restart) * (ub - lb)
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
                ls_freq = min(max_gen // 4, ls_freq + 2)
                ls_success = True

        return self.f_opt, self.x_opt