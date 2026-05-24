import numpy as np

class Refined_SHADE_ES:
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

        # Latin hypercube initialization
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

        # Stagnation and local search control
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.05 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))
        success_rates = []

        # (1+1)-ES state for local search (reused)
        es_sigma = None
        es_path = None

        while evals < self.budget:
            gen += 1

            # Population reduction (convex schedule)
            ratio = 1.0 - (evals / self.budget)**0.5
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate
            p = 0.1 + 0.4 * (evals / self.budget)**1.2
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

                # Sample F, CR
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial or exponential
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

            # ---------- Adaptive local search (1+1)-ES ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) < 5 or np.mean(success_rates[-5:]) < 0.15)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 30 and diversity and low_success):

                # Initialize or continue (1+1)-ES parameters
                if es_sigma is None or (gen % (ls_freq * 2) == 0):
                    es_mean = self.x_opt.copy()
                    es_sigma = 0.2 * domain_range.mean()
                    es_path = np.zeros(dim)
                else:
                    es_mean = self.x_opt.copy()

                # Number of ES iterations
                es_iters = min(10, max(2, int(0.02 * budget_left / dim)))
                for it in range(es_iters):
                    if evals + 1 >= self.budget:
                        break
                    # Sample offspring
                    z = np.random.randn(dim)
                    candidate = es_mean + es_sigma * z
                    candidate = np.clip(candidate, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1

                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = candidate.copy()

                    # Selection and path update
                    if f_candidate < fitness[0]:  # compare to best in population
                        diff = candidate - es_mean
                        # Update evolution path
                        es_path = (1.0 - 1.0 / dim) * es_path + np.sqrt(1.0 / dim) * diff / (es_sigma + 1e-30)
                        es_mean = candidate
                    else:
                        es_path = (1.0 - 1.0 / dim) * es_path - np.sqrt(1.0 / dim) * z

                    # Step size adaptation (CSA)
                    path_len = np.linalg.norm(es_path)
                    expected_len = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim))
                    es_sigma *= np.exp((path_len - expected_len) / (expected_len * dim**0.5))

                # Inject best found into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
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

                # Adapt LS frequency
                if self.f_opt < best_old - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.08 * max_gen)):
                # Full restart with mixed population
                n_restart = pop_size
                # 1/3 near best, 2/3 LHS
                n_near = n_restart // 3
                n_lhs = n_restart - n_near
                # LHS part
                perm2 = np.tile(np.arange(1, n_lhs + 1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs = (perm2 - 0.5) / n_lhs
                new_pop = np.empty((n_restart, dim))
                for idx in range(n_near):
                    scale = 0.1 * domain_range * (1 - evals / self.budget) + 0.01
                    new_pop[idx] = self.x_opt + np.random.randn(dim) * scale
                for idx in range(n_lhs):
                    new_pop[n_near + idx] = lb + lhs[idx] * domain_range
                new_pop = np.clip(new_pop, lb, ub)
                new_fitness = np.empty(n_restart)
                for idx in range(n_restart):
                    if evals < self.budget:
                        new_fitness[idx] = func(new_pop[idx])
                        evals += 1
                        if new_fitness[idx] < self.f_opt:
                            self.f_opt = new_fitness[idx]
                            self.x_opt = new_pop[idx].copy()
                pop = new_pop
                fitness = new_fitness
                pop_size = n_restart
                # Reset SHADE memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                es_sigma = None
                es_path = None
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt