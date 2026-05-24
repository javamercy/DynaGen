import numpy as np

class Refined_SHADE_NM:
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
        N_init = max(10, int(20 * np.sqrt(dim)))
        N_min = 5
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 3.0)

        # SHADE memory
        mem_size = 8
        mem_F = np.full(mem_size, 0.6)
        mem_CR = np.full(mem_size, 0.9)
        mem_idx = 0

        # Mutation strategy probabilities (initially 0.5 for each)
        prob_cur2pbest = 0.5
        success_cur = []
        success_rand = []

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
        archive_size = int(2.5 * pop_size)

        # Local search control
        best_old = self.f_opt
        stagnation_counter = 0
        ls_freq = max(10, int(0.05 * max_gen))
        min_freq = 5
        max_freq = max(30, int(0.3 * max_gen))

        # Success rates for LS trigger
        success_rates = []

        # Nelder-Mead parameters
        nm_rho = 1.0  # reflection
        nm_chi = 2.0  # expansion
        nm_gamma = 0.5  # contraction
        nm_sigma = 0.5  # shrink

        # Keep track of generation
        gen = 0

        while evals < self.budget:
            gen += 1

            # Nonlinear population reduction based on remaining budget
            remaining_budget = self.budget - evals
            ratio = remaining_budget / self.budget
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate: increases with generation
            p = 0.1 + 0.4 * (gen / max_gen) ** 1.2
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            n_success = 0

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Select mutation strategy adaptively
                if np.random.rand() < prob_cur2pbest:
                    use_cur2pbest = True
                else:
                    use_cur2pbest = False

                # pbest selection (for cur2pbest)
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

                if use_cur2pbest:
                    # current-to-pbest/1
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:
                    # rand/1
                    mutant = x_r1 + F * (x_r2 - pop[np.random.randint(pop_size)])  # actually rand/1 uses random base
                    # fix: rand/1: base = x_r2? better: choose third random
                    # proper: base = pop[random], diff = x_pbest - pop[random]? No, standard rand/1: v = x_r0 + F*(x_r1-x_r2)
                    # For simplicity, use base = pop[ random from union ], diff = x_r1 - x_r2
                    base_idx = np.random.choice(union)
                    base = get_ind(base_idx)
                    mutant = base + F * (x_r1 - x_r2)

                # Crossover: binomial with probability 0.8 for exponential
                if np.random.rand() < 0.8:
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
                    # Archive insertion
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

                    if use_cur2pbest:
                        success_cur.append(1)
                    else:
                        success_rand.append(1)

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                else:
                    if use_cur2pbest:
                        success_cur.append(0)
                    else:
                        success_rand.append(0)

            # Update mutation probability
            if len(success_cur) > 20 and len(success_rand) > 20:
                rate_cur = np.mean(success_cur[-20:])
                rate_rand = np.mean(success_rand[-20:])
                if rate_cur + rate_rand > 0:
                    prob_cur2pbest = rate_cur / (rate_cur + rate_rand)
                prob_cur2pbest = np.clip(prob_cur2pbest, 0.1, 0.9)

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

            # ---------- Local search: Nelder-Mead simplex ----------
            budget_left = self.budget - evals
            low_success = (len(success_rates) >= 5 and np.mean(success_rates[-5:]) < 0.15) or (len(success_rates) < 5)
            diversity = np.std(fitness) < 0.5 * np.mean(domain_range)
            if (gen % ls_freq == 0 and budget_left > 10 * (dim + 1) and diversity and low_success):
                # Start Nelder-Mead from current best
                x_center = self.x_opt.copy()
                f_center = self.f_opt

                # Build initial simplex: center + scaled identity
                step = min(0.05, 0.5 * np.sqrt(np.mean(domain_range))) * domain_range
                simplex = np.zeros((dim + 1, dim))
                simplex[0] = x_center
                for k in range(dim):
                    v = x_center.copy()
                    v[k] = np.clip(v[k] + step[k], lb[k], ub[k])
                    simplex[k+1] = v

                f_simplex = np.empty(dim + 1)
                f_simplex[0] = f_center
                for j in range(1, dim + 1):
                    if evals >= self.budget:
                        break
                    f_simplex[j] = func(simplex[j])
                    evals += 1
                    if f_simplex[j] < self.f_opt:
                        self.f_opt = f_simplex[j]
                        self.x_opt = simplex[j].copy()
                        f_center = self.f_opt
                        x_center = self.x_opt

                # Run limited NM iterations
                max_nm_iter = max(2, min(15, int(0.02 * budget_left / (dim + 1))))
                for it in range(max_nm_iter):
                    if evals >= self.budget:
                        break
                    # Order
                    order = np.argsort(f_simplex)
                    simplex = simplex[order]
                    f_simplex = f_simplex[order]

                    x0 = simplex[0]
                    f0 = f_simplex[0]
                    x_last = simplex[-1]
                    f_last = f_simplex[-1]

                    # Centroid (excluding worst)
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflect
                    xr = np.clip(centroid + nm_rho * (centroid - x_last), lb, ub)
                    fr = func(xr)
                    evals += 1
                    if fr < self.f_opt:
                        self.f_opt = fr
                        self.x_opt = xr.copy()

                    if fr < f0:
                        # Expand
                        xe = np.clip(centroid + nm_chi * (xr - centroid), lb, ub)
                        fe = func(xe)
                        evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            f_simplex[-1] = fe
                        else:
                            simplex[-1] = xr
                            f_simplex[-1] = fr
                        if fe < self.f_opt:
                            self.f_opt = fe
                            self.x_opt = xe.copy()
                    else:
                        if fr < f_simplex[-2]:
                            # Accept reflection
                            simplex[-1] = xr
                            f_simplex[-1] = fr
                        else:
                            # Contract
                            if fr >= f_last:
                                # Outside contraction
                                xc = np.clip(centroid + nm_gamma * (xr - centroid), lb, ub)
                                fc = func(xc)
                                evals += 1
                                if fc < f_last:
                                    simplex[-1] = xc
                                    f_simplex[-1] = fc
                                else:
                                    # Shrink
                                    for j in range(1, dim + 1):
                                        simplex[j] = np.clip(simplex[0] + nm_sigma * (simplex[j] - simplex[0]), lb, ub)
                                        f_simplex[j] = func(simplex[j])
                                        evals += 1
                                        if f_simplex[j] < self.f_opt:
                                            self.f_opt = f_simplex[j]
                                            self.x_opt = simplex[j].copy()
                            else:
                                # Inside contraction
                                xc = np.clip(centroid - nm_gamma * (centroid - x_last), lb, ub)
                                fc = func(xc)
                                evals += 1
                                if fc < f_last:
                                    simplex[-1] = xc
                                    f_simplex[-1] = fc
                                else:
                                    # Shrink
                                    for j in range(1, dim + 1):
                                        simplex[j] = np.clip(simplex[0] + nm_sigma * (simplex[j] - simplex[0]), lb, ub)
                                        f_simplex[j] = func(simplex[j])
                                        evals += 1
                                        if f_simplex[j] < self.f_opt:
                                            self.f_opt = f_simplex[j]
                                            self.x_opt = simplex[j].copy()

                # Inject best into population
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                    # also inject a perturbed version
                    if evals < self.budget:
                        perturb = self.x_opt + 0.01 * np.random.randn(dim) * domain_range
                        perturb = np.clip(perturb, lb, ub)
                        f_pert = func(perturb)
                        evals += 1
                        if f_pert < fitness.max():
                            worst2 = np.argmax(fitness)
                            pop[worst2] = perturb
                            fitness[worst2] = f_pert
                            if f_pert < self.f_opt:
                                self.f_opt = f_pert
                                self.x_opt = perturb.copy()

                # Adapt LS frequency based on improvement
                if f_simplex[0] < best_old - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.85))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(12, int(0.1 * max_gen)):
                n_restart = max(2, int(0.5 * pop_size))
                # Generate new points: half near best, half using Latin hypercube
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
                mem_F[:] = 0.6
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0
                ls_freq = max(ls_freq, min_freq)
                prob_cur2pbest = 0.5

        return self.f_opt, self.x_opt