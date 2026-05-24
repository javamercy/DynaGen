import numpy as np

class SHADE_NelderMead:
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

        # Population size: nonlinear reduction from ~14*sqrt(dim) to N_min
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)  # rough budget per generation

        # SHADE memory
        mem_size = 10  # larger memory
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin hypercube initialization
        n = pop_size
        perm = np.tile(np.arange(1, n+1), (dim, 1)).T
        for j in range(dim):
            perm[:, j] = np.random.permutation(perm[:, j])
        lhs = (perm - 0.5) / n
        pop = lb + lhs * domain_range

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

        # Control parameters for local search
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        ls_freq = max(8, int(0.08 * max_gen))
        min_freq = 4
        max_freq = max(30, int(0.2 * max_gen))

        # Probability for pbest (increases over generations)
        p = 0.1

        # For Nelder-Mead local search
        def nelder_mead(x0, f0, max_evals_local=50):
            nonlocal evals
            nm_dim = dim
            # Build initial simplex: x0 and scaled perturbations
            delta = 0.01 * domain_range
            simplex = np.zeros((nm_dim+1, nm_dim))
            simplex[0] = x0.copy()
            for i in range(nm_dim):
                pnt = x0.copy()
                pnt[i] = np.clip(pnt[i] + delta[i], lb[i], ub[i])
                simplex[i+1] = pnt
            f_vals = np.empty(nm_dim+1)
            f_vals[0] = f0
            for i in range(1, nm_dim+1):
                if evals < self.budget:
                    f_vals[i] = func(simplex[i])
                    evals += 1
                else:
                    return x0, f0, evals
            # Alpha, gamma, rho, sigma
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            local_evals = nm_dim  # already evaluated
            while local_evals < max_evals_local and evals < self.budget:
                # Order
                order = np.argsort(f_vals)
                simplex = simplex[order]
                f_vals = f_vals[order]
                # Centroid of all but worst
                cen = np.mean(simplex[:-1], axis=0)
                worst_idx = -1
                # Reflection
                xr = cen + alpha * (cen - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                evals += 1
                local_evals += 1
                if f_vals[0] <= fr < f_vals[-2]:
                    simplex[-1] = xr
                    f_vals[-1] = fr
                elif fr < f_vals[0]:
                    # Expansion
                    xe = cen + gamma * (xr - cen)
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    evals += 1
                    local_evals += 1
                    if fe < fr:
                        simplex[-1] = xe
                        f_vals[-1] = fe
                    else:
                        simplex[-1] = xr
                        f_vals[-1] = fr
                else:  # fr >= f_vals[-2]
                    if fr < f_vals[-1]:
                        # Outside contraction
                        xoc = cen + rho * (xr - cen)
                        xoc = np.clip(xoc, lb, ub)
                        foc = func(xoc)
                        evals += 1
                        local_evals += 1
                        if foc <= fr:
                            simplex[-1] = xoc
                            f_vals[-1] = foc
                        else:
                            # Shrink
                            for i in range(1, nm_dim+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                f_vals[i] = func(simplex[i])
                                evals += 1
                                local_evals += 1
                    else:
                        # Inside contraction
                        xic = cen - rho * (cen - simplex[-1])
                        xic = np.clip(xic, lb, ub)
                        fic = func(xic)
                        evals += 1
                        local_evals += 1
                        if fic <= f_vals[-1]:
                            simplex[-1] = xic
                            f_vals[-1] = fic
                        else:
                            for i in range(1, nm_dim+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                f_vals[i] = func(simplex[i])
                                evals += 1
                                local_evals += 1
                # Check best improvement
                if f_vals[0] < f0:
                    return simplex[0], f_vals[0], evals
            # Return best point after local search
            order = np.argsort(f_vals)
            return simplex[order[0]], f_vals[order[0]], evals

        # Main loop
        while evals < self.budget:
            gen += 1

            # Population reduction (nonlinear)
            ratio = max(0, 1 - (gen / max_gen) ** 1.2)
            new_pop_size = max(N_min, int(N_min + (N_init - N_min) * ratio))
            if new_pop_size < pop_size:
                idx = np.argsort(fitness)
                pop = pop[idx[:new_pop_size]].copy()
                fitness = fitness[idx[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # Adaptive pbest rate
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

                # r1, r2 from pop+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                try:
                    union.remove(i)
                except ValueError:
                    pass
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    get = lambda idx: pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1, x_r2 = get(r1), get(r2)
                else:
                    idxs = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F, CR from memory
                rmem = np.random.randint(mem_size)
                F = mem_F[rmem] + 0.1 * np.random.randn()
                CR = mem_CR[rmem] + 0.1 * np.random.randn()
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
                    idxs = (np.arange(dim) + start) % dim
                    mask = np.zeros(dim, dtype=bool)
                    mask[idxs[:L]] = True
                    trial = np.where(mask, mutant, pop[i])
                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_rem = np.random.randint(len(archive))
                        archive[idx_rem] = pop[i].copy()
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

            # Stagnation detection
            if self.f_opt < best_old - 1e-8:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Local search condition: progress low, diversity low, or regular generation
            budget_left = self.budget - evals
            diversity_ratio = np.std(fitness) / np.mean(domain_range + 1e-30)
            do_ls = (gen % ls_freq == 0 and budget_left > 30) or (stagnation_counter > max(5, int(0.03 * max_gen)) and budget_left > 30)
            if do_ls and diversity_ratio < 0.5:
                # Run Nelder-Mead on current best
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                max_nm_evals = min(50, int(budget_left / 3))  # limit evals for local search
                x_new, f_new, evals_new = nelder_mead(x_best, f_best, max_nm_evals)
                if f_new < self.f_opt - 1e-12:
                    self.f_opt = f_new
                    self.x_opt = x_new.copy()
                    # Inject into population (replace worst)
                    if evals < self.budget and f_new < fitness.max():
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = x_new.copy()
                        fitness[worst_idx] = f_new
                # Adapt local search frequency
                if f_new < f_best - 1e-8:
                    ls_freq = max(min_freq, int(ls_freq * 0.9))
                else:
                    ls_freq = min(max_freq, int(ls_freq * 1.1))

            # Restart if stagnation persists
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                perm2 = np.tile(np.arange(1, n_restart+1), (dim, 1)).T
                for j in range(dim):
                    perm2[:, j] = np.random.permutation(perm2[:, j])
                lhs_restart = (perm2 - 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * domain_range * (1 - gen / max_gen) + 0.01
                        pop[idx] = self.x_opt + np.random.randn(dim) * scale
                    else:
                        pop[idx] = lb + lhs_restart[idx] * domain_range
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
                ls_freq = max(ls_freq, min_freq)

        return self.f_opt, self.x_opt