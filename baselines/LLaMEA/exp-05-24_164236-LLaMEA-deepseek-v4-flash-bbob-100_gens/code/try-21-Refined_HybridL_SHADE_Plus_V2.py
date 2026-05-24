import numpy as np

class Refined_HybridL_SHADE_Plus_V2:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim

        # Population size (L-SHADE style reduction)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # Memory for successful parameters (SHADE)
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive (L-SHADE)
        archive = []
        archive_size = pop_size

        # Tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Main loop
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
                    archive = archive[-archive_size:]

            # pbest rate (time-dependent + diversity-based)
            fitness_range = np.max(fitness) - np.min(fitness) + 1e-30
            diversity = np.std(fitness) / fitness_range
            p = 0.2 * (1 - diversity) + 0.05
            p = min(max(p, 0.05), 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Choose r1,r2 from union of pop and archive (excluding i)
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_individual(idx):
                        if idx < pop_size:
                            return pop[idx]
                        else:
                            return archive[idx - pop_size]
                    x_r1 = get_individual(r1)
                    x_r2 = get_individual(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Sample F and CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial with probability 0.7, exponential with 0.3
                trial = np.zeros(dim)
                if np.random.rand() < 0.7:
                    # binomial
                    j_rand = np.random.randint(dim)
                    mask = np.random.rand(dim) < CR
                    mask[j_rand] = True
                    trial = np.where(mask, mutant, pop[i])
                else:
                    # exponential
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
                    # Add parent to archive (replace random if full)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_remove = np.random.randint(archive_size)
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update memory (weighted Lehmer mean for F and CR)
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                # Lehmer mean for F
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                # Lehmer mean for CR
                CR_lehmer = np.sum(w * np.array(success_CR)**2) / (np.sum(w * np.array(success_CR)) + 1e-30)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_lehmer
                mem_idx = (mem_idx + 1) % mem_size

            # ---- Local search: quasi-Newton (BFGS) with finite-difference gradient ----
            ls_budget = int(0.10 * (self.budget - evals))
            if ls_budget > dim + 1 and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt

                # Compute gradient via forward differences (cost: dim evals)
                eps = 1e-8 * (ub - lb)
                grad = np.zeros(dim)
                g_evals = 0
                for k in range(dim):
                    x_plus = x_best.copy()
                    x_plus[k] = np.clip(x_plus[k] + eps[k], lb[k], ub[k])
                    f_plus = func(x_plus)
                    g_evals += 1
                    if evals + g_evals >= self.budget:
                        break
                    grad[k] = (f_plus - f_best) / eps[k]
                evals += g_evals
                if g_evals < dim:
                    continue  # not enough budget

                # Quasi-Newton: BFGS with line search
                # Initialize Hessian approximation to identity
                H = np.eye(dim)
                x = x_best.copy()
                f = f_best
                nfev = 0
                max_iter = max(2, int(ls_budget / (dim + 5)))  # rough estimate
                for it in range(max_iter):
                    if evals >= self.budget:
                        break
                    # Compute search direction
                    p = -H @ grad
                    # Backtracking line search
                    alpha = 1.0
                    f_new = None
                    while alpha > 1e-12:
                        x_new = np.clip(x + alpha * p, lb, ub)
                        f_new = func(x_new)
                        nfev += 1
                        evals += 1
                        # Armijo condition: sufficient decrease
                        if f_new <= f + 1e-4 * alpha * np.dot(grad, p):
                            break
                        alpha *= 0.5
                    if alpha <= 1e-12 or f_new >= f:
                        break  # no improvement
                    # Compute new gradient
                    grad_new = np.zeros(dim)
                    for k in range(dim):
                        x_plus = x_new.copy()
                        x_plus[k] = np.clip(x_plus[k] + eps[k], lb[k], ub[k])
                        f_plus = func(x_plus)
                        nfev += 1
                        evals += 1
                        if evals >= self.budget:
                            break
                        grad_new[k] = (f_plus - f_new) / eps[k]
                    if evals >= self.budget:
                        break
                    # BFGS update
                    s = x_new - x
                    y = grad_new - grad
                    ynorm2 = np.dot(y, y)
                    if ynorm2 > 1e-12:
                        rho = 1.0 / ynorm2
                        Hy = H @ y
                        H = H + rho * np.outer(s, s) * (1 + np.dot(y, Hy) / ynorm2) - rho * (np.outer(s, Hy) + np.outer(Hy, s))
                    else:
                        break
                    # Update state
                    x = x_new
                    f = f_new
                    grad = grad_new
                    # Update best
                    if f_new < self.f_opt:
                        self.f_opt = f_new
                        self.x_opt = x_new.copy()

                # Inject best local search point into population if better than worst
                if self.f_opt < np.max(fitness):
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # Stagnation detection and restart
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(8, int(0.08 * max_gen)):
                # Diversity restoration: replace 60% of population
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Use Sobol-like low-discrepancy sequence if available
                try:
                    from scipy.stats import qmc
                    sampler = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler.random(n_restart)
                except:
                    # Fallback to LHS
                    sob = np.random.rand(n_restart, dim)
                    for j in range(dim):
                        sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # local perturbation around best
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.randn(dim) * scale
                    else:
                        # global quasi-random
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                stagnation_counter = 0

        return self.f_opt, self.x_opt