import numpy as np

class Enhanced_SHADE_Plus:
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

        # Population size settings
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory for F and CR (larger memory)
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling initial population
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

        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0

        # Local search parameters
        ls_freq = max(3, int(0.04 * max_gen))  # less frequent local search
        ls_budget_fraction = 0.1               # reduced budget for local search
        ls_center = self.x_opt.copy()
        ls_f_center = self.f_opt

        while evals < self.budget:
            gen += 1

            # Quadratic population size reduction (slower initial reduction)
            t = gen / max_gen
            new_pop_size = max(N_min, int(N_init * (1 - t**0.8) + N_min * t**0.8))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # pbest rate: adaptive based on remaining budget
            remaining = (self.budget - evals) / self.budget
            p = max(0.1, 0.2 * (1 - remaining) + 0.1 * remaining)
            p = min(p, 0.5)

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

                # Select r1, r2 from pop and archive (excluding i)
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

                # Sample F and CR from memory (with noise)
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial (65%) or exponential (35%)
                trial = np.zeros(dim)
                if np.random.rand() < 0.65:
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
                    # Add parent to archive (replace farthest to maintain diversity)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace the parent that is closest to the current best
                        dists = np.linalg.norm(np.array(archive) - pop[i], axis=1)
                        idx_remove = np.argmin(dists)
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

            # Update SHADE memory (using weighted median for robustness)
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                
                # Weighted median for F (more stable than Lehmer mean)
                sorted_idx = np.argsort(success_F)
                cumsum = np.cumsum(w[sorted_idx])
                median_val = np.interp(0.5, cumsum, np.array(success_F)[sorted_idx])
                F_mem = median_val
                
                # Weighted mean for CR
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = np.clip(F_mem, 0.1, 0.9)
                mem_CR[mem_idx] = np.clip(CR_mean, 0.1, 0.9)
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Local search (improved L-BFGS with cubic line search) ----------
            if gen % ls_freq == 0 and (self.budget - evals) > dim * 6 + 10:
                # Estimate gradient using central differences (with adaptive step)
                h = 1e-5 * (ub - lb) + 1e-8
                def grad(x):
                    g = np.zeros(dim)
                    x = np.asarray(x)
                    for d in range(dim):
                        xp = x.copy()
                        xn = x.copy()
                        xp[d] = np.clip(xp[d] + h[d], lb[d], ub[d])
                        xn[d] = np.clip(xn[d] - h[d], lb[d], ub[d])
                        g[d] = (func(xp) - func(xn)) / (2 * h[d])
                    return g

                x = self.x_opt.copy()
                f = self.f_opt
                H_inv = np.eye(dim)
                max_iter = int(ls_budget_fraction * (self.budget - evals) / (2 * dim + 5))
                max_iter = max(min(max_iter, 15), 3)

                for _ in range(max_iter):
                    if evals + 2 * dim + 2 >= self.budget:
                        break
                    g = grad(x)
                    evals += 2 * dim
                    if np.linalg.norm(g) < 1e-12:
                        break
                    d = -H_inv @ g
                    d_norm = np.linalg.norm(d)
                    if d_norm > 0.5 * (ub - lb).max():
                        d = d / d_norm * 0.5 * (ub - lb).max()
                    # Cubic line search with quadratic interpolation
                    alpha = 1.0
                    c = 1e-4
                    fx = f
                    f_prev = fx
                    alpha_prev = 0.0
                    for _ in range(15):
                        x_new = np.clip(x + alpha * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= fx + c * alpha * np.dot(g, x_new - x):
                            break
                        # Quadratic interpolation for better step
                        if _ == 0:
                            # Use derivative at 0
                            deriv = np.dot(g, d)
                            alpha_new = -deriv / (2 * (f_new - fx - deriv)) * alpha
                            alpha = np.clip(alpha_new, 0.1 * alpha, 0.5 * alpha)
                        else:
                            # Cubic interpolation
                            a = np.array([[alpha**3, alpha**2],
                                          [alpha_prev**3, alpha_prev**2]])
                            b = np.array([f_new - fx - deriv*alpha,
                                          f_prev - fx - deriv*alpha_prev])
                            try:
                                inv_a = np.linalg.inv(a)
                                coeff = inv_a @ b
                                disc = coeff[1]**2 - 3*coeff[0]*deriv
                                if disc > 0:
                                    alpha_new = (-coeff[1] + np.sqrt(disc)) / (3*coeff[0])
                                    alpha = np.clip(alpha_new, 0.1*alpha, 0.5*alpha)
                            except:
                                alpha *= 0.5
                        alpha_prev = alpha
                        f_prev = f_new
                        alpha *= 0.5
                        if alpha < 1e-12:
                            break
                    if alpha < 1e-12:
                        break
                    # Update inverse Hessian (BFGS with damping)
                    s = x_new - x
                    y = grad(x_new) - g
                    evals += 2 * dim
                    sy = np.dot(s, y)
                    if sy > 1e-10:
                        # Damped BFGS
                        theta = 1.0 if sy >= 0.2 * np.dot(y, y) else (0.8 * np.dot(y, y)) / (np.dot(y, y) - sy)
                        s_damped = theta * s + (1 - theta) * (H_inv @ y)
                        sy_damped = np.dot(s_damped, y)
                        if sy_damped > 1e-10:
                            rho = 1.0 / sy_damped
                            I = np.eye(dim)
                            H_inv = (I - rho * np.outer(s_damped, y)) @ H_inv @ (I - rho * np.outer(y, s_damped)) + rho * np.outer(s_damped, s_damped)
                    x = x_new
                    f = f_new
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()

                # Inject the best local search point into population (replace worst)
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
                ls_center = self.x_opt.copy()
                ls_f_center = self.f_opt
            else:
                stagnation_counter += 1

            # Stagnation tolerance depends on dimension
            stagnation_tol = max(5, int(0.06 * max_gen * (1 + dim/20)))
            if stagnation_counter > stagnation_tol:
                # Diversified restart: 60% near best with scaled Cauchy, 40% global Sobol
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Sobol-like low-discrepancy points (using LHS as proxy)
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < int(0.6 * n_restart):
                        # Local perturbation around best with Cauchy distribution (heavy tail)
                        scale = 0.05 * (ub - lb) * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.standard_cauchy(dim) * scale
                    else:
                        # Global quasi-random
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
                # Reset BFGS state
                ls_center = self.x_opt.copy()
                ls_f_center = self.f_opt

        return self.f_opt, self.x_opt