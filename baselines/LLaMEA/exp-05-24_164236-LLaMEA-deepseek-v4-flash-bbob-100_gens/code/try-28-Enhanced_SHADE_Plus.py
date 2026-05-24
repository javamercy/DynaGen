import numpy as np

class Enhanced_SHADE_Plus:
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
        span = ub - lb

        # Population size (L-SHADE style, smaller initial)
        N_init = max(10, int(10 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory for F and CR
        mem_size = 10
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling initial population
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * span

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

        # Stagnation detection
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        # Local search adaptive settings
        ls_success_streak = 0
        ls_fail_streak = 0
        ls_freq_max = max(10, int(0.06 * max_gen))
        ls_freq = ls_freq_max

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # Diversity injection counter
        last_improvement_gen = 0

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
                    np.random.shuffle(archive)
                    archive = archive[:archive_size]

            # pbest rate (adaptive, high early, low late)
            p = max(0.05, 0.25 * (1 - gen / max_gen)**2 + 0.1)
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # Select r1, r2 from pop and archive
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

                # Sample F, CR from memory (with noise)
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1 with Cauchy diversity injection (prob 10%)
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                if np.random.rand() < 0.1:  # Cauchy perturbation for diversity
                    cauchy = np.random.standard_cauchy(dim)
                    mutant += 0.5 * F * span * np.tanh(cauchy)  # bounded perturbation

                # Crossover: binomial (70%) or exponential (30%)
                trial = np.zeros(dim)
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
                    # Archive management: replace farthest (if full) else append
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # replace a random archive element to maintain diversity
                        idx_remove = np.random.randint(len(archive))
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
                        last_improvement_gen = gen

            # Update SHADE memory using weighted Lehmer mean
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Adaptive local search (L-BFGS) ----------
            # Trigger when enough budget left, population diversity low, and either stagnation or recent no improvement
            remaining_evals = self.budget - evals
            diversity = np.std(fitness)
            if (remaining_evals > dim * 6 + 30 and 
                diversity < 0.5 and 
                (stagnation_counter > 2 or gen - last_improvement_gen > 0.02 * max_gen) and
                np.random.rand() < 0.15):   # probabilistic trigger
                # Determine local search budget: max iterations scale with remaining evals
                ls_iter = max(1, min(4, int(0.03 * remaining_evals / (dim * 4 + 10))))
                h = 1e-7 * span + 1e-9
                def grad(x):
                    g = np.zeros(dim)
                    for d in range(dim):
                        xp = np.clip(x + np.eye(1,dim,d) * h[d], lb, ub)[0]
                        xn = np.clip(x - np.eye(1,dim,d) * h[d], lb, ub)[0]
                        g[d] = (func(xp) - func(xn)) / (2 * h[d])
                    return g

                x = self.x_opt.copy()
                f = self.f_opt
                improved = False
                for it in range(ls_iter):
                    if evals + 2*dim >= self.budget:
                        break
                    g = grad(x)
                    evals += 2*dim
                    if np.linalg.norm(g) < 1e-12:
                        break
                    # Compute search direction via L-BFGS two-loop recursion
                    q = g.copy()
                    alpha = np.zeros(len(s_list))
                    for i in range(len(s_list)-1, -1, -1):
                        alpha[i] = np.dot(s_list[i], q) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        q = q - alpha[i] * y_list[i]
                    d = -q
                    if len(s_list) > 0:
                        H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                        d = H0 * d
                    for i in range(len(s_list)):
                        beta = np.dot(y_list[i], d) / (np.dot(y_list[i], s_list[i]) + 1e-30)
                        d = d + (alpha[i] - beta) * s_list[i]
                    # Line search (Armijo)
                    alpha_step = 1.0
                    c = 1e-4
                    fx = f
                    for _ in range(10):
                        x_new = np.clip(x + alpha_step * d, lb, ub)
                        f_new = func(x_new)
                        evals += 1
                        if f_new <= fx + c * alpha_step * np.dot(g, x_new - x):
                            break
                        alpha_step *= 0.5
                    if alpha_step < 1e-12:
                        break
                    # Update L-BFGS history
                    s = x_new - x
                    y = grad(x_new) - g
                    evals += 2*dim
                    sy = np.dot(s, y)
                    if sy > 1e-10:
                        if len(s_list) >= L_mem:
                            s_list.pop(0)
                            y_list.pop(0)
                        s_list.append(s.copy())
                        y_list.append(y.copy())
                    x = x_new
                    f = f_new
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = x.copy()
                        improved = True
                # Inject best local point into population (replace worst)
                if self.f_opt < fitness.max():
                    worst = np.argmax(fitness)
                    pop[worst] = self.x_opt.copy()
                    fitness[worst] = self.f_opt
                # Adjust local search frequency based on success
                if improved:
                    ls_success_streak += 1
                    ls_fail_streak = 0
                    ls_freq = max(5, int(ls_freq * 0.9))
                else:
                    ls_fail_streak += 1
                    ls_success_streak = 0
                    ls_freq = min(ls_freq_max, int(ls_freq * 1.1))
                # Keep within bounds
                ls_freq = np.clip(ls_freq, 5, ls_freq_max)

            # ---------- Stagnation detection and Cauchy restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(6, int(0.06 * max_gen)):
                n_restart = max(1, int(0.6 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Quasi-random LHS for global points
                sob = np.random.rand(n_restart, dim)
                for j in range(dim):
                    sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / n_restart
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # Cauchy perturbation around best
                        scale = 0.05 * span * (1 - gen / max_gen) + 0.01
                        pop[idx] = best_copy + np.random.standard_cauchy(dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * span
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory, archive, L-BFGS history
                mem_F[:] = 0.5 + 0.1 * np.random.randn(mem_size)
                mem_CR[:] = 0.8 + 0.1 * np.random.randn(mem_size)
                mem_F[:] = np.clip(mem_F, 0.1, 1.0)
                mem_CR[:] = np.clip(mem_CR, 0.0, 1.0)
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                last_improvement_gen = gen
                ls_freq = ls_freq_max  # reset local search frequency after restart

        return self.f_opt, self.x_opt