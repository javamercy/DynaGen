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

        # ---------- population size (hyperbolic reduction) ----------
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 5
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # SHADE memory (F and CR) - use Cauchy for F
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # ---------- Initial population (Latin Hypercube) ----------
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
        archive_size = pop_size  # later resized proportionally

        # ---------- Stagnation detection ----------
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        # Local search parameters
        ls_freq = max(5, int(0.05 * max_gen))
        ls_count = 0
        ls_success_history = []  # recent success of LS

        # L-BFGS memory
        L_mem = 5
        s_list = []
        y_list = []

        # ---------- Main loop ----------
        while evals < self.budget:
            gen += 1

            # Hyperbolic population reduction (L-SHADE style)
            if pop_size > N_min:
                pop_size_new = int(N_init - (N_init - N_min) * (gen / max_gen)**0.7)
                pop_size_new = max(N_min, pop_size_new)
                if pop_size_new < pop_size:
                    idx_sorted = np.argsort(fitness)
                    pop = pop[idx_sorted[:pop_size_new]].copy()
                    fitness = fitness[idx_sorted[:pop_size_new]].copy()
                    pop_size = pop_size_new
                    archive_size = max(pop_size, int(1.5 * pop_size))
                    if len(archive) > archive_size:
                        np.random.shuffle(archive)
                        archive = archive[:archive_size]

            # pbest rate – decreasing over time
            p = 0.2 * (1.0 - gen / max_gen) + 0.1
            p = max(0.1, min(0.5, p))

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

                # r1, r2 from pop+archive (excluding current)
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

                # Sample F (Cauchy) and CR (Normal) from memory
                r = np.random.randint(mem_size)
                F = np.random.standard_cauchy() * 0.1 + mem_F[r]
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + mem_F[r]
                F = np.clip(F, 0.1, 1.0)
                CR = np.random.randn() * 0.1 + mem_CR[r]
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation: current-to-pbest/1
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Crossover: binomial with 70% probability, else exponential
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

                # Repair: reflection for bounds (better than clip)
                out_low = trial < lb
                out_high = trial > ub
                trial[out_low] = lb[out_low] + (lb[out_low] - trial[out_low])
                trial[out_high] = ub[out_high] - (trial[out_high] - ub[out_high])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive replacement: replace most similar point if full
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
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

            # ---------- Update SHADE memory ----------
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                # Weighted Lehmer mean for F
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Adaptive Local Search (L-BFGS) ----------
            # Trigger if stagnation > 5, enough budget, and population diversity low
            if (stagnation_counter >= 5 and
                (self.budget - evals) > dim * 10 + 30 and
                np.std(pop, axis=0).mean() < 0.3 * (ub.mean() - lb.mean())):
                # Perform local search from the best point (and sometimes second best)
                x0_list = [self.x_opt]
                if pop_size >= 3 and evals + dim*10 < self.budget:
                    # also try the second best
                    second_best_idx = np.argsort(fitness)[1] if len(fitness) > 1 else None
                    if second_best_idx is not None:
                        x0_list.append(pop[second_best_idx])
                for x_start in x0_list:
                    if evals + 2*dim + 5 >= self.budget:
                        break
                    x = x_start.copy()
                    f = func(x) if not np.array_equal(x, self.x_opt) else self.f_opt
                    if not np.array_equal(x, self.x_opt):
                        evals += 1
                    # Finite difference gradient
                    h = 1e-7 * (ub - lb) + 1e-9
                    def grad(x_in):
                        g = np.zeros(dim)
                        for d in range(dim):
                            xp = np.clip(x_in + np.eye(1,dim,d) * h[d], lb, ub)[0]
                            xn = np.clip(x_in - np.eye(1,dim,d) * h[d], lb, ub)[0]
                            g[d] = (func(xp) - func(xn)) / (2 * h[d])
                        return g

                    for it in range(5):  # at most 5 iterations per starting point
                        if evals + 2*dim >= self.budget:
                            break
                        g = grad(x)
                        evals += 2*dim
                        if np.linalg.norm(g) < 1e-12:
                            break
                        # Compute L-BFGS direction
                        q = g.copy()
                        alpha = np.zeros(len(s_list))
                        for i_ in range(len(s_list)-1, -1, -1):
                            alpha[i_] = np.dot(s_list[i_], q) / (np.dot(y_list[i_], s_list[i_]) + 1e-30)
                            q = q - alpha[i_] * y_list[i_]
                        d = -q
                        if len(s_list) > 0:
                            H0 = np.dot(s_list[-1], y_list[-1]) / (np.dot(y_list[-1], y_list[-1]) + 1e-30)
                            d = H0 * d
                        for i_ in range(len(s_list)):
                            beta = np.dot(y_list[i_], d) / (np.dot(y_list[i_], s_list[i_]) + 1e-30)
                            d = d + (alpha[i_] - beta) * s_list[i_]
                        # Line search (Armijo)
                        step = 1.0
                        c = 1e-4
                        fx = f
                        for _ in range(10):
                            x_new = np.clip(x + step * d, lb, ub)
                            f_new = func(x_new)
                            evals += 1
                            if f_new <= fx + c * step * np.dot(g, x_new - x):
                                break
                            step *= 0.5
                        if step < 1e-12:
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
                # Inject best into population (replace worst)
                worst = np.argmax(fitness)
                pop[worst] = self.x_opt.copy()
                fitness[worst] = self.f_opt
                ls_count += 1
                stagnation_counter = 0  # reset after LS

            # ---------- Stagnation detection and restart ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(12, int(0.1 * max_gen)):
                # Multi‑cluster restart: keep best 20% points, generate new ones around them
                n_keep = max(1, int(0.2 * pop_size))
                sorted_idx = np.argsort(fitness)
                keep_pop = pop[sorted_idx[:n_keep]].copy()
                keep_fit = fitness[sorted_idx[:n_keep]].copy()
                # For each kept point, generate several variants with decreasing radius
                new_pop = []
                new_fit = []
                # Radius based on average pairwise distance among kept points
                if n_keep > 1:
                    avg_dist = np.mean([np.linalg.norm(keep_pop[i] - keep_pop[j]) for i in range(n_keep) for j in range(i+1, min(4, n_keep))])
                else:
                    avg_dist = 0.5 * (ub - lb).mean()
                radius = max(0.02 * (ub - lb).mean(), avg_dist)
                for idx in range(pop_size):
                    if idx < n_keep:
                        # keep original
                        new_pop.append(keep_pop[idx].copy())
                        new_fit.append(keep_fit[idx])
                    elif idx < n_keep * 3 and n_keep > 0:
                        # around a random kept point
                        k_idx = np.random.randint(n_keep)
                        sigma = radius * (0.5 + 0.5 * np.random.rand())
                        x_new = keep_pop[k_idx] + np.random.randn(dim) * sigma
                        x_new = np.clip(x_new, lb, ub)
                        new_pop.append(x_new)
                        if evals < self.budget:
                            f_new = func(x_new)
                            evals += 1
                            new_fit.append(f_new)
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new.copy()
                        else:
                            new_fit.append(np.inf)
                    else:
                        # Sobol‑like quasi‑random points
                        sob = np.random.rand(1, dim)
                        for j in range(dim):
                            sob[:, j] = (np.argsort(sob[:, j]) + 0.5) / 1
                        x_new = lb + sob[0] * (ub - lb)
                        new_pop.append(x_new)
                        if evals < self.budget:
                            f_new = func(x_new)
                            evals += 1
                            new_fit.append(f_new)
                            if f_new < self.f_opt:
                                self.f_opt = f_new
                                self.x_opt = x_new.copy()
                        else:
                            new_fit.append(np.inf)
                # Convert to arrays
                pop = np.array(new_pop)
                fitness = np.array(new_fit)
                pop_size = len(pop)
                # Reset SHADE memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                s_list.clear()
                y_list.clear()
                stagnation_counter = 0
                # Increase local search frequency after restart
                ls_freq = max(5, int(ls_freq * 0.9))

        return self.f_opt, self.x_opt