import numpy as np

class Refined_SHADE_Plus_CMA:
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

        # Population size: larger initial for better exploration
        N_init = max(10, int(18 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2.5)  # generous generation budget

        # SHADE parameter memory
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.9)
        mem_idx = 0

        # Sobol initialization (fallback to LHS)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            init_pop = sampler.random(pop_size)
        except ImportError:
            init_pop = np.random.rand(pop_size, dim)
            for j in range(dim):
                init_pop[:, j] = (np.argsort(init_pop[:, j]) + 0.5) / pop_size
        pop = lb + init_pop * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive for diversity (L-SHADE)
        archive = []
        archive_size = int(1.5 * pop_size)  # larger archive

        # For stagnation detection
        best_old = self.f_opt
        stagnation = 0
        gen = 0

        # Local search – (1+1)-CMA with 1/5 rule
        def one_plus_one_cma(parent, f_parent, budget_local, sigma_init):
            sigma = sigma_init
            x = parent.copy()
            f = f_parent
            used = 0
            # Simple evolution path for step size adaptation
            path = np.zeros(dim)
            c_c = 2.0 / (dim + 2.0)  # cumulation parameter
            c_sigma = (dim + 2.0) / (dim + 4.0)  # damping
            # expected selection ratio for success
            target_success = 1.0 / 5.0
            success_cnt = 0
            total_cnt = 0

            while used < budget_local and self.f_opt > -1e100:
                # generate offspring by Gaussian mutation
                z = np.random.randn(dim)
                y = x + sigma * z
                y = np.clip(y, lb, ub)
                fy = func(y)
                evals += 1
                used += 1
                total_cnt += 1

                if fy <= f:
                    # success
                    success_cnt += 1
                    # update path for step size adaptation
                    path = (1.0 - c_c) * path + np.sqrt(c_c * (2.0 - c_c)) * z
                    # update mean
                    x = y
                    f = fy
                    if fy < self.f_opt:
                        self.f_opt = fy
                        self.x_opt = x.copy()
                else:
                    # failure: update path negatively
                    path = (1.0 - c_c) * path
                # adapt step size using success rate
                sigma = sigma * np.exp(c_sigma * (success_cnt / max(total_cnt, 1) - target_success))
                sigma = np.clip(sigma, 1e-8 * (ub - lb).mean(), 0.5 * (ub - lb).mean())
                # reset counters after adaptation window
                if total_cnt >= 20:
                    success_cnt = 0
                    total_cnt = 0
            return x, f

        # Main loop
        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                order = np.argsort(fitness)
                pop = pop[order[:new_pop_size]].copy()
                fitness = fitness[order[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # pbest rate – time decreasing
            p_min = 0.05
            p_max = 0.25
            p = p_max - (p_max - p_min) * (gen / max_gen)
            p = max(p_min, p)

            success_F = []
            success_CR = []
            weights = []

            f_min = np.min(fitness)

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # pbest selection
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                pbest_idx = np.random.choice(best_indices)
                x_pbest = pop[pbest_idx]

                # random individuals from pop ∪ archive (excluding i)
                union = list(range(pop_size)) + list(range(len(archive)))
                if i in union:
                    union.remove(i)
                if len(union) < 2:
                    # fallback to pure pop
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]
                else:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get(idx):
                        return pop[idx] if idx < pop_size else archive[idx - pop_size]
                    x_r1 = get(r1)
                    x_r2 = get(r2)

                # sample F and CR from memory
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # current-to-pbest/1 mutation
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)

                ftrial = func(trial)
                evals += 1

                if ftrial <= fitness[i]:
                    # archive replacement – keep diverse
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # replace the archive member most similar to trial (Euclidean)
                        dists = np.linalg.norm(np.array(archive) - trial, axis=1)
                        archive[np.argmin(dists)] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    improvement = max(fitness[i] - ftrial, 1e-12)
                    weights.append(improvement)

                    pop[i] = trial.copy()
                    fitness[i] = ftrial
                    if ftrial < self.f_opt:
                        self.f_opt = ftrial
                        self.x_opt = trial.copy()

            # Update memory with weighted Lehmer mean
            if len(success_F) > 0:
                w = np.array(weights)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = np.clip(CR_mean, 0.0, 1.0)
                mem_idx = (mem_idx + 1) % mem_size

            # ----- Local search: (1+1)-CMA on best point -----
            ls_budget = int(0.10 * (self.budget - evals))  # use up to 10% of remaining budget
            if ls_budget > dim and (gen % 5 == 0 or stagnation >= 5):
                # define initial step size as 15% of domain range
                sigma_init = 0.15 * (ub - lb).mean()
                x_new, f_new = one_plus_one_cma(self.x_opt.copy(), self.f_opt, ls_budget, sigma_init)
                # inject best point into population if it beats the worst
                worst_idx = np.argmax(fitness)
                if f_new < fitness[worst_idx]:
                    pop[worst_idx] = x_new.copy()
                    fitness[worst_idx] = f_new

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation = 0
            else:
                stagnation += 1

            # ----- Adaptive restart on severe stagnation -----
            if stagnation > max(10, int(0.12 * max_gen)) and evals < self.budget - 50:
                # keep best point, reinitialize a portion of pop and reset memory
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                n_restart = max(1, int(0.6 * pop_size))
                # generate quasi-random restarts
                try:
                    sampler_restart = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler_restart.random(n_restart)
                except:
                    sob = np.random.rand(n_restart, dim)
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # local perturbation around best
                        scale = 0.15 * (ub - lb) * (1 - gen / max_gen)
                        pop[idx] = best_copy + np.random.uniform(-1, 1, dim) * scale
                    else:
                        # scattered points full domain
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # reset memory
                mem_F[:] = 0.5
                mem_CR[:] = 0.9
                archive.clear()
                stagnation = 0

        return self.f_opt, self.x_opt