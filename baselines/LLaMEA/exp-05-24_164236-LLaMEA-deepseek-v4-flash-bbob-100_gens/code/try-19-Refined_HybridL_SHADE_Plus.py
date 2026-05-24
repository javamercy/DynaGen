import numpy as np

class Refined_HybridL_SHADE_Plus:
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
        N_init = max(10, int(18 * np.sqrt(dim)))   # increased from 14 to 18
        N_min = 4
        pop_size = N_init
        max_gen = int(self.budget / pop_size * 2)

        # Memory for successful parameters (SHADE)
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Sobol initialization (space-filling)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True)
            lhs = sampler.random(pop_size)
        except:
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

        # Archive (diversity) - random replacement to avoid clustering
        archive = []
        archive_size = pop_size

        # Stagnation tracking
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
                # Keep archive within size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            # Time-dependent pbest rate (decreasing)
            p = 0.2 * (1.0 - gen / max_gen) + 0.1
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

                # Random selection from union of pop and archive (excluding current)
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

                # Sample F and CR from memory with perturbation
                r = np.random.randint(mem_size)
                F = mem_F[r] + 0.1 * np.random.randn()
                CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Mutation (current-to-pbest/1)
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    # Archive update: random replacement (instead of closest distance)
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace a random archive element
                        idx_remove = np.random.randint(archive_size)
                        archive[idx_remove] = pop[i].copy()

                    # Record successful parameters
                    success_F.append(F)
                    success_CR.append(CR)
                    imp = fitness[i] - f_trial
                    weight.append(max(imp, 1e-12))

                    # Update population
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update memory with weighted Lehmer mean (SHADE style)
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---- Local search: Coordinate-wise pattern search on best ----
            nm_budget = int(0.1 * (self.budget - evals))   # reduced budget for local search
            if nm_budget > dim + 1 and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # Initial step size (10% of domain range)
                step = (ub - lb) * 0.1
                # Pattern search iterations
                used = 0
                max_iter = min(nm_budget, 20 * dim)   # cap to avoid excessive evaluations
                for _ in range(max_iter):
                    improved = False
                    # Coordinate-wise search (forward and backward)
                    for d in range(dim):
                        if used >= nm_budget or evals >= self.budget:
                            break
                        x_new = x_best.copy()
                        x_new[d] += step[d]
                        x_new[d] = np.clip(x_new[d], lb[d], ub[d])
                        f_new = func(x_new)
                        evals += 1; used += 1
                        if f_new < f_best:
                            f_best = f_new
                            x_best = x_new.copy()
                            improved = True
                            break   # accept first improvement and restart loop
                        # opposite direction
                        x_new[d] = x_best[d] - step[d]
                        x_new[d] = np.clip(x_new[d], lb[d], ub[d])
                        f_new = func(x_new)
                        evals += 1; used += 1
                        if f_new < f_best:
                            f_best = f_new
                            x_best = x_new.copy()
                            improved = True
                            break
                    if not improved:
                        # Reduce step size
                        step *= 0.9
                        if np.mean(step) < 1e-10 * np.mean(ub - lb):
                            break
                    else:
                        # Reset step to initial after improvement
                        step = (ub - lb) * 0.1
                # Update global best
                if f_best < self.f_opt:
                    self.f_opt = f_best
                    self.x_opt = x_best.copy()
                # Inject best into population if better than worst
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if severe stagnation
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                n_restart = max(1, int(0.5 * pop_size))
                # Quasi-random restart (Sobol)
                try:
                    sampler_restart = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler_restart.random(n_restart)
                except:
                    sob = np.random.rand(n_restart, dim)
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        # local perturbation
                        scale = 0.2 * (ub - lb) * (1 - gen / max_gen)
                        pop[idx] = best_copy + np.random.uniform(-1, 1, dim) * scale
                    else:
                        # scattered with Sobol
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