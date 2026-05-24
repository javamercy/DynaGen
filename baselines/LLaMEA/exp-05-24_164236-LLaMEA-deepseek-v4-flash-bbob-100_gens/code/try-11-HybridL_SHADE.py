import numpy as np

class HybridL_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim

        # --- initialization ---
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube Sampling
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

        archive = []  
        archive_size = min(2 * pop_size, int(self.budget * 0.2))
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # for local search
        local_search_period = max(5, int(0.02 * max_gen))
        ls_step = (ub - lb) * 0.02

        while evals < self.budget:
            gen += 1

            # --- nonlinear population size reduction ---
            if N_init != N_min:
                ratio = gen / max_gen
                new_pop_size = int(N_init * ((N_min / N_init) ** (ratio**1.5)))
                new_pop_size = max(N_min, min(pop_size, new_pop_size))
            else:
                new_pop_size = N_min
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # trim archive
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            success_F_gen = []
            success_CR_gen = []
            delta_f_gen = []  # for weighted adaptation

            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # --- p-best selection, adapt rate ---
                p = 0.3 * (1 - gen / max_gen) ** 0.7
                p = max(0.05, p)
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # --- choose two distinct indices from pop+archive ---
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    if r1 >= pop_size:
                        x_r1 = archive[r1 - pop_size]
                    else:
                        x_r1 = pop[r1]
                    if r2 >= pop_size:
                        x_r2 = archive[r2 - pop_size]
                    else:
                        x_r2 = pop[r2]
                else:
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # --- parameter generation (Cauchy for F) ---
                F = np.clip(np.random.cauchy(F_base, 0.1), 0.1, 1.0)
                CR = np.clip(np.random.normal(CR_base, 0.1), 0.0, 1.0)

                # --- mutation and crossover ---
                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    delta = max(1e-12, abs(fitness[i] - f_trial))
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    delta_f_gen.append(delta)
                    # archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_arch = np.random.randint(len(archive))
                        archive[idx_arch] = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # --- update memory with weighted Lehmer mean ---
            if len(success_F_gen) > 0:
                w = np.array(delta_f_gen) / np.sum(delta_f_gen)
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.average(success_CR_gen, weights=w)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # --- periodic local search (bounded random directions) ---
            if (gen % local_search_period == 0 or stagnation_counter >= 3) and evals < self.budget:
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # local search using random orthogonal directions
                ls_evals_budget = min(10 * dim, self.budget - evals)
                step = ls_step.copy()
                improvement = True
                while improvement and ls_evals_budget > 0:
                    improvement = False
                    # generate random orthonormal basis
                    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
                    for d in range(dim):
                        if ls_evals_budget <= 0:
                            break
                        direction = Q[:, d]
                        # try both signs
                        for sign in [1, -1]:
                            if ls_evals_budget <= 0:
                                break
                            trial = np.clip(x_best + sign * step * direction, lb, ub)
                            f_trial = func(trial)
                            evals += 1
                            ls_evals_budget -= 1
                            if f_trial < f_best:
                                f_best = f_trial
                                x_best = trial.copy()
                                improvement = True
                                step *= 1.2  # accelerate
                                break
                            else:
                                step *= 0.85  # shrink step
                    if improvement:
                        step = np.clip(step * 1.1, (ub-lb)*1e-4, (ub-lb)*0.2)
                # update global best if improved
                if f_best < self.f_opt:
                    self.f_opt = f_best
                    self.x_opt = x_best.copy()
                    # inject into population
                    worst_idx = np.argmax(fitness)
                    if f_best < fitness[worst_idx]:
                        pop[worst_idx] = self.x_opt.copy()
                        fitness[worst_idx] = self.f_opt

            # --- stagnation detection ---
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # --- restart if severe stagnation ---
            if stagnation_counter > max(10, int(0.1 * max_gen)) and evals < self.budget - pop_size:
                # keep best solution, reinitialize around it with larger spread
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # generate new population around best with Cauchy perturbation
                for i in range(pop_size):
                    if i == 0:
                        pop[i] = best_copy
                        fitness[i] = best_f
                        continue
                    scale = (ub - lb) * 0.3 * (stagnation_counter / max(10, 0.1*max_gen))
                    trial = best_copy + np.random.standard_cauchy(dim) * scale
                    trial = np.clip(trial, lb, ub)
                    if evals < self.budget:
                        f_trial = func(trial)
                        evals += 1
                        pop[i] = trial
                        fitness[i] = f_trial
                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial.copy()
                stagnation_counter = 0
                # reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive = []

        return self.f_opt, self.x_opt