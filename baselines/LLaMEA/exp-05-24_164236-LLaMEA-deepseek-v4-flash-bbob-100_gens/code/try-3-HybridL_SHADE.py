import numpy as np
from scipy.stats import qmc

class HybridL_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget

        # ---------- parameters ----------
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0
        max_gen = int(budget / pop_size * 2)

        # Latin hypercube initial population
        sampler = qmc.LatinHypercube(dim, seed=None)
        lhs = sampler.random(n=pop_size)
        pop = lb + (ub - lb) * lhs
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # local search success memory
        ls_step = (ub - lb) * 0.1
        ls_min_step = 1e-6 * (ub - lb).max()
        ls_success_rate = 0.0

        gen = 0
        stagnation_counter = 0
        best_old = self.f_opt
        archive = []

        while evals < budget:
            gen += 1

            # ---- linear population reduction ----
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size

            # ---- adapt F and CR ----
            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            success_F = []
            success_CR = []

            # ---- mutation and crossover ----
            for i in range(pop_size):
                if evals >= budget:
                    break

                # pbest selection (adaptive p)
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # random indices
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)

                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (pop[r1] - pop[r2])
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand) else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    success_F.append(F)
                    success_CR.append(CR)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                    pop[i] = trial
                    fitness[i] = f_trial

            # ---- update memory with successful parameters ----
            if len(success_F) > 0:
                w = np.ones(len(success_F))
                F_lehmer = np.sum(w * np.array(success_F)**2) / np.sum(w * np.array(success_F))
                CR_mean = np.mean(success_CR)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---- adaptive simplex local search on best ----
            if evals < budget and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt

                # build a small simplex around best
                simplex = [x_best.copy()]
                for k in range(dim):
                    step = max(ls_step[k] * 0.1, 1e-7)
                    p = x_best.copy()
                    p[k] = np.clip(p[k] + step, lb[k], ub[k])
                    simplex.append(p)

                # Nelder-Mead iterations (simplex)
                max_nm_iter = min(10 * dim, (budget - evals) // (dim+1))
                for _ in range(max_nm_iter):
                    # evaluate simplex points not yet evaluated
                    for s in range(len(simplex)):
                        # check if fitness already known (approx) – we always evaluate
                        if evals >= budget:
                            break
                        f = func(simplex[s])
                        evals += 1
                        if f < f_best:
                            f_best = f
                            x_best = simplex[s].copy()
                            if f_best < self.f_opt:
                                self.f_opt = f_best
                                self.x_opt = x_best.copy()

                    # order simplex by fitness
                    # we don't have full fitness history, so we rebuild it
                    fits = np.array([func(simplex[s]) for s in range(len(simplex))])
                    # this is expensive, but we only do it once per iteration
                    # better to keep track, but for clarity we do it
                    if evals >= budget:
                        break
                    order = np.argsort(fits)
                    simplex = [simplex[idx] for idx in order]
                    fits = fits[order]

                    # centroid
                    centroid = np.mean(simplex[:-1], axis=0)
                    # reflection
                    xr = centroid + (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr); evals += 1
                    if fr < fits[0]:
                        # expansion
                        xe = centroid + 2 * (xr - centroid)
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe); evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                        else:
                            simplex[-1] = xr
                    elif fr < fits[-2]:
                        simplex[-1] = xr
                    else:
                        # contraction
                        xc = centroid + 0.5 * (simplex[-1] - centroid)
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc); evals += 1
                        if fc < fits[-1]:
                            simplex[-1] = xc
                        else:
                            # shrink
                            for j in range(1, len(simplex)):
                                simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                    # update best
                    if fits[0] < f_best:
                        f_best = fits[0]
                        x_best = simplex[0].copy()
                        if f_best < self.f_opt:
                            self.f_opt = f_best
                            self.x_opt = x_best.copy()

                # replace worst individual with best from local search
                if f_best < self.f_opt:
                    pass  # already updated
                worst_idx = np.argmax(fitness)
                if f_best < fitness[worst_idx]:
                    pop[worst_idx] = x_best.copy()
                    fitness[worst_idx] = f_best

                # adapt step size for future pattern search (if any)
                ls_step = (ub - lb) * 0.1 * np.exp(-gen / max_gen)

            # ---- stagnation detection and restart ----
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            if stagnation_counter > max(10, int(0.1 * max_gen)):
                # restart: keep best 20%, reinitialize rest with LHS
                n_keep = max(1, int(0.2 * pop_size))
                idx_keep = np.argsort(fitness)[:n_keep]
                keep_pop = pop[idx_keep].copy()
                keep_fit = fitness[idx_keep].copy()
                n_new = pop_size - n_keep
                sampler = qmc.LatinHypercube(dim, seed=None)
                new_points = lb + (ub - lb) * sampler.random(n=n_new)
                pop = np.vstack((keep_pop, new_points))
                fitness_new = np.empty(n_new)
                for j in range(n_new):
                    if evals >= budget:
                        break
                    fitness_new[j] = func(pop[n_keep + j])
                    evals += 1
                    if fitness_new[j] < self.f_opt:
                        self.f_opt = fitness_new[j]
                        self.x_opt = pop[n_keep + j].copy()
                fitness = np.concatenate((keep_fit, fitness_new))
                stagnation_counter = 0
                # reset memory for increased exploration
                mem_F[:] = 0.8
                mem_CR[:] = 0.9
                mem_idx = 0

        return self.f_opt, self.x_opt