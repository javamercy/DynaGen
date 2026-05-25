import numpy as np

class SHADE_restart_local:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Split budget: 90% main DE, 10% local search (multi-start NM)
        local_budget_frac = 0.10
        main_budget = int(budget * (1 - local_budget_frac))
        local_budget = budget - main_budget
        if main_budget < 20:
            # Pure random search if too small
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Initialization ----
        NP_init = max(10, min(100, 18 * int(np.log(dim)) if dim > 1 else 18))
        NP = NP_init

        def latin_hypercube(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = latin_hypercube(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive and memory
        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation detection
        stagnation_limit = max(100, int(0.05 * main_budget))
        best_f_prev = self.best_f
        no_improve_count = 0
        restart_count = 0
        max_restarts = 3

        # ---- Main DE loop ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals

            # Stagnation check and restart
            if no_improve_count >= stagnation_limit and restart_count < max_restarts and fevals > 0.2*main_budget:
                # Reinitialize population except best, reset memory
                restart_count += 1
                NP = NP_init
                pop = latin_hypercube(NP, dim, lb, ub)
                # Insert best known solution
                pop[0] = self.best_x.copy()
                pop[1:] = latin_hypercube(NP-1, dim, lb, ub)
                fitness = np.array([func(x) for x in pop])
                fevals += NP
                # Reset archive and memory
                archive = np.empty((0, dim))
                max_archive = NP
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()
                no_improve_count = 0
                best_f_prev = self.best_f
                continue

            # Linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0., 1.)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Boundary handling: reflect then random
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            # Update stagnation counter
            if self.best_f < best_f_prev - 1e-12:
                no_improve_count = 0
                best_f_prev = self.best_f
            else:
                no_improve_count += 1

            if fevals >= main_budget:
                break

            # Update memory
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Multi-start Nelder-Mead local search ----
        if local_budget > 0:
            # Use top 3 individuals as starting points
            num_starts = min(3, len(pop))
            start_indices = np.argsort(fitness)[:num_starts]
            budget_per_nm = max(1, local_budget // num_starts)

            for idx in start_indices:
                if budget_per_nm <= 0:
                    break
                x0 = pop[idx].copy() if idx != best_idx else self.best_x.copy()
                f0 = fitness[idx] if idx != best_idx else self.best_f

                step = 0.05 * (ub - lb)
                n = dim
                simplex = np.zeros((n + 1, n))
                simplex[0] = x0
                for i in range(n):
                    simplex[i + 1] = x0.copy()
                    simplex[i + 1][i] += step[i]
                simplex = np.clip(simplex, lb, ub)

                fvals = np.array([func(x) for x in simplex])
                evals = n + 1

                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                best_f_local = fvals[0]
                best_x_local = simplex[0].copy()

                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma = 0.5

                while evals < budget_per_nm:
                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, lb, ub)
                    fr = func(xr)
                    evals += 1

                    if fr < fvals[0]:
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, lb, ub)
                        fe = func(xe)
                        evals += 1
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        if fr < fvals[-1]:
                            xc = centroid + rho * (centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc <= fr:
                                simplex[-1] = xc
                                fvals[-1] = fc
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    fvals[i] = func(simplex[i])
                                    evals += 1
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc < fvals[-1]:
                                simplex[-1] = xc
                                fvals[-1] = fc
                            else:
                                for i in range(1, n + 1):
                                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                    simplex[i] = np.clip(simplex[i], lb, ub)
                                    fvals[i] = func(simplex[i])
                                    evals += 1

                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    if fvals[0] < best_f_local:
                        best_f_local = fvals[0]
                        best_x_local = simplex[0].copy()

                if best_f_local < self.best_f:
                    self.best_f = best_f_local
                    self.best_x = best_x_local.copy()

        return self.best_f, self.best_x