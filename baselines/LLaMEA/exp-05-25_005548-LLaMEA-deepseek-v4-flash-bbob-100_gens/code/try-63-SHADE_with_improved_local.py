import numpy as np

class SHADE_with_improved_local:
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

        # Reserve budget for local search (spread across run)
        local_budget_each = max(10, int(0.02 * budget)) if budget > 200 else 0
        num_local_steps = 5
        total_local_budget = local_budget_each * num_local_steps
        if total_local_budget >= 0.3 * budget:
            num_local_steps = max(1, int(0.2 * budget / local_budget_each))
            total_local_budget = local_budget_each * num_local_steps
        main_budget = budget - total_local_budget
        if main_budget < 100:
            main_budget = budget
            total_local_budget = 0
            num_local_steps = 0

        # ---- Latin Hypercube Sampling for initial population ----
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init

        def lhs(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive for DE
        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # For local search triggering
        eval_counter_since_local = 0
        local_step = 0

        # For restart detection
        stagnate_evals = 0
        best_f_old = np.inf
        no_improve_threshold = max(200, int(0.05 * main_budget))

        # Main DE loop with linear population reduction
        while fevals < main_budget and fevals < budget:
            remaining_evals = main_budget - fevals
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

            # Adaptive pbest ratio from 0.2 to 0.05
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Success history memory
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy sampling for CR
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Cauchy sampling for F, clamp to [0.1, 1]
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.1 or F > 1.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                    F = np.clip(F, 0.1, 1.0)

                # Mutation: current-to-pbest/1 with archive
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
                r2_vec = combined[idx]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
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
                eval_counter_since_local += 1

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

            # Check for stagnation
            if self.best_f >= best_f_old:
                stagnate_evals += NP
            else:
                stagnate_evals = 0
                best_f_old = self.best_f

            if stagnate_evals > no_improve_threshold:
                # Restart: keep best, reinitialize 75% population
                keep = int(0.25 * NP)
                sorted_idx = np.argsort(fitness)
                best_indices = sorted_idx[:keep]
                new_pop = pop[best_indices].copy()
                new_fitness = fitness[best_indices].copy()
                # Generate new random points
                new_pts = lhs(NP - keep, dim, lb, ub)
                new_pop = np.vstack((new_pop, new_pts))
                new_fitness = np.concatenate([new_fitness, np.array([func(x) for x in new_pts])])
                fevals += (NP - keep)
                pop = new_pop
                fitness = new_fitness
                stagnate_evals = 0
                # Reset memory
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0

            # Update success-history memory
            if S_CR and fevals < main_budget:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                # Lehmer mean for F = sum(w * F^2)/ sum(w * F)
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Trigger local search periodically
            if total_local_budget > 0 and local_step < num_local_steps:
                if eval_counter_since_local >= int(main_budget / (num_local_steps + 1)):
                    eval_counter_since_local = 0
                    local_step += 1
                    # Perform local search on current best
                    x0 = self.best_x.copy()
                    f0 = self.best_f
                    step = 0.05 * (ub - lb)
                    n = dim
                    simplex = np.zeros((n + 1, n))
                    simplex[0] = x0
                    for i in range(n):
                        simplex[i + 1] = x0.copy()
                        simplex[i + 1][i] += step[i]
                    simplex = np.clip(simplex, lb, ub)

                    fvals = np.array([func(x) for x in simplex])
                    fevals += (n + 1)

                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    local_evals = n + 1

                    alpha = 1.0
                    gamma = 2.0
                    rho = 0.5
                    sigma = 0.5

                    # Use a smaller local budget per trigger (local_budget_each)
                    while local_evals < local_budget_each and fevals < budget:
                        centroid = np.mean(simplex[:-1], axis=0)

                        # Reflection
                        xr = centroid + alpha * (centroid - simplex[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        fevals += 1
                        local_evals += 1

                        if fr < fvals[0]:
                            # Expansion
                            xe = centroid + gamma * (centroid - simplex[-1])
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            fevals += 1
                            local_evals += 1
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
                                # Outside contraction
                                xc = centroid + rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                fevals += 1
                                local_evals += 1
                                if fc <= fr:
                                    simplex[-1] = xc
                                    fvals[-1] = fc
                                else:
                                    # Shrink
                                    for i in range(1, n + 1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i])
                                        fevals += 1
                                        local_evals += 1
                            else:
                                # Inside contraction
                                xc = centroid - rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc)
                                fevals += 1
                                local_evals += 1
                                if fc < fvals[-1]:
                                    simplex[-1] = xc
                                    fvals[-1] = fc
                                else:
                                    # Shrink
                                    for i in range(1, n + 1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i])
                                        fevals += 1
                                        local_evals += 1

                        order = np.argsort(fvals)
                        simplex = simplex[order]
                        fvals = fvals[order]

                        if fvals[0] < f0:
                            f0 = fvals[0]
                            x0 = simplex[0].copy()
                            # Update global best if improved
                            if f0 < self.best_f:
                                self.best_f = f0
                                self.best_x = x0.copy()

                    # After local search, update best if improved
                    if f0 < self.best_f:
                        self.best_f = f0
                        self.best_x = x0.copy()

            # Check budget
            if fevals >= budget:
                break

        # Final local search with remaining budget (if any left)
        remaining = budget - fevals
        if remaining > 10 and dim < 50:
            # Perform a short Nelder-Mead on the current best
            x0 = self.best_x.copy()
            f0 = self.best_f
            step = 0.05 * (ub - lb)
            n = dim
            simplex = np.zeros((n + 1, n))
            simplex[0] = x0
            for i in range(n):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += step[i]
            simplex = np.clip(simplex, lb, ub)

            fvals = np.array([func(x) for x in simplex])
            fevals += (n + 1)

            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5

            while fevals < budget:
                centroid = np.mean(simplex[:-1], axis=0)
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, lb, ub)
                fr = func(xr)
                fevals += 1
                if fr < fvals[0]:
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, lb, ub)
                    fe = func(xe)
                    fevals += 1
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
                        fevals += 1
                        if fc <= fr:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i])
                                fevals += 1
                    else:
                        xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, lb, ub)
                        fc = func(xc)
                        fevals += 1
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            for i in range(1, n+1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], lb, ub)
                                fvals[i] = func(simplex[i])
                                fevals += 1
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                if fvals[0] < self.best_f:
                    self.best_f = fvals[0]
                    self.best_x = simplex[0].copy()

        return self.best_f, self.best_x