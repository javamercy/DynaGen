import numpy as np

class LSHADE_CD:
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

        # Reserve budget for coordinate descent local search (few evaluations per dimension)
        local_budget = max(5 * dim, int(0.1 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling for initial population ----
        NP_init = max(10, int(18 * (np.log(dim) if dim > 1 else 0) + 4))
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

        # Archive and memory
        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop with linear population reduction ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (0.2 -> 0.05)
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
                CR = max(0., min(1., CR))
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

            if fevals >= main_budget:
                break

            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Coordinate descent local search using golden-section line searches ----
        if local_budget > 0:
            x = self.best_x.copy()
            f = self.best_f
            evals = 0
            max_cycles = 3  # number of full coordinate cycles
            for _ in range(max_cycles):
                improved = False
                for j in range(dim):
                    # Bounded golden-section search along coordinate j
                    low_j = lb[j]
                    high_j = ub[j]
                    # use best point as initial guess
                    a = low_j
                    b = high_j
                    # ensure x[j] inside [a,b]
                    xj = np.clip(x[j], a, b)
                    # golden ratio
                    phi = (np.sqrt(5) - 1) / 2  # 0.618...
                    tol = 1e-6 * (b - a)
                    # Evaluate at two interior points
                    x1 = b - phi * (b - a)
                    x2 = a + phi * (b - a)
                    # Evaluate f at x1 and x2 (keeping other dimensions fixed)
                    x_trial = x.copy()
                    x_trial[j] = x1
                    f1 = func(x_trial)
                    evals += 1
                    if evals >= local_budget:
                        break
                    x_trial[j] = x2
                    f2 = func(x_trial)
                    evals += 1
                    if evals >= local_budget:
                        break
                    # Iterate until interval small enough or budget exhausted
                    while (b - a) > tol and f1 != f2:
                        if f1 < f2:
                            b = x2
                            x2 = x1
                            f2 = f1
                            x1 = b - phi * (b - a)
                            x_trial[j] = x1
                            f1 = func(x_trial)
                            evals += 1
                        else:
                            a = x1
                            x1 = x2
                            f1 = f2
                            x2 = a + phi * (b - a)
                            x_trial[j] = x2
                            f2 = func(x_trial)
                            evals += 1
                        if evals >= local_budget:
                            break
                    if evals >= local_budget:
                        break
                    # Best point in interval
                    mid = (a + b) / 2
                    x_trial[j] = mid
                    f_mid = func(x_trial)
                    evals += 1
                    # Actually we need to compare f_mid with current f at x (best point)
                    # But we already have evaluations along the line; we can pick the best.
                    # Simpler: after convergence, evaluate at midpoint and accept if better.
                    # But we already used many evaluations; we can directly update x to the best known point along the line.
                    # For simplicity, we evaluate at the midpoint and if better, update.
                    if f_mid < f:
                        f = f_mid
                        x[j] = mid
                        improved = True
                    # else no change (we could also try the best of f1,f2,mid)
                    # The golden-section already converged to a local minimum, so midpoint is likely good.
                if not improved or evals >= local_budget:
                    break
            # Update best solution if improved
            if f < self.best_f:
                self.best_f = f
                self.best_x = x.copy()

        return self.best_f, self.best_x