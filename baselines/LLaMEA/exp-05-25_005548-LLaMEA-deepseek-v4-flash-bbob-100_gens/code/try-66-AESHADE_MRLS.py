import numpy as np

class AESHADE_MRLS:
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

        # Reserve budget for local search (multi-restart Nelder-Mead)
        local_budget = max(20 * dim, int(0.2 * budget))
        main_budget = budget - local_budget
        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube initialization
        NP_init = max(25, int(25 * np.log(dim) if dim > 1 else 25))
        NP_init = min(NP_init, 200)  # cap for high dimensions
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

        # Archive (double population size for diversity)
        max_archive = 2 * NP
        archive = np.empty((0, dim))

        # Success-history memory
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Strategy adaptation (ensemble of three mutation strategies)
        # 0: current-to-pbest/1 (archive), 1: current-to-rand/1, 2: rand/1 (with archive)
        strat_prob = np.array([0.8, 0.1, 0.1])
        strat_success = np.zeros(3)
        strat_total = np.ones(3) * 1e-10
        strat_idx = 0  # current strategy for selection

        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Linear population size reduction
            NP_new = max(6, int(6 + (NP_init - 6) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > 2 * NP:
                    np.random.shuffle(archive)
                    archive = archive[:2 * NP]
                max_archive = 2 * NP

            # Adaptive pbest ratio (from 0.2 to 0.05)
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = [[], [], []]   # for each strategy
            S_F = [[], [], []]
            delta_fitness = [[], [], []]

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Choose strategy via roulette wheel
                prob = strat_prob / strat_prob.sum()
                strategy = np.random.choice(3, p=prob)

                # Generate CR and F using history
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                if strategy == 0:  # current-to-pbest/1 (with archive)
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
                    r2 = combined[idx] if idx < NP else archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

                elif strategy == 1:  # current-to-rand/1 (no archive)
                    r1, r2 = np.random.choice(NP, 2, replace=False)
                    while r1 == i: r1 = np.random.randint(NP)
                    while r2 == i or r2 == r1: r2 = np.random.randint(NP)
                    v = pop[i] + F * (pop[r1] - pop[r2]) + F * (np.random.rand(dim) * (pop[r1] - pop[r2]))

                else:  # rand/1 (with archive)
                    r1, r2 = np.random.choice(NP, 2, replace=False)
                    while r1 == i: r1 = np.random.randint(NP)
                    while r2 == i or r2 == r1: r2 = np.random.randint(NP)
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx == i or idx == r1 or idx == r2:
                            continue
                        break
                    r3 = combined[idx] if idx < NP else archive[idx - NP]
                    v = pop[r1] + F * (pop[r2] - r3)

                # Crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Boundary handling (reflect to random)
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
                    S_CR[strategy].append(CR)
                    S_F[strategy].append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness[strategy].append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # Update archive with the replaced individual
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                    # Record success for strategy adaptation
                    strat_success[strategy] += 1
                strat_total[strategy] += 1

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update strategy probabilities (exponential moving average)
            for s in range(3):
                if strat_total[s] > 0:
                    sr = strat_success[s] / strat_total[s]
                    strat_prob[s] = 0.9 * strat_prob[s] + 0.1 * sr
            # Reset counters for next generation
            strat_success[:] = 0
            strat_total[:] = 1e-10

            # Update success-history memories per strategy (only strategy 0 used for memory? Actually we update based on all successes)
            # But standard SHADE uses all successful F/CR; we keep that for simplicity
            all_CR = []
            all_F = []
            all_delta = []
            for s in range(3):
                all_CR.extend(S_CR[s])
                all_F.extend(S_F[s])
                all_delta.extend(delta_fitness[s])
            if all_CR:
                w = np.array(all_delta) / np.sum(all_delta)
                mean_CR = np.sum(w * np.array(all_CR))
                # Lehmer mean for F
                sum_sq = np.sum(w * np.array(all_F) ** 2)
                sum_w = np.sum(w * np.array(all_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Multi-restart Nelder-Mead local search using remaining budget ----
        if local_budget > 0:
            candidates = [self.best_x]
            # Add a few random points to increase diversity
            for _ in range(min(3, dim)):
                candidates.append(self.best_x + 0.1 * np.random.uniform(lb - self.best_x, ub - self.best_x))
                candidates[-1] = np.clip(candidates[-1], lb, ub)

            best_local_f = self.best_f
            best_local_x = self.best_x.copy()
            per_restart_budget = local_budget // len(candidates)
            if per_restart_budget > 10:
                for x0 in candidates:
                    # Initialize simplex
                    step = 0.02 * (ub - lb)
                    n = dim
                    simplex = np.zeros((n + 1, n))
                    simplex[0] = x0
                    for i in range(n):
                        simplex[i + 1] = x0.copy()
                        simplex[i + 1][i] += step[i]
                    simplex = np.clip(simplex, lb, ub)
                    fvals = np.array([func(x) for x in simplex])
                    evals = n + 1
                    # Sort
                    order = np.argsort(fvals)
                    simplex = simplex[order]
                    fvals = fvals[order]
                    if fvals[0] < best_local_f:
                        best_local_f = fvals[0]
                        best_local_x = simplex[0].copy()

                    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                    while evals < per_restart_budget:
                        centroid = np.mean(simplex[:-1], axis=0)
                        xr = centroid + alpha * (centroid - simplex[-1])
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr); evals += 1
                        if fr < fvals[0]:
                            xe = centroid + gamma * (centroid - simplex[-1])
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe); evals += 1
                            if fe < fr:
                                simplex[-1] = xe; fvals[-1] = fe
                            else:
                                simplex[-1] = xr; fvals[-1] = fr
                        elif fr < fvals[-2]:
                            simplex[-1] = xr; fvals[-1] = fr
                        else:
                            if fr < fvals[-1]:
                                xc = centroid + rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc); evals += 1
                                if fc <= fr:
                                    simplex[-1] = xc; fvals[-1] = fc
                                else:
                                    for i in range(1, n+1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i]); evals += 1
                            else:
                                xc = centroid - rho * (centroid - simplex[-1])
                                xc = np.clip(xc, lb, ub)
                                fc = func(xc); evals += 1
                                if fc < fvals[-1]:
                                    simplex[-1] = xc; fvals[-1] = fc
                                else:
                                    for i in range(1, n+1):
                                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                        simplex[i] = np.clip(simplex[i], lb, ub)
                                        fvals[i] = func(simplex[i]); evals += 1
                        order = np.argsort(fvals)
                        simplex = simplex[order]; fvals = fvals[order]
                        if fvals[0] < best_local_f:
                            best_local_f = fvals[0]
                            best_local_x = simplex[0].copy()
                    # Update overall best if this restart found better
                    if best_local_f < self.best_f:
                        self.best_f = best_local_f
                        self.best_x = best_local_x.copy()

        return self.best_f, self.best_x