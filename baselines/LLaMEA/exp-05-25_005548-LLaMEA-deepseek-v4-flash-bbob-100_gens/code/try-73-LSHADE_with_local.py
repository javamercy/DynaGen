import numpy as np

class LSHADE_with_local:
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

        # Reserve budget for local search (sep-CMA-ES)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling ----
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
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

        # Stagnation tracking
        best_no_improve = 0
        stagnation_limit = max(100, int(0.15 * main_budget / NP_init))

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
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = max(NP, len(archive))

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
                r2_vec = combined[idx]

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
                        best_no_improve = 0
                    else:
                        best_no_improve += 1
                else:
                    best_no_improve += 1

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Parameter adaptation
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / (sum_w + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Restart on stagnation
            if best_no_improve > stagnation_limit and fevals < main_budget * 0.8:
                best_no_improve = 0
                # Keep the best solution, replace worst half with LHS samples
                n_replace = max(1, NP // 2)
                worst_idx = np.argsort(fitness)[-n_replace:]
                new_samples = lhs(n_replace, dim, lb, ub)
                for idx, x in zip(worst_idx, new_samples):
                    pop[idx] = x
                    fitness[idx] = func(x)
                    fevals += 1
                    if fevals >= main_budget:
                        break
                # Reset memories (optional) - keep archive but clear it
                archive = np.empty((0, dim))
                max_archive = NP

        # ---- sep-CMA-ES local search ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f

            # Initialize sep-CMA-ES parameters
            sigma = 0.3 * (ub - lb).mean()
            mean = x_best.copy()
            pc = np.zeros(dim)
            B = np.eye(dim)  # identity for separable
            D = np.ones(dim)
            C = np.eye(dim)  # diagonal only
            invsqrtC = np.eye(dim)

            # Strategy constants
            n = dim
            lambd = 4 + int(3 * np.log(n))
            mu = lambd // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / (weights ** 2).sum()
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs

            evals = 0
            while evals < local_budget:
                # Sample lambda offspring
                arz = np.random.randn(lambd, n)
                arx = mean + sigma * (arz * D)  # separable: D scales each dimension
                arx = np.clip(arx, lb, ub)
                ary = np.array([func(x) for x in arx])
                evals += lambd

                # Sort
                sorted_idx = np.argsort(ary)
                arx = arx[sorted_idx]
                ary = ary[sorted_idx]

                # Update mean
                old_mean = mean.copy()
                mean = weights @ arx[:mu]

                # Update evolution paths
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (mean - old_mean) / sigma
                hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (2 * evals / lambd)) / (1.4 + 2 / (n + 1))) < 1
                # Update D (diagonal variances) - for separable CMA
                # Use rank-one update on diagonal
                delta = (mean - old_mean) / sigma
                dC = np.zeros(n)
                for i in range(mu):
                    dC += weights[i] * (arz[sorted_idx[i]] ** 2 - 1.0)
                dC *= cmu
                D2 = D ** 2
                D2 = D2 * (1 - c1 - cmu) + c1 * (pc ** 2) + dC
                D2 = np.maximum(D2, 1e-20)
                D = np.sqrt(D2)

                # Step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * B @ np.linalg.solve(B @ np.diag(D), (mean - old_mean) / sigma)
                sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps) / np.sqrt(n) - 1))

                # Check for improvement
                if ary[0] < f_best:
                    f_best = ary[0]
                    x_best = arx[0].copy()
                    if f_best < self.best_f:
                        self.best_f = f_best
                        self.best_x = x_best.copy()

                # Early stop if sigma too small
                if sigma < 1e-12 * (ub - lb).mean():
                    break

            # Final evaluation of local best if not already done
            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best

        return self.best_f, self.best_x