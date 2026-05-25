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

        # allocate budget: 80% main DE, 20% local search (mini-CMA-ES)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Initialization ----
        NP_init = max(10, 20 * int(np.log(dim)) if dim > 1 else 20)
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

        archive = np.empty((0, dim))
        max_archive = NP
        H = 50  # increased history size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main jSO-inspired DE loop with rank-based weighting ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # linear population size reduction
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

            # adaptive pbest ratio (jSO style)
            ratio = 0.25 - 0.20 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_rank = []  # rank of improvement for weighting

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

                # reflected boundary handling
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
                    # use rank based on delta f (higher delta -> higher weight)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_rank.append(delta)
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

            # update memory with rank-based weighting (Lehmer for F, weighted arithmetic for CR)
            if S_CR:
                w = np.array(S_rank) / np.sum(S_rank)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / (sum_w + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Enhanced Local Search: mini-CMA-ES ----
        if local_budget > 0:
            # restart if previous best can be improved
            for restart in range(3):
                if local_budget <= 0:
                    break
                x_mean = self.best_x.copy()
                f_mean = self.best_f
                # initial step size = 0.1 * domain range
                sigma = 0.1 * np.mean(ub - lb)
                # population size for CMA-ES
                lam = max(6, 4 + int(3 * np.log(dim)))
                # weights for recombination
                w = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
                w = w / np.sum(w)
                mu = lam // 2
                weights = w[:mu]
                mueff = 1.0 / np.sum(weights ** 2)
                # adaptation parameters
                cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
                cs = (mueff + 2) / (dim + mueff + 5)
                c1 = 2 / ((dim + 1.3) ** 2 + mueff)
                cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
                damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

                pc = np.zeros(dim)
                ps = np.zeros(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                C = B @ np.diag(D ** 2) @ B.T
                invsqrtC = B @ np.diag(1 / D) @ B.T
                eigeneval = 0

                evals_local = 0
                gen = 0
                while evals_local < local_budget:
                    # sample new points
                    arz = np.random.randn(lam, dim)
                    arx = x_mean + sigma * (arz @ (B * D).T)
                    # boundary handling: clamp
                    arx = np.clip(arx, lb, ub)
                    # evaluate
                    ary = np.array([func(x) for x in arx])
                    evals_local += lam
                    fevals += lam

                    # sort
                    order = np.argsort(ary)
                    ary = ary[order]
                    arx = arx[order]

                    # update mean
                    xold = x_mean.copy()
                    x_mean = np.sum(weights[:, None] * arx[:mu], axis=0)

                    # update evolution paths
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (x_mean - xold) / sigma
                    hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1))
                    pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - xold) / sigma

                    # update covariance matrix
                    artmp = (arx[:mu] - xold).T / sigma
                    delta = (1 - hsig) * cc * (2 - cc)
                    C = (1 - c1 - cmu) * C + c1 * (pc[:, None] @ pc[None, :] + delta * C) + \
                        cmu * (artmp @ np.diag(weights) @ artmp.T)

                    # update step size
                    sigma *= np.exp((cs / damps) * (np.linalg.norm(ps) / (np.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) - 1))

                    # eigen decomposition if needed
                    if gen - eigeneval > dim / (c1 + cmu) / 10:
                        eigeneval = gen
                        D, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D, 1e-20))
                        invsqrtC = B @ np.diag(1 / D) @ B.T

                    # update best found
                    if ary[0] < f_mean:
                        f_mean = ary[0]
                        self.best_f = ary[0]
                        self.best_x = arx[0].copy()

                    gen += 1
                    if evals_local >= local_budget:
                        break

                # reduce remaining local budget for possible restarts
                local_budget -= evals_local
                if local_budget <= 0:
                    break

        return self.best_f, self.best_x