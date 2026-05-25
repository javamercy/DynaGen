import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        # initial population size (larger than basic LSHADE)
        NP_init = max(10, int(18 * np.sqrt(dim)) if dim > 1 else 18)
        NP = NP_init
        max_archive = NP_init * 2  # larger archive for more diversity

        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # SHADE memories
        H = 8  # a bit larger memory
        M_CR = 0.5 * np.ones(H)
        M_F  = 0.5 * np.ones(H)
        mem_idx = 0

        # main loop
        while fevals < budget:
            remaining = budget - fevals
            # non‑linear population reduction: quadratic decrease
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget) ** 2))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # adaptive pbest rate (decreases linearly from 0.2 to 0.1)
            p = 0.2 - 0.1 * (1.0 - remaining / budget)
            pbest_num = max(1, int(p * NP))

            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR, S_F, delta_fit = [], [], []
            new_pop = pop.copy()
            new_fit = fitness.copy()

            for i in range(NP):
                # memory index
                r = np.random.randint(H)

                # sample CR from normal, truncated to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # sample F from Cauchy, location M_F[r], scale 0.1, force >0, cap at 1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 distinct from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # also avoid using the same vector as pbest? Not enforced, acceptable
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # current-to-pbest/1 mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                # bound clipping
                u = np.clip(u, lb, ub)

                # evaluation
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fit.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fit[i] = f_u

                    # archive update (add parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # remove the worst archived solution (by fitness) – better than random
                        if len(archive) > 0:
                            worst_idx = np.argmax(fitness) if len(archive) <= len(pop) else 0
                            archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fit

            if fevals >= budget:
                break

            # memory update with weighted Lehmer for F and weighted arithmetic for CR
            if S_CR:
                w = np.array(delta_fit) / np.sum(delta_fit)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x