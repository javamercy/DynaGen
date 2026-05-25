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

        # initial population size: standard LSHADE uses 18*dim, but clamped to budget/2
        NP_init = max(10, min(int(18 * dim), budget // 2))
        NP = NP_init
        # archive size equals current population size
        # memory size
        H = 6
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # if budget too small, random search
        if budget < NP:
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # archive (start empty)
        archive = np.empty((0, dim))

        # main loop
        while fevals < budget:
            # Non‑linear population reduction (quadratic)
            remaining = budget - fevals
            NP_target = max(4, int(4 + (NP_init - 4) * (remaining / budget) ** 2))
            if NP_target < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_target]]
                fitness = fitness[sorted_idx[:NP_target]]
                NP = NP_target
                # reduce archive size to current NP
                if len(archive) > NP:
                    # random removal
                    while len(archive) > NP:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

            # pbest selection (constant 0.2)
            p = 0.2
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # success lists
            S_CR, S_F, delta_fitness = [], [], []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # choose memory index
                r = np.random.randint(H)
                # sample CR (normal truncated to [0,1])
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F (Cauchy truncated to (0,1])
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 (different from i)
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # ensure if idx corresponds to archive, it is not equal to i (already covered)
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # boundary handling: reflection
                for j in range(dim):
                    if u[j] < lb:
                        u[j] = lb + (lb - u[j])
                        if u[j] < lb:  # in case reflection still out of bounds
                            u[j] = lb + np.random.rand() * (ub - lb)
                    elif u[j] > ub:
                        u[j] = ub - (u[j] - ub)
                        if u[j] > ub:
                            u[j] = lb + np.random.rand() * (ub - lb)
                u = np.clip(u, lb, ub)  # final safety clip

                # evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # add parent to archive (if space)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > NP:
                        # remove random archived solution
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
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