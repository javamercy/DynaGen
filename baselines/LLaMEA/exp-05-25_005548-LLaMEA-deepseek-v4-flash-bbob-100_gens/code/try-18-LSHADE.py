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

        # initial population size: min(4 + 18*log(dim), budget/2)
        NP_init = max(4, int(18 * np.log(dim) if dim > 1 else 18))
        NP_init = min(NP_init, budget // 2)
        NP = NP_init
        NP_min = 4

        # archive size equals population size
        max_archive = NP_init

        if budget < NP:
            # budget too small: random search
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # initial population (uniform)
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # archive
        archive = np.empty((0, dim))

        # SHADE memory
        H = 10  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # main loop
        while fevals < budget:
            # ---- Non‑linear population reduction ----
            remaining = budget - fevals
            factor = (remaining / budget) ** 2  # exponential reduction
            NP_new = max(NP_min, int(NP_min + (NP_init - NP_min) * factor))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # ---- Adaptive pbest rate ----
            # starts at 0.2, linearly decreases to 0.1 over runtime
            p = 0.2 - 0.1 * (1 - remaining / budget)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # lists for successful parameters
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # choose strategy: current-to-pbest/1 with probability 0.8, else current-to-rand/1
                use_pbest = np.random.rand() < 0.8

                # select random memory index
                r = np.random.randint(H)
                # sample CR (normal truncated to [0,1])
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F (Cauchy, scale 0.1, truncated to >0 and <=1.2)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.2)

                if use_pbest:
                    # current-to-pbest/1 with archive
                    pbest = pop[np.random.choice(pbest_pool)]
                    # r1 ≠ i
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    # r2 from pop ∪ archive, distinct from i and r1
                    combined = np.vstack((pop, archive))
                    idx = np.random.randint(len(combined))
                    while idx == i or idx == r1 or (idx < NP and idx == i) or (idx >= NP and (idx - NP) == i):
                        idx = np.random.randint(len(combined))
                    if idx < NP:
                        r2 = combined[idx]
                    else:
                        r2 = archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)
                else:
                    # current-to-rand/1 (no archive, more explorative)
                    r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    while r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    v = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                # clamp to bounds
                u = np.clip(u, lb, ub)

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

                    # update archive (add parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            # replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory if any successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                wF = w
                Fs = np.array(S_F)
                mean_F = np.sum(wF * Fs**2) / np.sum(wF * Fs) if np.sum(wF * Fs) > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x