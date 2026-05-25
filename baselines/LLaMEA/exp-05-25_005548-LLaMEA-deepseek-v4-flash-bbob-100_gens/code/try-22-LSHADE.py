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

        # initial population size from LSHADE (18 log dim, min 10)
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP_init  # archive size equal to current NP (will update later)

        # insufficient budget -> random search
        if budget < NP:
            for i in range(budget):
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

        # archive (empty)
        archive = np.empty((0, dim))

        # SHADE memory (larger H=10 for better adaptation)
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # main loop
        while fevals < budget:
            # linear population reduction: target NP linearly to 4
            remaining_evals = budget - fevals
            NP_target = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_target < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_target]]
                fitness = fitness[sorted_idx[:NP_target]]
                NP = NP_target
                # resize archive to match new NP (max archive = NP)
                if len(archive) > NP:
                    archive = archive[-NP:]  # keep last NP entries (oldest? better: random)
                    # Actually random removal is safer:
                    # archive = archive[np.random.choice(len(archive), NP, replace=False)]
                max_archive = NP

            # dynamic pbest ratio: from 0.2 to 0.05 linearly
            p = 0.2 - 0.15 * (1 - remaining_evals / budget)
            p = max(0.05, p)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # success data
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # sample CR from Gaussian clipped to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F from Cauchy truncated to [0,1]
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                # retry until positive
                iteration = 0
                while F <= 0. and iteration < 100:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                    iteration += 1
                F = min(F, 1.)

                # pbest selection
                pbest = pop[np.random.choice(pbest_pool)]

                # select r1 ≠ i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # select r2 from pop∪archive, differ from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    if idx < NP:
                        r2_vec = combined[idx]
                    else:
                        r2_vec = archive[idx - NP]
                    break

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                u = np.clip(u, lb, ub)

                # evaluation
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    # square the delta to weight larger improvements more
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(delta ** 2)  # squared weight

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # remove random entry
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            # update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory if there were successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5

                # smoothed update with learning rate 0.1 to avoid overfitting
                M_CR[mem_idx] = (1 - 0.1) * M_CR[mem_idx] + 0.1 * mean_CR
                M_F[mem_idx] = (1 - 0.1) * M_F[mem_idx] + 0.1 * mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x