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

        # --- Latin Hypercube sampling for initial population ---
        def latin_hypercube(n, d, low, high):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = low[j] + (perm + np.random.uniform(size=n)) / n * (high[j] - low[j])
            return samples

        # initial population size: slightly larger than standard LSHADE
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP_init = min(NP_init, budget // 2)  # avoid too large population
        NP = NP_init
        max_archive = NP_init

        # handle tiny budget
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # initial population with LHS
        pop = latin_hypercube(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # archive (empty initially)
        archive = np.empty((0, dim))

        # SHADE memory (increased size)
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # pbest ratio starts at 0.1, can adapt
        p_best_init = 0.1

        # Exponential reduction exponent
        remaining_budget = budget - fevals
        while fevals < budget:
            # --- Exponential population size reduction ---
            remaining_budget = budget - fevals
            ratio = remaining_budget / budget
            NP_new = max(4, int(4 + (NP_init - 4) * (ratio ** 0.5)))  # sqrt gives slower reduction early
            if NP_new < NP:
                # sort by fitness and keep best
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # pbest selection (adaptive: decrease with generations)
            p_best = p_best_init - 0.05 * (1 - remaining_budget / budget)  # from 0.1 to 0.05
            p_best = max(0.05, min(0.2, p_best))
            pbest_num = max(1, int(p_best * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # select memory index randomly
                r = np.random.randint(H)
                # sample CR from truncated Cauchy (scale 0.1)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F from Cauchy
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # choose pbest (random from top)
                pbest = pop[np.random.choice(pbest_pool)]

                # choose r1 different from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # choose r2 from pop ∪ archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                n_combined = len(combined)
                idx = np.random.randint(n_combined)
                # rejection until valid
                while idx == i or idx == r1 or (idx >= NP and idx - NP == i): # extra check for archive
                    idx = np.random.randint(n_combined)
                r2_vec = combined[idx]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
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

                    # archive update
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_remove = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_remove, axis=0)

                    # update best
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

            # update memory if successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x