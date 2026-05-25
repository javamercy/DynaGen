import numpy as np

class SHADE:
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

        # population size: min 10, but use common formula
        NP = max(10, int(4 + 3 * np.log(dim)))
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

        # archive for inferior solutions
        archive = np.empty((0, dim))
        max_archive_size = NP

        # SHADE memory parameters
        H = 6  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        memory_idx = 0  # circular index

        # main loop
        while fevals < budget:
            # pbest selection (top p% individuals)
            p = 0.1
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # lists to store successful parameters
            S_CR = []
            S_F = []
            delta_fitness = []  # for weighting

            # generate trial vectors
            new_pop = np.copy(pop)
            new_fitness = np.copy(fitness)

            for i in range(NP):
                # select random memory index
                r = np.random.randint(H)
                # sample CR from Cauchy distribution
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)
                # sample F from Cauchy distribution
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # choose pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # choose r1 != i
                indices = list(range(NP))
                indices.remove(i)
                r1 = np.random.choice(indices)

                # choose r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    r2 = np.random.randint(len(combined))
                    if r2 != i and r2 != r1:
                        # Note: r2 indexes in combined; careful with archive indices
                        if len(archive) > 0 and r2 >= NP and (r2 - NP) == i:
                            continue
                        break
                # actual index mapping: r1 and i refer to original pop, r2 is index in combined
                mut_base = combined[r2] if r2 < NP else archive[r2 - NP]

                # mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - mut_base)

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
                    # success
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive_size:
                        archive = archive[np.random.choice(len(archive), size=max_archive_size, replace=False)]

                    # update best
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                # else: keep old

                if fevals >= budget:
                    break

            # replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory if any successful parameters
            if S_CR:
                # weighted mean for CR (weights based on delta_f or just arithmetic mean for simplicity)
                # use arithmetic mean to avoid instability
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                mean_F = np.sum(w * np.array(S_F))  # Lehmer mean recommended but arithmetic is simpler
                # For F, use weighted Lehmer mean
                sum_F = np.sum(w * np.array(S_F)**2) + 1e-30
                sum_wF = np.sum(w * np.array(S_F)) + 1e-30
                lehmer_F = sum_F / sum_wF
                # update memory at current index
                M_CR[memory_idx] = mean_CR
                M_F[memory_idx] = lehmer_F
                memory_idx = (memory_idx + 1) % H

        return self.best_f, self.best_x