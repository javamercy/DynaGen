import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def exponential_crossover(self, donor, target, CR):
        dim = len(target)
        start = np.random.randint(dim)
        L = 0
        while True:
            L += 1
            if np.random.rand() >= CR or L >= dim:
                break
        child = target.copy()
        for j in range(dim):
            idx = (start + j) % dim
            if j < L:
                child[idx] = donor[idx]
        return child

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget

        # initial population size (LSHADE standard: 18*dim)
        NP_init = max(10, int(18 * dim))
        if budget < NP_init:
            NP_init = max(4, budget // 2)
        NP = NP_init
        max_archive = NP_init  # archive size

        # if budget too small, random search
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

        archive = np.empty((0, dim))

        # SHADE memory
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # stagnation tracking
        stagnation_evals = 0
        last_improvement_evals = fevals

        # main loop
        while fevals < budget:
            remaining_evals = budget - fevals
            # linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # adaptive pbest rate (0.2 -> 0.1 linearly)
            p = 0.2 - 0.1 * (1.0 - fevals / budget)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # memory index
                r = np.random.randint(H)
                # generate CR from Cauchy (approximated with normal) truncated to [0,1]
                CR = min(max(np.random.normal(M_CR[r], 0.1), 0.0), 1.0)
                # generate F from Cauchy with scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 random different from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # if idx corresponds to archive element, check it's not same as i (already)
                    # also ensure that if idx >= NP, it's archive, but i is < NP, so fine
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # crossover: binomial with 50% chance of exponential crossover
                if np.random.rand() < 0.5:
                    u = self.exponential_crossover(v, pop[i], CR)
                else:
                    u = pop[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]
                u = np.clip(u, lb, ub)

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # update archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improvement_evals = fevals

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory with successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # restart if stagnation (no improvement for 10% of budget evaluations)
            if fevals - last_improvement_evals > 0.1 * budget:
                # reinitialize half of population (except best)
                num_restart = NP // 2
                # keep best individual
                sorted_idx = np.argsort(fitness)
                best_idx = sorted_idx[0]
                worst_indices = sorted_idx[-num_restart:]
                for idx in worst_indices:
                    if idx == best_idx:
                        continue
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fevals >= budget:
                        break
                # reset stagnation counter
                last_improvement_evals = fevals
                # also clear archive to avoid bias
                archive = np.empty((0, dim))

        return self.best_f, self.best_x