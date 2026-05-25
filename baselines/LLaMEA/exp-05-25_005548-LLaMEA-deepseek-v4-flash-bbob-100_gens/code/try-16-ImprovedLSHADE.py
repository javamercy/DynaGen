import numpy as np

class ImprovedLSHADE:
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

        # initial population size (as in LSHADE)
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        # archive size enlarged to 2*NP_init
        max_archive = 2 * NP_init

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

        # archive
        archive = np.empty((0, dim))

        # SHADE memory
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # stagnation detection
        best_unchanged_evals = 0
        best_f_last = self.best_f

        # main loop
        while fevals < budget:
            # compute current evaluation progress ratio
            eval_ratio = fevals / budget

            # linear population reduction
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # dynamic pbest ratio: linear from 0.2 to 0.05
            p_min, p_max = 0.05, 0.2
            p = p_min + (p_max - p_min) * (1 - eval_ratio)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # memory sampling
                r = np.random.randint(H)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 distinct from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = archive[idx - NP]

                # current-to-pbest/1 mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
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

                    # add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
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
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # stagnation restart
            if self.best_f < best_f_last:
                best_unchanged_evals = 0
                best_f_last = self.best_f
            else:
                best_unchanged_evals += NP  # roughly one generation

            if best_unchanged_evals > 0.15 * budget and fevals < budget - 10:
                # restart: keep best, reinitialize population and archive
                best_f_last = self.best_f
                best_unchanged_evals = 0
                # random new population (except keep best)
                pop = np.random.uniform(lb, ub, (NP, dim))
                # keep best individual
                pop[0] = self.best_x.copy()
                for i in range(NP):
                    if i == 0:
                        fitness[i] = self.best_f
                    else:
                        fitness[i] = func(pop[i])
                        fevals += 1
                        if fevals >= budget:
                            break
                # clear archive
                archive = np.empty((0, dim))
                # reset memory
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0

        return self.best_f, self.best_x