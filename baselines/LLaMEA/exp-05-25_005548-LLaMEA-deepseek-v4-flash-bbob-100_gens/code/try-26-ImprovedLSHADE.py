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

        # ----- initialization -----
        # Latin Hypercube Sampling for better initial coverage
        NP_init = max(10, int(18 * np.log(dim))) if dim > 1 else 18
        NP_init = min(NP_init, budget // 2)  # ensure at least half budget for evolution
        NP = NP_init
        max_archive = NP_init

        # generate LHS points
        pop = np.zeros((NP, dim))
        for j in range(dim):
            perm = np.random.permutation(NP)
            for i in range(NP):
                pop[i, j] = lb[j] + (ub[j] - lb[j]) * (perm[i] + np.random.uniform()) / NP
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # archive
        archive = np.empty((0, dim))

        # SHADE memory
        H = 6
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # stagnation detection
        stagnation_limit = max(5, int(0.15 * budget))  # evaluations without improvement
        last_improve_evals = fevals

        # main loop
        while fevals < budget:
            # linear population reduction: current NP depends on remaining evaluations
            remaining = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # adjust archive size
                if len(archive) > max_archive:
                    archive = archive[:max_archive]

            # adaptive pbest parameter (from jSO)
            p = 0.2 * (1 - (fevals / budget))  # linearly decreases from 0.2 to 0
            p = max(0.05, p)  # floor at 0.05
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # success storage
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            any_improvement = False

            for i in range(NP):
                # sample parameters from memory
                r = np.random.randint(H)
                # CR from Cauchy truncated
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # F from Cauchy (scale 0.1) with location M_F
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # select donors
                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
                # use a simple loop to avoid duplicates (population small enough)
                # combine and select
                combined = np.vstack((pop, archive))
                # ensure we get a unique index not equal to i or r1 (if they are in pop part)
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # if idx >= NP, it's archive; no direct conflict
                    break
                if idx < NP:
                    r2 = combined[idx]
                else:
                    r2 = combined[idx]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

                # binomial crossover
                j_rand = np.random.randint(dim)
                u = pop[i].copy()
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
                    any_improvement = True

                    # add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improve_evals = fevals

                if fevals >= budget:
                    break

            # update population
            pop[:] = new_pop
            fitness[:] = new_fitness

            if fevals >= budget:
                break

            # update memory if there were successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_w = np.sum(w * np.array(S_F))
                sum_sq = np.sum(w * np.array(S_F)**2)
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # stagnation detection and restart
            if (fevals - last_improve_evals) > stagnation_limit and not any_improvement:
                # reinitialize half the population (except best)
                num_restart = max(1, int(NP * 0.5))
                worst_indices = np.argsort(fitness)[-num_restart:]  # worst individuals
                for idx in worst_indices:
                    # generate random point in the whole space, but possibly with a small bias toward best
                    pop[idx] = np.random.uniform(lb, ub, dim)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fevals >= budget:
                        break
                last_improve_evals = fevals  # reset stagnation counter after restart
                # clear archive to allow diversity
                archive = np.empty((0, dim))
                # update best if any restart point improved
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()

        return self.best_f, self.best_x