import numpy as np

class LSHADE_RS:
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

        # initial population size (typical LSHADE: 18*log(dim) but with min)
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init

        # archive size factor (original LSHADE uses 2.6)
        arc_rate = 2.6
        max_archive = int(NP * arc_rate)

        # if budget too small, just random search
        if budget < NP:
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
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # stagnation detection
        stagnation_evals = 0
        stagnation_limit = min(budget // 5, max(500 + 100 * dim, 1000))

        # main loop
        while fevals < budget:
            # dynamic pbest ratio: decreases from 0.2 to 0.1 over budget
            p = 0.2 - 0.1 * (fevals / budget)

            # linear population reduction: update NP
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            # sort population by fitness for reduction
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # archive size also shrinks proportionally? keep arc_rate
                max_archive = int(NP * arc_rate)
                if len(archive) > max_archive:
                    archive = archive[np.random.choice(len(archive), max_archive, replace=False)]

            # pbest selection (dynamic p)
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
                # select random memory index
                r = np.random.randint(H)
                # sample CR from Cauchy (normal approx) truncated to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F from Cauchy with scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # choose pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # choose r1 different from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # choose r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # ensure not the same as i (already checked) and r1 (checked)
                    break
                # get r2 vector
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = archive[idx - NP]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

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
                stagnation_evals += 1

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
                        # remove random element if archive exceeds size
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # update best and reset stagnation counter
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stagnation_evals = 0

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
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # check stagnation and restart if needed
            if stagnation_evals >= stagnation_limit and remaining_evals > NP * 2:
                # restart: keep best, generate new random individuals
                NP_restart = NP  # keep current population size
                # create new population with NP-1 random points, keep best
                new_pop = [self.best_x] + [np.random.uniform(lb, ub) for _ in range(NP_restart - 1)]
                new_pop = np.array(new_pop)
                # evaluate new individuals (except best already evaluated)
                for idx in range(1, NP_restart):
                    f = func(new_pop[idx])
                    fevals += 1
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = new_pop[idx].copy()
                fitness = np.array([self.best_f] + [func(x) for x in new_pop[1:]])  # but best already evaluated
                # correct: best already known, assign dummy best for index 0, but we know its fitness
                fitness[0] = self.best_f
                pop = new_pop
                # reset archive
                archive = np.empty((0, dim))
                # reset memory
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                # reset stagnation counter
                stagnation_evals = 0

        return self.best_f, self.best_x