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

        # initial population size - slightly larger than classical LSHADE
        NP_init = max(10, int(4 + dim * 2.5))
        NP = NP_init

        # archive size: proportional to current NP
        max_archive = NP
        # stagnation detection
        stagnation_limit = max(50, int(0.1 * budget))
        no_improve_count = 0
        prev_best = np.inf

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
        prev_best = self.best_f

        # archive
        archive = np.empty((0, dim))

        # SHADE memory
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # main loop
        while fevals < budget:
            # linear population reduction
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                # sort and keep best
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                max_archive = NP

            # adaptive pbest size: starts at 20% and decreases to 10%
            p = max(0.1, 0.2 - 0.1 * (fevals / budget))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # success memories
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # choose memory index
                r = np.random.randint(H)
                # sample CR from normal, truncated to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)
                # sample F from Cauchy, truncate to >0 and <=1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # select r1 ≠ i
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # select r2 from pop ∪ archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                if idx < NP:
                    r2 = combined[idx]
                else:
                    r2 = archive[idx - NP]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

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

            # stagnation detection and restart
            if self.best_f < prev_best - 1e-12:
                no_improve_count = 0
                prev_best = self.best_f
            else:
                no_improve_count += NP  # increment by population size

            if no_improve_count >= stagnation_limit and fevals < budget - NP:
                # partial restart: replace worst 30% of population with random points
                n_replace = max(1, int(0.3 * NP))
                worst_idx = np.argsort(fitness)[-n_replace:]
                for idx in worst_idx:
                    new_pop[idx] = np.random.uniform(lb, ub)
                    # evaluate new point immediately
                    new_fitness[idx] = func(new_pop[idx])
                    fevals += 1
                    if new_fitness[idx] < self.best_f:
                        self.best_f = new_fitness[idx]
                        self.best_x = new_pop[idx].copy()
                    if fevals >= budget:
                        break
                if fevals >= budget:
                    break
                pop = new_pop
                fitness = new_fitness
                # reset stagnation counter
                no_improve_count = 0
                # also reset memory by perturbing M_CR and M_F
                for j in range(H):
                    M_CR[j] = np.random.uniform(0.3, 0.8)
                    M_F[j] = np.random.uniform(0.3, 0.9)
                mem_idx = 0

            if fevals >= budget:
                break

            # update memory if successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x