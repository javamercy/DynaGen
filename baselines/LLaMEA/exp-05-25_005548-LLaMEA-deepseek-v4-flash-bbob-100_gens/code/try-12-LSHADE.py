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

        # initial population size
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # early exit if budget too small
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

        # stagnation handling
        stagnation_limit = max(0.1 * budget, 50 * dim)
        stagnation_counter = 0
        restart_count = 0
        max_restarts = 2

        while fevals < budget:
            # --- restart if stagnation and budget allows ---
            remaining_evals = budget - fevals
            if stagnation_counter >= stagnation_limit and restart_count < max_restarts and remaining_evals > NP_init:
                # keep the best individual, reinitialize the rest
                new_pop = [self.best_x.copy()] + [np.random.uniform(lb, ub, dim) for _ in range(NP_init - 1)]
                pop = np.array(new_pop)
                # evaluate new individuals (best already known)
                new_fitness = [self.best_f] + [func(x) for x in pop[1:]]
                fevals += NP_init - 1
                fitness = np.array(new_fitness)
                NP = NP_init
                # reset memory and archive
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                archive = np.empty((0, dim))
                stagnation_counter = 0
                restart_count += 1
                # update best if any new point is better (unlikely)
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()
                continue

            # --- linear population size reduction ---
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # trim archive size if needed
                if len(archive) > max_archive:
                    archive = archive[np.random.choice(len(archive), max_archive, replace=False)]

            # adaptive pbest ratio
            p = 0.1 + 0.1 * (remaining_evals / budget)  # more exploration early
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR, S_F, delta_fitness = [], [], []
            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved = False

            for i in range(NP):
                # sample CR and F
                r = np.random.randint(H)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # select r1 (different from i)
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # select r2 from union of pop and archive
                if len(archive) > 0:
                    combined = np.vstack((pop, archive))
                else:
                    combined = pop
                while True:
                    idx = np.random.randint(len(combined))
                    if idx < NP:
                        if idx == i or idx == r1:
                            continue
                    # archive indices (>= NP) are automatically distinct from i and r1
                    break
                r2_vec = combined[idx]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # boundary reflection (then clamp as safety)
                for j in range(dim):
                    if u[j] < lb:
                        u[j] = 2 * lb - u[j]
                    elif u[j] > ub:
                        u[j] = 2 * ub - u[j]
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
                    archive = np.vstack((archive, pop[i].reshape(1,-1)))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        improved = True

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness
            if fevals >= budget:
                break

            # update stagnation counter
            if improved:
                stagnation_counter = 0
            else:
                stagnation_counter += NP

            # update memory with successful parameters
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