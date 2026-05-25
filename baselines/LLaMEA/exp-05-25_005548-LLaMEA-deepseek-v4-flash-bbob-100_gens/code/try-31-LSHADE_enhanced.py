import numpy as np

class LSHADE_enhanced:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Initial population size (larger for higher dimensions)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init
        H = 15  # larger memory for F/CR
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart parameters
        restart_period = max(200, budget // 10)
        best_no_improve = 0

        # Feasibility helper
        def repair(x):
            x = np.clip(x, lb, ub)
            return x

        # If budget too small, just random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population (quasi-random Sobol-like)
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP
        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()
        archive = np.empty((0, dim))

        # Main loop
        while fevals < budget:
            prev_best = self.best_f

            # Linear population reduction (NP -> 4)
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    # Keep diverse archive: remove random individuals
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio
            ratio = 0.2 - 0.12 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR and F from Cauchy truncated
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from union of pop and archive
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx != i and idx != r1:
                        break
                r2 = combined[idx] if idx < NP else archive[idx - NP]

                # Mutation with optional exponential crossover
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

                # Crossover: binomial with occasional exponential
                if np.random.rand() < 0.1:
                    # Exponential crossover
                    u = pop[i].copy()
                    L = 1
                    while np.random.rand() < CR and L < dim:
                        L += 1
                    start = np.random.randint(dim)
                    for j in range(dim):
                        if (j - start) % dim < L:
                            u[j] = v[j]
                else:
                    u = pop[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]

                # Bound handling: reflect then random if still out
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                # If still out, project to feasible random point
                still_low = u < lb
                still_high = u > ub
                if np.any(still_low):
                    u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                if np.any(still_high):
                    u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove a random element to maintain archive size
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

            # Update memory for F/CR
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Restart if no improvement for long time
            if self.best_f < prev_best:
                best_no_improve = 0
            else:
                best_no_improve += NP
            if best_no_improve > restart_period and fevals < budget - 500:
                # Restart: keep best solution, reinitialize population
                NP = NP_init
                pop = np.random.uniform(lb, ub, (NP, dim))
                fitness = np.array([func(x) for x in pop])
                fevals += NP
                # Update best if any new point is better
                min_idx = np.argmin(fitness)
                if fitness[min_idx] < self.best_f:
                    self.best_f = fitness[min_idx]
                    self.best_x = pop[min_idx].copy()
                archive = np.empty((0, dim))
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                best_no_improve = 0
                # Update NP for future reduction
                NP_init = NP
                max_archive = NP
                continue  # restart loop, will reduce again

        return self.best_f, self.best_x