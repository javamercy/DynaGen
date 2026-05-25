import numpy as np

class LSHADE_enhanced:
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

        # Initial population size
        NP_init = max(10, int(18 * np.log(dim)) if dim > 1 else 18)
        NP = NP_init
        max_archive = NP_init

        # If budget too small, random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin hypercube initialisation
        def lhs(n, d, lb, ub):
            points = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                points[:, j] = (perm + np.random.uniform(size=n)) / n
            return lb + points * (ub - lb)

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation detection
        stagnation_counter = 0
        max_stagnation = max(50, int(0.02 * budget))
        last_improvement_fevals = 0

        # Success rate tracking for memory adaptation
        success_rates = []

        while fevals < budget:
            remaining = budget - fevals

            # Linear population reduction (NP from NP_init to max(4, dim))
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)[:NP_new]
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                NP = NP_new
                if len(archive) > NP:
                    archive = archive[np.random.choice(len(archive), size=NP, replace=False)]
                max_archive = NP

            # Adaptive pbest ratio: decreasing from 0.2 to 0.05
            ratio = 0.2 - 0.15 * (1 - remaining / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []
            success_count = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Parameter selection
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0.0, 1.0)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # Choose pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Combine pop and archive for r2
                combined = np.vstack((pop, archive))
                while True:
                    idx2 = np.random.randint(len(combined))
                    if idx2 != i and idx2 != r1:
                        break
                if idx2 < NP:
                    r2 = combined[idx2]
                else:
                    r2 = archive[idx2 - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                # Evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))
                    success_count += 1

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive (parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = archive[np.random.choice(len(archive), size=max_archive, replace=False)]

                    # Update global best
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stagnation_counter = 0
                        last_improvement_fevals = fevals
                    else:
                        stagnation_counter += 1

                if fevals >= budget:
                    break

            # Apply stagnation restart if needed
            if (fevals - last_improvement_fevals) > max_stagnation and fevals < budget - NP:
                # Restart: keep best and replace half of population
                stagnation_counter = 0
                last_improvement_fevals = fevals
                keep_num = max(1, NP // 4)
                sorted_idx = np.argsort(fitness)[:keep_num]
                best_parts = pop[sorted_idx]
                best_fits = fitness[sorted_idx]
                # Keep archive size moderate
                archive = np.empty((0, dim))
                # Generate new random individuals
                new_inds = lhs(NP - keep_num, dim, lb, ub)
                new_fits = np.array([func(x) for x in new_inds])
                fevals += len(new_inds)
                pop = np.vstack((best_parts, new_inds))
                fitness = np.concatenate((best_fits, new_fits))
                # Reset memory partially
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                continue

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory with weighted averages
            if len(S_CR) > 0:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x