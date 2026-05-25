import numpy as np

class LSHADE_advanced:
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

        # Population size: smaller for high dimensions, but at least 10
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        # Handle tiny budgets
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling for initial population
        def lhs_sample(n, d, low, high):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = (perm + np.random.uniform(size=n)) / n
            return low + samples * (high - low)

        pop = lhs_sample(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        H = 15  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Success rate tracker
        success_rate = 0.5
        generation = 0

        while fevals < budget:
            generation += 1
            remaining = budget - fevals

            # Non-linear population reduction: keep large pop longer
            alpha = 0.4  # shape parameter
            ratio = remaining / budget
            NP_new = max(4, int(4 + (NP_init - 4) * (ratio ** alpha)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: decreases non-linearly
            p_base = 0.2 - 0.1 * (ratio ** 0.5)
            # Scale by success rate: if high success, smaller p (more exploitation)
            p = p_base * (1.0 - 0.3 * success_rate)
            p = max(0.05, min(0.25, p))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []
            success_count = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # Cauchy scale decays over generations
            cauchy_scale = 0.1 * ratio + 0.05

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy sampling for CR and F with decaying scale
                CR = np.random.standard_cauchy() * cauchy_scale + M_CR[r]
                CR = np.clip(CR, 0., 1.)
                F = np.random.standard_cauchy() * cauchy_scale + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * cauchy_scale + M_F[r]
                F = min(F, 1.)

                # pbest selection
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

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

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)
                    success_count += 1

                    new_pop[i] = u
                    new_fitness[i] = f_u

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

            # Update success rate with exponential smoothing
            gen_success = success_count / max(1, NP)
            success_rate = 0.7 * success_rate + 0.3 * gen_success

            if fevals >= budget:
                break

            # Update memory with weighted averages (same as SHADE)
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