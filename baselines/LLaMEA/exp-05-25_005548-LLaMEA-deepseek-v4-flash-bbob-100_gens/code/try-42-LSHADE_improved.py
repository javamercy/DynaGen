import numpy as np

class LSHADE_improved:
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

        # Initial population size (jSO style: larger initial pop)
        NP_init = int(10 * dim) if dim > 5 else 40
        NP_init = min(NP_init, 200)  # cap for high dim
        NP = NP_init
        max_archive = NP_init

        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory (increased size)
        H = 15
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation tracking
        stagnation = 0
        prev_best = self.best_f

        while fevals < budget:
            # Nonlinear population reduction (jSO-like: parabolic)
            remaining = budget - fevals
            NP_new = max(4, int(NP_init * (remaining / budget) ** (2/3)))
            NP_new = max(4, min(NP_new, NP_init))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # Trim archive if needed
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: decreases from 0.2 to 0.05
            ratio = 0.2 * (1 - remaining / budget) ** 0.5
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
                # Sample CR and F with increased scale for exploration
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.2 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.2 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.2 + M_F[r]
                F = min(F, 1.)

                # pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # r1 from population (distinct)
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from pop+archive (distinct from i and r1)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx]

                # current-to-pbest/1 mutation
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling, fallback to random near bound
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                # For those still out, sample between parent and bound
                u[still_low] = np.random.uniform(lb[still_low], pop[i][still_low])
                u[still_high] = np.random.uniform(ub[still_high], pop[i][still_high])
                # Ensure safety
                u = np.clip(u, lb, ub)

                # Evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Archive insertion
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            # Replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory with weighted averages
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Simple restart if stagnation (optional, but keep for robustness)
            if self.best_f < prev_best:
                prev_best = self.best_f
                stagnation = 0
            else:
                stagnation += 1
            if stagnation > 10 * NP / dim:
                # Reinitialize 20% of population randomly, keep best
                reinit_num = max(1, int(0.2 * NP))
                worst_idx = np.argsort(fitness)[-reinit_num:]
                for idx in worst_idx:
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fevals >= budget:
                        break
                stagnation = 0
                if fevals >= budget:
                    break

        return self.best_f, self.best_x