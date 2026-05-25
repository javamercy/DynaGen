import numpy as np

class SHADE_restart:
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

        # Larger initial population for better exploration
        NP_init = max(10, int(25 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP

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

        # Memory for CR and F
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation detection
        stall_counter = 0
        stall_limit = max(100, int(0.1 * budget))
        best_old = np.inf
        diversity_threshold = 1e-4 * (ub - lb).mean()  # per-dim average scale

        while fevals < budget:
            # Stagnation and diversity check -> possible restart
            if stall_counter >= stall_limit and len(pop) > 4:
                # Half population reinitialized around best + random
                half = len(pop) // 2
                # Keep best untouched
                new_indices = np.random.choice(len(pop), half, replace=False)
                # Reinitialize with random points near best (small perturbation) and pure random
                for idx in new_indices:
                    if np.random.rand() < 0.5:
                        # near best
                        pop[idx] = np.clip(self.best_x + 0.1 * (ub - lb) * np.random.randn(dim), lb, ub)
                    else:
                        # pure random
                        pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fitness[idx] < self.best_f:
                        self.best_f = fitness[idx]
                        self.best_x = pop[idx].copy()
                    if fevals >= budget:
                        break
                # Reset stall counter and reduce archive size
                stall_counter = 0
                if len(archive) > len(pop):
                    np.random.shuffle(archive)
                    archive = archive[:len(pop)]
                continue

            # Linear population reduction from NP_init to 4
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < len(pop):
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                if len(archive) > NP_new:
                    np.random.shuffle(archive)
                    archive = archive[:NP_new]
                max_archive = NP_new

            # Adaptive pbest ratio: decreasing from 0.2 to 0.05
            ratio = 0.2 - 0.1 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * len(pop)))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []
            successful = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(len(pop)):
                # Adaptive Cauchy scaling factor (smaller perturbation as budget depletes)
                scale = 0.1 * (1 - fevals / budget) + 0.01  # between 0.01 and 0.1

                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * scale + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * scale + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * scale + M_F[r]
                F = min(F, 1.)

                # pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # r1, r2 distinct
                r1 = np.random.randint(len(pop))
                while r1 == i:
                    r1 = np.random.randint(len(pop))
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2 = combined[idx] if idx < len(pop) else archive[idx - len(pop)]

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

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    successful += 1

                    # Archive parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stall_counter = 0  # reset on improvement
                    else:
                        stall_counter += 1

                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory with weighted averages
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Update stall counter based on best improvement
            if successful == 0:
                stall_counter += len(pop)
            else:
                stall_counter = max(0, stall_counter - 2*successful)  # reduce if successes

            # If best not improved for a while, increase counter
            if self.best_f == best_old:
                stall_counter += 1
            best_old = self.best_f

        return self.best_f, self.best_x