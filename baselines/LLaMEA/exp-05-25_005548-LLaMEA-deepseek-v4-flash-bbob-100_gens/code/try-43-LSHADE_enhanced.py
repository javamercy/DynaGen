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

        # Initial population size (slightly larger than classic LSHADE)
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP_init

        # Budget too small -> pure random search
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

        # Parameter memory – increased size for better history
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation detection
        stagnation_counter = 0
        last_improvement_eval = 0

        # Main loop
        while fevals < budget:
            # Cubic population reduction: stays larger for longer
            remaining = budget - fevals
            NP_new = max(4, int(round(4 + (NP_init - 4) * (1 - (remaining / budget) ** 2))))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: quadratic decrease
            ratio = 0.2 - 0.15 * (1 - remaining / budget) ** 2
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
                # Sample CR and F from truncated Cauchy
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]

                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Combine pop and archive for r2
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
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improvement_eval = fevals

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory with success history
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Restart if stagnation
            stagnation_counter = fevals - last_improvement_eval
            if stagnation_counter > 0.1 * budget and fevals < 0.9 * budget:
                # Save best solution and reinitialize population except best
                best_x_saved = self.best_x.copy()
                best_f_saved = self.best_f
                pop = np.random.uniform(lb, ub, (NP, dim))
                # Inject best into population
                pop[0] = best_x_saved
                fitness = np.array([func(x) for x in pop[1:]])
                fevals += len(fitness)
                fitness = np.concatenate(([best_f_saved], fitness))
                # Reset archive, memory, and stagnation counter
                archive = np.empty((0, dim))
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                stagnation_counter = 0
                last_improvement_eval = fevals
                # Re-sort for pbest
                best_idx = np.argmin(fitness)
                self.best_f = fitness[best_idx]
                self.best_x = pop[best_idx].copy()
                # Recompute pbest pools will happen next iteration

        return self.best_f, self.best_x