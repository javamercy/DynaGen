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

        # Larger initial population: 18 * dim (common for LSHADE)
        NP_init = int(18 * dim)
        # Ensure population does not exceed budget
        NP_init = max(10, min(NP_init, budget // 2))
        NP = NP_init
        max_archive = NP_init

        # If budget too small, do random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population (quasi-random not used for simplicity but can be added)
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F (size H=10)
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation tracking
        last_improvement_evals = 0
        restart_interval = max(1, budget // 15)

        # Main loop
        while fevals < budget:
            # Linear population reduction (NP from NP_init to 4)
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: quadratic decrease from 0.25 to 0.05
            ratio = 0.2 * (1 - (1 - remaining_evals / budget) ** 2) + 0.05
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR from Cauchy truncated to [0,1]
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Sample F from Cauchy truncated to >0 and <=1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
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

                # Reflected bound handling
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

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive (append parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove random element
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # Update best
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improvement_evals = fevals

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
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Restart if no improvement for a long time
            if (fevals - last_improvement_evals) >= restart_interval and fevals < 0.9 * budget:
                # Keep best individual, reinitialize rest
                best_individual = self.best_x.copy()
                # New population size stays the same (NP may have been reduced)
                NP_restart = max(4, NP)
                new_pop = np.empty((NP_restart, dim))
                new_fitness = np.empty(NP_restart)
                # Keep best
                new_pop[0] = best_individual
                new_fitness[0] = self.best_f
                # Half random, half around best with small perturbation
                half = max(1, (NP_restart - 1) // 2)
                for i in range(1, half+1):
                    x = np.random.uniform(lb, ub)
                    new_pop[i] = x
                    new_fitness[i] = func(x)
                    fevals += 1
                    if new_fitness[i] < self.best_f:
                        self.best_f = new_fitness[i]
                        self.best_x = new_pop[i].copy()
                    if fevals >= budget:
                        break
                if fevals < budget:
                    sigma = 0.05 * (ub - lb)
                    for i in range(half+1, NP_restart):
                        x = best_individual + np.random.normal(0, sigma)
                        x = np.clip(x, lb, ub)
                        new_pop[i] = x
                        new_fitness[i] = func(x)
                        fevals += 1
                        if new_fitness[i] < self.best_f:
                            self.best_f = new_fitness[i]
                            self.best_x = new_pop[i].copy()
                        if fevals >= budget:
                            break
                # Replace population
                pop = new_pop
                fitness = new_fitness
                NP = NP_restart
                archive = np.empty((0, dim))
                # Reset memory to default
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                last_improvement_evals = fevals

        return self.best_f, self.best_x