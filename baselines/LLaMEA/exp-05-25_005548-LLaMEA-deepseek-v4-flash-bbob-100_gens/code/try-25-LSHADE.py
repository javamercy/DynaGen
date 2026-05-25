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

        # Initial population size (larger for exploration, but cap to budget/2)
        NP_init = min(max(4, int(18 * dim)), budget // 2, 1000)
        NP_init = max(10, NP_init)
        NP = NP_init
        NP_min = 4
        max_archive = int(2.6 * NP_init)  # jSO style archive size

        # if budget too small, random search
        if budget < NP:
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initialize population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # SHADE memory
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart tracking
        no_improve_gen = 0
        restart_limit = max(50, 10 * dim)

        # Main loop
        while fevals < budget:
            # Non-linear population reduction (quadratic)
            remaining_evals = budget - fevals
            ratio = remaining_evals / budget
            NP_new = max(NP_min, int(NP_min + (NP_init - NP_min) * (ratio ** 2)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # Adaptive pbest ratio: starts high for exploration, decreases to 0.1
            p = 0.1 + (0.5 - 0.1) * (1 - fevals / budget)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # CR from normal truncated to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # F from Cauchy truncated to >0 and <=1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Select r1 distinct from i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx % NP == i or idx % len(combined) == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover with improved boundary handling (reflect at bounds)
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                    if u[j] < lb:
                        u[j] = (lb + pop[i][j]) / 2
                    elif u[j] > ub:
                        u[j] = (ub + pop[i][j]) / 2

                # Clamp as safety
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

                    # Add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove the closest to the best (crowding removal)
                        # For simplicity, remove random
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        no_improve_gen = 0
                    else:
                        no_improve_gen += 1
                else:
                    no_improve_gen += 1

                if fevals >= budget:
                    break

            if fevals >= budget:
                break

            # Replace population
            pop = new_pop
            fitness = new_fitness

            # Update memory if successful
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
            if no_improve_gen > restart_limit and fevals < budget:
                # Reinitialize population except the best solution
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx]
                fitness = fitness[sorted_idx]
                # Keep best, reinitialize the rest
                best_ind = pop[0].copy()
                best_fit = fitness[0]
                for i in range(1, NP):
                    pop[i] = np.random.uniform(lb, ub, dim)
                    fitness[i] = func(pop[i])
                    fevals += 1
                    if fitness[i] < best_fit:
                        best_fit = fitness[i]
                        best_ind = pop[i].copy()
                    if fevals >= budget:
                        break
                pop[0] = best_ind
                fitness[0] = best_fit
                # Reset memories
                M_CR = 0.5 * np.ones(H)
                M_F = 0.5 * np.ones(H)
                mem_idx = 0
                # Clear archive
                archive = np.empty((0, dim))
                no_improve_gen = 0
                # Update best if improved
                if best_fit < self.best_f:
                    self.best_f = best_fit
                    self.best_x = best_ind.copy()

        return self.best_f, self.best_x