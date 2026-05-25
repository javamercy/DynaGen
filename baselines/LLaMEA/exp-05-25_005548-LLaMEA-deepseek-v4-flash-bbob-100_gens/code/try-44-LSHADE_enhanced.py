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

        # Population size: typical LSHADE size, but larger for low dim
        NP_init = max(12, int(18 * np.log(dim) if dim > 1 else 18))
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

        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        H = 10  # memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart tracking
        best_since_restart = self.best_f
        evals_since_improvement = 0
        restart_threshold = max(100, int(0.05 * budget))

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

            # Adaptive pbest ratio: starts higher, decreases linearly
            ratio = 0.2 - 0.1 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Check for stagnation and possibly restart
            if (fevals - evals_since_improvement) >= restart_threshold and self.best_f < best_since_restart * 0.9999:
                # restart: keep best, reinitialize population (except best)
                best_since_restart = self.best_f
                evals_since_improvement = fevals
                # keep best individual, reinitialize others randomly
                keep_idx = np.argmin(fitness)
                pop_new = np.random.uniform(lb, ub, (NP, dim))
                pop_new[0] = pop[keep_idx].copy()
                fitness_new = np.full(NP, np.inf)
                fitness_new[0] = fitness[keep_idx]
                for i in range(1, NP):
                    if fevals >= budget:
                        break
                    f = func(pop_new[i])
                    fevals += 1
                    fitness_new[i] = f
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = pop_new[i].copy()
                pop = pop_new
                fitness = fitness_new
                # Reset memory to encourage exploration
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                archive = np.empty((0, dim))
                # Recompute pbest pool with new population
                sorted_idx = np.argsort(fitness)
                pbest_pool = sorted_idx[:pbest_num]
                continue

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Sample CR from Cauchy truncated to [0,1]
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Sample F from Cauchy truncated to (0,1]
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # jSO mutation: v = x_i + F_w*(pbest - x_i) + F*(x_r1 - x_r2)
                # F_w = 0.5 * F (can be adapted, but fixed ratio works well)
                F_w = 0.5 * F

                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)
                r2 = np.random.randint(NP)
                while r2 == i or r2 == r1:
                    r2 = np.random.randint(NP)

                # Mutation with jSO
                v = pop[i] + F_w * (pbest - pop[i]) + F * (pop[r1] - pop[r2])

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

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        evals_since_improvement = fevals

                if fevals >= budget:
                    break

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

        return self.best_f, self.best_x