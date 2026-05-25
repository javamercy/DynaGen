import numpy as np

class RefinedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dim = self.dim
        budget = self.budget

        # Initial population size (typical for LSHADE variants)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        # memory size
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive (starts empty, max size = NP)
        archive = np.empty((0, dim))

        # Count generations without improvement for restart trigger
        stagnation_count = 0
        stagnation_limit = max(5, int(budget / 200))

        while fevals < budget:
            # Linear population reduction
            remaining = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # Adaptive pbest ratio (decreases over time)
            p = 0.2 - 0.1 * (fevals / budget)
            p = np.clip(p, 0.05, 0.2)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Lists for successful parameters
            S_CR = []
            S_F = []
            delta_f = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved_this_gen = False

            for i in range(NP):
                # Sample CR: 10% chance uniform, else Gaussian with memory mean
                r = np.random.randint(H)
                if np.random.rand() < 0.1:
                    CR = np.random.rand()
                else:
                    CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)

                # Sample F: 10% chance small uniform, else Cauchy with memory mean
                if np.random.rand() < 0.1:
                    F = np.random.uniform(0, 0.1)
                else:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = max(0., min(F, 1.))  # truncate to [0,1]

                # Choose pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Choose r1 != i
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Choose r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx % NP == i or idx == r1:
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
                u = np.clip(u, lb, ub)

                # Evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_f.append(delta)

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > NP:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        improved_this_gen = True

                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory if successes
            if S_CR:
                w = np.array(delta_f) / np.sum(delta_f)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Stagnation detection and restart
            if improved_this_gen:
                stagnation_count = 0
            else:
                stagnation_count += 1

            if stagnation_count >= stagnation_limit and fevals < budget - NP:
                # Reinitialize population keeping the best solution
                stagnation_count = 0
                pop[0] = self.best_x.copy()
                fitness[0] = self.best_f
                for i in range(1, NP):
                    pop[i] = np.random.uniform(lb, ub, dim)
                    fitness[i] = func(pop[i])
                    fevals += 1
                    if fevals >= budget:
                        break
                # Reset memory
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                archive = np.empty((0, dim))

        return self.best_f, self.best_x