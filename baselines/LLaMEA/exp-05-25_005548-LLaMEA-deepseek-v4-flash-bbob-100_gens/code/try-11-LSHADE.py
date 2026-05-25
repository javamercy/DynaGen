import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        dim = self.dim
        budget = self.budget

        # Initial population size (same as LSHADE)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        if budget < NP:
            # pure random search
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling for initial population
        def lhs(n, d, lb, ub):
            samples = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                samples[:, i] = (perm + np.random.uniform(size=n)) / n
            return lb + samples * (ub - lb)

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive
        archive = np.empty((0, dim))

        # SHADE memory
        H = 10  # increased memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Restart / stagnation tracking
        no_improve_gens = 0
        max_stagnation = max(20, min(100, 20 + dim * 2))

        # Main loop
        while fevals < budget:
            # Linear population size reduction
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # Adaptive pbest ratio (linear from 0.2 to 0.1)
            p = max(0.05, 0.2 - 0.1 * (fevals / budget))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Successful parameter lists
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy sampling for CR (normal approximation) truncated to [0,1]
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # Cauchy sampling for F with scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

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
                    if idx == i or idx == r1:
                        continue
                    # If idx belongs to archive, it cannot be i (i < NP)
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

                # Reflection bound repair (bounce back)
                for j in range(dim):
                    if u[j] < lb:
                        u[j] = lb + (lb - u[j])
                    elif u[j] > ub:
                        u[j] = ub - (u[j] - ub)
                    # Final clamp for safety
                    u[j] = np.clip(u[j], lb, ub)

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

                    # Add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory if successes
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Check for stagnation and restart
            if self.best_f == fitness.min():
                no_improve_gens += 1
            else:
                no_improve_gens = 0

            if no_improve_gens >= max_stagnation:
                # Restart: keep best, reinitialize population (except best)
                best_x = self.best_x.copy()
                best_f = self.best_f
                # Reinitialize population with LHS (including best)
                new_pop = lhs(NP, dim, lb, ub)
                new_fitness = np.array([func(x) for x in new_pop])
                fevals += NP
                # Replace worst with best
                worst_idx = np.argmax(new_fitness)
                new_pop[worst_idx] = best_x
                new_fitness[worst_idx] = best_f
                # Update population and archive
                pop = new_pop
                fitness = new_fitness
                archive = np.empty((0, dim))
                # Reset memory
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                no_improve_gens = 0

        return self.best_f, self.best_x