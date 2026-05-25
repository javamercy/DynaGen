import numpy as np

class LSHADE_APR:
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

        # Latin Hypercube Sampling initial population
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init

        # If budget too small, random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # LHS initialization
        def lhs_sample(n, d, lb, ub):
            samples = np.empty((n, d))
            for j in range(d):
                interval = np.linspace(lb[j], ub[j], n + 1)
                points = interval[:-1] + np.random.uniform(0, (interval[1]-interval[0]), n)
                np.random.shuffle(points)
                samples[:, j] = points
            return samples

        pop = lhs_sample(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()
        last_improvement_evals = fevals

        # Archive
        archive = np.empty((0, dim))
        max_archive = NP_init

        # SHADE memory
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        stagnation_limit = max(50, int(0.1 * budget))
        restart_flag = False

        while fevals < budget:
            # Linear population reduction
            remaining = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # Dynamic pbest proportion (0.2 → 0.05)
            p = 0.2 - 0.15 * (fevals / budget)
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
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)

                # Cauchy with scale 0.5 for better diversity
                F = np.random.standard_cauchy() * 0.5 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.5 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]

                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                idx2 = np.random.randint(len(combined))
                # Ensure distinct
                while idx2 == i or idx2 == r1:
                    idx2 = np.random.randint(len(combined))
                if idx2 < NP:
                    r2_vec = combined[idx2]
                else:
                    r2_vec = archive[idx2 - NP]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
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

                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improvement_evals = fevals

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory
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

            # Restart if stagnation
            if (fevals - last_improvement_evals) > stagnation_limit and np.random.rand() < 0.5:
                # Reinitialize population via LHS, keep best solution
                pop_new = lhs_sample(NP_init, dim, lb, ub)
                fitness_new = np.array([func(x) for x in pop_new])
                fevals += NP_init
                # Merge best individual
                pop = np.vstack((pop_new, self.best_x.reshape(1, -1)))
                fitness = np.concatenate((fitness_new, [self.best_f]))
                # Resize to NP_init (but maybe keep NP = NP_init)
                NP = min(NP_init, NP)
                # Sort and truncate to NP
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP]]
                fitness = fitness[sorted_idx[:NP]]
                # Reset archive and memory
                archive = np.empty((0, dim))
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                last_improvement_evals = fevals
                # Update best if any better found
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()

        return self.best_f, self.best_x