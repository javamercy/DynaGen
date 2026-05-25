import numpy as np

class jSO_BBOB:
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

        # Initial population size (jSO heuristic: 25 * log(dim) * sqrt(dim))
        if dim > 1:
            NP_init = int(25 * np.log(dim) * np.sqrt(dim))
        else:
            NP_init = 25
        NP_init = np.clip(NP_init, 4, budget // 2)
        NP = NP_init
        NP_min = 4

        # Archive size proportion (jSO uses 2.6)
        ar = 2.6
        max_archive = int(ar * NP)

        # SHADE memory (H = 6)
        H = 6
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # If budget too small, random search
        if budget < NP_init:
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

        # Best so far
        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive (empty)
        archive = np.empty((0, dim))

        # Main loop
        while fevals < budget:
            # Quadratically reduce population size (jSO style)
            ratio = fevals / budget
            NP_new = int(round(NP_init - (NP_init - NP_min) * ratio * ratio))
            NP_new = max(NP_min, NP_new)
            if NP_new < NP:
                # Keep best individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # Reduce archive to keep proportion
                max_archive = int(ar * NP)
                if len(archive) > max_archive:
                    # Randomly remove excess archive members
                    np.random.shuffle(archive)
                    archive = archive[:max_archive]

            # Adaptive p_best (linearly from 0.2 to 0.05)
            p = 0.2 - 0.15 * ratio
            p = max(0.05, min(0.2, p))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Success records
            S_CR = []
            S_F = []
            delta_fitness = []

            # Offspring
            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Select memory entry
                r = np.random.randint(H)
                # Generate CR (truncated normal)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # Generate F (Cauchy, truncated to >=0, <=1)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Mutation: current-to-pbest/1 with archive
                pbest = pop[np.random.choice(pbest_pool)]
                # r1: distinct from i
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)
                # r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                # We need to ensure r2 is not i or r1
                candidates = list(range(len(combined)))
                # Remove i and r1 (if r1 < NP) from candidates
                candidates.remove(i)
                if r1 < NP:
                    candidates.remove(r1)
                idx = np.random.choice(candidates)
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = archive[idx - NP]

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
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove random entry
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # Update global best
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

            # Update memory if successful
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                # Lehmer mean for F
                sum_w = np.sum(w * np.array(S_F))
                sum_sq = np.sum(w * np.array(S_F)**2)
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x