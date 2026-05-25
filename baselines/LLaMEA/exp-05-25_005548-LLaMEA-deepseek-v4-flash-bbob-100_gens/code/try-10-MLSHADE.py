import numpy as np

class MLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def _latin_hypercube(self, lb, ub, n):
        """Generate n points using Latin hypercube sampling in [lb, ub]."""
        dim = self.dim
        points = np.zeros((n, dim))
        for d in range(dim):
            # Partition each dimension into n intervals
            j = np.random.permutation(n)  # random ordering
            offset = np.random.uniform(0, 1, n)
            points[:, d] = lb[d] + (ub[d] - lb[d]) * (j + offset) / n
        return points

    def _reflect_bounds(self, u, lb, ub):
        """Reflect out-of-bounds coordinates back into the domain (only one reflection allowed)."""
        u = np.where(u < lb, 2 * lb - u, u)
        u = np.where(u > ub, 2 * ub - u, u)
        # Final clamping if still out (rare)
        return np.clip(u, lb, ub)

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Initial population size (typical LSHADE)
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP_init

        # Budget too small -> random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin hypercube initialization
        pop = self._latin_hypercube(lb, ub, NP)
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

            # Adaptive pbest selection ratio: linear from 0.2 to 0.05
            progress = fevals / budget
            p = 0.2 - 0.15 * progress
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Successful parameters for this generation
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample crossover type: 0 = binomial, 1 = exponential
                use_exp = np.random.rand() < 0.5  # 50% exponential, 50% binomial

                # Sample CR and F from memory
                r = np.random.randint(H)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0, 1)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1)

                # Choose pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # Choose r1 != i
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Choose r2 from union of pop and archive, distinct from i and r1
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # Ensure r2 is not the same as i (already covered) and r1 (covered)
                    break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover
                u = pop[i].copy()
                if use_exp:  # exponential crossover
                    start = np.random.randint(dim)
                    L = 1
                    while np.random.rand() < CR and L < dim:
                        L += 1
                    for j in range(dim):
                        if (j - start) % dim < L:
                            u[j] = v[j]
                else:  # binomial crossover
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]

                # Reflection bound repair
                u = self._reflect_bounds(u, lb, ub)

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

                    # Archive update
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # Update best
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

            # Update memory if there were successful parameters
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

        return self.best_f, self.best_x