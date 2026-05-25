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

        # Initial population size
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        NP_min = 4

        # If budget too small, fallback to random search
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

        # Archive
        archive = np.empty((0, dim))
        # archive size = current NP (updated per generation)
        max_archive = NP

        # SHADE memory (H = 5)
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Main loop
        while fevals < budget:
            # Linear population size reduction
            remaining = budget - fevals
            NP_new = int(NP_min + (NP_init - NP_min) * (remaining / budget))
            NP_new = max(NP_min, NP_new)

            # Reduce population if needed (sort by fitness, keep best)
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]].copy()
                fitness = fitness[sorted_idx[:NP_new]].copy()
                NP = NP_new
                # Reduce archive size to match new NP
                max_archive = NP
                if len(archive) > max_archive:
                    # Randomly remove excess archive members
                    indices = np.random.choice(len(archive), len(archive) - max_archive, replace=False)
                    archive = np.delete(archive, indices, axis=0)

            # pbest selection (top p=0.1)
            p = 0.1
            pbest_num = max(2, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Success tracking
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR from normal distribution (truncated)
                r = np.random.randint(H)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # Sample F from Cauchy distribution (truncated to [0,1])
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # Select pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # Select r1 (different from i)
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from pop ∪ archive (different from i and r1)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx != i and idx != r1:
                        break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = combined[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                u = np.clip(u, lb, ub)

                # Evaluation
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
                        # Remove a random element to keep archive size bounded
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # Update best solution
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                # Budget check
                if fevals >= budget:
                    break

            # Replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory only if there were successful trials
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x