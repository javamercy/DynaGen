import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def _latin_hypercube(self, lb, ub, n):
        """Generate Latin Hypercube samples in [lb, ub]."""
        dim = len(lb)
        samples = np.random.uniform(low=0.0, high=1.0, size=(n, dim))
        for j in range(dim):
            samples[:, j] = (np.random.permutation(n) + samples[:, j]) / n
        # scale to bounds
        return lb + samples * (ub - lb)

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Initial population size (use LSHADE rule: 18*log(dim), but ensure minimal)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP  # archive size = current NP

        # If budget too small, random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population with LHS
        pop = self._latin_hypercube(lb, ub, NP)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))  # empty archive

        # SHADE memory (H = 5)
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        p = 0.2  # pbest ratio (fixed)

        while fevals < budget:
            # Linear population reduction based on used evaluations
            NP_new = max(4, int(NP_init - (NP_init - 4) * (fevals / budget)))
            if NP_new < NP:
                # sort by fitness and keep best
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # also trim archive to current NP if needed
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # pbest indices
            pbest_num = max(1, int(p * NP))
            sorted_indices = np.argsort(fitness)
            pbest_pool = sorted_indices[:pbest_num]

            # Successful parameter lists
            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR and F from memory with Cauchy/Normal
                r = np.random.randint(H)
                CR = np.clip(np.random.normal(M_CR[r], 0.1), 0., 1.)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Select pbest, r1, r2
                pbest = pop[np.random.choice(pbest_pool)]

                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of pop and archive, distinct from i and r1
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

                # Bounce-back bound repair
                for j in range(dim):
                    if u[j] < lb[j]:
                        u[j] = lb[j] + np.random.rand() * (pop[i][j] - lb[j])
                    elif u[j] > ub[j]:
                        u[j] = ub[j] - np.random.rand() * (ub[j] - pop[i][j])

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

                    # Add parent to archive, trim if needed
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory if successes
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