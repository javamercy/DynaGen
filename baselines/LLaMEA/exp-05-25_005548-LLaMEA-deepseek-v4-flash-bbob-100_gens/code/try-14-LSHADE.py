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

        # Standard LSHADE initial population size: 18 * dim (min 10)
        NP_init = max(10, 18 * dim)
        NP = NP_init

        # if budget is too small, just random search
        if budget < NP:
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive (dynamic size = current NP)
        archive = np.empty((0, dim))

        # SHADE memory
        H = 10                      # memory size increased
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Evaluate all individuals (already done) and continue
        while fevals < budget:
            # Linear population reduction
            remaining_evals = budget - fevals
            NP_target = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_target < NP:
                # Sort by fitness and keep best NP_target individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_target]]
                fitness = fitness[sorted_idx[:NP_target]]
                NP = NP_target
                # Also reduce archive size to current NP (if archive too large)
                if len(archive) > NP:
                    archive = archive[np.random.choice(len(archive), NP, replace=False)]

            # pbest selection (top 10%)
            p = 0.1
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR from normal with mean M_CR and std 0.1, truncated to [0,1]
                r = np.random.randint(H)
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0.0, 1.0)

                # Sample F from Cauchy with location M_F and scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # Random indices r1 (different from i) and r2 (from pop+archive, different from i and r1)
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # r2 from union of population and archive
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # if idx corresponds to archive entry, it's still valid
                    break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Clamp to bounds
                u = np.clip(u, lb, ub)

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    # Update the child and fitness
                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Archive: add parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > NP:
                        # Remove random element if archive exceeds current NP
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

            # Update memory if any successful parameters
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