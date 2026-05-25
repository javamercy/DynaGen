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

        # Initial population size: larger for higher dim but safe
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        # Archive size limit (dynamic)
        max_archive = NP_init

        # If budget too small, do random search with Latin hypercube
        if budget < NP:
            n = budget
            # Latin hypercube sampling
            segments = np.linspace(0, 1, n+1)
            for i in range(n):
                u = np.random.uniform(segments[i], segments[i+1], dim)
                x = lb + u * (ub - lb)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin hypercube initial population
        n = NP
        segments = np.linspace(0, 1, n+1)
        pop = np.empty((n, dim))
        for i in range(n):
            u = np.random.uniform(segments[i], segments[i+1], dim)
            np.random.shuffle(u)  # permute dimensions for LHS
            pop[i] = lb + u * (ub - lb)

        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F: increased size H=20
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Main loop
        while fevals < budget:
            remaining_evals = budget - fevals
            # Power-law population reduction (exponent 1.5 for slower decline)
            ratio = remaining_evals / budget
            NP_new = max(4, int(4 + (NP_init - 4) * (ratio ** 1.5)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # Trim archive to match new population size
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: cubic decay from 0.2 to 0.05
            base = 0.2 - 0.15 * ((1 - ratio) ** 3)
            p = max(0.05, min(0.2, base))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR from truncated Cauchy with narrower spread
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Sample F from truncated Cauchy (positive, <=1)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Combine pop and archive for r2
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                if idx < NP:
                    r2_vec = combined[idx]
                else:
                    r2_vec = combined[idx]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected bound handling, then clamp to bounds if still out
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                # If still out of bounds, random replacement
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

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

                    # Update archive (append parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove random element
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

            # Update memory with weighted averages
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / max(sum_w, 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = min(mean_F, 1.0)
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x