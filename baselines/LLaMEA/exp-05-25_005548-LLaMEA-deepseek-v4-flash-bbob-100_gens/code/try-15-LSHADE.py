import numpy as np
from scipy.stats import qmc  # For Latin Hypercube sampling

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

        # Parameters
        NP_init = max(10, int(18 * np.log(dim))) if dim > 1 else 18
        NP_min = 4
        H = 10                     # memory size
        max_archive = NP_init

        # If budget too small, random search
        if budget < NP_init:
            for i in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population (Latin Hypercube for better coverage)
        sampler = qmc.LatinHypercube(dim, seed=None)
        samples = sampler.random(n=NP_init)
        pop = lb + (ub - lb) * samples
        fitness = np.array([func(x) for x in pop])
        fevals = NP_init

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive
        archive = np.empty((0, dim))

        # SHADE memory
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Main loop
        while fevals < budget:
            # Linear population size reduction (over entire budget)
            NP = max(NP_min, int(round(NP_init - (NP_init - NP_min) * (fevals / budget))))
            current_NP = len(pop)
            if NP < current_NP:
                # Keep best NP individuals
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP]]
                fitness = fitness[sorted_idx[:NP]]
            elif NP > current_NP:
                # Rarely needed; just continue
                pass
            # pbest fraction decreases linearly from 0.2 to 0.1
            p = 0.2 - 0.1 * (fevals / budget)
            p = max(0.1, p)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Adaptive crossover control: switch between binomial (0) and exponential (1)
            # Use exponential with probability 0.5 (can be tuned)
            use_exp = np.random.rand() < 0.5

            # Success lists
            S_CR, S_F, delta_f = [], [], []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Select memory entry
                r = np.random.randint(H)
                # Sample CR (normal truncated to [0,1])
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # Sample F (Cauchy with location M_F[r], scale 0.1)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                # Resample if non-positive (limit tries)
                tries = 0
                while F <= 0. and tries < 10:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                    tries += 1
                if F <= 0.:
                    F = 0.1
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Select r1 (different from i)
                r1 = i
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from pop + archive (distinct from i and r1)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover
                if use_exp:  # Exponential crossover
                    u = pop[i].copy()
                    n = np.random.randint(dim)
                    L = 0
                    while np.random.rand() < CR and L < dim:
                        L += 1
                    for j in range(dim):
                        if j % dim >= n and j % dim < (n + L) % dim:
                            u[j] = v[j]
                else:  # Binomial crossover
                    u = pop[i].copy()
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]

                # Reflection repair (avoid clamping to bounds)
                u = np.where(u < lb, lb + (lb - u) % (ub - lb), u)
                u = np.where(u > ub, ub - (u - ub) % (ub - lb), u)

                # Evaluate
                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_f.append(max(delta, 1e-30))

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

            # Update memory if successful
            if S_CR:
                w = np.array(delta_f) / np.sum(delta_f)
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                wF = w * np.array(S_F)
                mean_F = np.sum(wF * np.array(S_F)) / (np.sum(wF) + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x