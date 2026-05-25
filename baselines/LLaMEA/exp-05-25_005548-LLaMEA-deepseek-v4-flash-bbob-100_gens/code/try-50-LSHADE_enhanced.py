import numpy as np
from scipy.stats import qmc

class LSHADE_enhanced:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Adaptive initial population size
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        # Small budget handling
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin hypercube initial population for better coverage
        sampler = qmc.LatinHypercube(d=dim, seed=None)
        sample = sampler.random(n=NP)
        pop = qmc.scale(sample, lb, ub)  # shape (NP, dim)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F (larger memory)
        H = 15
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation detection
        stagnation_counter = 0
        stagnation_limit = max(50, dim*20)
        best_prev = self.best_f

        # Main loop
        while fevals < budget:
            # Linear population reduction
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio: starts high, decreases linearly
            ratio = 0.2 - 0.1 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR and F from Cauchy
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Hybrid mutation: occasionally use rand/1 for exploration
                if np.random.rand() < 0.1:  # 10% probability
                    # rand/1
                    idxs = np.random.choice(NP, 3, replace=False)
                    a, b, c = pop[idxs[0]], pop[idxs[1]], pop[idxs[2]]
                    v = a + F * (b - c)
                else:
                    # current-to-pbest/1 with archive
                    pbest = pop[np.random.choice(pbest_pool)]
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx == i or idx == r1:
                            continue
                        break
                    r2_vec = combined[idx] if idx < NP else archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected bound handling (if out of bounds)
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                # Clamp if still out
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

                    # Update archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

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
            if S_CR and S_F:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Stagnation detection and restart
            if abs(self.best_f - best_prev) < 1e-12:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
                best_prev = self.best_f

            if stagnation_counter >= stagnation_limit:
                # Restart: reset memories, keep best solution, reinitialize most of population
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                # Keep the best, reinitialize rest
                NP_keep = NP
                new_pop2 = [self.best_x.copy()]
                # Generate new individuals using local perturbation around best and global sampling
                for _ in range(1, NP_keep):
                    if np.random.rand() < 0.5:
                        # local perturbation around best
                        new_pop2.append(self.best_x + 0.1 * np.random.uniform(lb, ub))
                    else:
                        new_pop2.append(np.random.uniform(lb, ub))
                pop = np.array(new_pop2)
                # Evaluate new individuals (except best already evaluated)
                for idx in range(1, NP_keep):
                    f = func(pop[idx])
                    fevals += 1
                    fitness[idx] = f
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = pop[idx].copy()
                # Re-evaluate best (optional, but already known)
                fitness[0] = self.best_f
                # Clear archive
                archive = np.empty((0, dim))
                if fevals >= budget:
                    break
                stagnation_counter = 0
                best_prev = self.best_f

        return self.best_f, self.best_x