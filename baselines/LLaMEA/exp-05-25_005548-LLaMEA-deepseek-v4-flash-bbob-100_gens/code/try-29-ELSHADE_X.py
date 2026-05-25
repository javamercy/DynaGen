import numpy as np

class ELSHADE_X:
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

        # Latin hypercube initialization
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, min(NP_init, budget // 2))
        NP = NP_init

        # Generate Latin hypercube samples
        pop = np.zeros((NP, dim))
        for j in range(dim):
            seq = np.random.permutation(NP)
            for i in range(NP):
                pop[i, j] = lb[j] + (seq[i] + np.random.uniform(0, 1)) * (ub[j] - lb[j]) / NP
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP_init

        # Memory for CR and F
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Local search trigger
        stagnation_counter = 0
        local_search_freq = max(1, int(0.02 * budget))

        while fevals < budget:
            # Population reduction (non‑linear: faster at beginning)
            remaining = budget - fevals
            sigma = remaining / budget
            NP_new = max(4, int(4 + (NP_init - 4) * (sigma ** 1.5)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (sigmoid shape, more exploitation later)
            p = 0.05 + 0.15 / (1 + np.exp(10 * (sigma - 0.5)))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR, S_F, delta_fitness = [], [], []
            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # Scale for Cauchy sampling: larger early, smaller later
            scale = 0.1 * (1 + 0.5 * (1 - sigma))

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * scale + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * scale + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * scale + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]

                # r1 and r2 with archive
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

                # Reflected bound handling (mirror)
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

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
                        stagnation_counter = 0
                    else:
                        if i == np.argmin(fitness):
                            stagnation_counter += 1

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Memory update
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Local search around best after stagnation
            if stagnation_counter >= 5:
                # Perform a few local steps
                for _ in range(min(3, budget - fevals)):
                    trial = self.best_x + np.random.normal(0, 0.1 * (1 - fevals / budget), dim)
                    trial = np.clip(trial, lb, ub)
                    ft = func(trial)
                    fevals += 1
                    if ft < self.best_f:
                        self.best_f = ft
                        self.best_x = trial.copy()
                        stagnation_counter = 0
                    if fevals >= budget:
                        break
                stagnation_counter = 0

        return self.best_f, self.best_x