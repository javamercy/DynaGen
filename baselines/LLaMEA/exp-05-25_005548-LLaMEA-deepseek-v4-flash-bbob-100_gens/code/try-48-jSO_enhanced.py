import numpy as np

class jSO_enhanced:
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
        maxfes = budget

        # --- Initialization ---
        NP_init = int(25 * np.sqrt(dim) * np.log(dim) if dim > 1 else 25)
        NP_init = max(10, min(NP_init, maxfes // 2))  # avoid too large
        NP = NP_init
        H = 6  # memory size (jSO uses 6)
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = int(2.6 * NP)  # archive size ratio

        # stagnation control
        last_improve_fevals = 0
        stagnation_threshold = 0.1 * maxfes

        # --- Main loop ---
        while fevals < maxfes:
            # remaining evaluations for linear population reduction
            remaining = maxfes - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / maxfes)))
            if NP_new < NP:
                # sort and truncate
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # reduce archive to ratio
                max_archive = int(2.6 * NP)
                if len(archive) > max_archive:
                    np.random.shuffle(archive)
                    archive = archive[:max_archive]

            # adaptive pbest ratio (jSO style)
            p_ratio = 0.2
            if fevals >= 0.2 * maxfes:
                if fevals < 0.6 * maxfes:
                    p_ratio = 0.2 - (0.2 - 0.1) * ((fevals - 0.2 * maxfes) / (0.4 * maxfes))
                else:
                    p_ratio = 0.1
            pbest_num = max(1, int(p_ratio * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            improved_this_gen = False

            for i in range(NP):
                # select memory index
                r = np.random.randint(H)

                # --- F generation (jSO: scaled Cauchy) ---
                if fevals < 0.2 * maxfes:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                elif fevals < 0.6 * maxfes:
                    F = np.random.standard_cauchy() * 0.1 + (0.5 * M_F[r] + 0.5)
                else:
                    F = np.random.standard_cauchy() * 0.05 + (0.5 * M_F[r] + 0.5)
                # clamp F to (0,1]
                while F <= 0.0:
                    if fevals < 0.2 * maxfes:
                        F = np.random.standard_cauchy() * 0.1 + M_F[r]
                    elif fevals < 0.6 * maxfes:
                        F = np.random.standard_cauchy() * 0.1 + (0.5 * M_F[r] + 0.5)
                    else:
                        F = np.random.standard_cauchy() * 0.05 + (0.5 * M_F[r] + 0.5)
                F = min(F, 1.0)

                # --- CR generation (jSO: normal early, Cauchy later) ---
                if fevals < 0.25 * maxfes:
                    CR = np.random.randn() * 0.1 + M_CR[r]
                else:
                    CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0.0, min(1.0, CR))

                # pbest individual
                pbest = pop[np.random.choice(pbest_pool)]

                # random indices for mutation
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # combine pop and archive for r2
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # reflected boundary handling
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
                    # update archive (parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improve_fevals = fevals
                        improved_this_gen = True

                if fevals >= maxfes:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= maxfes:
                break

            # update memory with weighted means
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                # Lehmer mean for F
                sum_f_sq = np.sum(w * np.array(S_F) ** 2)
                sum_f_w = np.sum(w * np.array(S_F))
                mean_F = sum_f_sq / sum_f_w if sum_f_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # --- Stagnation restart: replace worst half with random points ---
            if not improved_this_gen and (fevals - last_improve_fevals) > stagnation_threshold and fevals > 0.1 * maxfes:
                # replace bottom 50% (excluding best) with random solutions
                # but keep the best copy
                order = np.argsort(fitness)
                replace_idx = order[max(1, NP // 2):]  # worst half, but retain at least best
                for idx in replace_idx:
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fevals >= maxfes:
                        break
                # reset stagnation counter
                last_improve_fevals = fevals
                # also clear archive to avoid old stagnation memory? partially
                archive = np.empty((0, dim))

        return self.best_f, self.best_x