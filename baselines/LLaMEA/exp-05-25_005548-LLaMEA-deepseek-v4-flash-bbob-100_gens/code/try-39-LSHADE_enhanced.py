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

        # Initial population size
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP_init

        # If budget too small, random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling for initial population
        def latin_hypercube(n, d, lb, ub):
            samples = np.empty((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                for i in range(n):
                    samples[i, j] = lb[j] + (perm[i] + np.random.uniform()) / n * (ub[j] - lb[j])
            return samples

        pop = latin_hypercube(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Historical memories for two strategies
        H = 10
        # Strategy 0: current-to-pbest/1, Strategy 1: current-to-rand/1
        M_CR = np.array([[0.5] * H, [0.5] * H])  # 2 x H
        M_F = np.array([[0.5] * H, [0.5] * H])   # 2 x H
        mem_idx = [0, 0]

        # Adaptation for strategy selection
        n_strategies = 2
        prob_strat = np.ones(n_strategies) / n_strategies  # initial equal
        success_counts = np.zeros(n_strategies)
        failure_counts = np.zeros(n_strategies)

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

            # Adaptive pbest ratio
            ratio = 0.2 - 0.1 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = [[], []]  # per strategy
            S_F = [[], []]
            delta_fitness = [[], []]

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # Update strategy probabilities based on success/failure counts
            if fevals > 100:
                total_success = np.sum(success_counts) + np.sum(failure_counts)
                if total_success > 0:
                    for s in range(n_strategies):
                        if success_counts[s] + failure_counts[s] > 0:
                            prob_strat[s] = (success_counts[s] + 1) / (success_counts[s] + failure_counts[s] + n_strategies)
                        else:
                            prob_strat[s] = 1.0 / n_strategies
                    prob_strat /= np.sum(prob_strat)

            for i in range(NP):
                # Select strategy using probability
                strat = np.random.choice(n_strategies, p=prob_strat)

                # Sample parameters from historical memory
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[strat, r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[strat, r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[strat, r]
                F = min(F, 1.)

                if strat == 0:  # current-to-pbest/1 with archive
                    pbest = pop[np.random.choice(pbest_pool)]
                    # Random distinct indices
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    # Select r2 from union of pop and archive
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx == i or idx == r1:
                            continue
                        break
                    if idx < NP:
                        r2_vec = pop[idx]
                    else:
                        r2_vec = archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                else:  # current-to-rand/1 (no archive)
                    idxs = [i]
                    while len(idxs) < 3:
                        rnd = np.random.randint(NP)
                        if rnd not in idxs:
                            idxs.append(rnd)
                    r1, r2 = idxs[1], idxs[2]
                    v = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r1])

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
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

                # Selection
                if f_u <= fitness[i]:
                    S_CR[strat].append(CR)
                    S_F[strat].append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness[strat].append(max(delta, 1e-30))
                    success_counts[strat] += 1
                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Archive for strategy 0 only
                    if strat == 0:
                        archive = np.vstack((archive, pop[i]))
                        if len(archive) > max_archive:
                            archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                else:
                    failure_counts[strat] += 1

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memories for each strategy
            for strat in range(n_strategies):
                if S_CR[strat]:
                    w = np.array(delta_fitness[strat]) / np.sum(delta_fitness[strat])
                    mean_CR = np.sum(w * np.array(S_CR[strat]))
                    sum_sq = np.sum(w * np.array(S_F[strat])**2)
                    sum_w = np.sum(w * np.array(S_F[strat]))
                    mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                    M_CR[strat, mem_idx[strat]] = mean_CR
                    M_F[strat, mem_idx[strat]] = mean_F
                    mem_idx[strat] = (mem_idx[strat] + 1) % H
                    # Reset success/failure counts periodically (every generation)
            success_counts.fill(0)
            failure_counts.fill(0)

        return self.best_f, self.best_x