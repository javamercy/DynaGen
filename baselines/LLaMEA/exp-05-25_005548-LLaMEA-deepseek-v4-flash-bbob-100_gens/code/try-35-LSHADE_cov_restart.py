import numpy as np

class LSHADE_cov_restart:
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

        # initial population size
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        # if budget too small -> random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # initial population
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # memory for CR and F
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # strategy pool: 0 = current-to-pbest/1, 1 = current-to-rand/1 with jitter
        # probability of each strategy, initial equal
        strategy_prob = np.array([0.5, 0.5])
        strategy_success = np.zeros(2)
        strategy_trials = np.zeros(2)
        strategy_alpha = 0.1  # learning rate

        # covariance matrix for mutation direction (used in strategy 1)
        C = np.eye(dim)
        # update frequency for covariance (every 5*dim evaluations, but adaptive)
        cov_update_counter = 0
        cov_update_interval = max(1, int(0.1 * budget / (dim + 1)))

        # stagnation detection
        stagnation_counter = 0
        stagnation_limit = max(100, int(0.05 * budget))
        prev_best_f = self.best_f

        # main loop
        while fevals < budget:
            # degenerate case
            if NP < 5:
                # supplement with random points
                needed = 5 - NP
                new_pop = np.random.uniform(lb, ub, (needed, dim))
                pop = np.vstack((pop, new_pop))
                for x in new_pop:
                    f = func(x); fevals += 1
                    fitness = np.append(fitness, f)
                    if f < self.best_f: self.best_f = f; self.best_x = x.copy()
                NP = len(pop)

            # linear population size reduction
            remaining_evals = budget - fevals
            NP_new = max(5, int(5 + (NP_init - 5) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # adaptive pbest ratio
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

            # strategy selection probabilities: softmax
            # W = np.exp(strategy_success / (strategy_trials + 1e-10))
            # strategy_prob = W / W.sum()
            # but we use simple additive learning
            strat_probs = strategy_prob / (strategy_prob.sum() + 1e-30)

            for i in range(NP):
                # choose strategy
                rng = np.random.rand()
                if rng < strat_probs[0]:
                    strat = 0
                else:
                    strat = 1

                # sample CR and F (common for both strategies)
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                strategy_trials[strat] += 1

                # select base vectors
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

                if strat == 0:
                    # current-to-pbest/1
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                else:
                    # current-to-rand/1 with jitter from covariance
                    # sample direction from multivariate normal with covariance C
                    try:
                        d = np.random.multivariate_normal(np.zeros(dim), C)
                    except:
                        d = np.random.randn(dim)
                    v = pop[i] + F * (pop[r1] - r2_vec) + 0.5 * F * d

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # reflected bound handling
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

                    # archive update
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                    # update strategy success
                    strategy_success[strat] += 1

                if fevals >= budget:
                    break

            # update strategy probabilities with exponential learning
            eps = 1e-10
            for s in range(2):
                if strategy_trials[s] > 0:
                    rate = strategy_success[s] / (strategy_trials[s] + eps)
                    strategy_prob[s] = (1 - strategy_alpha) * strategy_prob[s] + strategy_alpha * rate

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # update memory for CR and F
            if S_CR:
                w = np.array(delta_fitness) / (np.sum(delta_fitness) + 1e-30)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / (sum_w + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # update covariance matrix (every so often)
            cov_update_counter += 1
            if cov_update_counter >= cov_update_interval and len(pop) > dim:
                cov_update_counter = 0
                # compute weighted covariance of pbest subset
                pbest_indices = pbest_pool
                if len(pbest_indices) > dim:
                    mean_pbest = np.mean(pop[pbest_indices], axis=0)
                    centered = pop[pbest_indices] - mean_pbest
                    # use only up to 2*dim points for speed
                    n_cov = min(len(centered), 2*dim)
                    if n_cov > dim:
                        C = np.cov(centered[:n_cov].T, rowvar=False)
                        # regularization
                        C += 1e-10 * np.eye(dim)
                    # else keep previous C

            # stagnation check and restart
            if self.best_f < prev_best_f - 1e-15:
                stagnation_counter = 0
                prev_best_f = self.best_f
            else:
                stagnation_counter += 1

            if stagnation_counter >= stagnation_limit:
                # restart: keep best, reinitialize population
                if budget - fevals > NP:
                    # store best
                    best_x_keep = self.best_x.copy()
                    best_f_keep = self.best_f
                    # reinitialize 80% of population from scratch, keep 20% best (except best)
                    keep_num = max(1, NP // 5)
                    # select top keep_num excluding best? we keep best separately
                    sorted_idx = np.argsort(fitness)
                    # keep some of the best individuals (but not the very best to avoid collapse)
                    keep_idx = sorted_idx[:keep_num] if len(sorted_idx) > 1 else sorted_idx
                    new_pop = pop[keep_idx]
                    new_fitness = fitness[keep_idx]
                    # fill rest with random
                    needed = NP - len(new_pop)
                    if needed > 0:
                        random_pop = np.random.uniform(lb, ub, (needed, dim))
                        new_pop = np.vstack((new_pop, random_pop))
                        for x in random_pop:
                            f = func(x); fevals += 1
                            new_fitness = np.append(new_fitness, f)
                            if f < self.best_f: self.best_f = f; self.best_x = x.copy()
                    pop = new_pop
                    fitness = new_fitness[:NP]
                    # reinject best
                    pop[0] = best_x_keep
                    fitness[0] = best_f_keep
                    # reset archive
                    archive = np.empty((0, dim))
                    # reset memory
                    M_CR[:] = 0.5
                    M_F[:] = 0.5
                    mem_idx = 0
                    # reset covariance
                    C = np.eye(dim)
                    # reset stagnation
                    stagnation_counter = 0
                    prev_best_f = self.best_f
                    # reduce NP slightly to avoid huge restarts
                    NP = max(5, int(NP * 0.8))

        return self.best_f, self.best_x