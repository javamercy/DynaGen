import numpy as np

class LSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.asarray(func.bounds.lb)
        ub = np.asarray(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # initial population size (LSHADE typical: ~ 18*log(dim) with min 10)
        NP_init = max(10, int(18 * np.log(dim))) if dim > 1 else 18
        NP = NP_init
        max_archive = int(2.0 * NP_init)  # JADE-like archive size

        # initial population via Latin Hypercube Sampling for better coverage
        pop = np.empty((NP, dim))
        for i in range(dim):
            perm = np.random.permutation(NP)
            pop[:, i] = lb[i] + (ub[i] - lb[i]) * (perm + np.random.uniform(size=NP)) / NP

        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # SHADE memory
        H = 10
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Ensemble mutation strategy parameters
        # probabilities for strategy 1 (current-to-pbest/1) and strategy 2 (rand/1)
        prob1 = 0.5
        prob2 = 0.5
        # success counters for each strategy
        ns1 = 0
        ns2 = 0
        nf1 = 0
        nf2 = 0
        learning_rate = 0.1  # for updating probabilities

        # main loop
        while fevals < budget:
            # Linear population size reduction
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new

            # Dynamic pbest proportion: starts at 0.2, decreases to 0.05
            progress = fevals / budget
            p = 0.2 - 0.15 * progress

            sorted_idx = np.argsort(fitness)
            pbest_num = max(1, int(p * NP))
            pbest_pool = sorted_idx[:pbest_num]

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            S_CR = []
            S_F = []
            delta_fitness = []
            strategy_used = []  # which strategy was successful

            for i in range(NP):
                r = np.random.randint(H)
                # sample CR from truncated Cauchy
                CR = np.random.normal(M_CR[r], 0.1)
                CR = np.clip(CR, 0., 1.)
                # sample F from Cauchy with location M_F[r] and scale 0.1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Choose mutation strategy based on probabilities
                if np.random.rand() < prob1 / (prob1 + prob2):
                    # Strategy 1: current-to-pbest/1
                    pbest = pop[np.random.choice(pbest_pool)]
                    r1 = i
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    # r2 from union of population and archive
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx == i or idx == r1:
                            continue
                        break
                    if idx < NP:
                        r2_vec = combined[idx]
                    else:
                        r2_vec = archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                    strategy = 1
                else:
                    # Strategy 2: rand/1
                    # select three distinct indices different from i
                    candidates = list(range(NP))
                    candidates.remove(i)
                    r1, r2, r3 = np.random.choice(candidates, size=3, replace=False)
                    v = pop[r1] + F * (pop[r2] - pop[r3])
                    strategy = 2

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflection boundary handling
                for j in range(dim):
                    if u[j] < lb[j]:
                        u[j] = 2 * lb[j] - u[j]
                    elif u[j] > ub[j]:
                        u[j] = 2 * ub[j] - u[j]
                    # clamp if still out (rare)
                    u[j] = np.clip(u[j], lb[j], ub[j])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = max(abs(fitness[i] - f_u), 1e-30)
                    delta_fitness.append(delta)
                    strategy_used.append(strategy)

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # add parent to archive
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

            # Update memory for successful parameters
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Update strategy probabilities using Success-History Adaptation
            # Count successful and failed trials for each strategy this generation
            ns1 = strategy_used.count(1)
            ns2 = strategy_used.count(2)
            nf1 = NP - ns1  # approximate fails per strategy (rough)
            nf2 = NP - ns2  # but each trial used one strategy; we can compute accurately
            # Actually we need exact counts: we know how many times each strategy was used.
            # We'll accumulate over generations using exponential smoothing.
            # Instead, we use the principle of probability update from SHADE-ensemble:
            #   prob1 = ns1*(ns1+nf1) / (ns1*(ns1+nf1) + ns2*(ns2+nf2))
            # but we need nf1, nf2 per strategy. We can keep running counters.
            # For simplicity, we use the success rate over last generation:
            #   sr1 = ns1 / max(1, (ns1+nf1)), sr2 = ns2 / max(1, (ns2+nf2))
            # then prob1 = (1 - learning_rate) * prob1 + learning_rate * sr1/(sr1+sr2+1e-30)
            # But we don't have nf1,nf2 exactly recorded. Let's record usage counts.
            # We'll maintain usage counters and reset each generation.
            # Since we only have strategy_used list of successes, we need to count how many times each strategy was used.
            # We can count total trials per strategy in this generation.
            # We'll compute usage1 and usage2 by counting how many trials were assigned to each strategy.
            # But we didn't record usage per trial. We'll add a small overhead: record usage in the loop.
            # To avoid increasing code complexity, we'll use a heuristic: update prob1 based on ratio of successes.
            # Another way: use the approach from SaNSDE: update prob1 = prob1 + 0.1 * (ns1/len(S_CR) - prob1) if S_CR nonempty.
            # That is simpler.
            if S_CR:
                sr1 = ns1 / len(S_CR) if len(S_CR) > 0 else 0
                prob1 = (1 - learning_rate) * prob1 + learning_rate * sr1
                prob1 = np.clip(prob1, 0.1, 0.9)
                prob2 = 1 - prob1

        return self.best_f, self.best_x