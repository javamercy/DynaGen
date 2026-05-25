import numpy as np

class LSHADE_AER:
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

        # initial population size (LSHADE typical: min(18*log(dim), dim*2) etc.)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        if budget < NP:
            for i in range(budget):
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

        # SHADE memory
        H = 5
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # strategy probabilities (0: current-to-pbest/1, 1: current-to-rand/1)
        strat_prob = np.array([0.5, 0.5])
        strat_success = np.zeros(2)
        strat_attempts = np.zeros(2)

        # reset detection
        stall_count = 0
        stall_threshold = max(10, int(budget * 0.03 / NP_init))  # ~3% budget in generations

        # linear population reduction limits
        NP_min = 4

        # keep best_f for stall detection
        prev_best_f = self.best_f

        def generate_offspring(i, strategy):
            """Generate offspring for individual i using given strategy."""
            nonlocal mem_idx, pop, fitness, archive, M_CR, M_F

            r = np.random.randint(H)
            CR = np.random.normal(M_CR[r], 0.1)
            CR = np.clip(CR, 0., 1.)
            F = np.random.standard_cauchy() * 0.1 + M_F[r]
            while F <= 0.:
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
            F = min(F, 1.)

            # adaptive pbest ratio: linearly decreasing from 0.2 to 0.1
            p = 0.2 - 0.1 * (fevals / budget)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]
            pbest = pop[np.random.choice(pbest_pool)]

            # select distinct r1
            r1 = np.random.randint(NP)
            while r1 == i:
                r1 = np.random.randint(NP)

            # select r2 from pop ∪ archive
            combined = np.vstack((pop, archive))
            idx = np.random.randint(len(combined))
            while idx == i or idx == r1:
                idx = np.random.randint(len(combined))
            if idx < NP:
                r2_vec = combined[idx]
            else:
                r2_vec = archive[idx - NP]

            if strategy == 0:  # current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
            else:               # current-to-rand/1 (no pbest bias)
                v = pop[i] + F * (pop[r1] - r2_vec)

            # binomial crossover
            u = pop[i].copy()
            j_rand = np.random.randint(dim)
            for j in range(dim):
                if np.random.rand() < CR or j == j_rand:
                    u[j] = v[j]
            u = np.clip(u, lb, ub)
            return u, CR, F

        # main loop
        while fevals < budget:
            # linear population reduction
            remaining_evals = budget - fevals
            NP_new = max(NP_min, int(NP_min + (NP_init - NP_min) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # also adjust archive size? keep as is (random removal later)
                if len(archive) > max_archive:
                    archive = archive[np.random.choice(len(archive), size=max_archive, replace=False)]

            # reset stall counters if improved
            if self.best_f < prev_best_f:
                stall_count = 0
                prev_best_f = self.best_f
            else:
                stall_count += 1

            # restart if stuck
            if stall_count >= stall_threshold:
                # keep best individual, reinitialize rest
                best = pop[np.argmin(fitness)]
                new_pop = [best]
                for _ in range(NP - 1):
                    if np.random.rand() < 0.5:
                        # Gaussian around best
                        x = best + np.random.normal(0, 0.2 * (ub - lb))  # scale 0.2 * range
                    else:
                        # uniform random
                        x = np.random.uniform(lb, ub)
                    x = np.clip(x, lb, ub)
                    new_pop.append(x)
                pop = np.array(new_pop)
                fitness = np.array([func(x) for x in pop])
                fevals += NP - 1  # careful: best already evaluated, we evaluate the rest
                # reset memory
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0
                # reset strategy probabilities
                strat_prob[:] = 0.5
                strat_success[:] = 0
                strat_attempts[:] = 0
                stall_count = 0
                prev_best_f = self.best_f
                # re-evaluate best? already has its fitness
                continue

            # strategy success accumulation
            S_CR = [[], []]
            S_F = [[], []]
            delta_fitness = [[], []]

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # assign strategy to each individual
            strategies = np.random.choice(2, size=NP, p=strat_prob)

            for i in range(NP):
                strat = strategies[i]
                u, CR, F = generate_offspring(i, strat)
                f_u = func(u)
                fevals += 1

                strat_attempts[strat] += 1

                if f_u <= fitness[i]:
                    strat_success[strat] += 1
                    S_CR[strat].append(CR)
                    S_F[strat].append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness[strat].append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # archive
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

            # update strategy probabilities (weighted success rates)
            for s in range(2):
                if strat_attempts[s] > 0:
                    sr = strat_success[s] / strat_attempts[s]
                else:
                    sr = 0.0
                # smoothing
                strat_prob[s] = 0.5 * strat_prob[s] + 0.5 * sr
                # ensure sum=1
                strat_prob = strat_prob / strat_prob.sum()
                # reset counters (for next generation)
                strat_success[s] = 0
                strat_attempts[s] = 0

            # update SHADE memory (use all successful strategies combined, weighted by delta)
            all_CR = []
            all_F = []
            all_delta = []
            for s in range(2):
                all_CR.extend(S_CR[s])
                all_F.extend(S_F[s])
                all_delta.extend(delta_fitness[s])
            if all_CR:
                w = np.array(all_delta) / np.sum(all_delta)
                mean_CR = np.sum(w * np.array(all_CR))
                sum_sq = np.sum(w * np.array(all_F)**2)
                sum_w = np.sum(w * np.array(all_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x