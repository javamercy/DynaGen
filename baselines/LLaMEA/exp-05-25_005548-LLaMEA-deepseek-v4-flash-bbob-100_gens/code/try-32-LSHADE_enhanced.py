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

        # Population size: slightly larger than before
        NP_init = max(10, int(20 * np.log(dim) if dim > 1 else 20))
        NP = NP_init
        max_archive = NP

        # Budget too small -> random search
        if budget < NP:
            for _ in range(budget):
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

        archive = np.empty((0, dim))

        # Memory for CR, F and crossover type success
        H = 10
        M_CR = 0.8 * np.ones(H)   # start high for exploitation
        M_F = 0.5 * np.ones(H)
        # Crossover selection: probability of using binomial (0) vs exponential (1)
        M_cross = 0.5 * np.ones(H)  # probability of binomial
        mem_idx = 0

        # Track best improvement for stagnation detection
        stagnation_counter = 0
        stagnation_limit = max(1, int(0.15 * budget))

        # Main loop
        while fevals < budget:
            remaining_evals = budget - fevals
            # Linear population reduction
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

            # pbest ratio: quadratic decay from 0.2 to 0.05
            ratio = 0.2 - 0.15 * (1 - remaining_evals / budget) ** 2
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []
            S_cross = []  # 0 for binomial, 1 for exponential

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # Adaptive crossover probability from memory (mean)
            cross_prob = np.mean(M_cross)

            for i in range(NP):
                # Sample CR
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Sample F
                scale_F = 0.15 * (1 - remaining_evals / budget) + 0.05
                F = np.random.standard_cauchy() * scale_F + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * scale_F + M_F[r]
                F = min(F, 1.)

                # Decide crossover type: 0=binomial, 1=exponential
                use_exponential = np.random.rand() < cross_prob

                # Select pbest
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

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Crossover
                u = pop[i].copy()
                if use_exponential:
                    # Exponential crossover
                    j_rand = np.random.randint(dim)
                    j = j_rand
                    L = 0
                    while np.random.rand() < CR and L < dim:
                        u[j] = v[j]
                        j = (j + 1) % dim
                        L += 1
                else:
                    # Binomial crossover
                    j_rand = np.random.randint(dim)
                    for j in range(dim):
                        if np.random.rand() < CR or j == j_rand:
                            u[j] = v[j]

                # Reflected bound handling
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
                    S_cross.append(1 if use_exponential else 0)  # 1 for successful exponential
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stagnation_counter = 0
                    # else stagnation not reset here, but we count later

                # Check budget
                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            # Stagnation detection: if best not improved for many evals, trigger restart
            if fevals < budget:
                # Check if best improved during this generation
                if self.best_f == fitness[np.argmin(fitness)]:  # no improvement? Actually best_f is global, compare with old best?
                    stagnation_counter += NP  # approximate count
                else:
                    stagnation_counter = 0

                if stagnation_counter >= stagnation_limit:
                    # Restart worst 30% with random points (keep best)
                    replace_num = max(1, int(0.3 * NP))
                    worst_idx = np.argsort(fitness)[-replace_num:]
                    for idx in worst_idx:
                        pop[idx] = np.random.uniform(lb, ub)
                        fitness[idx] = func(pop[idx])
                        fevals += 1
                        if fevals >= budget:
                            break
                    # Reset archive
                    archive = np.empty((0, dim))
                    stagnation_counter = 0
                    # Optionally reset memory? Keep as is.
                    continue  # skip parameter update? Better to apply update after restart.

            if fevals >= budget:
                break

            # Update memories if successes exist
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                # Weighted mean for crossover type
                mean_cross = np.sum(w * np.array(S_cross))  # proportion of exponential successes

                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                M_cross[mem_idx] = mean_cross
                mem_idx = (mem_idx + 1) % H

        return self.best_f, self.best_x