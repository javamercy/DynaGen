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
        NP_init = max(10, int(18 * np.log(dim)) if dim > 1 else 18)
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

        # Initial population and fitness
        pop = np.random.uniform(lb, ub, (NP, dim))
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        # Archive of parent solutions
        archive = np.empty((0, dim))

        # Success-history memory for CR and F (one per strategy)
        H = 10
        M_CR = [0.5] * H
        M_F = [0.5] * H
        mem_idx = 0

        # Dual strategy probabilities (pbest vs rand)
        prob_pbest = 0.5
        # History for strategy success (window of 50 generations)
        success_pbest = 0.0
        success_rand = 0.0
        count_pbest = 0
        count_rand = 0
        learn_period = 50
        gen_since_learn = 0

        # Stagnation detection for restart
        stagnation_limit = max(500, budget // 20)
        fevals_since_improvement = 0

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

            # Store for generation statistics
            S_CR = []
            S_F = []
            delta_fitness = []
            # For strategy adaptation
            gen_success_pbest = 0
            gen_success_rand = 0
            gen_count_pbest = 0
            gen_count_rand = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Choose strategy
                if np.random.rand() < prob_pbest:
                    strategy = 'pbest'
                    gen_count_pbest += 1
                else:
                    strategy = 'rand'
                    gen_count_rand += 1

                # Sample CR and F
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Mutation
                if strategy == 'pbest':
                    pbest = pop[np.random.choice(pbest_pool)]
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx != i and idx != r1:
                            break
                    r2 = combined[idx] if idx < NP else archive[idx - NP]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)
                else:  # current-to-rand/1 (no archive)
                    r1, r2, r3 = np.random.choice([j for j in range(NP) if j != i], 3, replace=False)
                    v = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

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
                # If still out, random within bounds
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

                f_u = func(u)
                fevals += 1

                # Selection
                if f_u <= fitness[i]:
                    # Successful trial
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Update archive with parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        fevals_since_improvement = 0
                    # Track strategy success
                    if strategy == 'pbest':
                        gen_success_pbest += 1
                    else:
                        gen_success_rand += 1

                else:
                    if f_u == fitness[i] and f_u < self.best_f:
                        # Handle equality but still count as no improvement? keep best unchanged
                        pass
                    # No improvement from this trial

                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory for successful parameters
            if len(S_CR) > 0:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Update strategy probabilities (window-based)
            gen_since_learn += 1
            success_pbest += gen_success_pbest
            success_rand += gen_success_rand
            count_pbest += gen_count_pbest
            count_rand += gen_count_rand
            if gen_since_learn >= learn_period:
                # Update probability based on success rates
                rate_pbest = success_pbest / max(count_pbest, 1)
                rate_rand = success_rand / max(count_rand, 1)
                # Smooth update
                prob_pbest = 0.8 * prob_pbest + 0.2 * (rate_pbest / max(rate_pbest + rate_rand, 1e-10))
                prob_pbest = max(0.1, min(0.9, prob_pbest))
                # Reset counters
                success_pbest = 0
                success_rand = 0
                count_pbest = 0
                count_rand = 0
                gen_since_learn = 0

            # Restart if stagnation
            fevals_since_improvement += NP  # approximate
            if fevals_since_improvement >= stagnation_limit and NP > 4:
                # Keep best solution, reinitialize population (smaller size)
                new_NP = max(4, NP // 2)
                new_pop = np.random.uniform(lb, ub, (new_NP, dim))
                new_fitness = np.array([func(x) for x in new_pop])
                fevals += new_NP
                # Keep best from new population
                local_best_idx = np.argmin(new_fitness)
                if new_fitness[local_best_idx] < self.best_f:
                    self.best_f = new_fitness[local_best_idx]
                    self.best_x = new_pop[local_best_idx].copy()
                # Merge with best solution from previous run
                new_pop = np.vstack([self.best_x.reshape(1, -1), new_pop])
                new_fitness = np.append(np.array([self.best_f]), new_fitness)
                pop = new_pop
                fitness = new_fitness
                NP = new_NP + 1
                # Reset archive and memories
                archive = np.empty((0, dim))
                mem_idx = 0
                M_CR = [0.5] * H
                M_F = [0.5] * H
                # Reset stagnation counter
                fevals_since_improvement = 0
                # Reset strategy adaptation
                prob_pbest = 0.5
                success_pbest = 0
                success_rand = 0
                count_pbest = 0
                count_rand = 0
                gen_since_learn = 0
                # Update archive size
                max_archive = NP
                # Continue loop

        return self.best_f, self.best_x