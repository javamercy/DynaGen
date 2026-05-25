import numpy as np

class LSHADE_improved:
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

        # --- Initial population size (LSHADE style) ---
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP_init

        # Latin Hypercube Sampling (LHS) for better initial coverage
        def lhs_uniform(n, d, low, high):
            samples = np.zeros((n, d))
            for j in range(d):
                perm = np.random.permutation(n)
                samples[:, j] = (perm + np.random.uniform(size=n)) / n
            # scale to bounds
            for j in range(d):
                samples[:, j] = low[j] + (high[j] - low[j]) * samples[:, j]
            return samples

        if budget < NP:  # trivial random search
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        pop = lhs_uniform(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Memory for CR and F
        H = 20       # larger memory
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation tracking
        stall_evals = 0
        max_stall = max(10, int(0.03 * budget))  # restart if no improvement for 3% budget

        # --- Main loop ---
        while fevals < budget:
            # Linear population reduction (NP from NP_init to 4)
            remaining = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                # reduce archive size
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # --- Diversity‑based pbest ratio ---
            # compute average std across dimensions, normalized by range width
            if NP > 1:
                std_dim = np.std(pop, axis=0)
                avg_std = np.mean(std_dim) / (np.mean(ub - lb) + 1e-30)
                # low diversity => increase pbest (focus exploitation), high diversity => decrease
                p_base = 0.1 + 0.15 * (1.0 - min(avg_std, 1.0))
                p = np.clip(p_base, 0.05, 0.2)
            else:
                p = 0.2
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # --- Mutation and crossover ---
            for i in range(NP):
                # Sample CR and F from memory using Cauchy
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random indices
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

                # Mutation: current‑to‑pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling (mirror twice, then random fallback)
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                still_low = u < lb
                still_high = u > ub
                u[still_low] = np.random.uniform(lb[still_low], ub[still_low])
                u[still_high] = np.random.uniform(lb[still_high], ub[still_high])

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
                        stall_evals = 0   # reset stagnation counter on improvement
                    else:
                        stall_evals += 1
                else:
                    stall_evals += 1

                if fevals >= budget:
                    break

            # Update population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory using weighted means
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # weighted Lehmer mean for F
                F_arr = np.array(S_F)
                sum_sq = np.sum(w * F_arr**2)
                sum_w = np.sum(w * F_arr)
                mean_F = sum_sq / (sum_w + 1e-30)
                M_CR[mem_idx] = np.clip(mean_CR, 0.0, 1.0)
                M_F[mem_idx] = np.clip(mean_F, 0.0, 1.0)
                mem_idx = (mem_idx + 1) % H

            # --- Stagnation restart ---
            if stall_evals >= max_stall and fevals < budget - 10:
                # restart: keep best 30% individuals, generate 70% new random points
                keep_num = max(1, int(0.3 * NP))
                sorted_idx = np.argsort(fitness)
                keep_idx = sorted_idx[:keep_num]
                new_pop = pop[keep_idx].copy()
                new_fitness = fitness[keep_idx].copy()
                # generate additional individuals via LHS (or random)
                extra_num = NP - keep_num
                if extra_num > 0:
                    extra = np.random.uniform(lb, ub, (extra_num, dim))
                    # add small perturbation to kept individuals to avoid exact duplicates
                    for i in range(keep_num):
                        noise = np.random.normal(0, 0.1 * (ub - lb), dim)
                        perturbed = new_pop[i] + noise
                        # clip to bounds
                        perturbed = np.clip(perturbed, lb, ub)
                        new_pop = np.vstack((new_pop, perturbed))
                    # trim if overshoot
                    if len(new_pop) > NP:
                        new_pop = new_pop[:NP]
                        new_fitness = new_fitness[:NP]
                    else:
                        # evaluate extra individuals
                        for j in range(extra_num):
                            x = extra[j]
                            f = func(x)
                            fevals += 1
                            if fevals > budget:
                                break
                            new_pop = np.vstack((new_pop, x[np.newaxis, :]))
                            new_fitness = np.append(new_fitness, f)
                            if f < self.best_f:
                                self.best_f = f
                                self.best_x = x.copy()
                # replace population and reset archive partly
                pop = new_pop[:NP]
                fitness = new_fitness[:NP]
                archive = np.empty((0, dim))   # clear archive to avoid outdated vectors
                max_archive = NP
                stall_evals = 0
                # reset memory to initial values to encourage exploration
                M_CR[:] = 0.5
                M_F[:] = 0.5
                mem_idx = 0

        return self.best_f, self.best_x