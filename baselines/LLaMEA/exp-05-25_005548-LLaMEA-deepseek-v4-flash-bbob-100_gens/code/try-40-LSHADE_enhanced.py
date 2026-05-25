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

        # Population size: typical LSHADE size
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
        NP = NP_init
        max_archive = NP_init

        # If budget too small, do random search
        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Initial population via Sobol-like low-discrepancy sampling (using random shuffling as approximation)
        # Here we use uniform random in each dimension, but we can add a simple Latin hypercube style
        # To improve coverage, generate points in a stratified way
        # For simplicity, we use a crude Sobol approximation: divide each axis into sqrt(NP) intervals
        # But with higher dimensions, use random permutation per dimension.
        # Real Sobol requires scipy; we use a simple quasi-random: each dimension random but with offsets.
        # We'll generate points using a small perturbation of a grid to cover space better.
        pop = np.random.uniform(lb, ub, (NP, dim))
        # Improve initial diversity: use random shifts per dimension
        # Actually keep as uniform random for simplicity, but ensure no duplicates? Not needed.
        # Alternatively, use a better initialization: generate NP points by perturbing a grid
        # Since dim can be high, we stick to uniform random, but we add a small diversity trick:
        # after generation, if any two points are very close, perturb one slightly.
        # Not necessary.

        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()
        last_improve_evals = 0

        archive = np.empty((0, dim))

        # Memory for CR and F - larger memory for adaptation
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Cauchy scale parameter: adaptive decay
        scale_init = 0.1
        scale_final = 0.05

        # Main loop
        while fevals < budget:
            remaining_evals = budget - fevals
            # Linear population reduction (NP from NP_init to 4)
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

            # Adaptive pbest ratio: starts at 0.2, ends at 0.05
            ratio = 0.2 - 0.15 * (1 - remaining_evals / budget)
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Cauchy scale for this generation (linear decay)
            scale = scale_init - (scale_init - scale_final) * (1 - remaining_evals / budget)

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                # Sample CR from Cauchy truncated to [0,1]
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * scale + M_CR[r]
                CR = max(0., min(1., CR))
                # Sample F from Cauchy truncated to >0 and <=1
                F = np.random.standard_cauchy() * scale + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * scale + M_F[r]
                F = min(F, 1.)

                # Select pbest
                pbest = pop[np.random.choice(pbest_pool)]

                # Random distinct indices
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Combine pop and archive for r2
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

                # Binomial crossover
                u = pop[i].copy()
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

                    # Update archive (append parent)
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    # Update best and reset stagnation counter
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        last_improve_evals = fevals

                if fevals >= budget:
                    break

            # Replace population
            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory with weighted averages
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Stagnation detection: if no improvement in 5% of budget, restart some worst points
            if fevals - last_improve_evals > 0.05 * budget:
                # Keep best 20% and reinitialize the rest randomly (preserving archive)
                num_keep = max(1, int(0.2 * NP))
                sorted_idx = np.argsort(fitness)
                keep_indices = sorted_idx[:num_keep]
                keep_pop = pop[keep_indices].copy()
                keep_fit = fitness[keep_indices].copy()
                # Generate new random points for the rest
                new_idx = sorted_idx[num_keep:]
                for idx in new_idx:
                    pop[idx] = np.random.uniform(lb, ub)
                    fitness[idx] = func(pop[idx])
                    fevals += 1
                    if fevals >= budget:
                        break
                # Preserve the kept best
                pop[:num_keep] = keep_pop
                fitness[:num_keep] = keep_fit
                last_improve_evals = fevals  # reset counter, avoid multiple restarts

                if fevals >= budget:
                    break

        return self.best_f, self.best_x