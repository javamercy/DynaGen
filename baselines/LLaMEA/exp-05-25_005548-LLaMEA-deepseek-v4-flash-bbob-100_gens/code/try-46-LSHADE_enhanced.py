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

        # Initial population size (logarithmic in dim)
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init
        max_archive = NP

        if budget < NP:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Improved Latin Hypercube initialization
        pop = np.empty((NP, dim))
        for j in range(dim):
            perm = np.random.permutation(NP)
            pop[:, j] = lb[j] + (perm + np.random.uniform(size=NP)) / NP * (ub[j] - lb[j])
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))

        # Parameter memory
        H = 12
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation counters
        stagnation = 0
        last_improvement = 0

        while fevals < budget:
            # Goal: reduce population linearly to 4
            remaining_evals = budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    # Prune archive by worst fitness to keep best diversity
                    # Only keep individuals with fitness in archive? Not trivial.
                    # Simply keep random subset (as original) but we can do better by keeping those with best parent fitness? Not stored. Keep random.
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Sigmoid pbest ratio (remains high early, decays fast then levels off)
            progress = 1.0 - remaining_evals / budget
            ratio = 0.2 / (1.0 + np.exp(4.0 * (0.5 - progress)))
            p = max(0.05, min(0.2, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []
            # For rank-based weighting: store improvement values
            imp = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                pbest = pop[np.random.choice(pbest_pool)]

                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx != i and idx != r1:
                        break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)

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

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u)
                    delta_fitness.append(max(delta, 1e-30))
                    imp.append(delta)

                    new_pop[i] = u
                    new_fitness[i] = f_u

                    # Archive: append parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        # Remove the parent with worst fitness among archive (approximate: choose from archive based on some proxy)
                        # Since we don't store fitness, we randomly pick (original method)
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        stagnation = 0
                    else:
                        stagnation += 1
                else:
                    stagnation += 1

                if fevals >= budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= budget:
                break

            # Update memory using rank-based weighting (improves over delta weighting)
            if len(S_CR) > 0:
                # Rank-based weights (exponential ranking, 1/(i+1) style)
                idx_sorted = np.argsort(imp)[::-1]  # larger imp first
                ranks = np.argsort(idx_sorted) + 1  # rank 1 for largest improvement
                w = (1.0 / ranks) / np.sum(1.0 / ranks)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F)**2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Restart if stagnation for many function evaluations
            if stagnation > 0.1 * budget:
                # Keep best half, reinitialize the rest with random
                sorted_idx = np.argsort(fitness)
                keep = max(2, NP // 2)
                new_pop = pop[sorted_idx[:keep]].copy()
                new_fitness = fitness[sorted_idx[:keep]].copy()
                for _ in range(NP - keep):
                    x = np.random.uniform(lb, ub)
                    f = func(x)
                    fevals += 1
                    new_pop = np.vstack((new_pop, x))
                    new_fitness = np.append(new_fitness, f)
                    if f < self.best_f:
                        self.best_f = f
                        self.best_x = x.copy()
                    if fevals >= budget:
                        break
                pop = new_pop
                fitness = new_fitness
                NP = pop.shape[0]
                archive = np.empty((0, dim))
                stagnation = 0
                # Reset memory? Keep memory to retain learned parameters.

        return self.best_f, self.best_x