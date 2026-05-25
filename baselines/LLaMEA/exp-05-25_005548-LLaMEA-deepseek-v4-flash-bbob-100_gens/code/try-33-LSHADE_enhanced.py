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

        H = 20  # increased memory size
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Stagnation control
        no_improve_evals = 0
        best_f_prev = self.best_f

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

            # Dynamic pbest ratio: increase from 0.1 to 0.2
            p_min, p_max = 0.1, 0.2
            p = p_min + (p_max - p_min) * (fevals / budget)
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            # Decaying F scale: larger early, smaller late
            scale_F = max(0.02, 0.1 * (1 - fevals / budget))

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            any_improvement = False

            for i in range(NP):
                r = np.random.randint(H)
                # CR from Cauchy truncated to [0,1]
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # F from Cauchy with decaying scale
                F = np.random.standard_cauchy() * scale_F + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * scale_F + M_F[r]
                F = min(F, 1.)

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

                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling (with fallback)
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                # If still out, random re-initialization
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

                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        archive = np.delete(archive, np.random.randint(len(archive)), axis=0)

                    if f_u < self.best_f:
                        any_improvement = True
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= budget:
                    break

            if fevals >= budget:
                break

            # Stagnation detection and restart (reinitialize worst individuals)
            if any_improvement:
                no_improve_evals = 0
                best_f_prev = self.best_f
            else:
                no_improve_evals += NP  # each generation uses NP evals
                if no_improve_evals > 0.02 * budget and NP > 4:
                    # Reinitialize worst 30% of population
                    n_reinit = max(1, int(0.3 * NP))
                    sorted_idx = np.argsort(fitness)
                    worst_idx = sorted_idx[-n_reinit:]
                    pop[worst_idx] = np.random.uniform(lb, ub, (n_reinit, dim))
                    fitness[worst_idx] = np.array([func(x) for x in pop[worst_idx]])
                    fevals += n_reinit
                    no_improve_evals = 0
                    # Also reset some memory entries to encourage exploration
                    for k in range(min(3, H)):
                        M_CR[np.random.randint(H)] = 0.5
                        M_F[np.random.randint(H)] = 0.5
                    if fevals >= budget:
                        break

            # Update population
            pop = new_pop
            fitness = new_fitness

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

        return self.best_f, self.best_x