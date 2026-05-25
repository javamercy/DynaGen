import numpy as np

class LSHADE_with_local:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim
        budget = self.budget

        # Reserve budget for local search (Hooke-Jeeves)
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling for initial population ----
        NP_init = max(10, int(18 * np.log(dim) if dim > 1 else 18))
        NP = NP_init

        def lhs(n, d, low, high):
            result = np.zeros((n, d))
            for i in range(d):
                perm = np.random.permutation(n)
                result[:, i] = low[i] + (perm + np.random.uniform(size=n)) / n * (high[i] - low[i])
            return result

        pop = lhs(NP, dim, lb, ub)
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop (jSO-like adaptation) ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Population size reduction (jSO style)
            NP_new = max(4, int(4 + (NP_init - 4) * (1 - (remaining_evals / max(1, main_budget))**0.3)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # Adaptive pbest ratio (jSO: decreases nonlinearly)
            p = 0.2 * (1 - (remaining_evals / main_budget)**0.3)
            p = max(0.05, min(0.2, p))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            delta_fitness = []

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # Cauchy for both CR and F (jSO style)
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
                    if idx == i or idx == r1:
                        continue
                    break
                r2_vec = combined[idx] if idx < NP else archive[idx - NP]

                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
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

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            if S_CR:
                # Weighted Lehmer mean for F, arithmetic for CR (jSO)
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_F2 = np.sum(w * np.array(S_F)**2)
                sum_F = np.sum(w * np.array(S_F))
                mean_F = sum_F2 / sum_F if sum_F > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Hooke-Jeeves pattern search local refinement ----
        if local_budget > 0:
            x_start = self.best_x.copy()
            f_start = self.best_f
            x_current = x_start.copy()
            f_current = f_start

            # Initial step size
            step0 = 0.1 * (ub - lb)
            step = step0.copy()

            min_step = 1e-5 * (ub - lb)
            evals = 0
            improved = True

            while evals < local_budget and np.any(step > min_step):
                x_try = x_current.copy()
                step_taken = False

                # Exploratory moves along each dimension
                for j in range(dim):
                    # Positive direction
                    x_test = x_try.copy()
                    x_test[j] += step[j]
                    x_test = np.clip(x_test, lb, ub)
                    f_test = func(x_test)
                    evals += 1
                    if f_test < f_current:
                        x_try[j] += step[j]
                        f_current = f_test
                        step_taken = True
                        continue
                    # Negative direction
                    x_test = x_try.copy()
                    x_test[j] -= step[j]
                    x_test = np.clip(x_test, lb, ub)
                    f_test = func(x_test)
                    evals += 1
                    if f_test < f_current:
                        x_try[j] -= step[j]
                        f_current = f_test
                        step_taken = True
                    # If no improvement, leave coordinate unchanged

                if step_taken:
                    # Pattern move: accelerate from x_current to x_try
                    x_move = 2 * x_try - x_current
                    x_move = np.clip(x_move, lb, ub)
                    f_move = func(x_move)
                    evals += 1
                    if f_move < f_current:
                        x_current = x_move
                        f_current = f_move
                        # increase step slightly
                        step = np.minimum(step * 2.0, 0.5 * (ub - lb))
                    else:
                        x_current = x_try
                        # step unchanged (or slight reduction?)
                    improved = True
                else:
                    # No improvement in exploratory moves -> shrink step
                    step = step * 0.5
                    improved = False

                if evals >= local_budget:
                    break

            if f_current < self.best_f:
                self.best_f = f_current
                self.best_x = x_current.copy()

        return self.best_f, self.best_x