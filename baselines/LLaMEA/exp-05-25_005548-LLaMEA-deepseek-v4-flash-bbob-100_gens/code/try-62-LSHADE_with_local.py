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

        # Budget allocation: 80% main DE, 20% local pattern search
        local_frac = 0.2
        local_budget = max(dim * 5, int(local_frac * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling
        NP_init = max(10, min(80, int(18 * np.log(dim)) if dim > 1 else 18))
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

        # Stagnation tracking for restart
        stagnation_counter = 0
        stagnation_limit = max(10, int(0.02 * main_budget / NP))
        prev_best_f = self.best_f
        eval_since_last_restart = 0

        # Main LSHADE loop
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
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
            ratio = 0.2 - 0.15 * (1 - remaining_evals / main_budget)
            p = max(0.05, min(0.2, ratio))
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
                u = np.clip(u, lb, ub)

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

            # Stagnation check
            if self.best_f < prev_best_f:
                stagnation_counter = 0
                prev_best_f = self.best_f
            else:
                stagnation_counter += 1

            # Restart if stagnation
            if stagnation_counter >= stagnation_limit and fevals < main_budget * 0.9:
                # Keep best, reinitialize rest around best with reduced radius
                radius = 0.15 * (ub - lb)
                new_pop_size = NP
                new_pop = [self.best_x.copy()]
                for _ in range(new_pop_size - 1):
                    offset = np.random.uniform(-radius, radius, size=dim)
                    new_x = np.clip(self.best_x + offset, lb, ub)
                    new_pop.append(new_x)
                pop = np.array(new_pop)
                fitness = np.array([func(x) for x in pop])
                fevals += NP
                # Reset archive
                archive = np.empty((0, dim))
                max_archive = NP
                stagnation_counter = 0
                # Update best again (best already included)
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()

            if fevals >= main_budget:
                break

            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                sum_sq = np.sum(w * np.array(S_F) ** 2)
                sum_w = np.sum(w * np.array(S_F))
                mean_F = sum_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # Pattern search local refinement (Hooke-Jeeves style)
        if local_budget > 0:
            x_ref = self.best_x.copy()
            f_ref = self.best_f
            step_size = 0.1 * (ub - lb)
            min_step = 1e-7 * (ub - lb)
            evals = 0
            # Pattern search loop
            while evals < local_budget:
                improved = False
                for d in range(dim):
                    if evals >= local_budget:
                        break
                    # Positive direction
                    x_candidate = x_ref.copy()
                    x_candidate[d] += step_size[d]
                    x_candidate[d] = np.clip(x_candidate[d], lb[d], ub[d])
                    f_candidate = func(x_candidate)
                    evals += 1
                    if f_candidate < f_ref:
                        x_ref = x_candidate
                        f_ref = f_candidate
                        improved = True
                        step_size[d] *= 1.2  # expand step
                        continue
                    # Negative direction
                    x_candidate[d] = x_ref[d] - step_size[d]
                    x_candidate[d] = np.clip(x_candidate[d], lb[d], ub[d])
                    f_candidate = func(x_candidate)
                    evals += 1
                    if f_candidate < f_ref:
                        x_ref = x_candidate
                        f_ref = f_candidate
                        improved = True
                        step_size[d] *= 1.2
                    else:
                        step_size[d] *= 0.5  # contract step
                # If no improvement in any direction, shrink all steps
                if not improved:
                    step_size *= 0.5
                # Clamp step sizes to min/max
                step_size = np.clip(step_size, min_step, 0.5 * (ub - lb))
                if np.all(step_size < min_step * 2):
                    break
            if f_ref < self.best_f:
                self.best_f = f_ref
                self.best_x = x_ref.copy()

        return self.best_f, self.best_x