import numpy as np

class LSHADE_with_CSA_ES:
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

        # Reserve budget for local search (cumulative step-size ES)
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        if main_budget < 10:
            # Pure random search if budget too small
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # Latin Hypercube Sampling for initial population
        NP_init = int(18 * np.log(dim) if dim > 1 else 18)
        NP_init = max(10, NP_init)
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

        # Archive and memory (improved: archive size tracks current NP)
        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # Main DE loop with linear population reduction
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

            # Adaptive pbest ratio (decreasing from 0.2 to 0.05)
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
                # Generate CR from Cauchy, clamp to [0,1]
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                # Generate F from Cauchy, ensure >0, clamp to <=1
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # Mutation strategy: current-to-pbest/1 with archive
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
                    # Update archive
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

            # Update memory with success-based adaptation (weighted Lehmer for F, arithmetic for CR)
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                wF = np.array(S_F) * w
                mean_F = np.sum(wF * np.array(S_F)) / (np.sum(wF) + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- Local search: (1+1)-ES with cumulative step-size adaptation ----
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f

            # Initialize (1+1)-ES parameters
            mean = x_best.copy()
            sigma = 0.2 * (ub - lb)  # initial step size as 20% of domain
            p_path = np.zeros(dim)   # evolution path
            c_c = 2.0 / (dim + 2.0)  # cumulation rate
            d = dim                   # damping factor
            chi_N = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim * dim))

            evals_local = 0
            while evals_local < local_budget:
                # Sample candidate
                z = np.random.normal(0, 1, size=dim)
                x_new = mean + sigma * z
                # Boundary handling: reflect and clip
                out_low = x_new < lb
                out_high = x_new > ub
                x_new[out_low] = 2 * lb[out_low] - x_new[out_low]
                x_new[out_high] = 2 * ub[out_high] - x_new[out_high]
                x_new = np.clip(x_new, lb, ub)
                f_new = func(x_new)
                evals_local += 1

                # Selection and step-size adaptation
                if f_new < f_best:
                    f_best = f_new
                    x_best = x_new.copy()
                    # Success -> update mean
                    mean = x_new
                    p_path = (1 - c_c) * p_path + np.sqrt(c_c * (2 - c_c)) * z
                else:
                    p_path = (1 - c_c) * p_path

                # Update step size
                sigma = sigma * np.exp((c_c / d) * (np.linalg.norm(p_path) / chi_N - 1.0))

                if evals_local >= local_budget:
                    break

            if f_best < self.best_f:
                self.best_f = f_best
                self.best_x = x_best.copy()

        return self.best_f, self.best_x