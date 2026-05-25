import numpy as np

class LSHADE_with_local_improved:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_f = np.inf
        self.best_x = None

    def __call__(self, func):
        lb = np.array(func.bounds.lb, dtype=float)
        ub = np.array(func.bounds.ub, dtype=float)
        dim = self.dim
        budget = self.budget

        # Reserve budget for local search (CMA-ES)
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget

        # If budget is too small, use pure random search
        if main_budget < 20:
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling for initial population ----
        NP_init = max(10, int(18 * np.log(dim))) if dim > 1 else 18
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

        # Archive for DE (size up to 2*NP)
        archive = np.empty((0, dim))
        max_archive = 2 * NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop with linear population reduction (jSO style) ----
        while fevals < main_budget:
            remaining_evals = main_budget - fevals
            # Linear population reduction from NP_init to 4
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining_evals / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]].copy()
                fitness = fitness[sorted_idx[:NP_new]].copy()
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = 2 * NP

            # Adaptive pbest ratio: linear from 0.2 to 0.05
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
                # Generate CR with Cauchy distribution
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = np.clip(CR, 0.0, 1.0)
                # Generate F with Cauchy distribution (jSO uses Lehmer mean later)
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.0:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.0)

                # Choose pbest individual
                pbest = pop[np.random.choice(pbest_pool)]
                # Random index different from i
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)
                # Random index from union of pop and archive (avoid i and r1)
                combined = np.vstack((pop, archive))
                # Select a random vector different from pop[i] and pop[r1]
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or idx == r1:
                        continue
                    # Check if it's a valid index in archive mapping
                    break
                r2_vec = combined[idx]

                # Mutation: current-to-pbest/1 with archive
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                # Binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # Reflected boundary handling
                u = np.where(u < lb, 2*lb - u, u)
                u = np.where(u > ub, 2*ub - u, u)
                # If still out of bounds, reset uniformly in bounds
                out = (u < lb) | (u > ub)
                u[out] = np.random.uniform(lb[out], ub[out])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        del_idx = np.random.randint(len(archive))
                        archive = np.delete(archive, del_idx, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update success memories using weighted Lehmer mean (jSO style)
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                # Weighted arithmetic mean for CR
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                mean_F = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

        # ---- (1+1)-CMA-ES local search using remaining budget ----
        if local_budget > 0:
            # Initialize (1+1)-CMA-ES
            x_mean = self.best_x.copy()
            sigma = 0.2 * (ub - lb)  # initial step size, scaled to domain
            # Diagonal covariance matrix initially identity
            C = np.eye(dim)
            # Evolution path
            pc = np.zeros(dim)
            # Learning rates
            c_c = 4.0 / (dim + 4.0)  # cumulation for covariance
            c_sigma = 2.0 / (dim + 2.0)  # step size adaptation
            damps = 2.0 + dim * 0.3  # damping for sigma
            # Expected length of evolution path
            chi_n = np.sqrt(dim) * (1.0 - 1.0 / (4.0*dim) + 1.0 / (21.0*dim*dim))

            evals = 0
            while evals < local_budget:
                # Sample a candidate
                z = np.random.randn(dim)
                x_trial = x_mean + sigma * (C @ z)  # because C is symmetric, use matrix multiplication
                x_trial = np.clip(x_trial, lb, ub)
                f_trial = func(x_trial)
                evals += 1

                if f_trial < self.best_f:
                    # Success: update mean, evolution paths, covariance, step size
                    delta = x_trial - x_mean
                    x_mean = x_trial.copy()
                    self.best_f = f_trial
                    self.best_x = x_trial.copy()

                    # Update evolution paths
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c)) * delta / sigma
                    # Covariance matrix rank-1 update
                    C = (1 - 1/dimps) * C + (1/dimps) * np.outer(pc, pc)

                    # Step size adaptation
                    sigma *= np.exp((np.linalg.norm(pc) / chi_n - 1) * c_sigma / damps)
                else:
                    # Failure: update evolution path and step size (pc unchanged)
                    # Update step size using negative update
                    sigma *= np.exp((np.linalg.norm(pc) / chi_n - 1) * c_sigma / damps)
                    # Optionally adjust covariance? Not in (1+1)-CMA-ES standard, but we can leave it

                # Enforce minimal step size to avoid complete stall
                sigma = max(sigma, 1e-12 * (ub - lb).mean())

            # Final best is already updated inside loop if improved

        return self.best_f, self.best_x