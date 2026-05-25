import numpy as np

class LSHADE_CMA:
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

        # Reserve budget for CMA-ES local search (20% of total)
        local_budget = max(10 * dim, int(0.20 * budget))
        main_budget = budget - local_budget

        if main_budget < 20:
            # Pure random search if budget very small
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---- Latin Hypercube Sampling for initial population ----
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

        # Archive and memory (improved)
        archive = np.empty((0, dim))
        max_archive = NP
        H = 20
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # ---- Main DE loop with linear population reduction (L-SHADE) ----
        generation = 0
        no_improve_count = 0
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

            # Adaptive pbest ratio (decreases from 0.2 to 0.05)
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
            improved = False

            for i in range(NP):
                # Sample CR and F from memory
                r = np.random.randint(H)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(1., F)

                pbest = pop[np.random.choice(pbest_pool)]
                r1 = np.random.randint(NP)
                while r1 == i:
                    r1 = np.random.randint(NP)

                # Select r2 from union of pop and archive (excluding current and r1)
                combined = np.vstack((pop, archive))
                while True:
                    idx = np.random.randint(len(combined))
                    if idx == i or (idx < NP and idx == r1):
                        continue
                    break
                r2_vec = combined[idx]
                # Mutation: current-to-pbest/1
                v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2_vec)
                # Crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]
                # Bounce-back boundary handling
                for j in range(dim):
                    if u[j] < lb[j]:
                        u[j] = lb[j] + np.random.rand() * (pop[i][j] - lb[j])
                    elif u[j] > ub[j]:
                        u[j] = ub[j] - np.random.rand() * (ub[j] - pop[i][j])

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    delta_fitness.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    # Archive old parent
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                        improved = True

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            if fevals >= main_budget:
                break

            # Update memory (Lehmer mean)
            if S_CR:
                w = np.array(delta_fitness) / np.sum(delta_fitness)
                mean_CR = np.sum(w * np.array(S_CR))
                # Weighted Lehmer mean for F
                sum_wF = np.sum(w * np.array(S_F))
                sum_wF2 = np.sum(w * np.array(S_F) ** 2)
                mean_F = sum_wF2 / sum_wF if sum_wF > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # Stagnation detection: if no improvement for 30% of generations, reinitialize half population
            generation += 1
            if improved:
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count > max(10, int(0.3 * generation)):
                # Replace half of the population with LHS points around best
                num_repl = max(1, NP // 3)
                repl = lhs(num_repl, dim, lb, ub)
                pop[-num_repl:] = repl
                fitness[-num_repl:] = np.array([func(x) for x in repl])
                fevals += num_repl
                no_improve_count = 0
                # Reset memory partially
                M_CR[:] = 0.5
                M_F[:] = 0.5

        # ---- Bounded CMA-ES local search using remaining budget ----
        if local_budget > 0:
            x0 = self.best_x.copy()
            f0 = self.best_f

            # Initialize CMA-ES parameters
            N = dim
            m = x0.copy()
            sigma = 0.3  # initial step size (relative to range 10)
            C = np.eye(N)
            chiN = np.sqrt(N) * (1.0 - 1.0/(4.0*N) + 1.0/(21.0*N**2))
            pc = np.zeros(N)
            ps = np.zeros(N)
            cc = 4.0 / (N + 4.0)
            cs = (N + 2.0) / (N + 3.0)
            c1 = 2.0 / ((N + 1.3)**2 + 1.0)
            cmu = min(1 - c1, 2.0 * ((N**2 - 2.0*N + 2.0) / (N**2 + 4.0*N + 4.0)))
            damps = 1.0 + 2.0 * max(0.0, np.sqrt((N-1.0)/(N+1.0)) - 1.0) + cs  # damping
            lambda_ = 4 + int(3 * np.log(N))
            mu = max(1, lambda_ // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights**2)

            evals = 0
            while evals < local_budget:
                # Sample offspring
                arz = np.random.randn(lambda_, N)
                arx = m + sigma * (arz @ C.T)  # actually arz @ C.T (since C is symmetric)
                # Actually: need to compute B*D? Use Cholesky
                # Simplify: use eigendecomposition but costly; for small dim we can compute B=U, D=sqrt(eigvals)
                # Better: use Cholesky decomposition
                A = np.linalg.cholesky(C)
                arx = m + sigma * (arz @ A.T)

                # Boundary handling: sample until within bounds (simple truncation with mirror)
                for i in range(lambda_):
                    out_low = arx[i] < lb
                    out_high = arx[i] > ub
                    arx[i, out_low] = lb[out_low] + np.random.rand(np.sum(out_low)) * (ub[out_low] - lb[out_low])
                    arx[i, out_high] = lb[out_high] + np.random.rand(np.sum(out_high)) * (ub[out_high] - lb[out_high])

                # Evaluate
                arf = np.array([func(x) for x in arx])
                evals += lambda_

                # Sort
                idx = np.argsort(arf)
                arx = arx[idx]
                arf = arf[idx]

                # Update mean
                m_old = m.copy()
                m = np.dot(weights, arx[:mu])

                # Update evolution paths
                invsqrtC = np.linalg.inv(np.linalg.cholesky(C))
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (invsqrtC @ (m - m_old)) / sigma
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2*evals/lambda_)) / chiN < 1.4 + 2/(N+1)
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (m - m_old) / sigma

                # Update covariance matrix
                artmp = (arx[:mu] - m_old) / sigma
                C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + cmu * (artmp.T @ np.diag(weights) @ artmp)

                # Update step size
                sigma *= np.exp((np.linalg.norm(ps) / chiN - 1) * cs / damps)

                # Track best
                if arf[0] < self.best_f:
                    self.best_f = arf[0]
                    self.best_x = arx[0].copy()

                # Early stop if budget exhausted
                if evals >= local_budget:
                    break

        return self.best_f, self.best_x