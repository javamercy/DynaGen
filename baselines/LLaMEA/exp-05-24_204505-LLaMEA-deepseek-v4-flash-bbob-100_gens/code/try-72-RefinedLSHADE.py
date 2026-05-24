import numpy as np

class RefinedLSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        max_evals = self.budget

        # Population size parameters
        N_init = min(max(10 * dim, 50), max_evals // 2)
        N_min = max(4, int(dim / 5))
        N = N_init

        # Latin hypercube initialization
        samples = np.random.uniform(0, 1, (N, dim))
        samples = lb + samples * (ub - lb)
        pop = samples.copy()
        fitness = np.full(N, np.inf)
        for i in range(N):
            fitness[i] = func(pop[i])
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()
        n_evals = N

        # Archive for DE mutation
        archive = np.empty((0, dim))
        archive_max = N

        # Success-history memory for F and CR
        H = 10
        MF = np.ones(H) * 0.5
        MCR = np.ones(H) * 0.8
        memory_idx = 0

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.15 * max_evals
        no_improve_local = 0
        restart_local_threshold = 0.1 * max_evals

        # CMA-ES local search parameters
        def cma_es(mean, sigma, max_evals_local, lambda_=None, mu_=None):
            """Lightweight CMA-ES from best point."""
            if lambda_ is None:
                lambda_ = 4 + int(3 * np.log(dim))
            if mu_ is None:
                mu_ = lambda_ // 2
            w = np.log(mu_ + 0.5) - np.log(np.arange(1, mu_ + 1))
            w = w / np.sum(w)
            mueff = 1.0 / np.sum(w ** 2)
            cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            cs = (mueff + 2) / (dim + mueff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            dams = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

            xmean = mean.copy()
            sigma = sigma
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            invsqrtC = np.eye(dim)
            eigeneval = 0
            max_evals_used = 0
            best_val = np.inf
            best_x = xmean.copy()

            while max_evals_used < max_evals_local:
                # Generate offspring
                arx = np.zeros((dim, lambda_))
                arfitness = np.full(lambda_, np.inf)
                for k in range(lambda_):
                    arx[:, k] = xmean + sigma * B @ (D * np.random.randn(dim))
                    arx[:, k] = np.clip(arx[:, k], lb, ub)
                    arfitness[k] = func(arx[:, k])
                    max_evals_used += 1
                    if arfitness[k] < best_val:
                        best_val = arfitness[k]
                        best_x = arx[:, k].copy()
                # Sort
                idx = np.argsort(arfitness)
                arfitness = arfitness[idx]
                arx = arx[:, idx]
                # Update mean
                xold = xmean.copy()
                xmean = np.dot(arx[:, :mu_], w)
                # Update evolution paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * invsqrtC @ (xmean - xold) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * max_evals_used / lambda_))
                        < (1.4 + 2 / (dim + 1))) * 1.0
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma
                # Update covariance matrix
                artmp = (arx[:, :mu_] - xold.reshape(-1, 1)) / sigma
                C = (1 - c1 - cmu) * C \
                    + c1 * (pc.reshape(-1, 1) @ pc.reshape(1, -1) \
                            + (1 - hsig) * cc * (2 - cc) * C) \
                    + cmu * np.dot(artmp * w, artmp.T)
                # Update sigma
                sigma = sigma * np.exp((cs / dams) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                # Enforce bounds on sigma (avoid too small/large)
                sigma = max(sigma, 1e-15)
                # Enforce positive definiteness and eigendecomposition
                if max_evals_used - eigeneval > lambda_ / (c1 + cmu) / dim / 10:
                    eigeneval = max_evals_used
                    C = (C + C.T) / 2
                    try:
                        D, B = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(D, 1e-20))
                        invsqrtC = B @ np.diag(D ** -1) @ B.T
                    except:
                        D = np.ones(dim)
                        B = np.eye(dim)
                        invsqrtC = np.eye(dim)
                # Early stop if sigma very small
                if sigma * np.max(D) < 1e-12 * (np.max(ub - lb)):
                    break
                # Update best global
                if best_val < self.f_opt:
                    self.f_opt = best_val
                    self.x_opt = best_x.copy()
            return best_x, best_val, max_evals_used

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: decreasing from 0.2 to 0.05
            p = 0.2 * (1 - (n_evals / max_evals) ** 1.5) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = []
            S_CR = []
            delta_f = []

            # Generate offspring
            for i in range(N):
                # Choose r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)
                # r2 from union of population and archive
                if archive.size > 0:
                    union = np.vstack((pop, archive))
                else:
                    union = pop
                r2 = np.random.randint(union.shape[0])
                # pbest index
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)
                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[mem] + 0.1 * np.random.standard_cauchy(), 0, 1)
                CR = np.clip(MCR[mem] + 0.1 * np.random.randn(), 0, 1)
                # Mutation: current-to-pbest/1/archive
                base = pop[i]
                diff1 = pop[pbest_idx] - base
                diff2 = pop[r1] - union[r2]
                mutant = base + F * diff1 + F * diff2
                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, base)
                trial[j_rand] = mutant[j_rand]
                # Reflection boundary handling
                for _ in range(10):
                    out_low = trial < lb
                    out_high = trial > ub
                    if not (np.any(out_low) or np.any(out_high)):
                        break
                    trial = np.where(out_low, 2 * lb - trial, trial)
                    trial = np.where(out_high, 2 * ub - trial, trial)
                trial = np.clip(trial, lb, ub)
                # Evaluate
                trial_f = func(trial)
                n_evals += 1
                if trial_f < self.f_opt:
                    self.f_opt = trial_f
                    self.x_opt = trial.copy()
                    evals_no_improve = 0
                else:
                    evals_no_improve += 1

                if trial_f < fitness[i]:
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    # Add parent to archive
                    archive = np.vstack((archive, pop[i].reshape(1, -1)))
                    if archive.shape[0] > archive_max:
                        remove_idx = np.random.randint(archive.shape[0])
                        archive = np.delete(archive, remove_idx, axis=0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update memory with weighted Lehmer means
            if len(S_F) > 0:
                sorted_order = np.argsort(delta_f)[::-1]
                S_F = np.array(S_F)[sorted_order]
                S_CR = np.array(S_CR)[sorted_order]
                w = np.array(delta_f)[sorted_order] / (np.sum(delta_f) + 1e-30)
                MF[memory_idx] = np.sum(w * S_F ** 2) / (np.sum(w * S_F) + 1e-30)
                MCR[memory_idx] = np.sum(w * S_CR ** 2) / (np.sum(w * S_CR) + 1e-30)
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (quadratic schedule)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 2
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if archive.shape[0] > archive_max:
                    perm = np.random.permutation(archive.shape[0])[:archive_max]
                    archive = archive[perm]
                N = N_new

            # Periodic local refinement with CMA-ES (instead of pattern search)
            # Trigger whenever best has not improved for some evaluations or after many generations
            if (n_evals > 0.1 * max_evals) and (n_evals % max(30, int(0.02 * max_evals)) == 0):
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                # Allocate budget for CMA-ES: proportional to remaining (but limited)
                remaining = max_evals - n_evals
                cma_budget = min(int(0.1 * remaining), dim * 20)
                if cma_budget >= 4 * dim:
                    sigma0 = 0.2 * np.mean(ub - lb)  # initial step size
                    new_pos, new_val, used = cma_es(best_pos, sigma0, cma_budget)
                    n_evals += used
                    if new_val < best_val:
                        best_val = new_val
                        best_pos = new_pos
                        if best_val < self.f_opt:
                            self.f_opt = best_val
                            self.x_opt = best_pos.copy()
                            evals_no_improve = 0
                        # Replace worst individual
                        worst_idx = np.argmax(fitness)
                        if best_val < fitness[worst_idx]:
                            pop[worst_idx] = best_pos
                            fitness[worst_idx] = best_val

            # Restart if stagnation detected (no improvement for long time)
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8:
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init * 2, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                if new_N > N:
                    # Increase population with quasi-random samples
                    samples = np.random.uniform(0, 1, (new_N, dim))
                    samples = lb + samples * (ub - lb)
                    pop = samples.copy()
                    fitness = np.full(new_N, np.inf)
                    pop[0] = best_ind
                    fitness[0] = best_fit
                    for j in range(1, new_N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                    N = new_N
                else:
                    # Partial restart: keep best, reset others
                    pop = lb + np.random.uniform(0, 1, (N, dim)) * (ub - lb)
                    pop[0] = best_ind
                    for j in range(1, N):
                        fitness[j] = func(pop[j])
                        n_evals += 1
                        if fitness[j] < self.f_opt:
                            self.f_opt = fitness[j]
                            self.x_opt = pop[j].copy()
                # Reset memory parameters
                MF[:] = 0.5
                MCR[:] = 0.5
                memory_idx = 0
                archive = np.empty((0, dim))
                archive_max = N
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt