import numpy as np
from scipy.stats import cauchy

class AdvancedLSHADE:
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

        # Population size
        N_init = min(max(10 * dim, 60), max_evals // 2)
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

        # Archive (FIFO)
        archive = []
        archive_max = N

        # Memory for F and CR: 3 strategies (0: current-to-pbest, 1: rand/1, 2: best/1)
        H = 25
        MF = np.ones((3, H)) * 0.5
        MCR = np.ones((3, H)) * 0.8
        memory_idx = 0
        # Strategy probabilities (adaptive)
        strategy_prob = np.ones(3) / 3.0
        strategy_success = np.zeros(3)
        strategy_attempts = np.zeros(3) + 1e-10

        # Stagnation detection
        best_fitness_hist = [self.f_opt]
        evals_no_improve = 0
        restart_threshold = 0.08 * max_evals
        restart_count = 0
        max_restarts = 2

        # Local search using simplified CMA-ES
        def cma_local_search(best_pos, best_val, sigma_init, max_evals_local):
            # Initialize CMA parameters
            N_local = min(4 + int(3 * np.log(dim)), max_evals_local // 2)  # small population
            if N_local < 3:
                N_local = 3
            mean = best_pos.copy()
            sigma = sigma_init * np.mean(ub - lb)
            C = np.eye(dim)
            # Evolution path for rank-one update
            pc = np.zeros(dim)
            # Learning rates
            cc = 2.0 / (dim + 2.0)
            c1 = 0.1
            cmu = 0.9
            # Limits
            used_local = 0
            # Sample and evaluate
            for gen in range(50):
                if used_local >= max_evals_local:
                    break
                # Generate offspring
                A = np.linalg.cholesky(C)
                offspring = np.zeros((N_local, dim))
                for i in range(N_local):
                    z = np.random.randn(dim)
                    offspring[i] = mean + sigma * A @ z
                    offspring[i] = np.clip(offspring[i], lb, ub)
                # Evaluate all
                vals = np.array([func(x) for x in offspring])
                used_local += N_local
                if used_local > max_evals_local:
                    break
                # Update mean
                idx = np.argsort(vals)
                best_new = offspring[idx[0]]
                best_new_val = vals[idx[0]]
                if best_new_val < best_val:
                    best_val = best_new_val
                    best_pos = best_new.copy()
                # Selection: use top ceil(N_local/2) individuals
                mu = int(np.ceil(N_local / 2))
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
                weights = weights / np.sum(weights)
                old_mean = mean.copy()
                mean = np.dot(weights, offspring[idx[:mu]])
                # Update evolution path
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mu) * (mean - old_mean) / sigma
                # Update covariance (rank-one and rank-mu)
                C = (1 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                # Rank-mu update
                art = (offspring[idx[:mu]] - old_mean) / sigma
                C += cmu * np.dot((weights * art.T), art)
                # Enforce symmetry and positive definiteness
                C = (C + C.T) / 2
                # Ensure numerical stability
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-20:
                    C += np.eye(dim) * 1e-12
                # Update step size (simple adaptation)
                sigma *= np.exp(0.2 * (np.linalg.norm(pc) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-12, 0.3 * np.mean(ub - lb))
            return best_pos, best_val, used_local

        # Main loop
        while n_evals < max_evals:
            # pbest ratio: aggressive quadratic decay
            p = 0.2 * (1 - (n_evals / max_evals) ** 2) + 0.05

            new_pop = pop.copy()
            new_fitness = fitness.copy()
            S_F = [[], [], []]
            S_CR = [[], [], []]
            delta_f = [[], [], []]

            # Generate offspring
            for i in range(N):
                # Choose strategy via probability
                strategy = np.random.choice(3, p=strategy_prob)
                strategy_attempts[strategy] += 1

                # Setup mutation depending on strategy
                # r1 different from i
                idxs = list(range(N))
                idxs.remove(i)
                r1 = np.random.choice(idxs)

                # Archive union
                if archive:
                    union = np.vstack((pop, np.array(archive)))
                else:
                    union = pop

                # pbest for strategy 0 and 2 (when needed)
                pbest_size = max(1, int(p * N))
                sorted_idx = np.argsort(fitness)
                pbest_candidates = sorted_idx[:pbest_size]
                pbest_idx = np.random.choice(pbest_candidates)

                # Sample F and CR
                mem = np.random.randint(H)
                F = np.clip(MF[strategy, mem] + 0.1 * cauchy.rvs(), 0, 1)
                while F <= 0:
                    F = np.clip(MF[strategy, mem] + 0.1 * cauchy.rvs(), 0, 1)
                CR = np.clip(MCR[strategy, mem] + 0.1 * np.random.randn(), 0, 1)

                # Mutation
                if strategy == 0:  # current-to-pbest/1 with archive
                    r2 = np.random.randint(union.shape[0])
                    base = pop[i]
                    diff1 = pop[pbest_idx] - base
                    diff2 = pop[r1] - union[r2]
                    mutant = base + F * diff1 + F * diff2
                elif strategy == 1:  # rand/1 (no archive)
                    # select two random distinct from pop (excluding i and r1)
                    candidates = list(set(range(N)) - {i, r1})
                    r2 = np.random.choice(candidates)
                    base = pop[r1]
                    diff = pop[r2] - base
                    mutant = base + F * diff
                else:  # best/1 with archive (pbest as best)
                    r2 = np.random.randint(union.shape[0])
                    base = pop[pbest_idx]
                    diff1 = pop[r1] - base
                    diff2 = pop[r1] - union[r2]   # second diff from archive
                    mutant = base + F * diff1 + F * diff2

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # Boundary handling: reflect and clamp
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

                # Selection
                if trial_f < fitness[i]:
                    S_F[strategy].append(F)
                    S_CR[strategy].append(CR)
                    delta_f[strategy].append(fitness[i] - trial_f)
                    new_fitness[i] = trial_f
                    new_pop[i] = trial
                    strategy_success[strategy] += 1
                    # Add parent to archive
                    archive.append(pop[i].copy())
                    if len(archive) > archive_max:
                        archive.pop(0)

            # Update population and fitness
            pop = new_pop
            fitness = new_fitness

            # Update strategy probabilities
            for s in range(3):
                if strategy_attempts[s] > 0:
                    rate = strategy_success[s] / strategy_attempts[s]
                    strategy_prob[s] = max(0.01, rate)
            strategy_prob /= np.sum(strategy_prob)
            # Avoid zero probability
            strategy_prob = np.clip(strategy_prob, 0.01, 1.0)
            strategy_prob /= np.sum(strategy_prob)

            # Update memory for each strategy
            for s in range(3):
                if len(S_F[s]) > 0:
                    sorted_order = np.argsort(delta_f[s])[::-1]
                    S_F_s = np.array(S_F[s])[sorted_order]
                    S_CR_s = np.array(S_CR[s])[sorted_order]
                    w = np.array(delta_f[s])[sorted_order] / (np.sum(delta_f[s]) + 1e-30)
                    if np.sum(w * S_F_s) > 1e-30:
                        MF[s, memory_idx] = np.sum(w * S_F_s ** 2) / (np.sum(w * S_F_s) + 1e-30)
                    else:
                        MF[s, memory_idx] = 0.5
                    if np.sum(w * S_CR_s) > 1e-30:
                        MCR[s, memory_idx] = np.sum(w * S_CR_s ** 2) / (np.sum(w * S_CR_s) + 1e-30)
                    else:
                        MCR[s, memory_idx] = 0.8
            # Advance memory index (shared across strategies for simplicity)
            if any(len(s) > 0 for s in S_F):
                memory_idx = (memory_idx + 1) % H

            # Population size reduction (cubic)
            N_new = N_min + (N_init - N_min) * ((max_evals - n_evals) / max_evals) ** 3
            N_new = int(np.round(N_new))
            N_new = max(N_min, min(N_new, N_init))
            if N_new < N:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:N_new]]
                fitness = fitness[sorted_idx[:N_new]]
                archive_max = N_new
                if len(archive) > archive_max:
                    archive = archive[-archive_max:]
                N = N_new

            # Periodic CMA local search
            if (n_evals < max_evals * 0.95) and (n_evals % max(30, int(0.01 * max_evals)) == 0):
                best_idx = np.argmin(fitness)
                best_pos = pop[best_idx].copy()
                best_val = fitness[best_idx]
                max_local = min(dim * 5, max_evals - n_evals - 5)
                sigma_init = 0.1 * (1 - n_evals / max_evals) + 0.02
                new_pos, new_val, used = cma_local_search(best_pos, best_val, sigma_init, max_local)
                n_evals += used
                if new_val < best_val:
                    if new_val < self.f_opt:
                        self.f_opt = new_val
                        self.x_opt = new_pos.copy()
                        evals_no_improve = 0
                    # Replace worst individual if improved
                    if new_val < np.max(fitness):
                        worst_idx = np.argmax(fitness)
                        pop[worst_idx] = new_pos
                        fitness[worst_idx] = new_val

            # Diversity injection: Cauchy mutation of worst 15% every ~200 evals
            if n_evals % max(200, int(0.02 * max_evals)) == 0 and n_evals < max_evals * 0.9:
                n_replace = max(1, int(0.15 * N))
                worst_idx = np.argsort(fitness)[-n_replace:]
                for idx in worst_idx:
                    # Cauchy perturbation around best
                    scale = 0.1 * (ub - lb) * (1 - n_evals / max_evals)
                    candidate = self.x_opt + cauchy.rvs(size=dim) * scale
                    candidate = np.clip(candidate, lb, ub)
                    cand_val = func(candidate)
                    n_evals += 1
                    if cand_val < self.f_opt:
                        self.f_opt = cand_val
                        self.x_opt = candidate.copy()
                    if cand_val < fitness[idx]:
                        pop[idx] = candidate
                        fitness[idx] = cand_val

            # Restart if stagnation
            if evals_no_improve > restart_threshold and n_evals < max_evals * 0.8 and restart_count < max_restarts:
                restart_count += 1
                best_idx = np.argmin(fitness)
                best_ind = pop[best_idx].copy()
                best_fit = fitness[best_idx]
                remaining = max_evals - n_evals
                new_N = min(N_init, N * 2, remaining // 2)
                new_N = max(new_N, N_min)
                # Generate new population using Latin hypercube and reflection around best
                samples = np.random.uniform(0, 1, (new_N, dim))
                samples = lb + samples * (ub - lb)
                pop = samples.copy()
                fitness = np.full(new_N, np.inf)
                pop[0] = best_ind
                fitness[0] = best_fit
                for j in range(1, new_N):
                    # Optionally reflect some points around best
                    if np.random.rand() < 0.3:
                        reflected = 2 * best_ind - pop[j]
                        pop[j] = np.clip(reflected, lb, ub)
                    fitness[j] = func(pop[j])
                    n_evals += 1
                    if fitness[j] < self.f_opt:
                        self.f_opt = fitness[j]
                        self.x_opt = pop[j].copy()
                N = new_N
                # Reset archives and memory
                archive = []
                archive_max = N
                # Refresh memory with random values
                for s in range(3):
                    MF[s, :] = 0.5 + 0.2 * np.random.rand(H)
                    MCR[s, :] = 0.8 + 0.2 * np.random.rand(H)
                memory_idx = 0
                strategy_prob[:] = 1/3
                strategy_success[:] = 0
                strategy_attempts[:] = 1e-10
                evals_no_improve = 0

            if n_evals >= max_evals:
                break

        return self.f_opt, self.x_opt