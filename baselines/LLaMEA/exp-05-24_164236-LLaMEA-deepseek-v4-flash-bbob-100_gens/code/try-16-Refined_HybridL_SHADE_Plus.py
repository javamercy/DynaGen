import numpy as np
from scipy.stats import qmc
from scipy.optimize import minimize

class Refined_HybridL_SHADE_Plus:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed()
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        dim = self.dim

        # Population size (L-SHADE style)
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        max_evals = self.budget

        # Memory for successful parameters (SHADE)
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Strategy pool: 0 = current-to-pbest/1, 1 = current-to-rand/1
        num_strategies = 2
        strategy_memory = [np.full(mem_size, 0.5) for _ in range(num_strategies)]  # success rates
        strategy_prob = np.ones(num_strategies) / num_strategies
        strategy_success = np.zeros(num_strategies)
        strategy_total = np.zeros(num_strategies)

        # Sobol initialization
        try:
            sampler = qmc.Sobol(d=dim, scramble=True)
            lhs = sampler.random(pop_size)
        except:
            lhs = np.random.rand(pop_size, dim)
            for j in range(dim):
                lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)

        # Evaluate initial population
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # Archive for diversity
        archive = []
        archive_size = pop_size

        # Stagnation tracking
        best_old = self.f_opt
        stagnation_counter = 0
        gen = 0
        max_gen = int(max_evals / pop_size * 2)

        while evals < max_evals:
            gen += 1

            # Linear population size reduction (based on evaluation budget)
            remaining = max_evals - evals
            total_evals = max_evals
            new_pop_size = max(N_min, int(N_init + (N_min - N_init) * (1 - evals / total_evals)))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]].copy()
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    # Keep only most diverse
                    dists = np.array([np.min(np.linalg.norm(np.array(archive) - a, axis=1)) for a in archive])
                    idx_keep = np.argsort(-dists)[:archive_size]
                    archive = [archive[i] for i in idx_keep]

            # Update pbest rate (time-dependent)
            p = 0.2 * (gen / max_gen) ** 2 + 0.1
            p = min(p, 0.5)

            success_F = []
            success_CR = []
            weight = []
            used_strategies = []

            f_min = np.min(fitness)

            for i in range(pop_size):
                if evals >= max_evals:
                    break

                # Choose strategy with probability proportional to past success
                strategy = np.random.choice(num_strategies, p=strategy_prob / strategy_prob.sum())

                # Draw F and CR from memory
                r = np.random.randint(mem_size)
                if strategy == 0:
                    F = mem_F[r] + 0.1 * np.random.randn()
                    CR = mem_CR[r] + 0.1 * np.random.randn()
                else:
                    # current-to-rand/1 uses larger F to encourage exploration
                    F = mem_F[r] + 0.2 * np.random.randn()
                    CR = mem_CR[r] + 0.1 * np.random.randn()
                F = np.clip(F, 0.1, 1.0)
                CR = np.clip(CR, 0.0, 1.0)

                # Selection of indices
                pbest_size = max(2, int(p * pop_size))
                best_indices = np.argsort(fitness)[:pbest_size]
                # Fitness-weighted selection for pbest
                weights = np.max(fitness[best_indices]) - fitness[best_indices] + 1e-10
                pbest_idx = np.random.choice(best_indices, p=weights/weights.sum())
                x_pbest = pop[pbest_idx]

                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    def get_individual(idx):
                        if idx < pop_size:
                            return pop[idx]
                        else:
                            return archive[idx - pop_size]
                    x_r1 = get_individual(r1)
                    x_r2 = get_individual(r2)
                else:
                    indices = [j for j in range(pop_size) if j != i]
                    r1, r2 = np.random.choice(indices, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Mutation
                if strategy == 0:
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                else:
                    # current-to-rand/1 (no bias towards best)
                    mutant = pop[i] + F * (x_r1 - pop[i]) + F * (x_r2 - pop[i])

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.where(np.random.rand(dim) < CR, mutant, pop[i])
                trial[j_rand] = mutant[j_rand]

                # Reflection at boundaries
                for d in range(dim):
                    if trial[d] < lb[d]:
                        trial[d] = lb[d] + (lb[d] - trial[d])
                    elif trial[d] > ub[d]:
                        trial[d] = ub[d] - (trial[d] - ub[d])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                strategy_total[strategy] += 1
                if f_trial <= fitness[i]:
                    # Store parent to archive with diversity
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        # Replace the individual with smallest crowding distance
                        archive_arr = np.array(archive)
                        # Compute distances to nearest neighbor
                        dists = []
                        for j in range(len(archive_arr)):
                            others = np.delete(archive_arr, j, axis=0)
                            d = np.min(np.linalg.norm(others - archive_arr[j], axis=1))
                            dists.append(d)
                        idx_remove = np.argmin(dists)
                        archive[idx_remove] = pop[i].copy()

                    success_F.append(F)
                    success_CR.append(CR)
                    imp = max(fitness[i] - f_trial, 1e-12)
                    weight.append(imp)
                    used_strategies.append(strategy)

                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update memory with weighted Lehmer mean
            if len(success_F) > 0:
                w = np.array(weight)
                w = w / (np.sum(w) + 1e-30)
                F_lehmer = np.sum(w * np.array(success_F)**2) / (np.sum(w * np.array(success_F)) + 1e-30)
                CR_mean = np.sum(w * np.array(success_CR))
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

                # Update strategy probabilities (smoothed)
                for s in range(num_strategies):
                    success_count = sum(1 for strat in used_strategies if strat == s)
                    total_count = strategy_total[s]
                    if total_count > 0:
                        strategy_success[s] = success_count / total_count
                # Use softmax over recent success rates
                strategy_prob = np.exp(strategy_success) / np.sum(np.exp(strategy_success))

            # Local search using Powell's method, triggered by stagnation or periodic
            nm_budget = int(0.15 * (max_evals - evals))
            if nm_budget > dim + 1 and (gen % 5 == 0 or stagnation_counter >= 3):
                # Use Powell with small max function evaluations
                res = minimize(func, self.x_opt, method='Powell',
                               bounds=list(zip(lb, ub)),
                               options={'maxfev': min(nm_budget, 100+10*dim), 'xtol': 1e-6, 'ftol': 1e-12})
                if res.fun < self.f_opt:
                    self.f_opt = res.fun
                    self.x_opt = res.x.copy()
                evals_used = res.nfev if hasattr(res, 'nfev') else nm_budget
                evals += evals_used

                # Inject back into population if better than worst
                worst_idx = np.argmax(fitness)
                if self.f_opt < fitness[worst_idx]:
                    pop[worst_idx] = self.x_opt.copy()
                    fitness[worst_idx] = self.f_opt

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if severe stagnation
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                n_restart = max(1, int(0.5 * pop_size))
                try:
                    sampler_restart = qmc.Sobol(d=dim, scramble=True)
                    sob = sampler_restart.random(n_restart)
                except:
                    sob = np.random.rand(n_restart, dim)
                for idx in range(n_restart):
                    if idx < n_restart // 2:
                        scale = 0.1 * (ub - lb) * (1 - gen / max_gen)
                        pop[idx] = best_copy + np.random.uniform(-1, 1, dim) * scale
                    else:
                        pop[idx] = lb + sob[idx] * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < max_evals:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                # Reset memory and archive
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive.clear()
                strategy_success[:] = 0
                strategy_total[:] = 0
                strategy_prob[:] = 1.0 / num_strategies
                stagnation_counter = 0

        return self.f_opt, self.x_opt