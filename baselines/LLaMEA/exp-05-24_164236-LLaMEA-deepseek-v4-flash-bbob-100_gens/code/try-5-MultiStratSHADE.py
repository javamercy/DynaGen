import numpy as np

class MultiStratSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim

        # ------------------ Initialisation ------------------
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Latin Hypercube sampling
        lhs = np.random.rand(pop_size, dim)
        for j in range(dim):
            lhs[:, j] = (np.argsort(lhs[:, j]) + 0.5) / pop_size
        pop = lb + lhs * (ub - lb)
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        archive = []
        archive_size = pop_size
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # Multi-strategy setup: three strategies with probabilities
        num_strategies = 3
        strategy_probs = np.ones(num_strategies) / num_strategies
        strategy_success = np.zeros(num_strategies)
        strategy_total = np.zeros(num_strategies)

        # Diversity threshold for restart
        diversity_threshold = 0.05 * (ub - lb).mean()

        while evals < self.budget:
            gen += 1

            # Linear population size reduction
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]
            success_F_gen = []
            success_CR_gen = []
            strategy_success_gen = np.zeros(num_strategies)
            strategy_total_gen = np.zeros(num_strategies)

            for i in range(pop_size):
                if evals >= self.budget:
                    break

                # Choose strategy for this individual (roulette wheel)
                s = np.random.choice(num_strategies, p=strategy_probs)
                strategy_total_gen[s] += 1

                # pbest selection for strategies that need it
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Select random individuals from population+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    x_r1 = archive[r1 - pop_size] if r1 >= pop_size else pop[r1]
                    x_r2 = archive[r2 - pop_size] if r2 >= pop_size else pop[r2]
                else:
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Adaptive parameters
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                # Mutation based on selected strategy
                if s == 0:  # current-to-pbest/1
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                elif s == 1:  # current-to-rand/1 (explorative)
                    mutant = pop[i] + F * (x_r1 - pop[i]) + F * (x_r2 - x_r1)
                else:  # s == 2: rand-to-pbest/1
                    mutant = x_r1 + F * (x_pbest - x_r1) + F * (x_r2 - x_pbest)

                # Binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    strategy_success_gen[s] += 1
                    # Update archive
                    if len(archive) < archive_size:
                        archive.append(pop[i].copy())
                    else:
                        idx_arch = np.random.randint(len(archive))
                        archive[idx_arch] = pop[i].copy()
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

            # Update strategy probabilities (adaptive selection)
            eps = 1e-10
            for s in range(num_strategies):
                if strategy_total_gen[s] > 0:
                    strategy_success[s] += strategy_success_gen[s] / (strategy_total_gen[s] + eps)
                strategy_total[s] += strategy_total_gen[s]
            # Recompute probabilities using softmax over success rates
            if np.sum(strategy_success) > eps:
                exp_rates = np.exp(strategy_success / (strategy_total + eps))
                strategy_probs = exp_rates / np.sum(exp_rates)

            # Update memory with successful parameters
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---------- Local perturbation around best (light replacement for Nelder-Mead) ----------
            if evals < self.budget and (gen % 5 == 0 or stagnation_counter >= 3):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                step = (ub - lb) * 0.02  # small step size
                # Sample a few candidates around best
                num_local = min(5 * dim, self.budget - evals)
                for _ in range(num_local):
                    perturb = np.random.normal(0, step, dim)
                    candidate = np.clip(x_best + perturb, lb, ub)
                    f_candidate = func(candidate)
                    evals += 1
                    if f_candidate < f_best:
                        f_best = f_candidate
                        x_best = candidate.copy()
                        self.f_opt = f_best
                        self.x_opt = x_best.copy()
                # Replace worst in population if local search found better
                if f_best < self.f_opt:
                    worst_idx = np.argmax(fitness)
                    if f_best < fitness[worst_idx]:
                        pop[worst_idx] = x_best
                        fitness[worst_idx] = f_best

            # ---------- Stagnation detection ----------
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # ---------- Diversity-guided restart ----------
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                # Check population diversity
                centroid = np.mean(pop, axis=0)
                diversity = np.mean([np.linalg.norm(p - centroid) for p in pop])
                if diversity < diversity_threshold:
                    # Restart: keep best, reinitialize rest
                    best_copy = self.x_opt.copy()
                    best_f = self.f_opt
                    n_restart = max(1, int(0.5 * pop_size))
                    # Reinitialize with Latin Hypercube around best
                    for idx in range(pop_size):
                        if idx == np.argmin(fitness):
                            continue  # keep the current best individual
                        if n_restart > 0:
                            # Use LHS in a small region around best
                            lhs_sample = np.random.rand(1, dim)
                            lhs_sample = (np.argsort(lhs_sample[0]) + 0.5) / 1.0
                            radius = (ub - lb) * 0.1
                            pop[idx] = best_copy + (lhs_sample - 0.5) * radius
                            pop[idx] = np.clip(pop[idx], lb, ub)
                            if evals < self.budget:
                                fitness[idx] = func(pop[idx])
                                evals += 1
                                if fitness[idx] < self.f_opt:
                                    self.f_opt = fitness[idx]
                                    self.x_opt = pop[idx].copy()
                            n_restart -= 1
                    stagnation_counter = 0
                    # Reset strategy statistics
                    strategy_success[:] = 0
                    strategy_total[:] = 1  # avoid division by zero
                    strategy_probs[:] = 1.0 / num_strategies
                    # Clear archive
                    archive = []

        return self.f_opt, self.x_opt