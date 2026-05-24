import numpy as np

class SAEM_LR:
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

        # Latin Hypercube initialization
        N_init = max(10, int(14 * np.sqrt(dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.ones(mem_size) * 0.5
        mem_CR = np.ones(mem_size) * 0.8
        mem_idx = 0

        # LHS
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

        archive = []  # stores replaced parents
        archive_size = pop_size  # initial archive size, will be proportionally updated
        success_F = []
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt
        gen = 0
        max_gen = int(self.budget / pop_size * 2)

        # Success rates for mutation strategies (0: pbest/1, 1: rand/1)
        strategy_success = [1e-10, 1e-10]
        strategy_fail = [1e-10, 1e-10]
        strategy_prob = [0.5, 0.5]

        while evals < self.budget:
            gen += 1
            # Exponential population size reduction
            if pop_size > N_min:
                factor = (N_min / N_init) ** (gen / max_gen)
                new_pop_size = max(N_min, int(N_init * factor))
            else:
                new_pop_size = N_min
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size
                # adjust archive size proportionally
                archive_size = pop_size
                if len(archive) > archive_size:
                    archive = archive[-archive_size:]

            success_F_gen = []
            success_CR_gen = []
            gen_strategy_success = 0
            gen_strategy_fail = 0

            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection (adaptive rate)
                p = 0.2 * (1 - gen / max_gen) + 0.1
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # choose mutation strategy based on adaptive probabilities
                if np.random.rand() < strategy_prob[0]:
                    # current-to-pbest/1 (exploitation)
                    strategy_used = 0
                else:
                    # current-to-rand/1 (exploration, no crossover needed)
                    strategy_used = 1

                # select two distinct individuals from pop + archive
                union = list(range(pop_size)) + list(range(len(archive)))
                union.remove(i)
                if len(union) >= 2:
                    r1, r2 = np.random.choice(union, 2, replace=False)
                    if r1 >= pop_size:
                        x_r1 = archive[r1 - pop_size]
                    else:
                        x_r1 = pop[r1]
                    if r2 >= pop_size:
                        x_r2 = archive[r2 - pop_size]
                    else:
                        x_r2 = pop[r2]
                else:
                    idxs = list(range(pop_size))
                    idxs.remove(i)
                    r1, r2 = np.random.choice(idxs, 2, replace=False)
                    x_r1, x_r2 = pop[r1], pop[r2]

                # Adaptive parameters (Cauchy sampling for F, normal for CR)
                F = np.clip(np.random.cauchy(F_base, 0.1), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                if strategy_used == 0:
                    # mutant = pop[i] + F*(x_pbest - pop[i]) + F*(x_r1 - x_r2)
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                    # binomial crossover
                    j_rand = np.random.randint(dim)
                    trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                      else pop[i][j] for j in range(dim)])
                else:
                    # current-to-rand/1 (rotation invariant, no crossover)
                    K = np.random.uniform(0, 1, dim)  # component-wise scaling
                    trial = pop[i] + K * (x_pbest - pop[i]) + F * (x_r1 - x_r2)

                trial = np.clip(trial, lb, ub)
                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    # archive update
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
                    gen_strategy_success += 1
                else:
                    gen_strategy_fail += 1

            # Update strategies probabilities based on success rates
            if gen_strategy_success + gen_strategy_fail > 0:
                if strategy_used == 0:
                    strategy_success[0] += gen_strategy_success
                    strategy_fail[0] += gen_strategy_fail
                else:
                    strategy_success[1] += gen_strategy_success
                    strategy_fail[1] += gen_strategy_fail
                total_success = max(1e-10, strategy_success.sum())
                strategy_prob[0] = strategy_success[0] / total_success
                strategy_prob[1] = 1.0 - strategy_prob[0]

            # Update memory with successful parameters (Lehmer mean)
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / max(1e-10, np.sum(w * np.array(success_F_gen)))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # --- Lightweight Random Walk Local Search (triggered by stagnation) ---
            if evals < self.budget and stagnation_counter >= 5:
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                step_size = 0.1 * (ub - lb)  # initial step
                eval_budget_ls = min(20 * dim, self.budget - evals)
                ls_evals = 0
                for _ in range(eval_budget_ls):
                    # random walk with decreasing step
                    perturbation = np.random.randn(dim) * step_size * (1 - ls_evals / eval_budget_ls)
                    trial = np.clip(x_best + perturbation, lb, ub)
                    f_trial = func(trial)
                    evals += 1
                    ls_evals += 1
                    if f_trial < f_best:
                        x_best = trial.copy()
                        f_best = f_trial
                        if f_best < self.f_opt:
                            self.f_opt = f_best
                            self.x_opt = x_best.copy()
                    # early stop if no improvement and step size is too small?
                    if ls_evals > 10 and (f_best - self.f_opt) < 1e-12:
                        break
                # Replace worst if best improved
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
            if stagnation_counter > max(10, int(0.2 * max_gen)):
                n_restart = max(1, int(0.4 * pop_size))
                best_copy = self.x_opt.copy()
                # reinitialize part of population around best with random perturbations
                idx_keep = np.random.choice(pop_size, n_restart, replace=False)
                for idx in idx_keep:
                    sigma = 0.1 * (ub - lb) * (1 + np.random.rand())
                    pop[idx] = np.clip(best_copy + np.random.randn(dim) * sigma, lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                stagnation_counter = 0
                # Reset memories
                mem_F[:] = 0.5
                mem_CR[:] = 0.8
                archive = []  # clear archive
                strategy_success = [1e-10, 1e-10]
                strategy_fail = [1e-10, 1e-10]
                strategy_prob = [0.5, 0.5]

        return self.f_opt, self.x_opt