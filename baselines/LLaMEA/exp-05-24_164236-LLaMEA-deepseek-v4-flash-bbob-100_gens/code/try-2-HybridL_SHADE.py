import numpy as np

class HybridL_SHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Population size: linear reduction from N_init to N_min
        N_init = max(10, int(14 * np.sqrt(self.dim)))
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.empty(pop_size)
        evals = 0
        for i in range(pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        gen = 0
        max_gen = int(self.budget / pop_size * 2)  # estimate
        archive = []  # not used in current version, but placeholder
        archive_size = pop_size
        success_F = []  # per generation success
        success_CR = []
        stagnation_counter = 0
        best_old = self.f_opt

        while evals < self.budget:
            gen += 1
            # --- Linear population size reduction ---
            new_pop_size = max(N_min, int(N_init - gen * (N_init - N_min) / max_gen))
            if new_pop_size < pop_size:
                idx_sorted = np.argsort(fitness)
                pop = pop[idx_sorted[:new_pop_size]]
                fitness = fitness[idx_sorted[:new_pop_size]]
                pop_size = new_pop_size

            # Reset per-generation success lists
            success_F_gen = []
            success_CR_gen = []

            # Adaptive parameter generation
            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            # For each individual
            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Choose two distinct random indices different from i
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, 2, replace=False)

                # current-to-pbest/1
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                mutant = pop[i] + F * (x_pbest - pop[i]) + F * (pop[r1] - pop[r2])
                # Binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(self.dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()
                    # Replace in population
                    pop[i] = trial
                    fitness[i] = f_trial

            # Update memory with successful parameters
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # --- Local search (Hooke-Jeeves pattern search) on best ---
            if evals < self.budget and (gen % 6 == 0 or stagnation_counter >= 2):
                # Perform a limited pattern search
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                step = (ub - lb) * 0.1
                min_step = 1e-5 * (ub - lb).max()
                max_local_evals = min(30 * self.dim, self.budget - evals)
                local_evals = 0
                improved = True
                while local_evals < max_local_evals and np.any(step > min_step):
                    improved_flag = False
                    for d in range(self.dim):
                        for sign in [1, -1]:
                            x_try = x_best.copy()
                            x_try[d] = np.clip(x_best[d] + sign * step[d], lb[d], ub[d])
                            f_try = func(x_try)
                            local_evals += 1
                            evals += 1
                            if f_try < f_best:
                                x_best = x_try
                                f_best = f_try
                                improved_flag = True
                                if f_best < self.f_opt:
                                    self.f_opt = f_best
                                    self.x_opt = x_best.copy()
                                break  # greedy accept first improvement
                        if improved_flag:
                            break
                    if not improved_flag:
                        step *= 0.5
                    else:
                        improved = True
                    if evals >= self.budget:
                        break
                # Update best in population (replace worst individual)
                if f_best < self.f_opt:  # already updated above
                    pass
                # Replace worst individual with local search result (if better)
                worst_idx = np.argmax(fitness)
                if f_best < fitness[worst_idx]:
                    pop[worst_idx] = x_best.copy()
                    fitness[worst_idx] = f_best

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if stagnation is severe
            if stagnation_counter > max(10, int(0.1 * max_gen)):
                # Reinitialize part of population (keep best)
                n_restart = max(1, int(0.5 * pop_size))
                idx_random = np.random.choice(pop_size, n_restart, replace=False)
                for idx in idx_random:
                    pop[idx] = np.random.uniform(lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                stagnation_counter = 0
                # Reset memory to avoid getting stuck
                mem_F[:] = 0.5
                mem_CR[:] = 0.8

        return self.f_opt, self.x_opt