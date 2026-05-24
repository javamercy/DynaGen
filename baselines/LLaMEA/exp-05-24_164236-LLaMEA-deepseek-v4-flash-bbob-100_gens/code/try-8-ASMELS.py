import numpy as np

class ASMELS:
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

        # Population parameters
        N_init = max(10, int(10 + 15 * np.sqrt(dim)))  # larger initial pop
        N_min = 4
        pop_size = N_init
        mem_size = 6
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.8)
        mem_idx = 0

        # Strategy adaptation
        prob_strat = np.array([0.5, 0.5])  # [p1 for current-to-pbest, p2 for current-to-rand]
        ns1, ns2 = 0, 0  # successes per strategy
        nf1, nf2 = 0, 0  # failures per strategy

        # Latin Hypercube initialization
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

            # Update strategy probabilities based on success rates (every 10 gens)
            if gen % 10 == 0:
                if ns1 + nf1 > 0:
                    rate1 = ns1 / (ns1 + nf1)
                else:
                    rate1 = 0.5
                if ns2 + nf2 > 0:
                    rate2 = ns2 / (ns2 + nf2)
                else:
                    rate2 = 0.5
                total = rate1 + rate2
                if total > 1e-10:
                    prob_strat[0] = rate1 / total
                    prob_strat[1] = rate2 / total
                # reset counters
                ns1, ns2 = 0, 0
                nf1, nf2 = 0, 0

            success_F_gen = []
            success_CR_gen = []

            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            for i in range(pop_size):
                if evals >= self.budget:
                    break
                # pbest selection (adaptive rate)
                p = max(0.2, 0.2 * (1 - gen / max_gen))
                pbest_size = max(2, int(p * pop_size))
                idx_pbest = np.random.choice(pop_size, pbest_size, replace=False)
                best_p = np.argmin(fitness[idx_pbest])
                x_pbest = pop[idx_pbest[best_p]]

                # Choose two distinct random indices from pop+archive
                union = list(range(pop_size)) + list(range(len(archive)))
                if i in union:
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

                # Adaptive parameters
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                # Choose mutation strategy based on probabilities
                if np.random.rand() < prob_strat[0]:
                    # current-to-pbest/1 (original)
                    mutant = pop[i] + F * (x_pbest - pop[i]) + F * (x_r1 - x_r2)
                    chosen_strat = 1
                else:
                    # current-to-rand/1 (rotation-invariant)
                    mutant = pop[i] + F * (x_r1 - x_r2) + 0.5 * (x_pbest - pop[i])
                    chosen_strat = 2

                # Crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1

                if f_trial <= fitness[i]:
                    if chosen_strat == 1:
                        ns1 += 1
                    else:
                        ns2 += 1
                    success_F_gen.append(F)
                    success_CR_gen.append(CR)
                    # Archive update
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
                else:
                    if chosen_strat == 1:
                        nf1 += 1
                    else:
                        nf2 += 1

            # Update memory with successful parameters
            if len(success_F_gen) > 0:
                w = np.ones(len(success_F_gen))
                F_lehmer = np.sum(w * np.array(success_F_gen)**2) / np.sum(w * np.array(success_F_gen))
                CR_mean = np.mean(success_CR_gen)
                mem_F[mem_idx] = F_lehmer
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # Lightweight local search on best solution (random coordinate search)
            if evals < self.budget and (gen % 10 == 0 or stagnation_counter >= 5):
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                max_ls_evals = min(int(0.1 * (self.budget - evals)), 10 * dim, 200)
                for _ in range(max_ls_evals):
                    # choose a random coordinate to perturb
                    j = np.random.randint(dim)
                    sigma = 0.02 * (ub[j] - lb[j]) * (1 + np.random.randn() * 0.1)
                    x_new = x_best.copy()
                    x_new[j] = np.clip(x_best[j] + sigma, lb[j], ub[j])
                    f_new = func(x_new)
                    evals += 1
                    if f_new < f_best:
                        x_best = x_new.copy()
                        f_best = f_new
                        break  # accept and stop this iteration
                if f_best < self.f_opt:
                    self.f_opt = f_best
                    self.x_opt = x_best.copy()
                    # replace worst in population with new best
                    worst_idx = np.argmax(fitness)
                    pop[worst_idx] = x_best.copy()
                    fitness[worst_idx] = f_best

            # Stagnation detection
            if self.f_opt < best_old - 1e-12:
                best_old = self.f_opt
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Restart if stagnation severe or population diversity low
            if stagnation_counter > max(20, int(0.15 * max_gen)):
                # keep best solution
                n_restart = max(2, int(0.4 * pop_size))
                best_copy = self.x_opt.copy()
                best_f = self.f_opt
                # Reinitialize around best with small perturbations
                for idx in np.random.choice(pop_size, n_restart, replace=False):
                    pop[idx] = best_copy + np.random.uniform(-0.1, 0.1, dim) * (ub - lb)
                    pop[idx] = np.clip(pop[idx], lb, ub)
                    if evals < self.budget:
                        fitness[idx] = func(pop[idx])
                        evals += 1
                        if fitness[idx] < self.f_opt:
                            self.f_opt = fitness[idx]
                            self.x_opt = pop[idx].copy()
                stagnation_counter = 0
                # Reset memory (keep some randomness)
                mem_F[:] = np.random.uniform(0.3, 0.9, mem_size)
                mem_CR[:] = np.random.uniform(0.5, 0.9, mem_size)
                # Clear archive
                archive = []
                # Reset strategy probabilities
                prob_strat[:] = 0.5
                ns1, ns2 = 0, 0
                nf1, nf2 = 0, 0

        return self.f_opt, self.x_opt