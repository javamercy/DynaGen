import numpy as np

class MemeticAdaptiveDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)  # for reproducibility, can be removed
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size (linear scaling with dimension, minimal 7)
        pop_size = max(7, int(4 + 3 * np.log(self.dim)))
        # memory sizes for parameter adaptation
        mem_size = 5
        mem_F = np.full(mem_size, 0.5)
        mem_CR = np.full(mem_size, 0.5)
        mem_idx = 0

        # initialize population
        pop = np.random.uniform(lb, ub, (pop_size, self.dim))
        fitness = np.full(pop_size, np.inf)
        evals = 0

        # evaluate initial population
        for i in range(pop_size):
            if evals >= self.budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < self.f_opt:
                self.f_opt = fitness[i]
                self.x_opt = pop[i].copy()

        # main loop (generations + local search)
        gen = 0
        while evals < self.budget:
            gen += 1
            # ---- adaptive differential evolution (DE/current-to-best/1/bin) ----
            # store successful parameters for adaptation
            success_F = []
            success_CR = []
            new_pop = pop.copy()
            new_fitness = fitness.copy()

            # select a random memory slot for parameter generation
            r = np.random.randint(mem_size)
            F_base = mem_F[r]
            CR_base = mem_CR[r]

            # generate trial vectors
            for i in range(pop_size):
                # adapt F and CR per individual (dithering)
                F = np.clip(F_base + 0.1 * np.random.randn(), 0.1, 1.0)
                CR = np.clip(CR_base + 0.1 * np.random.randn(), 0.0, 1.0)

                # select two distinct random indices different from i and best
                idxs = list(range(pop_size))
                idxs.remove(i)
                r1, r2 = np.random.choice(idxs, size=2, replace=False)
                # DE/current-to-best/1
                best_idx = np.argmin(fitness)
                mutant = pop[i] + F * (pop[best_idx] - pop[i]) + F * (pop[r1] - pop[r2])
                # binomial crossover
                j_rand = np.random.randint(self.dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(self.dim)])
                # bound constraint handling: reflect back
                trial = np.clip(trial, lb, ub)

                # evaluate if budget allows
                if evals >= self.budget:
                    break
                trial_f = func(trial)
                evals += 1

                # selection
                if trial_f <= fitness[i]:
                    new_pop[i] = trial
                    new_fitness[i] = trial_f
                    success_F.append(F)
                    success_CR.append(CR)
                    if trial_f < self.f_opt:
                        self.f_opt = trial_f
                        self.x_opt = trial.copy()

            pop, fitness = new_pop, new_fitness

            # update memory with successful parameters (if any)
            if len(success_F) > 0:
                # compute weighted mean (Lehmer mean for F, arithmetic for CR)
                w = np.array([1.0] * len(success_F))  # uniform weights for simplicity
                # F: Lehmer mean
                if np.sum(w) > 0:
                    F_mean = np.sum(w * np.array(success_F)**2) / np.sum(w * np.array(success_F))
                else:
                    F_mean = 0.5
                CR_mean = np.mean(success_CR)
                mem_F[mem_idx] = F_mean
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % mem_size

            # ---- local search (pattern search) on best solution ----
            # perform every 2 generations or when best stagnates (simple: every 2 gens)
            if gen % 2 == 0 and evals < self.budget:
                x_best = self.x_opt.copy()
                f_best = self.f_opt
                # initial step size: 10% of domain range
                step = (ub - lb) * 0.1
                min_step = 1e-5 * (ub - lb).max()  # tolerance
                # pattern search (coordinate-wise with forward/backward)
                max_iter = min(20 * self.dim, self.budget - evals)  # limit local evals
                for _ in range(max_iter):
                    improved = False
                    for d in range(self.dim):
                        # try positive step
                        x_try = x_best.copy()
                        x_try[d] = np.clip(x_best[d] + step[d], lb[d], ub[d])
                        f_try = func(x_try)
                        evals += 1
                        if f_try < f_best:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            if f_best < self.f_opt:
                                self.f_opt = f_best
                                self.x_opt = x_best.copy()
                            break  # restart coordinate loop after success (greedy)
                        # try negative step
                        x_try[d] = np.clip(x_best[d] - step[d], lb[d], ub[d])
                        f_try = func(x_try)
                        evals += 1
                        if f_try < f_best:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            if f_best < self.f_opt:
                                self.f_opt = f_best
                                self.x_opt = x_best.copy()
                            break
                    if not improved:
                        step *= 0.5  # reduce step size
                    if np.all(step < min_step):
                        break
                # update best in population (optional: replace worst with x_best)
                # but we keep best separately

        return self.f_opt, self.x_opt