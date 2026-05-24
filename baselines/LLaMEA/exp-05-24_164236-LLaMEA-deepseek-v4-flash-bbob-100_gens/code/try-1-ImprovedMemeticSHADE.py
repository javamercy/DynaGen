import numpy as np

class ImprovedMemeticSHADE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)  # reproducibility
        lb, ub = func.bounds.lb, func.bounds.ub
        dim = self.dim
        budget = self.budget

        # population initial size, then linear reduction
        pop_size = max(7, int(4 + 3 * np.log(dim)))
        N_max = pop_size
        N_min = 4
        archive = []
        archive_max = pop_size

        # memory for parameter adaptation
        H = 6
        mem_F = np.full(H, 0.5)
        mem_CR = np.full(H, 0.5)
        mem_idx = 0

        # initialize population
        pop = np.random.uniform(lb, ub, (pop_size, dim))
        fit = np.full(pop_size, np.inf)
        evals = 0
        for i in range(pop_size):
            fit[i] = func(pop[i])
            evals += 1
            if fit[i] < self.f_opt:
                self.f_opt = fit[i]
                self.x_opt = pop[i].copy()

        best_f = self.f_opt
        best_x = self.x_opt.copy()
        last_improve_gen = 0  # generation of last best improvement

        gen = 0
        while evals < budget:
            gen += 1

            # ----- linear population size reduction -----
            if N_max > N_min:
                frac = evals / budget
                new_size = max(N_min, int(round(N_max - frac * (N_max - N_min))))
                if new_size < pop_size:
                    sorted_idx = np.argsort(fit)
                    pop = pop[sorted_idx][:new_size]
                    fit = fit[sorted_idx][:new_size]
                    if len(archive) > archive_max:
                        archive = archive[:archive_max]
                    pop_size = new_size

            # ----- DE generation with SHADE adaptation -----
            S_F = []
            S_CR = []
            delta_f = []  # improvement magnitudes

            for i in range(pop_size):
                if evals >= budget:
                    break

                # parameter generation
                k = np.random.randint(H)
                # F: Cauchy with location mem_F[k] and scale 0.1, truncated to [0.1, 1]
                F = np.clip(mem_F[k] + 0.1 * np.random.standard_cauchy(), 0.1, 1.0)
                # CR: Normal with mean mem_CR[k] and std 0.1, truncated to [0, 1]
                CR = np.clip(mem_CR[k] + 0.1 * np.random.randn(), 0.0, 1.0)

                # select pbest from top 10%
                p = 0.1
                pbest_size = max(1, int(p * pop_size))
                sorted_idx = np.argsort(fit)
                pbest = pop[np.random.choice(sorted_idx[:pbest_size])]

                # select r1 from population (exclude i)
                indices = list(range(pop_size))
                indices.remove(i)
                r1 = np.random.choice(indices)

                # select r2 from pop+archive (exclude i and r1)
                candidates = list(range(pop_size))
                if len(archive) > 0:
                    candidates += list(range(pop_size, pop_size + len(archive)))
                candidates.remove(i)
                candidates.remove(r1)
                r2 = np.random.choice(candidates)
                if r2 >= pop_size:
                    x_r2 = archive[r2 - pop_size]
                else:
                    x_r2 = pop[r2]

                # mutation: current-to-pbest/1
                mutant = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - x_r2)

                # binomial crossover
                j_rand = np.random.randint(dim)
                trial = np.array([mutant[j] if (np.random.rand() < CR or j == j_rand)
                                  else pop[i][j] for j in range(dim)])
                trial = np.clip(trial, lb, ub)

                # evaluation
                f_trial = func(trial)
                evals += 1

                # selection
                if f_trial <= fit[i]:
                    # archive old vector (if space)
                    if len(archive) < archive_max:
                        archive.append(pop[i].copy())
                    else:
                        archive[np.random.randint(archive_max)] = pop[i].copy()
                    # replace
                    pop[i] = trial
                    fit[i] = f_trial
                    S_F.append(F)
                    S_CR.append(CR)
                    delta_f.append(abs(f_trial - fit[i]))
                    if f_trial < best_f:
                        best_f = f_trial
                        best_x = trial.copy()
                        self.f_opt = best_f
                        self.x_opt = best_x.copy()
                        last_improve_gen = gen

            # update memory with successful parameters (weighted)
            if len(S_F) > 0:
                w = np.array(delta_f)
                w = w / max(w) if max(w) > 0 else np.ones_like(w)
                # Lehmer mean for F
                num = np.sum(w * np.array(S_F)**2)
                den = np.sum(w * np.array(S_F))
                F_mean = num / den if den > 0 else 0.5
                # arithmetic mean for CR
                CR_mean = np.average(S_CR, weights=w)
                mem_F[mem_idx] = F_mean
                mem_CR[mem_idx] = CR_mean
                mem_idx = (mem_idx + 1) % H

            # ----- local search (complete polling pattern search) on best -----
            # trigger every 5 generations or if no improvement for 10 generations
            if (gen % 5 == 0 or (gen - last_improve_gen) > 10) and evals < budget:
                x_best = best_x.copy()
                f_best = best_f
                step = (ub - lb) * 0.2
                min_step = 1e-4 * (ub - lb).max()
                max_evals = min(200 * dim, budget - evals)
                local_evals = 0
                while local_evals < max_evals:
                    improved = False
                    order = np.random.permutation(dim)
                    for d in order:
                        # positive step
                        x_try = x_best.copy()
                        x_try[d] = np.clip(x_best[d] + step[d], lb[d], ub[d])
                        f_try = func(x_try)
                        evals += 1
                        local_evals += 1
                        if f_try < f_best:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            break
                        # negative step
                        x_try[d] = np.clip(x_best[d] - step[d], lb[d], ub[d])
                        f_try = func(x_try)
                        evals += 1
                        local_evals += 1
                        if f_try < f_best:
                            x_best = x_try
                            f_best = f_try
                            improved = True
                            break
                    if improved:
                        if f_best < best_f:
                            best_f = f_best
                            best_x = x_best.copy()
                            self.f_opt = f_best
                            self.x_opt = x_best.copy()
                            last_improve_gen = gen
                    else:
                        step *= 0.5
                    if np.all(step < min_step) or evals >= budget:
                        break

        return self.f_opt, self.x_opt