import numpy as np
from scipy.stats import qmc  # for Sobol sequence


class lshade_ensemble_local:
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

        # budget allocation
        local_budget = max(10 * dim, int(0.15 * budget))
        main_budget = budget - local_budget
        if main_budget < 20:
            # fallback to random search
            for _ in range(budget):
                x = np.random.uniform(lb, ub)
                f = func(x)
                if f < self.best_f:
                    self.best_f = f
                    self.best_x = x.copy()
            return self.best_f, self.best_x

        # ---------- initialization ----------
        NP_init = max(10, min(200, 20 * int(np.log(dim)) if dim > 1 else 20))
        NP = NP_init

        # Sobol sequence initialization
        sobol = qmc.Sobol(d=dim, scramble=True)
        samples = sobol.random(NP)
        pop = lb + samples * (ub - lb)  # scale to bounds
        fitness = np.array([func(x) for x in pop])
        fevals = NP

        best_idx = np.argmin(fitness)
        self.best_f = fitness[best_idx]
        self.best_x = pop[best_idx].copy()

        archive = np.empty((0, dim))
        max_archive = NP
        H = 30
        M_CR = 0.5 * np.ones(H)
        M_F = 0.5 * np.ones(H)
        mem_idx = 0

        # parameters for ensemble of two strategies
        prob_strat = 0.5  # probability to use current-to-pbest; else current-to-rand
        S_prob = []

        # stagnation detection
        no_improve_evals = 0
        last_best_f = self.best_f

        # ---------- main DE loop ----------
        while fevals < main_budget:
            remaining = main_budget - fevals
            # linear population reduction
            NP_new = max(4, int(4 + (NP_init - 4) * (remaining / main_budget)))
            if NP_new < NP:
                sorted_idx = np.argsort(fitness)
                pop = pop[sorted_idx[:NP_new]]
                fitness = fitness[sorted_idx[:NP_new]]
                NP = NP_new
                if len(archive) > NP:
                    np.random.shuffle(archive)
                    archive = archive[:NP]
                max_archive = NP

            # pbest ratio (jSO style)
            ratio = 0.25 - 0.20 * (remaining / main_budget)
            p = max(0.05, min(0.25, ratio))
            pbest_num = max(1, int(p * NP))
            sorted_idx = np.argsort(fitness)
            pbest_pool = sorted_idx[:pbest_num]

            S_CR = []
            S_F = []
            S_df = []
            # ensemble success counters
            n_succ_pbest = 0
            n_succ_rand = 0
            n_attempt_pbest = 0
            n_attempt_rand = 0

            new_pop = pop.copy()
            new_fitness = fitness.copy()

            for i in range(NP):
                r = np.random.randint(H)
                # sample CR and F (Cauchy)
                CR = np.random.standard_cauchy() * 0.1 + M_CR[r]
                CR = max(0., min(1., CR))
                F = np.random.standard_cauchy() * 0.1 + M_F[r]
                while F <= 0.:
                    F = np.random.standard_cauchy() * 0.1 + M_F[r]
                F = min(F, 1.)

                # decide strategy
                use_pbest = np.random.rand() < prob_strat

                if use_pbest:
                    n_attempt_pbest += 1
                    # current-to-pbest/1 with archive
                    pbest = pop[np.random.choice(pbest_pool)]
                    r1 = np.random.randint(NP)
                    while r1 == i:
                        r1 = np.random.randint(NP)
                    combined = np.vstack((pop, archive))
                    while True:
                        idx = np.random.randint(len(combined))
                        if idx < NP:
                            if idx != i and idx != r1:
                                break
                        else:
                            break  # archive index always different from i
                    r2 = combined[idx]
                    v = pop[i] + F * (pbest - pop[i]) + F * (pop[r1] - r2)
                else:
                    n_attempt_rand += 1
                    # current-to-rand/1 (no archive, better diversity)
                    r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    while r1 == i or r2 == i or r3 == i:
                        r1, r2, r3 = np.random.choice(NP, 3, replace=False)
                    v = pop[i] + F * (pop[r1] - pop[i]) + F * (pop[r2] - pop[r3])

                # binomial crossover
                u = pop[i].copy()
                j_rand = np.random.randint(dim)
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        u[j] = v[j]

                # reflected boundary handling
                out_low = u < lb
                out_high = u > ub
                u[out_low] = 2 * lb[out_low] - u[out_low]
                u[out_high] = 2 * ub[out_high] - u[out_high]
                u = np.clip(u, lb, ub)

                f_u = func(u)
                fevals += 1

                if f_u <= fitness[i]:
                    S_CR.append(CR)
                    S_F.append(F)
                    delta = abs(fitness[i] - f_u) + 1e-30
                    S_df.append(delta)
                    new_pop[i] = u
                    new_fitness[i] = f_u
                    archive = np.vstack((archive, pop[i]))
                    if len(archive) > max_archive:
                        idx_del = np.random.randint(len(archive))
                        archive = np.delete(archive, idx_del, axis=0)
                    if f_u < self.best_f:
                        self.best_f = f_u
                        self.best_x = u.copy()
                    if use_pbest:
                        n_succ_pbest += 1
                    else:
                        n_succ_rand += 1

                if fevals >= main_budget:
                    break

            pop = new_pop
            fitness = new_fitness

            # update ensemble probability (roulette wheel based on success rate)
            eps = 1e-10
            rate_pbest = n_succ_pbest / max(1, n_attempt_pbest)
            rate_rand = n_succ_rand / max(1, n_attempt_rand)
            total = rate_pbest + rate_rand
            if total > 0:
                prob_strat = rate_pbest / total
                prob_strat = np.clip(prob_strat, 0.1, 0.9)

            # update memory with Lehmer for F and arithmetic for CR
            if S_CR:
                w = np.array(S_df) / np.sum(S_df)
                mean_CR = np.sum(w * np.array(S_CR))
                F_arr = np.array(S_F)
                sum_w = np.sum(w * F_arr)
                sum_w_sq = np.sum(w * F_arr ** 2)
                mean_F = sum_w_sq / sum_w if sum_w > 1e-30 else 0.5
                M_CR[mem_idx] = mean_CR
                M_F[mem_idx] = mean_F
                mem_idx = (mem_idx + 1) % H

            # stagnation check
            if self.best_f < last_best_f:
                last_best_f = self.best_f
                no_improve_evals = 0
            else:
                no_improve_evals += NP

            # restart if stagnation for a significant portion of remaining budget
            if no_improve_evals > 0.1 * remaining and remaining > 50*dim:
                # restart: reinitialize population while keeping best
                NP_restart = NP_init
                # generate new population around best with perturbation and random
                new_pop = np.tile(self.best_x, (NP_restart, 1))
                new_pop += np.random.randn(NP_restart, dim) * (0.05 * (ub - lb))
                new_pop = np.clip(new_pop, lb, ub)
                # add some pure random points
                n_random = max(1, NP_restart // 3)
                new_pop[:n_random] = np.random.uniform(lb, ub, (n_random, dim))
                # evaluate
                new_fitness = np.array([func(x) for x in new_pop])
                fevals += NP_restart
                # replace population
                pop = new_pop
                fitness = new_fitness
                NP = NP_restart
                archive = np.empty((0, dim))
                max_archive = NP
                # reset stagnation
                no_improve_evals = 0
                last_best_f = self.best_f
                # if best found in new pop, update
                best_idx = np.argmin(fitness)
                if fitness[best_idx] < self.best_f:
                    self.best_f = fitness[best_idx]
                    self.best_x = pop[best_idx].copy()

            if fevals >= main_budget:
                break

        # ---------- local search: adaptive Cauchy random walk ----------
        if local_budget > 0:
            x_best = self.best_x.copy()
            f_best = self.best_f
            evals = 0
            step = np.mean(ub - lb) * 0.02  # initial step size
            step_min = 1e-10
            step_max = np.mean(ub - lb) * 0.2
            success_rate = 0.5
            n_success = 0
            n_trials = 0

            while evals < local_budget:
                # generate candidate via Cauchy perturbation
                scale = max(step, step_min)
                noise = np.random.standard_cauchy(size=dim) * scale
                cand = x_best + noise
                cand = np.clip(cand, lb, ub)
                f_cand = func(cand)
                evals += 1
                n_trials += 1

                if f_cand < f_best:
                    x_best, f_best = cand, f_cand
                    n_success += 1
                    step = min(step * 1.2, step_max)
                else:
                    step = max(step * 0.85, step_min)

                # occasionally try coordinate-like moves
                if evals % max(1, local_budget // (10*dim)) == 0 and evals < local_budget:
                    for j in np.random.choice(dim, size=min(3, dim), replace=False):
                        if evals >= local_budget:
                            break
                        cand = x_best.copy()
                        cand[j] = np.random.uniform(lb[j], ub[j])
                        f_cand = func(cand)
                        evals += 1
                        if f_cand < f_best:
                            x_best, f_best = cand, f_cand
                            step = min(step * 1.2, step_max)
                            break
                        else:
                            cand[j] = 2 * x_best[j] - cand[j]  # reflect
                            cand = np.clip(cand, lb, ub)
                            f_cand = func(cand)
                            evals += 1
                            if f_cand < f_best:
                                x_best, f_best = cand, f_cand
                                step = min(step * 1.2, step_max)
                                break

                # compute running success rate and adjust step (optional)
                window = min(50, n_trials)
                if n_trials >= window:
                    success_rate = n_success / window
                    if success_rate < 0.2:
                        step = max(step * 0.9, step_min)
                    elif success_rate > 0.5:
                        step = min(step * 1.1, step_max)
                    # reset counters
                    n_success = 0
                    n_trials = 0

                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

            # final refinement if budget left
            while evals < local_budget:
                scale = max(step, step_min)
                cand = x_best + np.random.randn(dim) * scale
                cand = np.clip(cand, lb, ub)
                f_cand = func(cand)
                evals += 1
                if f_cand < f_best:
                    x_best, f_best = cand, f_cand
                    step = min(step * 1.1, step_max)
                else:
                    step = max(step * 0.9, step_min)
                if f_best < self.best_f:
                    self.best_f = f_best
                    self.best_x = x_best.copy()

        return self.best_f, self.best_x