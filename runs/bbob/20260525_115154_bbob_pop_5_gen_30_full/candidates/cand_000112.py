import numpy as np
from numpy.random import RandomState

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size
        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        # initial population
        pop = rng.uniform(lb, ub, size=(pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # archive
        archive = []
        max_archive = pop_size

        # JADE parameters
        mean_F = 0.5
        mean_CR = 0.5
        c = 0.1
        p_best = 0.1

        # stagnation
        max_stag = max(1, budget // (2 * pop_size))
        stag_gen = 0
        prev_best = best_val

        # diversity threshold
        diversity_threshold = 1e-3

        while evals < budget:
            # generate F and CR
            F = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_F, 0, 1)
            CR = np.clip(rng.standard_cauchy(pop_size) * 0.1 + mean_CR, 0, 1)
            if np.std(F) < 1e-10:
                F += rng.uniform(-0.1, 0.1, pop_size)
                F = np.clip(F, 0, 1)
            if np.std(CR) < 1e-10:
                CR += rng.uniform(-0.1, 0.1, pop_size)
                CR = np.clip(CR, 0, 1)

            # select pbest indices
            sorted_idx = np.argsort(fitness)
            n_pbest = max(1, int(p_best * pop_size))
            pbest_idx = sorted_idx[:n_pbest]

            # combine population and archive for mutation
            combined = list(pop) + archive
            if len(combined) < 2:
                combined = list(pop)

            succ_F = []
            succ_CR = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                # mutation: current-to-pbest/1 with archive
                pbest_i = pbest_idx[rng.randint(n_pbest)]
                # select two distinct indices from combined excluding i if in pop
                indices = [j for j in range(len(combined)) if j != i and j < pop_size]
                if len(indices) < 2:
                    indices = [j for j in range(len(combined)) if j != i]
                if len(indices) < 2:
                    continue
                r1, r2 = rng.choice(indices, size=2, replace=False)
                # ensure indices are within combined bounds
                if r1 >= pop_size:
                    r1 = r1 % pop_size
                if r2 >= pop_size:
                    r2 = r2 % pop_size
                mutant = pop[i] + F[i] * (pop[pbest_i] - pop[i]) + F[i] * (combined[r1] - combined[r2])
                mutant = np.clip(mutant, lb, ub)

                # crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR[i] or j == j_rand:
                        trial[j] = mutant[j]
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    # add old solution to archive
                    archive.append(pop[i].copy())
                    if len(archive) > max_archive:
                        archive.pop(0)
                    # update population
                    fitness[i] = trial_fit
                    pop[i] = trial
                    succ_F.append(F[i])
                    succ_CR.append(CR[i])
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # update mean_F and mean_CR
            if len(succ_F) > 0:
                mean_F = (1 - c) * mean_F + c * (np.sum(np.square(succ_F)) / np.sum(succ_F))
                mean_CR = (1 - c) * mean_CR + c * np.mean(succ_CR)

            # check diversity
            if best_val == np.inf:
                diversity = 0
            else:
                f_min = np.min(fitness[fitness < np.inf])
                f_max = np.max(fitness[fitness < np.inf])
                if f_max - f_min > 1e-12:
                    diversity = (f_max - f_min) / (abs(f_max) + 1e-12)
                else:
                    diversity = 0

            # stagnation or diversity trigger
            if best_val < prev_best:
                stag_gen = 0
                prev_best = best_val
            else:
                stag_gen += 1

            if (stag_gen >= max_stag or diversity < diversity_threshold) and evals < budget:
                # restart
                new_pop = np.empty((pop_size, dim))
                new_pop[0] = best_x
                std_pop = np.std(pop, axis=0)
                std_pop = np.maximum(std_pop, 1e-10 * (ub - lb))
                scale = 0.2
                for i in range(1, pop_size):
                    noise = rng.normal(0, std_pop * scale, dim) * 0.5 + rng.uniform(-1, 1, dim) * std_pop * scale * 0.5
                    new_pop[i] = best_x + noise
                    new_pop[i] = np.clip(new_pop[i], lb, ub)
                pop = new_pop
                fitness = np.full(pop_size, np.inf)
                fitness[0] = best_val
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                # reset archive
                archive = []
                stag_gen = 0
                prev_best = best_val

                # local refinement with Nelder-Mead (simplex)
                if evals < budget:
                    # allocate up to 5% of remaining budget
                    local_budget = max(1, int(0.05 * (budget - evals)))
                    simplex = [best_x.copy()]
                    for j in range(dim):
                        perturb = best_x.copy()
                        step = max(0.1 * (ub[j] - lb[j]), 1e-10)
                        perturb[j] = np.clip(perturb[j] + step, lb[j], ub[j])
                        simplex.append(perturb)
                    simplex_fitness = []
                    for s in simplex:
                        if evals >= budget:
                            break
                        f = func(s)
                        evals += 1
                        simplex_fitness.append(f)
                        if f < best_val:
                            best_val = f
                            best_x = s.copy()
                            report_best(best_val, best_x)
                    # simple Nelder-Mead iteration
                    for _ in range(local_budget):
                        if evals >= budget:
                            break
                        # order simplex by fitness
                        order = np.argsort(simplex_fitness)
                        simplex = [simplex[i] for i in order]
                        simplex_fitness = [simplex_fitness[i] for i in order]
                        # centroid of all but worst
                        centroid = np.mean(simplex[:-1], axis=0)
                        worst = simplex[-1]
                        # reflection
                        xr = centroid + (centroid - worst)
                        xr = np.clip(xr, lb, ub)
                        fr = func(xr)
                        evals += 1
                        if simplex_fitness[0] <= fr < simplex_fitness[-2]:
                            # accept reflection
                            simplex[-1] = xr
                            simplex_fitness[-1] = fr
                        elif fr < simplex_fitness[0]:
                            # expansion
                            xe = centroid + 2 * (centroid - worst)
                            xe = np.clip(xe, lb, ub)
                            fe = func(xe)
                            evals += 1
                            if fe < fr:
                                simplex[-1] = xe
                                simplex_fitness[-1] = fe
                            else:
                                simplex[-1] = xr
                                simplex_fitness[-1] = fr
                        elif fr >= simplex_fitness[-2]:
                            # contraction
                            xc = centroid + 0.5 * (worst - centroid)
                            xc = np.clip(xc, lb, ub)
                            fc = func(xc)
                            evals += 1
                            if fc < simplex_fitness[-1]:
                                simplex[-1] = xc
                                simplex_fitness[-1] = fc
                            else:
                                # shrink
                                for j in range(1, len(simplex)):
                                    simplex[j] = simplex[0] + 0.5 * (simplex[j] - simplex[0])
                                    simplex[j] = np.clip(simplex[j], lb, ub)
                                    ff = func(simplex[j])
                                    evals += 1
                                    simplex_fitness[j] = ff
                                    if ff < best_val:
                                        best_val = ff
                                        best_x = simplex[j].copy()
                                        report_best(best_val, best_x)
                        # update best
                        if simplex_fitness[0] < best_val:
                            best_val = simplex_fitness[0]
                            best_x = simplex[0].copy()
                            report_best(best_val, best_x)
                        # avoid extra evaluations if budget is near
                        if evals >= budget:
                            break

        return best_val, best_x