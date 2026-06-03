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

        pop_size = max(4 * dim, 10)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

        pop = rng.uniform(lb, ub, (pop_size, dim))
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            fit = func(pop[i])
            evals += 1
            fitness[i] = fit
            if fit < best_val:
                best_val = fit
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        archive = []
        max_archive = pop_size * 2
        mu_F = 0.5
        mu_CR = 0.5
        gen_no_improve = 0
        prev_best = best_val
        max_stagnation = max(1, budget // (3 * pop_size))
        strategies = ['current-to-pbest/1', 'rand/1', 'best/1']

        while evals < budget:
            F_success = []
            CR_success = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                F_i = mu_F + 0.5 * rng.standard_cauchy()
                F_i = np.clip(F_i, 0.001, 1.0)
                CR_i = mu_CR + 0.5 * rng.standard_cauchy()
                CR_i = np.clip(CR_i, 0, 1)
                strat = rng.choice(strategies)

                # common selections for all strategies
                sorted_idx = np.argsort(fitness)
                p_best_num = max(1, int(0.2 * pop_size))
                pbest_idx = rng.choice(sorted_idx[:p_best_num])
                x_pbest = pop[pbest_idx]

                candidates_a = [j for j in range(pop_size) if j != i]
                a = rng.choice(candidates_a)
                x_a = pop[a]

                candidates_b = [pop[j] for j in range(pop_size) if j != i] + archive
                if len(candidates_b) == 0:
                    b = pop[rng.choice(candidates_a)]
                else:
                    b = candidates_b[rng.randint(len(candidates_b))]

                x_i = pop[i]
                if strat == 'current-to-pbest/1':
                    mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_a - b)
                elif strat == 'rand/1':
                    # two distinct random indices from pop
                    candidates_rand = [j for j in range(pop_size) if j != i]
                    if len(candidates_rand) >= 2:
                        r1, r2 = rng.choice(candidates_rand, 2, replace=False)
                    else:
                        r1, r2 = candidates_rand[0], candidates_rand[0]
                    mutant = pop[r1] + F_i * (pop[r2] - b)
                elif strat == 'best/1':
                    best_pop_idx = sorted_idx[0]
                    x_best = pop[best_pop_idx]
                    mutant = x_best + F_i * (x_a - b)
                else:
                    mutant = x_i + F_i * (x_pbest - x_i) + F_i * (x_a - b)

                mutant = np.clip(mutant, lb, ub)
                j_rand = rng.randint(dim)
                trial = x_i.copy()
                for j in range(dim):
                    if rng.rand() < CR_i or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    if len(archive) >= max_archive:
                        archive.pop(0)
                    archive.append(pop[i].copy())
                    fitness[i] = trial_fit
                    pop[i] = trial
                    F_success.append(F_i)
                    CR_success.append(CR_i)
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if len(F_success) > 0:
                sum_F = sum(F_success)
                sum_F_sq = sum(f**2 for f in F_success)
                mu_F = sum_F_sq / sum_F if sum_F > 0 else 0.5
                mu_CR = (1 - 0.1) * mu_CR + 0.1 * np.mean(CR_success)

            if best_val < prev_best:
                gen_no_improve = 0
                prev_best = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= max_stagnation and evals < budget:
                new_pop = [best_x]
                new_fitness = [best_val]
                scale = 0.5 * (ub - lb)
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    if rng.rand() < 0.2:
                        new = best_x + rng.standard_cauchy(dim) * scale
                    else:
                        new = rng.uniform(lb, ub, dim)
                    new = np.clip(new, lb, ub)
                    fit = func(new)
                    evals += 1
                    new_fitness.append(fit)
                    new_pop.append(new)
                    if fit < best_val:
                        best_val = fit
                        best_x = new.copy()
                        report_best(best_val, best_x)
                pop = np.array(new_pop)
                fitness = np.array(new_fitness)
                archive = []
                mu_F = 0.5
                mu_CR = 0.5
                gen_no_improve = 0

        return best_val, best_x