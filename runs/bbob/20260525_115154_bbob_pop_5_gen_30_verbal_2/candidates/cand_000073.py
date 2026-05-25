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

        pop_size = max(4 * dim, 3)
        if pop_size > budget // 2:
            pop_size = max(3, budget // 2)
        if pop_size < 3:
            pop_size = 3

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

        archive = []
        max_archive = 2 * pop_size
        mu_F = 0.5
        mu_CR = 0.5
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        while evals < budget:
            F_success = []
            CR_success = []
            for i in range(pop_size):
                if evals >= budget:
                    break
                F_i = rng.standard_cauchy() * 0.1 + mu_F
                F_i = np.clip(F_i, 0.001, 1.0)
                CR_i = rng.standard_cauchy() * 0.1 + mu_CR
                CR_i = np.clip(CR_i, 0, 1)

                sorted_idx = np.argsort(fitness)
                p_best_num = max(1, int(0.2 * pop_size))
                pbest_idx = rng.choice(sorted_idx[:p_best_num])

                candidates_a = [j for j in range(pop_size) if j != i]
                a = rng.choice(candidates_a)

                union_pop_archive = []
                for j in range(pop_size):
                    if j != i:
                        union_pop_archive.append(pop[j])
                union_pop_archive.extend(archive)
                if len(union_pop_archive) == 0:
                    b = pop[rng.choice(candidates_a)]
                else:
                    b = union_pop_archive[rng.randint(len(union_pop_archive))]

                x_i = pop[i]
                x_pbest = pop[pbest_idx]
                x_a = pop[a]
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
                F_sum_f = sum(F_success)
                F_sum_f2 = sum(f**2 for f in F_success)
                mu_F = F_sum_f2 / F_sum_f if F_sum_f > 0 else 0.5
                mu_CR = sum(CR_success) / len(CR_success)  # arithmetic mean
            # else keep previous mu_F, mu_CR

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                new_pop = np.empty((pop_size, dim))
                new_fitness = np.full(pop_size, np.inf)
                new_pop[0] = best_x
                new_fitness[0] = best_val
                spread = np.std(pop, axis=0) if pop_size > 1 else 0.2 * (ub - lb)
                for i in range(1, pop_size):
                    if evals >= budget:
                        break
                    if rng.rand() < 0.3:  # 30% uniform sampling
                        new = rng.uniform(lb, ub, dim)
                    else:
                        if rng.rand() < 0.5:
                            noise = rng.standard_cauchy(dim) * spread
                        else:
                            noise = rng.standard_cauchy(dim) * 0.2 * (ub - lb)
                        new = best_x + noise
                        new = np.clip(new, lb, ub)
                    new_fitness[i] = func(new)
                    evals += 1
                    if new_fitness[i] < best_val:
                        best_val = new_fitness[i]
                        best_x = new.copy()
                        report_best(best_val, best_x)

                # Local search around best_x
                local_search_steps = min(5, budget - evals)
                for _ in range(local_search_steps):
                    if evals >= budget:
                        break
                    step = 0.01 * (ub - lb) * rng.standard_cauchy(dim)
                    trial = best_x + step
                    trial = np.clip(trial, lb, ub)
                    val = func(trial)
                    evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)

                pop = new_pop
                fitness = new_fitness
                archive = []
                mu_F = 0.5
                mu_CR = 0.5
                gen_no_improve = 0

        return best_val, best_x