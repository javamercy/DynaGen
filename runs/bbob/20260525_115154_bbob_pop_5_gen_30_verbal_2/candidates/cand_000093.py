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

        # population size: at least 4*dim, but cap to budget//2 and at least 3
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

        # evaluate initial population
        for i in range(pop_size):
            if evals >= budget:
                break
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        # DE parameters
        CR = 0.9
        F_low, F_high = 0.5, 1.0
        # success memory for CR
        success_CR = [CR] * 20
        # stagnation detection: restart if no improvement for this many generations
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        # main loop
        while evals < budget:
            # sample CR from success memory mean with small std (bounded in [0,1])
            if len(success_CR) > 0:
                cr_mean = np.mean(success_CR)
                cr_std = 0.1
                CR = rng.normal(cr_mean, cr_std)
                CR = np.clip(CR, 0.0, 1.0)
            else:
                CR = 0.9

            for i in range(pop_size):
                if evals >= budget:
                    break
                # select three distinct indices different from i
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, size=3, replace=False)
                a, b, c = ids
                # dithering F per individual
                F = rng.uniform(F_low, F_high)
                # mutant using rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                # clip
                mutant = np.clip(mutant, lb, ub)
                # binomial crossover
                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                # evaluate
                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    # successful update
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                    # update success memory with this CR
                    success_CR.append(CR)
                    if len(success_CR) > 100:
                        success_CR = success_CR[-100:]

            # check for stagnation (after each generation)
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            if gen_no_improve >= gen_max_restart and evals < budget:
                # restart: create diverse population
                new_pop = []
                # half uniform random, half perturbations around best
                half = pop_size // 2
                for _ in range(half):
                    new_pop.append(rng.uniform(lb, ub))
                for _ in range(pop_size - half):
                    # perturbation: best + random vector scaled by range*0.2
                    perturb = rng.uniform(-1, 1, size=dim) * (ub - lb) * 0.2
                    new_point = best_x + perturb
                    new_point = np.clip(new_point, lb, ub)
                    new_pop.append(new_point)
                new_pop = np.array(new_pop)
                pop = new_pop
                # reevaluate fitness
                fitness = np.full(pop_size, np.inf)
                for i in range(pop_size):
                    if evals >= budget:
                        break
                    fitness[i] = func(pop[i])
                    evals += 1
                    if fitness[i] < best_val:
                        best_val = fitness[i]
                        best_x = pop[i].copy()
                        report_best(best_val, best_x)
                # reset stagnation tracking
                prev_best_val = best_val
                gen_no_improve = 0
                # reset success memory to default
                success_CR = [0.9] * 20

        return best_val, best_x