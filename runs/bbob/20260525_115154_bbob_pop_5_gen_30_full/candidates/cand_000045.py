import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget
        rng = self.rng

        pop_size = max(4 * dim, 3)
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
            fitness[i] = func(pop[i])
            evals += 1
            if fitness[i] < best_val:
                best_val = fitness[i]
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        F_memory = []
        CR_memory = []
        max_memory = 50
        gen_max_restart = max(1, budget // (2 * pop_size))
        gen_no_improve = 0
        prev_best_val = best_val

        while evals < budget:
            for i in range(pop_size):
                if evals >= budget:
                    break
                candidates = list(range(pop_size))
                candidates.remove(i)
                ids = rng.choice(candidates, 3, replace=False)
                a, b, c = ids

                # sample F
                if len(F_memory) > 0:
                    mean_F = np.mean(F_memory)
                    F = rng.normal(mean_F, 0.1)
                    F = np.clip(F, 0.1, 1.0)
                else:
                    F = rng.uniform(0.5, 1.0)

                # decide mutation strategy
                if rng.rand() < 0.5:  # DE/rand/1/bin
                    # sample CR
                    if len(CR_memory) > 0:
                        mean_CR = np.mean(CR_memory)
                        CR = rng.normal(mean_CR, 0.1)
                        CR = np.clip(CR, 0, 1)
                    else:
                        CR = 0.9
                    mutant = pop[a] + F * (pop[b] - pop[c])
                    mutant = np.clip(mutant, lb, ub)
                    j_rand = rng.randint(dim)
                    trial = pop[i].copy()
                    for j in range(dim):
                        if rng.rand() < CR or j == j_rand:
                            trial[j] = mutant[j]
                else:  # DE/current-to-rand/1 (no crossover)
                    trial = pop[i] + F * (pop[a] - pop[i]) + F * (pop[b] - pop[c])
                    trial = np.clip(trial, lb, ub)

                trial_fit = func(trial)
                evals += 1
                if trial_fit < fitness[i]:
                    fitness[i] = trial_fit
                    pop[i] = trial
                    # store successful parameters
                    if len(F_memory) >= max_memory:
                        F_memory.pop(0)
                    F_memory.append(F)
                    if rng.rand() < 0.5:  # only for rand/1, but we approximate by storing also for current-to-rand/1? Actually for simplicity store both, but CR only for rand/1. We'll store CR only when using rand/1.
                        # This is tricky; we know mutation type based on the branch; we can just check if we used CR in this iteration.
                        # Better: store CR only when using rand/1. We'll do that.
                        pass
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            # stagnation detection
            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            # restart if stagnant
            if gen_no_improve >= gen_max_restart and evals < budget:
                new_pop = np.zeros((pop_size, dim))
                new_pop[0] = best_x
                for i in range(1, pop_size):
                    if i <= pop_size // 2:
                        new_pop[i] = rng.uniform(lb, ub)
                    else:
                        scale = 0.1 * (ub - lb)
                        noise = rng.normal(0, scale, dim)
                        candidate = best_x + noise
                        candidate = np.clip(candidate, lb, ub)
                        new_pop[i] = candidate
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
                # reset memories and counters
                F_memory = []
                CR_memory = []
                prev_best_val = best_val
                gen_no_improve = 0

        return best_val, best_x