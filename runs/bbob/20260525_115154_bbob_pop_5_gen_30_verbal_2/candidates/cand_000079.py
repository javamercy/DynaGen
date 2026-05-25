import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        lb = func.bounds.lb
        ub = func.bounds.ub

        pop_size_start = max(4 * dim, 5)
        pop_size_end = max(3, dim)
        pop_size = pop_size_start
        pop = rng.uniform(lb, ub, size=(pop_size, dim)).astype(float)
        fitness = np.full(pop_size, np.inf)
        best_val = np.inf
        best_x = None
        evals = 0

        for i in range(pop_size):
            if evals >= budget:
                break
            val = func(pop[i])
            evals += 1
            fitness[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)

        F = 0.8
        CR = 0.9
        archive = []
        prev_best_val = best_val
        gen_no_improve = 0

        while evals < budget:
            progress = evals / budget
            pop_size = int(pop_size_start + (pop_size_end - pop_size_start) * progress)
            pop_size = max(3, pop_size)
            archive_size = pop_size

            for i in range(pop_size):
                if evals >= budget:
                    break

                candidates = [j for j in range(pop_size) if j != i]
                if len(candidates) < 3:
                    continue
                r1, r2, r3 = rng.choice(candidates, 3, replace=False)
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                mutant = np.clip(mutant, lb, ub)

                j_rand = rng.randint(dim)
                trial = pop[i].copy()
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]

                trial_fit = func(trial)
                evals += 1

                if trial_fit < fitness[i]:
                    archive.append(pop[i].copy())
                    if len(archive) > archive_size:
                        archive.pop(rng.randint(len(archive)))
                    fitness[i] = trial_fit
                    pop[i] = trial
                    if trial_fit < best_val:
                        best_val = trial_fit
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if best_val < prev_best_val:
                gen_no_improve = 0
                prev_best_val = best_val
            else:
                gen_no_improve += 1

            remaining_evals = budget - evals
            if remaining_evals > 0:
                threshold_gen = max(1, int(0.1 * remaining_evals / pop_size))
                if gen_no_improve >= threshold_gen and evals < budget:
                    # restart: keep best 30%, reinitialize 70% uniformly
                    sort_idx = np.argsort(fitness)[:pop_size]
                    n_keep = max(1, int(0.3 * pop_size))
                    keep_idx = sort_idx[:n_keep]
                    new_pop = np.empty_like(pop)
                    new_pop[:n_keep] = pop[keep_idx]
                    new_fitness = np.full(pop_size, np.inf)
                    new_fitness[:n_keep] = fitness[keep_idx]
                    for i in range(n_keep, pop_size):
                        if evals >= budget:
                            break
                        new_pop[i] = rng.uniform(lb, ub, dim)
                        val = func(new_pop[i])
                        evals += 1
                        new_fitness[i] = val
                        if val < best_val:
                            best_val = val
                            best_x = new_pop[i].copy()
                            report_best(best_val, best_x)
                    pop = new_pop
                    fitness = new_fitness
                    gen_no_improve = 0
                    archive = []

        return best_val, best_x