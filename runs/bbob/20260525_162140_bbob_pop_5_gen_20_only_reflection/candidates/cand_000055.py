import numpy as np

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed

    def __call__(self, func):
        rng = np.random.RandomState(self.seed)
        lb = func.bounds.lb
        ub = func.bounds.ub
        dim = self.dim
        budget = self.budget

        NP = min(10 * dim, max(4, budget // 2 - 1))
        if budget < NP:
            NP = budget

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fit = np.full(NP, np.inf)
        best_val = np.inf
        best_x = None
        for i in range(NP):
            val = func(pop[i])
            fit[i] = val
            if val < best_val:
                best_val = val
                best_x = pop[i].copy()
                report_best(best_val, best_x)
        func_evals = NP

        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.5)
        k = 0
        archive = []
        archive_max = NP

        stall = 0
        max_stall = max(10 * dim, int(0.1 * budget))

        while func_evals < budget:
            # restart condition
            if stall >= max_stall and budget - func_evals > 2 * NP:
                remaining = budget - func_evals
                new_NP = min(10 * dim, max(4, remaining // 2 - 1))
                if new_NP < 2:
                    break
                new_pop = np.zeros((new_NP, dim))
                new_fit = np.full(new_NP, np.inf)
                new_pop[0] = best_x
                new_fit[0] = best_val
                archive.clear()
                for i in range(1, new_NP):
                    if func_evals >= budget:
                        break
                    new_pop[i] = lb + (ub - lb) * rng.rand(dim)
                    val = func(new_pop[i])
                    new_fit[i] = val
                    func_evals += 1
                    if val < best_val:
                        best_val = val
                        best_x = new_pop[i].copy()
                        report_best(best_val, best_x)
                pop = new_pop
                fit = new_fit
                NP = new_NP
                stall = 0
                continue

            # generate offspring
            for i in range(NP):
                if func_evals >= budget:
                    break
                # select pbest
                p = 0.2
                pbest_size = max(1, int(p * NP))
                sorted_idx = np.argsort(fit)
                pbest_idx = sorted_idx[:pbest_size]
                pbest = pop[pbest_idx[rng.randint(pbest_size)]]
                # select two random distinct individuals from population ∪ archive
                pool_indices = list(range(NP)) + list(range(len(archive)))
                # remove i from pool if it's in population
                # Note: indices for archive are offset by NP
                pool = [pop[j] if j < NP else archive[j - NP] for j in pool_indices]
                # ensure we don't select the same as current i
                candidate_indices = [j for j in range(len(pool)) if not (j < NP and j == i)]
                if len(candidate_indices) < 2:
                    continue
                selected = rng.choice(candidate_indices, 2, replace=False)
                a, b = pool[selected[0]], pool[selected[1]]
                # sample F and CR
                r = rng.randint(H)
                F = np.clip(rng.standard_cauchy() * 0.1 + M_F[r], 0, 1)
                CR = np.clip(rng.normal(M_CR[r], 0.1), 0, 1)
                # current-to-pbest/1
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)
                # binomial crossover
                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)
                val = func(trial)
                func_evals += 1
                if val < fit[i]:
                    # add parent to archive
                    if len(archive) >= archive_max:
                        archive.pop(rng.randint(len(archive)))
                    archive.append(pop[i].copy())
                    pop[i] = trial
                    fit[i] = val
                    if val < best_val:
                        best_val = val
                        best_x = trial.copy()
                        report_best(best_val, best_x)
                        stall = 0
                    # update memory (FIFO)
                    M_F = np.roll(M_F, -1)
                    M_F[-1] = F
                    M_CR = np.roll(M_CR, -1)
                    M_CR[-1] = CR
                else:
                    stall += 1

        return best_val, best_x