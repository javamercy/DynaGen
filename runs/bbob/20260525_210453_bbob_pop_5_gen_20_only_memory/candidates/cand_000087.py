import numpy as np
import random

class Optimizer:
    def __init__(self, budget: int, dim: int, seed: int):
        self.budget = budget
        self.dim = dim
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def __call__(self, func):
        budget = self.budget
        dim = self.dim
        lb = func.bounds.lb
        ub = func.bounds.ub

        # population size
        pop_size = min(10 * dim, max(4, budget // 10))

        # LHS initialization
        pop = np.zeros((pop_size, dim))
        for j in range(dim):
            rand_perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                pop[i, j] = lb[j] + (rand_perm[i] + np.random.uniform()) * (ub[j] - lb[j]) / pop_size
        pop = np.clip(pop, lb, ub)

        pop_f = np.full(pop_size, np.inf)
        best_x = None
        best_f = np.inf
        fcalls = 0

        # initial evaluation
        for i in range(pop_size):
            if fcalls >= budget:
                break
            x = pop[i]
            val = func(x)
            fcalls += 1
            pop_f[i] = val
            if val < best_f:
                best_f = val
                best_x = x.copy()
                report_best(best_f, best_x)

        # archive for inferior solutions (for diversity)
        archive = np.empty((0, dim))
        max_archive = pop_size

        # JADE memory for F and CR
        M_F = 0.5
        M_CR = 0.5

        stagnation = 0
        max_stag = 10

        while fcalls < budget:
            success_F = []
            success_CR = []
            improved = False

            for i in range(pop_size):
                if fcalls >= budget:
                    break

                # generate F and CR
                F = np.random.cauchy(M_F, 0.1)
                while F <= 0:
                    F = np.random.cauchy(M_F, 0.1)
                F = min(F, 1.0)

                CR = np.random.normal(M_CR, 0.1)
                CR = np.clip(CR, 0, 1)

                # choose mutation strategy
                if random.random() < 0.5:
                    # DE/rand/1 with archive
                    candidates = list(range(pop_size)) + list(range(pop_size, pop_size + len(archive)))
                    candidates.remove(i)
                    if len(candidates) < 3:
                        continue
                    idx = random.sample(candidates, 3)
                    r1 = idx[0] if idx[0] < pop_size else archive[idx[0] - pop_size]
                    r2 = idx[1] if idx[1] < pop_size else archive[idx[1] - pop_size]
                    r3 = idx[2] if idx[2] < pop_size else archive[idx[2] - pop_size]
                    mutant = r1 + F * (r2 - r3)
                else:
                    # DE/current-to-best/1
                    candidates = list(range(pop_size))
                    candidates.remove(i)
                    r1, r2 = random.sample(candidates, 2)
                    mutant = pop[i] + F * (best_x - pop[i]) + F * (pop[r1] - pop[r2])
                mutant = np.clip(mutant, lb, ub)

                # binomial crossover
                trial = pop[i].copy()
                j_rand = random.randint(0, dim-1)
                for j in range(dim):
                    if random.random() < CR or j == j_rand:
                        trial[j] = mutant[j]

                val = func(trial)
                fcalls += 1

                if val < pop_f[i]:
                    # success
                    success_F.append(F)
                    success_CR.append(CR)
                    # archive old individual
                    if len(archive) < max_archive:
                        archive = np.vstack([archive, pop[i]])
                    else:
                        # replace random archive member
                        ridx = random.randint(0, max_archive-1)
                        archive[ridx] = pop[i]
                    pop[i] = trial
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = trial.copy()
                        report_best(best_f, best_x)
                        improved = True

            # update memory
            if len(success_F) > 0:
                # Lehmer mean
                sum_F = sum(f**2 for f in success_F)
                sum_F_w = sum(f for f in success_F)
                M_F = sum_F / sum_F_w if sum_F_w > 0 else M_F
                M_CR = np.mean(success_CR) if len(success_CR) > 0 else M_CR

            if improved:
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= max_stag and fcalls < budget:
                # restart: keep best, reinitialize others uniformly
                best_idx = np.argmin(pop_f)
                for i in range(pop_size):
                    if fcalls >= budget:
                        break
                    if i == best_idx:
                        continue
                    new_x = np.random.uniform(lb, ub, dim)
                    new_x = np.clip(new_x, lb, ub)
                    val = func(new_x)
                    fcalls += 1
                    pop[i] = new_x
                    pop_f[i] = val
                    if val < best_f:
                        best_f = val
                        best_x = new_x.copy()
                        report_best(best_f, best_x)
                # reset archive and memory
                archive = np.empty((0, dim))
                M_F = 0.5
                M_CR = 0.5
                stagnation = 0

        return best_f, best_x