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
        if NP > budget:
            NP = budget
        NP = max(1, NP)

        pop = lb + (ub - lb) * rng.rand(NP, dim)
        fitness = np.array([func(pop[i]) for i in range(NP)])
        evals = NP

        best_idx = np.argmin(fitness)
        best_val = fitness[best_idx]
        best_x = pop[best_idx].copy()
        report_best(best_val, best_x)

        if NP == 1:
            return best_val, best_x

        archive = []
        archive_max = NP
        H = 5
        M_F = np.full(H, 0.5)
        M_CR = np.full(H, 0.9)
        p = 0

        gen = 0
        while evals < budget:
            # Diversity-based restart
            if gen > 0 and (evals < budget - 2 * NP):
                std_dim = np.std(pop, axis=0)
                diversity = np.mean(std_dim) / np.mean(ub - lb)
                if diversity < 1e-6:
                    remaining = budget - evals
                    # keep best, reinitialize others
                    new_pop = [best_x.copy()]
                    needed = NP - 1
                    if remaining >= needed:
                        new_points = lb + (ub - lb) * rng.rand(needed, dim)
                        for pt in new_points:
                            new_pop.append(pt)
                        pop = np.array(new_pop)
                        for i in range(1, NP):
                            fitness[i] = func(pop[i])
                            evals += 1
                        # reset memory
                        M_F[:] = 0.5
                        M_CR[:] = 0.9
                        p = 0
                        archive = []

            # Sort fitness for pbest
            idx_sorted = np.argsort(fitness)
            top_N = max(1, int(NP * 0.2))
            S_F = []
            S_CR = []
            delta_f = []
            for i in range(NP):
                if evals >= budget:
                    break
                r = rng.randint(H)
                F = M_F[r] + 0.1 * rng.standard_cauchy()
                if F <= 0:
                    F = 0.1
                elif F > 1:
                    F = 1.0
                CR = M_CR[r] + 0.1 * rng.randn()
                CR = np.clip(CR, 0, 1)

                # current-to-pbest/1 with archive
                pbest_idx = rng.randint(top_N)
                pbest = pop[idx_sorted[pbest_idx]]
                # Select r1 from population excluding i and pbest
                candidates_pop = list(range(NP))
                candidates_pop.remove(i)
                if idx_sorted[pbest_idx] in candidates_pop:
                    candidates_pop.remove(idx_sorted[pbest_idx])
                if len(candidates_pop) < 1:
                    candidates_pop = list(range(NP))
                    candidates_pop.remove(i)
                r1_idx = rng.choice(candidates_pop)
                # Select r2 from union of population and archive, excluding i, pbest, r1
                candidates_vectors = []
                for j in range(NP):
                    if j == i or j == idx_sorted[pbest_idx] or j == r1_idx:
                        continue
                    candidates_vectors.append(pop[j])
                for v in archive:
                    candidates_vectors.append(v)
                if len(candidates_vectors) < 1:
                    candidates_vectors = [pop[j] for j in range(NP) if j != i]
                r2_idx = rng.randint(len(candidates_vectors))
                a = pop[r1_idx]
                b = candidates_vectors[r2_idx]
                mutant = pop[i] + F * (pbest - pop[i]) + F * (a - b)

                trial = pop[i].copy()
                j_rand = rng.randint(dim)
                for j in range(dim):
                    if rng.rand() < CR or j == j_rand:
                        trial[j] = mutant[j]
                trial = np.clip(trial, lb, ub)

                f_trial = func(trial)
                evals += 1
                if f_trial < fitness[i]:
                    delta_f.append(fitness[i] - f_trial)
                    S_F.append(F)
                    S_CR.append(CR)
                    # add old individual to archive if not best
                    old = pop[i].copy()
                    if not np.allclose(old, best_x):
                        if len(archive) >= archive_max:
                            del archive[rng.randint(len(archive))]
                        archive.append(old)
                    pop[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_val:
                        best_val = f_trial
                        best_x = trial.copy()
                        report_best(best_val, best_x)

            if len(S_F) > 0:
                w = np.array(delta_f) / (np.sum(delta_f) + 1e-30)
                M_F[p] = np.sum(w * np.array(S_F)**2) / (np.sum(w * np.array(S_F)) + 1e-30)
                M_CR[p] = np.sum(w * np.array(S_CR)**2) / (np.sum(w * np.array(S_CR)) + 1e-30)
                p = (p + 1) % H
            gen += 1

        return best_val, best_x